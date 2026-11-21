"""Fine-tune bert-base-uncased on FNC-1 (4-class stance) on a single GPU.

Architecture as in the original notebook: BertForSequenceClassification,
num_labels=4, max 512 tokens, AdamW, LR 5e-5, 500 warm-up steps, linear decay,
weight decay 0.01, effective batch 16. Changes (docs/DECISIONS.md):
* fp16 mixed precision + micro-batch 8 x 2 accumulation (6 GB VRAM);
* sentence-pair encoding with segment ids (D1);
* model selection on validation macro-F1 (D3), grouped split (K2);
* --time-budget-min stops training gracefully so the deadline is respected;
* the model is saved as safetensors after each validation pass that improves.

Usage:
  python train_bert.py --smoke              # ~60 steps + tiny eval + save + reload
  python train_bert.py --epochs 1 --time-budget-min 90
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import config
from evaluation import compute_metrics
from preprocessing import bert_encode, load_fnc1, make_split


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PairDataset(Dataset):
    def __init__(self, enc, labels):
        self.ids = [np.asarray(x, dtype=np.int32) for x in enc["input_ids"]]
        self.tt = [np.asarray(x, dtype=np.int8) for x in enc["token_type_ids"]]
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.ids[i], self.tt[i], self.labels[i]


def collate(batch):
    n = max(len(b[0]) for b in batch)
    ids = torch.zeros(len(batch), n, dtype=torch.long)
    tt = torch.zeros(len(batch), n, dtype=torch.long)
    am = torch.zeros(len(batch), n, dtype=torch.long)
    for i, (a, t, _) in enumerate(batch):
        ids[i, : len(a)] = torch.from_numpy(a.astype(np.int64))
        tt[i, : len(t)] = torch.from_numpy(t.astype(np.int64))
        am[i, : len(a)] = 1
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return {"input_ids": ids, "token_type_ids": tt, "attention_mask": am, "labels": labels}


@torch.inference_mode()
def predict_logits(model, loader, device, fp16=True, limit=None):
    model.eval()
    out, seen = [], 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != "labels"}
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and device == "cuda"):
            out.append(model(**batch).logits.float().cpu())
        seen += len(batch["input_ids"])
        if limit and seen >= limit:
            break
    model.train()
    return torch.cat(out).numpy()


def save_model(model, tokenizer, out_dir: Path, info: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    for stray in list(out_dir.glob("*.bin")):
        stray.unlink()  # never keep pickle checkpoints
    (out_dir / "model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=float, default=config.BERT["epochs"])
    ap.add_argument("--max-length", type=int, default=config.BERT["max_length"])
    ap.add_argument("--time-budget-min", type=float, default=None)
    ap.add_argument("--evals-per-epoch", type=int, default=1)
    ap.add_argument("--out", type=Path, default=config.BERT_MODEL_DIR)
    args = ap.parse_args()

    from transformers import AutoTokenizer, BertForSequenceClassification

    hp = dict(config.BERT, max_length=args.max_length)
    set_seed(config.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = config.PROJECT_DIR / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S" + ("-smoke" if args.smoke else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir = run_dir / "model" if args.smoke else args.out
    log = open(run_dir / "train_log.jsonl", "a", encoding="utf-8")

    def logj(**kw):
        kw["t"] = round(time.time() - t_start, 1)
        log.write(json.dumps(kw) + "\n"); log.flush()
        print(json.dumps(kw), flush=True)

    t_start = time.time()
    df = load_fnc1("train")
    split = make_split(df)
    tr, va = df.iloc[split.train_idx], df.iloc[split.val_idx]
    if args.smoke:
        tr, va = tr.sample(2000, random_state=config.SEED), va.sample(400, random_state=config.SEED)

    tok = AutoTokenizer.from_pretrained(config.PRETRAINED_BERT_DIR, local_files_only=True)
    enc_tr = bert_encode(tok, tr["Headline"], tr["articleBody"], hp["max_length"])
    enc_va = bert_encode(tok, va["Headline"], va["articleBody"], hp["max_length"])
    ds_tr, ds_va = PairDataset(enc_tr, tr["label"]), PairDataset(enc_va, va["label"])
    logj(event="data", n_train=len(ds_tr), n_val=len(ds_va),
         mean_len=float(np.mean([len(x) for x in ds_tr.ids])),
         frac_truncated=float(np.mean([len(x) >= hp["max_length"] for x in ds_tr.ids])))

    g = torch.Generator().manual_seed(config.SEED)
    dl_tr = DataLoader(ds_tr, batch_size=hp["micro_batch_size"], shuffle=True, generator=g,
                       collate_fn=collate, pin_memory=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=hp["eval_batch_size"], shuffle=False, collate_fn=collate, pin_memory=True)

    model = BertForSequenceClassification.from_pretrained(
        config.PRETRAINED_BERT_DIR, num_labels=4, id2label=config.ID2LABEL, label2id=config.LABEL2ID,
        local_files_only=True, use_safetensors=True,
    ).to(device)
    model.train()

    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": hp["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=hp["learning_rate"])
    accum = hp["grad_accum_steps"]
    steps_per_epoch = math.ceil(len(dl_tr) / accum)
    total_steps = 60 if args.smoke else max(1, int(steps_per_epoch * args.epochs))
    warmup = min(hp["warmup_steps"], max(1, total_steps // 10)) if args.smoke else min(hp["warmup_steps"], total_steps // 5)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=hp["fp16"] and device == "cuda")
    eval_every = max(1, steps_per_epoch // args.evals_per_epoch) if not args.smoke else total_steps
    logj(event="plan", device=device, total_steps=total_steps, steps_per_epoch=steps_per_epoch,
         warmup=warmup, eval_every=eval_every, hp=hp)

    best_f1, step, micro, t_train, running = -1.0, 0, 0, time.time(), 0.0
    stop_reason = "completed"

    def evaluate_and_maybe_save(tag):
        nonlocal best_f1
        t0 = time.time()
        logits = predict_logits(model, dl_va, device, hp["fp16"])
        pred = [config.ID2LABEL[i] for i in logits.argmax(1)]
        gold = [config.ID2LABEL[i] for i in va["label"]]
        m = compute_metrics(gold, pred)
        improved = m["macro"]["f1"] > best_f1
        logj(event="eval", tag=tag, step=step, val_acc=m["accuracy"], val_macro_f1=m["macro"]["f1"],
             val_fnc_rel=m["fnc"]["relative"], eval_sec=round(time.time() - t0, 1), improved=improved)
        if improved:
            best_f1 = m["macro"]["f1"]
            save_model(model, tok, out_dir, {
                "display_name": "BERT (bert-base-uncased fine-tuned on FNC-1)",
                "base_model": f"{config.PRETRAINED_BERT_SOURCE}@{config.PRETRAINED_BERT_REVISION}",
                "trained_by": "this project (train_bert.py)", "smoke_test": args.smoke,
                "max_length": hp["max_length"], "hyperparameters": hp, "optimizer_steps": step,
                "planned_steps": total_steps, "epochs_completed": round(step / steps_per_epoch, 3),
                "selection": "best validation macro-F1", "validation_metrics": m,
                "split": "StratifiedGroupKFold by Body ID, seed 42 (data/splits/train_val_split.json)",
                "input_format": "sentence pair: [CLS] headline [SEP] body [SEP] with token_type_ids",
                "created": datetime.now().isoformat(timespec="seconds"),
            })
            logj(event="saved", dir=str(out_dir), step=step)

    done = False
    while not done:
        for batch in dl_tr:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=hp["fp16"] and device == "cuda"):
                loss = model(**batch).loss / accum
            scaler.scale(loss).backward()
            running += loss.item()
            micro += 1
            if micro % accum:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            step += 1
            if step % 50 == 0 or (args.smoke and step % 10 == 0):
                el = time.time() - t_train
                logj(event="train", step=step, loss=round(running / (50 if not args.smoke else 10), 4),
                     lr=sched.get_last_lr()[0], sec_per_step=round(el / step, 3),
                     eta_min=round(el / step * (total_steps - step) / 60, 1),
                     vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else 0)
                running = 0.0
            if step % eval_every == 0 or step >= total_steps:
                evaluate_and_maybe_save(f"step{step}")
            if step >= total_steps:
                done = True; break
            if args.time_budget_min and (time.time() - t_start) / 60 > args.time_budget_min:
                stop_reason = "time budget reached"
                evaluate_and_maybe_save(f"budget-step{step}")
                done = True; break

    logj(event="done", reason=stop_reason, steps=step, best_val_macro_f1=best_f1,
         train_min=round((time.time() - t_train) / 60, 1))
    if args.smoke:  # verify the full path: reload from disk and predict
        from model import BertStancePredictor
        p = BertStancePredictor(out_dir, device="cpu")
        r = p.predict(va["Headline"].iloc[0], va["articleBody"].iloc[0])
        logj(event="reload_ok", label=r.label, probs=r.probabilities, truncated=r.truncated)


if __name__ == "__main__":
    main()
