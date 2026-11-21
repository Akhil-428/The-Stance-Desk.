"""Score the fine-tuned BERT model ONCE on the official FNC-1 competition test set.

Writes outputs/predictions/bert_fnc1_test.csv and outputs/metrics/ours/bert_fnc1.json.
Run after train_bert.py has finished (model selection used validation only).
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
from evaluation import compute_metrics, save_metrics
from preprocessing import bert_encode, load_fnc1
from train_bert import PairDataset, collate


def main() -> None:
    from transformers import AutoTokenizer, BertForSequenceClassification

    device = "cuda" if torch.cuda.is_available() else "cpu"
    info = json.loads((config.BERT_MODEL_DIR / "model_info.json").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(config.BERT_MODEL_DIR, local_files_only=True)
    model = BertForSequenceClassification.from_pretrained(
        config.BERT_MODEL_DIR, local_files_only=True, use_safetensors=True).to(device).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    test = load_fnc1("test")
    enc = bert_encode(tok, test["Headline"], test["articleBody"], info["max_length"])
    ds = PairDataset(enc, test["label"])
    order = np.argsort([len(x) for x in ds.ids])          # length-sorted batches = less padding
    t0 = time.time()
    logits = np.zeros((len(ds), 4), dtype=np.float32)
    dl = DataLoader(torch.utils.data.Subset(ds, order), batch_size=64, collate_fn=collate)
    pos = 0
    with torch.inference_mode():
        for batch in dl:
            b = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                out = model(**b).logits.float().cpu().numpy()
            logits[order[pos:pos + len(out)]] = out
            pos += len(out)
    secs = time.time() - t0
    pred = [id2label[i] for i in logits.argmax(1)]
    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Headline": test["Headline"], "Body ID": test["Body ID"], "Stance": pred}).to_csv(
        config.PREDICTIONS_DIR / "bert_fnc1_test.csv", index=False, encoding="utf-8")
    m = compute_metrics(test["Stance"], pred)
    save_metrics(
        m, config.METRICS_DIR / "ours" / "bert_fnc1.json",
        source="this project (trained and evaluated here)",
        display_name="BERT (bert-base-uncased fine-tuned here)", model_key="bert_fnc1",
        trained_on="FNC-1 train fold (44,973 pairs, grouped split)",
        evaluated_on="official FNC-1 competition test set (25,413 pairs)",
        epochs_completed=info.get("epochs_completed"), optimizer_steps=info.get("optimizer_steps"),
        validation_metrics=info.get("validation_metrics"), inference_seconds=round(secs, 1), device=device,
        note=f"Checkpoint selected on validation macro-F1 after {info.get('epochs_completed')} epochs.",
    )
    print(f"TEST acc {m['accuracy']:.4f} macroF1 {m['macro']['f1']:.4f} FNC {m['fnc']['score']}/{m['fnc']['max']} "
          f"({m['fnc']['relative']:.2%}) in {secs:.0f}s on {device}")


if __name__ == "__main__":
    main()
