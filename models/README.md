# Models

Model weights are **not stored in git**: each `model.safetensors` is about 440 MB, above GitHub's 100 MB per-file limit. Only small metadata files (`model_info.json`) are tracked. This folder is where the app and the scripts expect the weights to live.

```
models/
├── pretrained/bert-base-uncased/   official starting weights (you download these)
├── bert-fnc1/                      BERT fine-tuned on FNC-1 (you train this)
└── lr-relational/                  Logistic Regression engine (you train this)
```

## Provenance and integrity

| Folder | Content | SHA-256 of the weights file |
|---|---|---|
| `pretrained/bert-base-uncased/` | [`google-bert/bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased) at revision `86b5e0934494bd15c9632b12f734a8a67f723594` (Apache-2.0). Files: `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `vocab.txt`. | `model.safetensors`: `68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3` (the value published by Hugging Face) |
| `bert-fnc1/` | Output of `train_bert.py` (437,964,776 bytes). Metadata in `model_info.json`. | `model.safetensors` of the run reported in the README: `2357cc2a518617e29dc2909d7b4c575285b7e92a6c0a48979be1b91b6d0a1494` |
| `lr-relational/` | Output of `train_baselines.py`. The `joblib` file is verified against the hash in `model_info.json` before it is loaded. | `model.joblib`: `f2c62d72e22ccd3d481d36c472e42208bad66d94df69128f2feaf5a101a75069` |

> A re-trained `bert-fnc1` will **not** have the same hash: GPU kernels, library versions and fp16 make training non-bit-reproducible even with a fixed seed. Expect metrics close to, but not identical with, the ones in the README.

## How to get the weights

1. **Pretrained BERT.** Download the five files listed above from the Hugging Face repository at the pinned revision into `models/pretrained/bert-base-uncased/` and compare the `model.safetensors` hash:

   ```powershell
   Get-FileHash models\pretrained\bert-base-uncased\model.safetensors -Algorithm SHA256
   ```

   With the Hugging Face client (installed with `transformers`), run this *before* switching to offline mode (`env.ps1` and `run_app.bat` set `HF_HUB_OFFLINE=1`):

   ```powershell
   python -c "from huggingface_hub import snapshot_download as d; d('google-bert/bert-base-uncased', revision='86b5e0934494bd15c9632b12f734a8a67f723594', local_dir='models/pretrained/bert-base-uncased', allow_patterns=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.txt'])"
   ```

   Always compare the hash afterwards, and do not download any `.bin` file.

2. **Fine-tuned BERT.** Fetch the data (`python tools/fetch_fnc1.py`), then run `python train_bert.py --epochs 3`. About 1 h 48 min on an RTX 3050 6 GB (see `docs/training_log.jsonl`). Use `python train_bert.py --smoke` for a one-minute end-to-end check.

3. **Logistic Regression engine.** `python train_baselines.py` (CPU, about 8 minutes for all baselines).

## Security note

The loader only accepts `safetensors` for BERT and refuses `.bin` / `.pt` checkpoints, because pickle-based checkpoint formats can execute arbitrary code when loaded. The LR engine is stored with `joblib` (pickle); it is only loaded after its SHA-256 matches `model_info.json`. Never place a `model.joblib` from an untrusted source in `models/lr-relational/`.
