# Contributing

Thanks for your interest in The Stance Desk. Bug reports, documentation fixes, tests and well-scoped improvements are welcome.

## Ground rules

- Be kind and constructive. See the [Code of Conduct](CODE_OF_CONDUCT.md).
- Open an issue before starting a large change so the approach can be agreed first.
- Keep pull requests small and focused: one logical change each.
- **Never commit** datasets, model weights, checkpoints, `runs/`, virtual environments or personal paths. `.gitignore` blocks most of them; see [models/README.md](models/README.md) and [data/README.md](data/README.md).
- Do not weaken the safety measures in [SECURITY.md](SECURITY.md) (safetensors-only loading, hash-checked LR file, escaped HTML).
- Reported numbers must come from `outputs/metrics/*.json` written by `evaluation.py`, never typed by hand. If your change alters results, regenerate the metrics and explain the difference in the PR.

## Development setup

```powershell
git clone https://github.com/Akhil-428/The-Stance-Desk.git
cd The-Stance-Desk
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt ruff==0.12.0
```

A CPU build of PyTorch is enough for development and the test suite. `requirements.txt` deliberately does not list `torch` (pip cannot choose the CPU/CUDA build); see its header for the tested version.

## Checks before you open a PR

```powershell
ruff check .                      # lint (configuration in pyproject.toml)
python -m pytest -m "not gpu and not slow"
```

Tests that need the FNC-1 CSVs or model weights are skipped automatically when those files are absent, with the reason printed (`pytest -rs`). If you have the files, run the complete suite with `python -m pytest`.

## Code style

- Python 3.11+ syntax, type hints on public functions, short docstrings that explain *why*.
- Lint rules are in `pyproject.toml` (`ruff check .` must pass). The code base is not auto-formatted, so please match the surrounding style and do not reformat unrelated lines.
- New behaviour needs a test. UI behaviour is tested with Streamlit's `AppTest` (see `tests/test_app.py`).

## Commit messages and pull requests

- Use the imperative mood, a short summary line (about 72 characters), and a body that says what and why when it is not obvious (`Fix off-by-one in split indices`).
- Fill in the pull request template: what changed, how you tested it, and any effect on reported metrics.

## Adding a new model engine

1. Implement a class in `model.py` with a `predict(headline, body) -> PredictionResult` method and register it in `ENGINES`.
2. Add a training script that writes weights to `models/<name>/` plus a `model_info.json`, and metrics through `evaluation.save_metrics` to `outputs/metrics/ours/`.
3. Add tests for loading errors, class mapping and probability normalisation (see `tests/test_model.py`).
4. Document it in the README and `docs/DECISIONS.md`.

## Academic integrity

This project is an independent re-implementation and study of a 2021 course project. If you are a student, do not submit this repository or code from it as your own coursework.
