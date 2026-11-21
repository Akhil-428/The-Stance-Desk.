# Security policy

## Supported versions

Only the latest release on the `main` branch receives fixes.

## Reporting a vulnerability

Please **do not open a public issue** for a security problem. Use GitHub's private reporting instead:
**Security tab → "Report a vulnerability"** on this repository. Include what you found, how to reproduce it, and the impact you expect. You can expect an acknowledgement within a few days; this is a volunteer-maintained project, so there is no formal response-time guarantee.

## Security design

These choices exist on purpose; reports that bypass them are in scope.

| Area | Measure |
|---|---|
| Model files | BERT is loaded from `safetensors` only. `.bin` / `.pt` checkpoints are refused, because pickle-based formats can run arbitrary code when loaded. `trust_remote_code=False`; loading is offline (`local_files_only=True`). |
| Logistic Regression engine | Stored with `joblib` (pickle). It is loaded only after its SHA-256 matches `models/lr-relational/model_info.json`. **Never place a `model.joblib` from an untrusted source in that folder.** |
| Dataset download | `tools/fetch_fnc1.py` fetches four fixed files from a pinned upstream commit and rejects any file whose SHA-256 differs from `data/raw/fnc1/SHA256SUMS.txt`. Nothing downloaded is executed. |
| Web app | Binds to `localhost` by default, telemetry is off. All user text is HTML-escaped before it is rendered. Input length is capped (`MAX_INPUT_CHARS`). The app makes no network calls at runtime. |
| Dependencies | Libraries are pinned to exact versions in `requirements.txt`; the full tested environment, including the PyTorch build, is frozen in `requirements-lock.txt`. The original 2021 pins (torch 1.9, transformers 4.12.5) carry many published advisories and are deliberately not used. |

## Out of scope

- Misclassifications: the model is a statistical classifier and will be wrong on some inputs. See the limitations in the README.
- Exposing the Streamlit app to the public internet without authentication or rate limiting. It is designed as a local tool.
