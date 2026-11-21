"""Recompute metrics for the ORIGINAL repository's committed prediction files.

Reads the audited clone (read-only), copies the CSVs into data/historical/ with
SHA-256 checks, and writes outputs/metrics/historical/*.json.

These are the authors' self-reported, *unreproduced* results. The file names in
the original repository are wrong for Logistic Regression (audit F.5): the file
called "..._with_CountVectorizer.csv" was produced with TF-IDF features and vice
versa. The two BernoulliNB files are byte-identical (one model).
"""
from __future__ import annotations

import hashlib
import json
import shutil

import config
from evaluation import save_metrics, score_prediction_files

HIST_METRICS = config.METRICS_DIR / "historical"

# file in repo -> (display name, actual features, note)
FILES = {
    "BERT.csv": ("BERT (authors' checkpoint)", "bert-base-uncased fine-tuned (checkpoint not public)", ""),
    "LogisticRegression_with_CountVectorizer.csv": (
        "Logistic Regression + TF-IDF", "TF-IDF (file is mislabelled 'CountVectorizer' in the original repo)",
        "Original OvR argmax bug: uncertain rows forced to 'unrelated'.",
    ),
    "LogisticRegression_with_TfidfVectorizer.csv": (
        "Logistic Regression + Count", "Count (file is mislabelled 'TfidfVectorizer' in the original repo)",
        "Original OvR argmax bug: uncertain rows forced to 'unrelated'.",
    ),
    "BernoulliNB_with_CountVectorizer.csv": (
        "Bernoulli Naive Bayes", "Binary bag of words (Count and TF-IDF files are byte-identical)",
        "Original OvR argmax bug: uncertain rows forced to 'unrelated'.",
    ),
}


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Recompute the historical metrics.

    Needs the original repository's prediction CSVs, either already copied to
    data/historical/ or readable from a local clone of
    zeyongj/Fake-News-Stance-Detection (set FNC_ORIGINAL_REPO to its path,
    commit config.ORIGINAL_REPO_COMMIT). The clone is only ever read.
    """
    names = list(FILES) + ["BernoulliNB_with_TfidfVectorizer.csv", "competition_test_stances.csv"]
    src_dir = config.ORIGINAL_REPO / "output" if config.ORIGINAL_REPO else None
    config.HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in names:
        src, dst = (src_dir / name if src_dir else None), config.HISTORICAL_DIR / name
        if not dst.exists():
            if src is None or not src.exists():
                raise SystemExit(
                    f"Missing {dst}. Clone https://github.com/zeyongj/Fake-News-Stance-Detection "
                    f"(commit {config.ORIGINAL_REPO_COMMIT}) and set FNC_ORIGINAL_REPO to its path."
                )
            shutil.copyfile(src, dst)  # read from the clone, write only into this workspace
        if src is not None and src.exists() and sha256(src) != sha256(dst):
            raise RuntimeError(f"Hash mismatch for {name}")
        manifest[name] = sha256(dst)
    (config.HISTORICAL_DIR / "SHA256SUMS.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    gold = config.HISTORICAL_DIR / "competition_test_stances.csv"
    for fname, (display, features, note) in FILES.items():
        m = score_prediction_files(gold, config.HISTORICAL_DIR / fname)
        stem = fname.removesuffix(".csv")
        save_metrics(
            m, HIST_METRICS / f"{stem}.json",
            source="original repository (historical, self-reported, not reproduced)",
            display_name=display, features=features, note=note, file=fname, sha256=manifest[fname],
            evaluated_on="official FNC-1 competition test set (25,413 pairs)",
        )
        print(f"{display:32s} acc={m['accuracy']:.6f} macroF1={m['macro']['f1']:.4f} "
              f"FNC={m['fnc']['score']}/{m['fnc']['max']} ({m['fnc']['relative']:.2%})")
    print("BNB files identical:", manifest["BernoulliNB_with_CountVectorizer.csv"] == manifest["BernoulliNB_with_TfidfVectorizer.csv"])


if __name__ == "__main__":
    main()
