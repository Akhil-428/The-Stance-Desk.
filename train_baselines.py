"""Train and evaluate the Logistic Regression / Naive Bayes models on CPU.

1. Original-style baselines (bug-fixed re-implementation of notebook cells 17-28):
   bag of words over "headline [SEP] body", unigrams+bigrams, English stop words,
   OneVsRest(LogisticRegression) with Count and with TF-IDF, OneVsRest(BernoulliNB).
   Fixes: correct encoder names (K3), argmax over predict_proba (K4), vectoriser
   fit inside a Pipeline on the training fold only (K7), BNB reported once (K5),
   same grouped split as BERT (K2). min_df=2 added to keep memory manageable.
2. The app's CPU engine: multinomial Logistic Regression on RelationalFeatures
   (headline/body TF-IDF + cosine + overlap/refuting features), saved to
   models/lr-relational with a SHA-256 integrity hash.

Each model is selected/inspected on validation only; the official test set is
scored once at the end. Metrics -> outputs/metrics/ours/*.json.
"""
from __future__ import annotations

import json
import time
from datetime import datetime

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

import config
from evaluation import compute_metrics, save_metrics
from features import RelationalFeatures, concat_text
from model import sha256_file
from preprocessing import load_fnc1, make_split

OURS = config.METRICS_DIR / "ours"


def bow(vec_cls):
    return vec_cls(ngram_range=(1, 2), stop_words="english", min_df=2)


def models():
    return {
        "lr_count_original_style": (
            "Logistic Regression + Count (original-style, bug-fixed)",
            Pipeline([("text", FunctionTransformer(concat_text)), ("vec", bow(CountVectorizer)),
                      ("clf", OneVsRestClassifier(LogisticRegression(max_iter=1000)))]),
        ),
        "lr_tfidf_original_style": (
            "Logistic Regression + TF-IDF (original-style, bug-fixed)",
            Pipeline([("text", FunctionTransformer(concat_text)), ("vec", bow(TfidfVectorizer)),
                      ("clf", OneVsRestClassifier(LogisticRegression(max_iter=1000)))]),
        ),
        "bnb_original_style": (
            "Bernoulli Naive Bayes (original-style, bug-fixed)",
            Pipeline([("text", FunctionTransformer(concat_text)), ("vec", bow(CountVectorizer)),
                      ("clf", OneVsRestClassifier(BernoulliNB()))]),
        ),
        "lr_relational": (
            "Logistic Regression (TF-IDF + relation features)",
            Pipeline([("feat", RelationalFeatures()),
                      ("clf", LogisticRegression(max_iter=3000, C=4.0))]),
        ),
    }


def main(only: list[str] | None = None) -> None:
    train = load_fnc1("train")
    test = load_fnc1("test")
    split = make_split(train)
    tr, va = train.iloc[split.train_idx], train.iloc[split.val_idx]
    cols = ["Headline", "articleBody"]
    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    for key, (display, pipe) in models().items():
        if only and key not in only:
            continue
        t0 = time.time()
        pipe.fit(tr[cols], tr["Stance"])
        fit_s = time.time() - t0
        classes = list(pipe.classes_)
        val_pred = [classes[i] for i in pipe.predict_proba(va[cols]).argmax(1)]
        test_pred = [classes[i] for i in pipe.predict_proba(test[cols]).argmax(1)]
        m_val = compute_metrics(va["Stance"], val_pred)
        m_test = compute_metrics(test["Stance"], test_pred)
        pd.DataFrame({"Headline": test["Headline"], "Body ID": test["Body ID"], "Stance": test_pred}).to_csv(
            config.PREDICTIONS_DIR / f"{key}_test.csv", index=False, encoding="utf-8")
        meta = dict(source="this project (trained and evaluated here)", display_name=display, model_key=key,
                    trained_on=f"FNC-1 train fold ({len(tr)} pairs, grouped split)", fit_seconds=round(fit_s, 1),
                    evaluated_on="official FNC-1 competition test set (25,413 pairs)",
                    validation_metrics=m_val)
        save_metrics(m_test, OURS / f"{key}.json", **meta)
        print(f"{display:58s} fit {fit_s:6.1f}s | val acc {m_val['accuracy']:.4f} mF1 {m_val['macro']['f1']:.4f} "
              f"rel/unrel {m_val['related_vs_unrelated_accuracy']:.4f} | TEST acc {m_test['accuracy']:.4f} "
              f"mF1 {m_test['macro']['f1']:.4f} FNC {m_test['fnc']['relative']:.2%}", flush=True)

        if key == "lr_relational":
            out = config.LR_MODEL_DIR
            out.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipe, out / "model.joblib", compress=3)
            info = {
                "display_name": display, "trained_by": "this project (train_baselines.py)",
                "sha256": sha256_file(out / "model.joblib"),
                "classes": classes, "validation_metrics": m_val,
                "created": datetime.now().isoformat(timespec="seconds"),
                "note": "joblib (pickle) file written by this project; integrity-checked by hash before loading.",
            }
            (out / "model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    import sys

    main(sys.argv[1:] or None)
