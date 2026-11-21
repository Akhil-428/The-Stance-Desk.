"""Feature transformers for the Logistic Regression engines (scikit-learn).

RelationalFeatures models the headline-body *relationship*, which a bag of
words over the concatenated text cannot do (audit C, K6). It is inspired by
the official FNC-1 baseline features: TF-IDF cosine similarity, word overlap,
refuting-word indicators and headline-word hits in the body.
"""
from __future__ import annotations

import re

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

_TOKEN = re.compile(r"[a-z0-9]+")
REFUTING = {
    "fake", "fraud", "hoax", "false", "deny", "denies", "denied", "not", "despite", "nope",
    "doubt", "doubts", "bogus", "debunk", "debunked", "debunks", "pranks", "prank", "retract",
    "retracts", "retracted", "untrue", "misleading", "rumor", "rumour", "rumors", "claims",
}
DISCUSS = {"reportedly", "allegedly", "according", "claim", "claims", "reports", "said", "says",
           "suggests", "rumored", "unconfirmed", "may", "might", "could"}


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _content(toks):
    return [t for t in toks if t not in ENGLISH_STOP_WORDS]


def hand_features(headline: str, body: str) -> list[float]:
    h, b = tokens(headline), tokens(body)
    hc, bc = set(_content(h)), set(_content(b))
    b_lead = set(_content(tokens(body[:255])))
    inter = hc & bc
    feats = [
        len(inter) / max(1, len(hc | bc)),            # Jaccard overlap
        len(inter) / max(1, len(hc)),                  # share of headline words in body
        len(hc & b_lead) / max(1, len(hc)),            # headline words in body lead
        float(sum(t in REFUTING for t in h) > 0),
        sum(t in REFUTING for t in b) / max(1, len(b)) * 100,
        float(sum(t in DISCUSS for t in h) > 0),
        sum(t in DISCUSS for t in b) / max(1, len(b)) * 100,
        float("?" in headline),
        np.log1p(len(b)),
        np.log1p(len(h)),
    ]
    # bigram overlap (headline bigrams appearing in body)
    hb = {(x, y) for x, y in zip(h, h[1:])}
    bb = {(x, y) for x, y in zip(b, b[1:])}
    feats.append(len(hb & bb) / max(1, len(hb)))
    return feats


class RelationalFeatures(BaseEstimator, TransformerMixin):
    """[TF-IDF(headline) | TF-IDF(body) | cosine | hand features] as a sparse matrix."""

    def __init__(self, max_features: int = 50000, ngram_range=(1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range

    def fit(self, X, y=None):
        self.tfidf_ = TfidfVectorizer(
            max_features=self.max_features, ngram_range=self.ngram_range, stop_words="english",
            sublinear_tf=True, min_df=2,
        )
        self.tfidf_.fit(list(X["Headline"]) + list(X["articleBody"]))
        dense = self._dense(X, fit=True)
        self.mean_, self.std_ = dense.mean(0), dense.std(0) + 1e-6
        return self

    def _dense(self, X, fit=False):
        H = self.tfidf_.transform(X["Headline"])
        B = self.tfidf_.transform(X["articleBody"])
        cos = np.asarray(H.multiply(B).sum(1)).ravel()   # rows are L2-normalised
        hand = np.array([hand_features(h, b) for h, b in zip(X["Headline"], X["articleBody"])])
        self._HB = (H, B)
        return np.column_stack([cos, hand])

    def transform(self, X):
        dense = (self._dense(X) - self.mean_) / self.std_
        H, B = self._HB
        return sp.hstack([H, B, sp.csr_matrix(dense * 3.0)], format="csr")


def concat_text(X) -> list[str]:
    """Original notebook input: Headline + "[SEP]" + body (bag-of-words baselines)."""
    return [f"{h} [SEP] {b}" for h, b in zip(X["Headline"], X["articleBody"])]
