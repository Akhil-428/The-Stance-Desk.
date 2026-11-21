"""8: dashboard metrics are computed correctly (incl. regression vs. audited figures)."""
import pytest
from _markers import requires_historical

import config
from evaluation import EvaluationError, compute_metrics, fnc_score, score_prediction_files

HIST = config.HISTORICAL_DIR
GOLD = HIST / "competition_test_stances.csv"


def test_fnc_score_rules():
    gold = ["agree", "disagree", "discuss", "unrelated", "agree", "unrelated"]
    pred = ["agree", "agree", "unrelated", "unrelated", "discuss", "agree"]
    # agree==agree: 1.0 | disagree->agree: .25 | discuss->unrelated: 0 | unrel ok: .25 | agree->discuss: .25 | unrel->agree: 0
    s = fnc_score(gold, pred)
    assert s["score"] == pytest.approx(1.75)
    assert s["max"] == pytest.approx(4 * 1.0 + 2 * 0.25)
    assert s["null"] == pytest.approx(0.5)


def test_compute_metrics_perfect_and_matrix():
    gold = ["agree", "disagree", "discuss", "unrelated"] * 3
    m = compute_metrics(gold, gold)
    assert m["accuracy"] == 1.0 and m["macro"]["f1"] == 1.0 and m["fnc"]["relative"] == 1.0
    assert m["confusion_matrix"]["rows_gold_cols_pred"] == [[3, 0, 0, 0], [0, 3, 0, 0], [0, 0, 3, 0], [0, 0, 0, 3]]


def test_errors_are_explicit():
    with pytest.raises(EvaluationError, match="Length mismatch"):
        fnc_score(["agree"], [])
    with pytest.raises(EvaluationError, match="unknown labels"):
        compute_metrics(["agree"], ["Agree"])


@requires_historical
def test_regression_original_bert_csv():
    """Must reproduce the independently audited figures exactly."""
    m = score_prediction_files(GOLD, HIST / "BERT.csv")
    assert m["fnc"]["score"] == pytest.approx(10068.25)
    assert m["fnc"]["max"] == pytest.approx(11651.25)
    assert m["fnc"]["null"] == pytest.approx(4587.25)
    assert m["accuracy"] == pytest.approx(22967 / 25413)
    assert m["macro"]["f1"] == pytest.approx(0.6910, abs=5e-5)
    assert m["confusion_matrix"]["rows_gold_cols_pred"][1] == [221, 229, 193, 54]


@requires_historical
def test_historical_lr_labels_are_corrected():
    tfidf = score_prediction_files(GOLD, HIST / "LogisticRegression_with_CountVectorizer.csv")
    count = score_prediction_files(GOLD, HIST / "LogisticRegression_with_TfidfVectorizer.csv")
    assert tfidf["accuracy"] == pytest.approx(0.706331, abs=1e-6)
    assert count["accuracy"] == pytest.approx(0.576713, abs=1e-6)
    from evaluate_historical import FILES

    assert "TF-IDF" in FILES["LogisticRegression_with_CountVectorizer.csv"][0]
    assert "Count" in FILES["LogisticRegression_with_TfidfVectorizer.csv"][0]


def test_utf8_file_with_cp1252_undecodable_bytes(tmp_path):
    p = tmp_path / "pred.csv"
    p.write_text('Headline,Body ID,Stance\n"Quote ”x”",1,agree\n', encoding="utf-8")
    assert b"\x9d" in p.read_bytes()  # byte undefined in cp1252
    m = score_prediction_files(p, p)
    assert m["accuracy"] == 1.0


def test_misaligned_and_truncated_files_rejected(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("Headline,Body ID,Stance\nh1,1,agree\nh2,2,discuss\n", encoding="utf-8")
    b.write_text("Headline,Body ID,Stance\nh1,1,agree\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="Row count"):
        score_prediction_files(a, b)
    b.write_text("Headline,Body ID,Stance\nh1,1,agree\nhX,2,discuss\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="misaligned"):
        score_prediction_files(a, b)


def test_dashboard_json_matches_recomputation():
    from evaluation import load_metrics_dir

    hist = load_metrics_dir(config.METRICS_DIR / "historical")
    assert hist["BERT"]["metrics"]["fnc"]["score"] == pytest.approx(10068.25)
    for key, entry in load_metrics_dir(config.METRICS_DIR / "ours").items():
        pred = config.PREDICTIONS_DIR / f"{key}_test.csv"
        if pred.exists():
            again = score_prediction_files(config.FNC1_DIR / "competition_test_stances.csv", pred)
            assert again["accuracy"] == pytest.approx(entry["metrics"]["accuracy"])
            assert again["fnc"]["score"] == pytest.approx(entry["metrics"]["fnc"]["score"])
