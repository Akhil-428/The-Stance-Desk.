"""4: input validation; data loading; leakage-free split; shared input builder."""
import numpy as np
import pytest
from _markers import requires_fnc1, requires_pretrained

import config
from preprocessing import InputValidationError, bert_encode, load_fnc1, make_split, normalize_text, validate_inputs


@pytest.mark.parametrize("h,b", [("", "body"), ("head", ""), ("   ", "body"), ("head", "\n\t "), (None, "b"),
                                 ("head", None), (123, "body"), ("!!!", "???")])
def test_invalid_inputs_rejected(h, b):
    with pytest.raises(InputValidationError):
        validate_inputs(h, b)


def test_too_long_input_rejected():
    with pytest.raises(InputValidationError, match="too long"):
        validate_inputs("headline", "x " * (config.MAX_INPUT_CHARS))


def test_valid_and_non_ascii_inputs_normalised():
    h, b = validate_inputs("  Café “quotes” — ünïcode  ", "Line one\r\nLine   two")
    assert h == "Café “quotes” — ünïcode"
    assert b == "Line one\nLine two"
    assert normalize_text("a\r\nb\rc") == "a\nb\nc"


@pytest.fixture(scope="module")
def train_df():
    return load_fnc1("train")


@requires_fnc1
def test_fnc1_sizes_and_labels(train_df):
    test = load_fnc1("test")
    assert len(train_df) == 49972 and len(test) == 25413
    assert set(train_df["Stance"]) == set(config.LABELS)
    assert test["Stance"].value_counts().to_dict() == {"unrelated": 18349, "discuss": 4464, "agree": 1903, "disagree": 697}
    assert train_df["articleBody"].notna().all()


@requires_fnc1
def test_split_grouped_by_body_and_persisted(train_df):
    s = make_split(train_df)
    assert len(np.intersect1d(s.train_idx, s.val_idx)) == 0
    assert len(s.train_idx) + len(s.val_idx) == len(train_df)
    tr_bodies = set(train_df.iloc[s.train_idx]["Body ID"])
    va_bodies = set(train_df.iloc[s.val_idx]["Body ID"])
    assert not tr_bodies & va_bodies, "an article body leaks across the split"
    assert set(train_df.iloc[s.val_idx]["Stance"]) == set(config.LABELS)
    s2 = make_split(train_df)  # loads the persisted split
    assert np.array_equal(s.val_idx, s2.val_idx)


@requires_pretrained
def test_bert_encode_pair_and_truncation():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.PRETRAINED_BERT_DIR, local_files_only=True)
    enc = bert_encode(tok, "Headline here", "word " * 2000, 512)
    ids, tt = enc["input_ids"][0], enc["token_type_ids"][0]
    assert len(ids) == 512
    assert ids[0] == tok.cls_token_id and ids[-1] == tok.sep_token_id
    text = tok.decode(ids)
    assert text.startswith("[CLS] headline here [SEP]")       # headline kept intact
    assert tt[1] == 0 and tt[-2] == 1                         # segment ids distinguish parts
