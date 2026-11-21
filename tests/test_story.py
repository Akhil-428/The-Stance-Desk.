"""The single story box is split into the (headline, body) pair the models expect."""
import pytest

from preprocessing import InputValidationError
from ui.story import join_story, parse_story, split_story


def test_first_line_is_headline_rest_is_body():
    assert parse_story("Headline here\nBody line one.\nBody line two.") == ("Headline here", "Body line one.\nBody line two.")


def test_paragraphs_and_line_breaks_are_preserved():
    h, b = parse_story("H\nPara one.\n\nPara two.\n  indented line")
    assert h == "H" and b == "Para one.\n\nPara two.\n  indented line"


def test_surrounding_whitespace_and_blank_lines_are_trimmed():
    assert parse_story("\n\n   Spaced   headline  \r\n\n\nBody.   \n\n") == ("Spaced headline", "Body.")


def test_windows_and_old_mac_newlines():
    assert parse_story("H\r\nB1\rB2") == ("H", "B1\nB2")


@pytest.mark.parametrize("text, msg", [("", "headline"), ("   \n  ", "headline"), ("Only a headline", "article body"),
                                       ("Headline\n\n   \n", "article body")])
def test_validation_messages(text, msg):
    with pytest.raises(InputValidationError, match=msg):
        split_story(text)


def test_join_round_trip():
    assert parse_story(join_story("H", "B1\n\nB2")) == ("H", "B1\n\nB2")
