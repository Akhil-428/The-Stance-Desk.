"""Split the single story box into the two inputs the models expect.

Format: the first non-empty line is the headline; everything after it is the
article body. Paragraphs and line breaks in the body are preserved; only
surrounding whitespace is trimmed. The models themselves are unchanged and
still receive (headline, body) through model.*Predictor.predict().
"""
from __future__ import annotations

from preprocessing import InputValidationError

FORMAT_HINT = "First line — headline | Below — article content"


def parse_story(text: str | None) -> tuple[str, str]:
    """Return (headline, body). Either may be empty; use `split_story` to validate."""
    lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return "", ""
    headline = " ".join(lines[0].split())
    body_lines = [ln.rstrip() for ln in lines[1:]]
    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)
    while body_lines and not body_lines[-1].strip():
        body_lines.pop()
    return headline, "\n".join(body_lines)


def split_story(text: str | None) -> tuple[str, str]:
    """Parse and validate. Raises InputValidationError with a user-facing message."""
    headline, body = parse_story(text)
    if not headline:
        raise InputValidationError("Please enter a headline on the first line, then the article text below it.")
    if not body.strip():
        raise InputValidationError("Please paste the article body below the headline (from the second line on).")
    return headline, body


def join_story(headline: str, body: str) -> str:
    """Inverse of parse_story, used by the sample-story loader."""
    return f"{headline.strip()}\n{body.strip()}"
