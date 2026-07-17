from __future__ import annotations

from dataclasses import dataclass
import re

ANSWER_RE = re.compile(r"<answer>\s*([^<]*?)\s*</answer>", re.IGNORECASE)
STRICT_ANSWER_RE = re.compile(r"^<answer>\s*([ABCD])\s*</answer>$", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedAnswer:
    choice: str | None
    valid: bool
    multiple_answer_tags: bool
    malformed_answer_tag: bool
    extra_text: bool
    raw_matches: list[str]


def parse_answer_with_metadata(text: str) -> ParsedAnswer:
    raw_text = text or ""
    stripped = raw_text.strip()
    matches = [match.strip() for match in ANSWER_RE.findall(raw_text)]
    strict = STRICT_ANSWER_RE.fullmatch(stripped)
    multiple = len(matches) > 1
    valid = strict is not None and len(matches) == 1
    choice = strict.group(1).upper() if valid else None
    has_answer_syntax = "<answer" in raw_text.lower() or "</answer>" in raw_text.lower()
    malformed = False
    if not valid:
        malformed = has_answer_syntax and (not matches or any(match.upper() not in {"A", "B", "C", "D"} for match in matches))
    extra_text = bool(matches) and not valid and not multiple and not malformed
    return ParsedAnswer(
        choice=choice,
        valid=valid,
        multiple_answer_tags=multiple,
        malformed_answer_tag=malformed,
        extra_text=extra_text,
        raw_matches=matches,
    )


def parse_answer(text: str) -> str | None:
    return parse_answer_with_metadata(text).choice
