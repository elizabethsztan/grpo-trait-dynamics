from __future__ import annotations

from dataclasses import dataclass
import re

ANSWER_RE = re.compile(r"<answer>\s*([ABCD])\s*</answer>", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedAnswer:
    choice: str | None
    multiple_answer_tags: bool


def parse_answer_with_metadata(text: str) -> ParsedAnswer:
    matches = ANSWER_RE.findall(text or "")
    if not matches:
        return ParsedAnswer(choice=None, multiple_answer_tags=False)
    return ParsedAnswer(choice=matches[0].upper(), multiple_answer_tags=len(matches) > 1)


def parse_answer(text: str) -> str | None:
    return parse_answer_with_metadata(text).choice
