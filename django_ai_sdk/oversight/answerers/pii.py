"""Personal data and secrets, found without a model and checked by checksum.

A number shaped like a BSN that fails the 11-check is not a BSN, so it is not
reported: checksums are what keep a regex detector usable at the gate.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar

from django_ai_sdk.oversight.answerers.rules import Detector
from django_ai_sdk.oversight.types import TextSpan

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from django_ai_sdk.oversight.lenses import Lens
    from django_ai_sdk.oversight.types import TurnView


def luhn_ok(digits: str) -> bool:
    """The Luhn check that every payment card number passes."""
    numbers = [int(d) for d in digits if d.isdigit()]
    if len(numbers) < 12:
        return False
    total = 0
    for index, digit in enumerate(reversed(numbers)):
        if index % 2:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def iban_ok(value: str) -> bool:
    """ISO 13616 mod-97: move the first four characters to the end, letters to numbers."""
    compact = re.sub(r"\s+", "", value).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}", compact):
        return False
    rearranged = compact[4:] + compact[:4]
    number = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(number) % 97 == 1


def bsn_ok(value: str) -> bool:
    """The Dutch citizen service number's 11-check (elfproef), last digit weighted -1."""
    digits = re.sub(r"\D", "", value)
    if len(digits) == 8:
        digits = "0" + digits
    if len(digits) != 9 or digits == "000000000":
        return False
    weights = [9, 8, 7, 6, 5, 4, 3, 2, -1]
    return sum(int(d) * w for d, w in zip(digits, weights, strict=True)) % 11 == 0


# label -> (pattern, validator). Checksummed kinds first: an overlap goes to the first.
KINDS: dict[str, tuple[str, Callable[[str], bool] | None]] = {
    "email": (r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b", None),
    "iban": (r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){2,7}(?:\s?[A-Z0-9]{1,4})?\b", iban_ok),
    "credit_card": (r"\b(?:\d[ -]?){12,18}\d\b", luhn_ok),
    "bsn": (r"\b\d{9}\b|\b\d{4}\.\d{2}\.\d{3}\b", bsn_ok),
    "phone": (
        r"(?<![\w+])(?:\+|00)\d{1,3}[\s-]?(?:\(0\)[\s-]?)?\d(?:[\s-]?\d){6,11}\b"
        r"|\b0(?:6|[1-9]\d)[\s-]?\d{3,4}[\s-]?\d{3,4}\b",
        None,
    ),
    "ip_address": (r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b", None),
    "postcode_nl": (r"\b[1-9]\d{3}\s?(?!SA|SD|SS)[A-Z]{2}\b", None),
}

SECRETS: dict[str, str] = {
    "openai_key": r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}\b",
    "github_token": r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b",
    "aws_key": r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "slack_token": r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b",
    "jwt": r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b",
    "private_key": r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----",
    "password_assignment": r"(?i)\b(?:password|passwd|wachtwoord|pwd)\s*[:=]\s*\S{6,}",
}


class PII(Detector):
    """Personal data by kind: `PII(kinds=["email", "iban"])`, or every kind by default.

    `extra` adds patterns of your own, label -> regex, without a checksum.
    """

    system_one: ClassVar[bool] = True
    miss_confidence = 0.85

    def __init__(
        self,
        kinds: Iterable[str] | None = None,
        *,
        extra: dict[str, str] | None = None,
        name: str = "",
    ) -> None:
        chosen = list(kinds) if kinds is not None else list(KINDS)
        unknown = set(chosen) - set(KINDS)
        if unknown:
            raise ValueError(f"Unknown PII kinds {sorted(unknown)}; known: {sorted(KINDS)}")
        self.patterns = [(kind, re.compile(KINDS[kind][0]), KINDS[kind][1]) for kind in chosen] + [
            (label, re.compile(pattern), None) for label, pattern in (extra or {}).items()
        ]
        self.name = name or "pii"

    def detect(self, text: str, view: TurnView, lens: Lens) -> list[TextSpan]:
        spans: list[TextSpan] = []
        taken: list[tuple[int, int]] = []
        for label, pattern, valid in self.patterns:
            for match in pattern.finditer(text):
                if valid is not None and not valid(match.group()):
                    continue
                # A card number also looks like a phone number: first kind wins.
                if any(match.start() < end and start < match.end() for start, end in taken):
                    continue
                taken.append((match.start(), match.end()))
                spans.append(TextSpan(start=match.start(), end=match.end(), label=label))
        return sorted(spans, key=lambda s: s.start)


class Secrets(Detector):
    """API keys, tokens and private keys pasted into a message or a tool argument."""

    system_one: ClassVar[bool] = True
    miss_confidence = 0.9

    def __init__(self, *, extra: dict[str, str] | None = None, name: str = "") -> None:
        self.patterns = [
            (label, re.compile(pattern)) for label, pattern in {**SECRETS, **(extra or {})}.items()
        ]
        self.name = name or "secrets"

    def detect(self, text: str, view: TurnView, lens: Lens) -> list[TextSpan]:
        return [
            TextSpan(start=m.start(), end=m.end(), label=label)
            for label, pattern in self.patterns
            for m in pattern.finditer(text)
        ]


def redact(text: str, spans: Iterable[TextSpan], template: str = "[{label}]") -> str:
    """`text` with every span replaced by its label; overlapping spans merge."""
    ordered = sorted(spans, key=lambda s: (s.start, -s.end))
    out: list[str] = []
    cursor = 0
    for span in ordered:
        if span.end <= cursor:
            continue
        start = max(span.start, cursor)
        out.append(text[cursor:start])
        out.append(template.format(label=(span.label or "redacted").upper()))
        cursor = span.end
    out.append(text[cursor:])
    return "".join(out)


__all__ = ["KINDS", "PII", "SECRETS", "Secrets", "bsn_ok", "iban_ok", "luhn_ok", "redact"]
