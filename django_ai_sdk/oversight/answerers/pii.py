"""Personal data and secrets, found without a model and checked by checksum.

A number shaped like an IBAN that fails mod-97 is not an IBAN, so it is not
reported: checksums are what keep a regex detector usable at the gate.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from django_ai_sdk.oversight.answerers.rules import Detector
from django_ai_sdk.oversight.types import TextSpan

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

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


# Letter pairs after four digits that are a unit or a word, not a postcode: "4096 MB",
# "8080 OK", "1984 IN". Dutch postcodes never use SA, SD or SS either.
NOT_POSTCODE_LETTERS = (
    "SA SD SS KB MB GB TB PB OK AM PM AD BC CE HZ MS PX PT EM KM CM MM KG MG ML HP KW MW GW "
    "TV PC VS AN AS AT BE BY DO GO HE IF IN IS IT ME MY NO OF ON OR SO TO UP US WE"
).split()

# label -> (pattern, validator). Checksummed kinds first: an overlap goes to the first.
KINDS: dict[str, tuple[str, Callable[[str], bool] | None]] = {
    # Bounded, so a long run of "a." cannot backtrack for seconds.
    "email": (r"\b[\w.+-]{1,64}@[\w-]{1,63}(?:\.[\w-]{1,63}){1,8}\b", None),
    "iban": (r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){2,7}(?:\s?[A-Z0-9]{1,4})?\b", iban_ok),
    "credit_card": (r"\b(?:\d[ -]?){12,18}\d\b", luhn_ok),
    "bsn": (r"\b\d{9}\b|\b\d{4}\.\d{2}\.\d{3}\b", bsn_ok),
    # International format: a country code after + or 00.
    "phone": (
        r"(?<![\w+])(?:\+|00)\d{1,3}[\s-]?(?:\(0\)[\s-]?)?\d(?:[\s-]?\d){6,11}\b",
        None,
    ),
    # Dutch national format: 06-12345678, 020 123 4567.
    "phone_nl": (r"\b0(?:6|[1-9]\d)[\s-]?\d{3,4}[\s-]?\d{3,4}\b", None),
    "ip_address": (r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b", None),
    "postcode_nl": (
        rf"\b[1-9]\d{{3}}\s?(?!(?:{'|'.join(NOT_POSTCODE_LETTERS)})\b)[A-Z]{{2}}\b",
        None,
    ),
}

# What PII() looks for without `kinds`: formats that mean the same in every country.
DEFAULT_KINDS: tuple[str, ...] = ("email", "iban", "credit_card", "phone", "ip_address")
# Country-specific kinds, only when named: PII(kinds=[*DEFAULT_KINDS, "bsn"]).
COUNTRY_KINDS: tuple[str, ...] = ("bsn", "phone_nl", "postcode_nl")

# Kinds whose shape can swallow what follows them ("BE68539007547034 EUR", an IBAN and
# its BIC, a card number and its expiry): when the checksum fails, shorter matches that
# end where the text has a space or a hyphen are tried, longest first.
SHORTER_KINDS = frozenset({"iban", "credit_card"})

# A value after "password:" that looks like one: a digit, a symbol inside it, or quotes.
# Prose ("my password: settings or profile?") has none of these.
_PASSWORD_VALUE = r"""(?:(["'])[^"'\s]{6,}\1|(?=\S*?(?:\d|[!@#$%^&*_+=~|\\/<>{}\[\]-]\S))\S{6,})"""

SECRETS: dict[str, str] = {
    # sk-..., sk-proj-..., sk-ant-...: the key shape several model providers share.
    "api_key": r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}\b",
    "stripe_key": r"\b(?:sk|rk)_live_[A-Za-z0-9]{20,}\b",
    "google_api_key": r"\bAIza[0-9A-Za-z_-]{35}(?![0-9A-Za-z_-])",
    "github_token": r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b",
    "aws_key": r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "slack_token": r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b",
    "jwt": r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b",
    "private_key": r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----",
    "password_assignment": (
        r"(?i)\b(?:password|passwd|wachtwoord|pwd)\s*[:=]\s*" + _PASSWORD_VALUE
    ),
}


def _shorter(
    pattern: re.Pattern[str], valid: Callable[[str], bool], text: str, start: int, end: int
) -> int:
    """The end of the longest valid match from `start` that stops at a space or hyphen
    before `end`; 0 when there is none."""
    for cut in range(end - 1, start, -1):
        if text[cut] not in " \t\n-" or not text[cut - 1].isalnum():
            continue
        if pattern.fullmatch(text, start, cut) and valid(text[start:cut]):
            return cut
    return 0


class PII(Detector):
    """Personal data by kind: `PII()` for the default kinds, `PII(kinds=[...])` to choose.

    Default kinds (DEFAULT_KINDS), the same in every country:
      email; iban (mod-97 checked); credit_card (Luhn checked); phone (international
      format, +31 6 1234 5678 or 0031...); ip_address (IPv4).
    Country-specific kinds, only when named (COUNTRY_KINDS):
      bsn (Dutch citizen service number, 11-check); phone_nl (Dutch national format,
      06-12345678); postcode_nl (1234 AB).

        PII(kinds=[*DEFAULT_KINDS, "bsn", "postcode_nl"])

    `extra` adds patterns of your own, label -> regex, without a checksum.
    """

    metered = False
    miss_confidence = 0.85

    def __init__(
        self,
        kinds: Iterable[str] | None = None,
        *,
        extra: dict[str, str] | None = None,
        name: str = "",
    ) -> None:
        # None or empty (the console's "none picked"): the default kinds.
        chosen = list(kinds or DEFAULT_KINDS)
        unknown = set(chosen) - set(KINDS)
        if unknown:
            raise ValueError(f"Unknown PII kinds {sorted(unknown)}; known: {sorted(KINDS)}")
        self.patterns = [(kind, re.compile(KINDS[kind][0]), KINDS[kind][1]) for kind in chosen] + [
            (label, re.compile(pattern), None) for label, pattern in (extra or {}).items()
        ]
        self.name = name or "pii"

    def looks_for(self) -> list[str]:
        return [label for label, _, _ in self.patterns]

    def detect(self, text: str, view: TurnView, lens: Lens) -> list[TextSpan]:
        spans: list[TextSpan] = []
        taken: list[tuple[int, int]] = []
        for label, pattern, valid in self.patterns:
            for start, end in _matches(label, pattern, valid, text):
                # A card number also looks like a phone number: first kind wins.
                if any(start < until and since < end for since, until in taken):
                    continue
                taken.append((start, end))
                spans.append(TextSpan(start=start, end=end, label=label))
        return sorted(spans, key=lambda s: s.start)


def _matches(
    label: str, pattern: re.Pattern[str], valid: Callable[[str], bool] | None, text: str
) -> Iterator[tuple[int, int]]:
    """(start, end) of each match of one kind that passes its checksum."""
    pos = 0
    while (match := pattern.search(text, pos)) is not None:
        start, end = match.span()
        pos = max(end, start + 1)
        if valid is not None and not valid(match.group()):
            end = _shorter(pattern, valid, text, start, end) if label in SHORTER_KINDS else 0
            if not end:
                continue
            # What the shorter match left over may hold the next one.
            pos = end
        yield start, end


class Secrets(Detector):
    """API keys, tokens and private keys pasted into a message or a tool argument."""

    metered = False
    miss_confidence = 0.9

    def __init__(self, *, extra: dict[str, str] | None = None, name: str = "") -> None:
        self.patterns = [
            (label, re.compile(pattern)) for label, pattern in {**SECRETS, **(extra or {})}.items()
        ]
        self.name = name or "secrets"

    def looks_for(self) -> list[str]:
        return [label for label, _ in self.patterns]

    def detect(self, text: str, view: TurnView, lens: Lens) -> list[TextSpan]:
        return [
            TextSpan(start=m.start(), end=m.end(), label=label)
            for label, pattern in self.patterns
            for m in pattern.finditer(text)
        ]


_SECRET_PATTERNS = [(label, re.compile(pattern)) for label, pattern in SECRETS.items()]


def mask_secrets(text: str) -> str:
    """`text` with every built-in secret pattern masked, before it leaves for a processor."""
    if not text:
        return text
    spans = [
        TextSpan(start=m.start(), end=m.end(), label=label)
        for label, pattern in _SECRET_PATTERNS
        for m in pattern.finditer(text)
    ]
    return mask(text, spans) if spans else text


def mask(text: str, spans: Iterable[TextSpan], template: str = "[{label}]") -> str:
    """`text` with every span replaced by its label; overlapping spans merge."""
    return mask_all(text, [(spans, template)])


def mask_all(text: str, groups: Iterable[tuple[Iterable[TextSpan], str]]) -> str:
    """`text` with the spans of every (spans, template) group masked, in one pass.

    Masking group after group would shift the offsets of the next; here all spans
    index the original text. Overlapping spans merge into the first one's label.
    """
    ordered = sorted(
        ((span, template) for spans, template in groups for span in spans),
        key=lambda item: (item[0].start, -item[0].end),
    )
    out: list[str] = []
    cursor = 0
    for span, template in ordered:
        start, end = max(span.start, 0), min(span.end, len(text))
        if end <= cursor or start >= end:
            continue
        if start < cursor:
            # Overlaps the span just masked: that label covers this one too.
            cursor = end
            continue
        out.append(text[cursor:start])
        out.append(template.format(label=(span.label or "masked").upper()))
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


__all__ = [
    "COUNTRY_KINDS",
    "DEFAULT_KINDS",
    "KINDS",
    "NOT_POSTCODE_LETTERS",
    "PII",
    "SECRETS",
    "Secrets",
    "bsn_ok",
    "iban_ok",
    "luhn_ok",
    "mask",
    "mask_all",
    "mask_secrets",
]
