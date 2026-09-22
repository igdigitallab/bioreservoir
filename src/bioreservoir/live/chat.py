"""Chat identity, session tokens, and nickname-impersonation checks for the live chat
(`POST /api/chat`; hardened 2026-09-18 per the security review, findings F3-F5). Pure, no I/O --
`api.py`'s routes own wiring these to the store and to `moderation.py`'s chat text/nickname
policy, the same split `moderation.py`/`api.py` already use for question moderation.

**Author identity (F4).** `LiveStore.hash_ip` rotates its salt once per UTC day -- fine for
question rate-limit buckets, but the security review reproduced it silently expiring a chat ban
at 00:00 UTC (4pm PST on US election day) and rotating every `fly-NNNN` handle at the same moment.
`author_id()` below is a SEPARATE, non-rotating identity: `HMAC-SHA256(CHAT_ID_PEPPER,
normalized_ip)`, IPv6 normalized to its /64 (a residential ISP hands out a fresh address from the
same /64 far more often than a fresh /64 itself). Bans, reports, message authorship, session
tokens, and anonymous handles ALL key off this value now, not off `hash_ip`.

**Session tokens (F3).** A client passes `turnstile_token` on its first message; on success the
server issues a `chat_token` back, bound to the SPECIFIC `author_id` that solved the captcha
(`"<author_id>.<issued_at>.<expires_at>.<sig>"`, sig over the first three fields). Verifying a
token recomputes `author_id` from the CURRENT request's IP and rejects on mismatch -- a token
solved from one IP cannot be replayed by a different IP/subnet, closing the "solve Turnstile once,
report from many IPs" flood-censorship exploit the review reproduced against the old (unbound)
token. `issued_at` lets callers (the report route) enforce a minimum session age before a report
counts toward auto-hide, without needing a second signed field.

**Report-group identity (F3, distinct from author_id).** "Reporters on the same IPv4 /24 or IPv6
/64 count once" is a COARSER grouping than author_id (whose IPv4 side is the full address, not a
/24) -- used only to dedupe distinct reporters for the auto-hide tally, never for bans/tokens.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import re
import time
import unicodedata
from dataclasses import dataclass

from bioreservoir.live import config

NICKNAME_RE = re.compile(
    rf"^[A-Za-z0-9_-]{{{config.NICKNAME_MIN_LENGTH},{config.NICKNAME_MAX_LENGTH}}}$"
)


def check_nickname_format(nickname: str) -> bool:
    """Length 2-20, `[A-Za-z0-9_-]` only -- deliberately excludes `@`, `.` and
    whitespace, so a nickname matching this can never itself be a URL/email/handle/phone number
    (moderation.py's chat contact-info check does not need to special-case nicknames)."""
    return bool(NICKNAME_RE.match(nickname))


# -- nickname impersonation (F5, ASCII-leetspeak-hardened R4) --------------------------------------

# Reserved substrings: the original list plus round-2's additions (flyoracle, igor -- the lab's
# founder -- support, verified, team, lab, bot). Matched after NFKC normalization,
# casefold, an ASCII leetspeak map (see below), and stripping zero-width characters and
# nickname-legal separators (`-`/`_`), so "B-io_Reservoir", "ADMIN", "0fficial", "adm1n" etc. all
# normalize to a form containing the bare reserved word.
# "ig digital lab" has internal spaces, which the nickname format regex already forbids -- the
# only way it could appear in a valid nickname is with separators the normalizer below also
# strips (ig-digital-lab, ig_digital_lab), so it collapses to the same "igdigitallab" string as
# the standalone "igdigitallab" entry -- both spellings covered by one substring.
_RESERVED_SUBSTRINGS = (
    "admin",
    "mod",
    "moderator",
    "staff",
    "official",
    "system",
    "bioreservoir",
    "igdigi",
    "igdigitallab",
    "thefly",
    "flyoracle",
    "igor",
    "support",
    "verified",
    "team",
    "lab",
    "bot",
)

# Only the server may assign a `fly-<digits>` handle (chat.anonymous_handle) -- a user-chosen
# nickname matching this shape would let a real anonymous visitor's words be put in someone else's
# mouth (security review F5). The middle letter is matched as a character class, not via string
# substitution (round-2 R4): "l"/"1"/"i"/"|" all read as the same glyph in most sans-serif UI
# fonts ("fIy-4821" casefolds to "fiy-4821", NOT "fly-4821", since capital I casefolds to i, not
# l) -- folding that ambiguity into the DIGIT-suffix leetspeak map instead would corrupt the
# numeric id itself ("4821" contains a "1" and a "4", both leetspeak targets below), so it is
# scoped to just this one regex instead.
_SERVER_HANDLE_RE = re.compile(r"^f[l1i|]y-?\d+$", re.IGNORECASE)

_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")

# ASCII leetspeak substitutions (round-2 security review R4's exact list) -- applied AFTER
# NFKC+casefold. "1" maps to "i" (not "l"): the review's own reproduced bypass list is dominated
# by "1"-for-"i" substitutions ("adm1n" -> "admin"), and folding "1" to "l" instead would turn
# "adm1n" into "admln", which no longer contains the reserved word "admin" at all. (The separate
# i/l/1/| ambiguity for the fly-<digits> handle check is handled in _SERVER_HANDLE_RE above, not
# here, so it never touches this table.)
#
# Replaces round-1's Cyrillic/Greek Unicode-confusables map, which round-2 found to be dead code:
# `check_nickname_format`'s `^[A-Za-z0-9_-]{2,20}$` gate runs BEFORE this function is ever reached
# and rejects all non-ASCII input outright, so a Cyrillic lookalike letter can never arrive here
# to be normalized -- there is no bug in the mapping itself, the input that would exercise it is
# unreachable by construction. ASCII leetspeak, by contrast, passes the format gate untouched and
# needed a real fix, not a mapping that only ever sees text nobody can submit.
_LEETSPEAK_MAP = str.maketrans(
    {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "|": "l"}
)


def _normalize_base(text: str) -> str:
    """NFKC + casefold + zero-width-strip + separator-strip -- shared by both checks below, with
    NO digit-leetspeak substitution (that step is applied on top, only for the reserved-word
    check; see `is_reserved_nickname`'s docstring for why the handle check must skip it)."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = _ZERO_WIDTH_RE.sub("", normalized)
    return normalized.replace("-", "").replace("_", "")


def is_reserved_nickname(nickname: str) -> bool:
    """True if `nickname` impersonates the lab/staff/operator or the server's own anonymous-handle
    namespace, after normalization (see module docstring, F5/R4).

    Two DIFFERENT normalizations, not one: the handle check (`f[l1i|]y-?\\d+`) needs its trailing
    digits to stay literal digits, so it runs against `_normalize_base`'s output with NO
    leetspeak substitution -- applying `_LEETSPEAK_MAP` first would mangle a genuine numeric id
    ("4821" contains a "1" and a "4", both leetspeak targets) into something that no longer looks
    like digits at all, and the handle would silently stop matching. The reserved-word substring
    check has no such constraint, so it runs on top of the leetspeak-substituted form."""
    base = _normalize_base(nickname)
    if _SERVER_HANDLE_RE.match(base):
        return True
    # In Inter, capital I and lowercase l are the same bar, so "OfficiaI" reads as "Official":
    # fold l to i on both sides (security review round 3, S2).
    leetspoken = base.translate(_LEETSPEAK_MAP).replace("l", "i")
    return any(word.replace("l", "i") in leetspoken for word in _RESERVED_SUBSTRINGS)


# -- author identity (F4) -------------------------------------------------------------------------


def _normalize_ipv6_to_64(addr: ipaddress.IPv6Address) -> str:
    return str(ipaddress.ip_network(f"{addr}/64", strict=False).network_address)


def normalize_ip_for_author(ip: str) -> str:
    """IPv6 -> its /64 network address; IPv4 unchanged (full address). Unparsable input (e.g. the
    "unknown"/"testclient" fallbacks `api._client_ip` can produce with no real client) passes
    through as-is -- `author_id` still hashes it deterministically, it just isn't IP-shaped."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return ip
    return _normalize_ipv6_to_64(addr) if addr.version == 6 else str(addr)


def normalize_ip_for_report_group(ip: str) -> str:
    """Coarser than `normalize_ip_for_author` on the IPv4 side (/24, not the full address) -- used
    ONLY to dedupe distinct reporters for the auto-hide tally (F3), never for bans/tokens."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return ip
    if addr.version == 6:
        return _normalize_ipv6_to_64(addr)
    return str(ipaddress.ip_network(f"{ip}/24", strict=False).network_address)


def author_id(ip: str, pepper: str) -> str:
    """Stable (non-daily-rotating) identity for bans, reports, message authorship, session
    tokens, and anonymous handles -- see module docstring."""
    return hmac.new(pepper.encode("utf-8"), normalize_ip_for_author(ip).encode("utf-8"), hashlib.sha256).hexdigest()[:32]


def report_group_id(ip: str, pepper: str) -> str:
    return hmac.new(pepper.encode("utf-8"), normalize_ip_for_report_group(ip).encode("utf-8"), hashlib.sha256).hexdigest()[:32]


def anonymous_handle(author_id_: str) -> str:
    """Stable per-author handle, e.g. "fly-4821" -- derived from the stable
    `author_id`, NOT the daily-rotating `hash_ip`, so it survives the UTC day boundary (F4)."""
    digest = hashlib.sha256(author_id_.encode("utf-8")).hexdigest()
    n = int(digest[:8], 16) % 9000 + 1000
    return f"fly-{n}"


# -- session tokens (F3: bound to author_id) -------------------------------------------------------


@dataclass(frozen=True)
class ChatTokenClaims:
    author_id: str
    issued_at: int
    expires_at: int


def issue_chat_token(secret: str, author_id_: str, ttl_s: float, now: float | None = None) -> str:
    now_i = int(now if now is not None else time.time())
    exp = now_i + int(ttl_s)
    payload = f"{author_id_}.{now_i}.{exp}"
    sig = hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def verify_chat_token(
    token: str, secret: str, author_id_: str, now: float | None = None
) -> ChatTokenClaims | None:
    """`None` if the token is malformed, forged, expired, or was minted for a DIFFERENT
    `author_id_` than the one presenting it now (F3: "reject on mismatch") -- a token solved by
    one client cannot be handed to or replayed from a different IP/subnet."""
    if not token:
        return None
    parts = token.split(".")
    if len(parts) != 4:
        return None
    token_author, iat_s, exp_s, sig = parts
    if token_author != author_id_:
        return None
    payload = f"{token_author}.{iat_s}.{exp_s}"
    expected = hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        iat, exp = int(iat_s), int(exp_s)
    except ValueError:
        return None
    now_f = now if now is not None else time.time()
    if exp <= now_f:
        return None
    return ChatTokenClaims(author_id=token_author, issued_at=iat, expires_at=exp)
