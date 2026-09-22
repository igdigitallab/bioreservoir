"""chat.py: chat session tokens (issue/verify, bound to author_id per security review F3), the
anonymous-handle generator, author identity (F4), and nickname-impersonation checks (F5). Pure,
no I/O.
"""

from __future__ import annotations

from bioreservoir.live.chat import (
    anonymous_handle,
    author_id,
    check_nickname_format,
    is_reserved_nickname,
    issue_chat_token,
    normalize_ip_for_author,
    normalize_ip_for_report_group,
    report_group_id,
    verify_chat_token,
)

SECRET = "test-secret-do-not-use-in-prod"
PEPPER = "test-pepper-do-not-use-in-prod"

# -- nickname format -------------------------------------------------------------------------


def test_nickname_format_accepts_valid_nicknames():
    for nick in ("ab", "a" * 20, "fly_fan-99", "MaleCNS", "a1"):
        assert check_nickname_format(nick), nick


def test_nickname_format_rejects_too_short():
    assert not check_nickname_format("a")


def test_nickname_format_rejects_too_long():
    assert not check_nickname_format("a" * 21)


def test_nickname_format_rejects_disallowed_characters():
    for nick in ("bad name", "bad@name", "bad.name", "bad/name", "über", "😀😀"):
        assert not check_nickname_format(nick), nick


# -- nickname impersonation (F5) ---------------------------------------------------------------


def test_reserved_nickname_blocks_the_task_brief_list():
    for nick in (
        "admin", "mod", "moderator", "staff", "official", "system",
        "bioreservoir", "igdigi", "igdigitallab", "ig-digital-lab", "ig_digital_lab",
        "the-fly", "the_fly",
    ):
        assert is_reserved_nickname(nick), nick


def test_reserved_nickname_blocks_case_variants():
    for nick in ("Admin", "ADMIN", "Moderator", "BioReservoir", "IGDigitalLab", "Official"):
        assert is_reserved_nickname(nick), nick


def test_reserved_nickname_blocks_the_server_handle_pattern():
    """Only the server may assign fly-<digits> (task brief F5) -- a user picking one is rejected."""
    for nick in ("fly-4821", "fly4821", "FLY-99", "fly-1"):
        assert is_reserved_nickname(nick), nick


def test_reserved_nickname_blocks_ascii_leetspeak_r4():
    """Round-2 security review R4: check_nickname_format's ASCII-only gate (^[A-Za-z0-9_-]{2,20}$)
    runs BEFORE is_reserved_nickname, so a Cyrillic/Greek confusable glyph can never reach this
    function in the real POST /api/chat flow -- round-1's Unicode-confusables map was reachable
    only via a direct call bypassing that gate, i.e. dead code for the real caller chain (see
    chat.py's own comment on _LEETSPEAK_MAP, which replaces it). ASCII leetspeak, by contrast,
    passes the format gate untouched and is the real bypass round-2 reproduced."""
    for nick in ("adm1n", "m0d", "0fficial", "st4ff", "sy5tem", "Bi0Reserv0ir", "1gdigi"):
        assert is_reserved_nickname(nick), nick


def test_reserved_nickname_leetspeak_does_not_corrupt_the_fly_handle_digit_suffix():
    """The digit-leetspeak map ("1"->"i", "4"->"a", ...) must not be applied to the numeric ID
    portion of a fly-<digits> handle attempt, or genuine impersonation attempts stop matching
    (regression this exact bug hit during implementation: "f1y-4821" -> leetspeak -> "fiy-a82i",
    which no longer looks like digits at all)."""
    for nick in ("fIy-4821", "f1y-4821", "FIy-4821", "fly-4821", "fly4821"):
        assert is_reserved_nickname(nick), nick


def test_reserved_nickname_blocks_zero_width_insertion():
    assert is_reserved_nickname("ad\u200bmin")


def test_reserved_nickname_allows_ordinary_nicknames():
    for nick in ("quantum_fan", "skeptic_sam", "flylover", "night_owl", "BioNerd22"):
        assert not is_reserved_nickname(nick), nick


def test_reserved_nickname_blocks_the_coordinator_r4_test_list():
    """The exact regression list from the round-2 security-review follow-up instructions."""
    for nick in ("fIy-4821", "f1y-4821", "0fficial", "Bi0Reservoir", "adm1n", "st4ff"):
        assert is_reserved_nickname(nick), nick


def test_reserved_nickname_blocks_round2_added_words():
    for nick in ("flyoracle", "FlyOracle", "igor", "Igor", "support", "verified", "team", "lab", "bot"):
        assert is_reserved_nickname(nick), nick


def test_reserved_nickname_folds_capital_i_and_lowercase_l_round3():
    # In Inter, "I" and "l" render as the same bar (security review round 3, S2).
    for nick in ("OfficiaI", "FIyOracle", "FlyOracIe", "Officlal", "Admln", "lgor"):
        assert is_reserved_nickname(nick), nick
    for nick in ("fly_fan", "neuro_nerd", "sky_watcher"):
        assert not is_reserved_nickname(nick), nick


# -- IP normalization (F3/F4) --------------------------------------------------------------------


def test_normalize_ipv4_for_author_is_unchanged():
    assert normalize_ip_for_author("203.0.113.42") == "203.0.113.42"


def test_normalize_ipv6_for_author_collapses_to_slash_64():
    a = normalize_ip_for_author("2001:db8::1")
    b = normalize_ip_for_author("2001:db8::2")
    assert a == b == "2001:db8::"


def test_normalize_ipv6_for_author_differs_across_slash_64s():
    a = normalize_ip_for_author("2001:db8:0:0::1")
    b = normalize_ip_for_author("2001:db8:0:1::1")
    assert a != b


def test_normalize_ip_for_author_passes_through_unparsable_input():
    assert normalize_ip_for_author("unknown") == "unknown"


def test_normalize_ipv4_for_report_group_is_a_slash_24():
    assert normalize_ip_for_report_group("203.0.113.42") == "203.0.113.0"
    assert normalize_ip_for_report_group("203.0.113.200") == "203.0.113.0"


def test_normalize_ipv4_for_report_group_differs_across_slash_24s():
    assert normalize_ip_for_report_group("203.0.113.1") != normalize_ip_for_report_group("203.0.114.1")


def test_normalize_ipv6_for_report_group_is_also_slash_64():
    assert normalize_ip_for_report_group("2001:db8::1") == normalize_ip_for_author("2001:db8::1")


# -- author_id / report_group_id (F4) ------------------------------------------------------------


def test_author_id_is_stable_for_the_same_ip():
    assert author_id("1.2.3.4", PEPPER) == author_id("1.2.3.4", PEPPER)


def test_author_id_is_stable_across_ipv6_slash_64():
    assert author_id("2001:db8::1", PEPPER) == author_id("2001:db8::2", PEPPER)


def test_author_id_differs_with_a_different_pepper_or_ip():
    base = author_id("1.2.3.4", PEPPER)
    assert base != author_id("1.2.3.4", "a-different-pepper")
    assert base != author_id("5.6.7.8", PEPPER)


def test_author_id_never_rotates_daily_unlike_hash_ip():
    """No day parameter exists at all -- author_id has nothing that could rotate (F4: bans must
    not evaporate at the UTC day boundary)."""
    assert author_id("1.2.3.4", PEPPER) == author_id("1.2.3.4", PEPPER)


def test_report_group_id_differs_from_author_id_for_ipv4():
    # Two IPs in the same /24 but different addresses: same report_group_id, different author_id.
    assert report_group_id("203.0.113.1", PEPPER) == report_group_id("203.0.113.2", PEPPER)
    assert author_id("203.0.113.1", PEPPER) != author_id("203.0.113.2", PEPPER)


# -- anonymous handle -------------------------------------------------------------------------


def test_anonymous_handle_is_stable_for_the_same_author_id():
    assert anonymous_handle("h1") == anonymous_handle("h1")


def test_anonymous_handle_differs_across_author_ids_in_general():
    handles = {anonymous_handle(f"h{i}") for i in range(20)}
    assert len(handles) > 1


def test_anonymous_handle_matches_the_documented_shape():
    handle = anonymous_handle("some-author-id")
    assert handle.startswith("fly-")
    assert handle[4:].isdigit()
    assert 1000 <= int(handle[4:]) <= 9999


# -- session token: issue / verify / expiry / author-binding (F3) -------------------------------


def test_issued_token_verifies_immediately_for_its_own_author():
    token = issue_chat_token(SECRET, "author-a", ttl_s=3600, now=1000.0)
    assert verify_chat_token(token, SECRET, "author-a", now=1000.0) is not None


def test_token_is_valid_until_just_before_expiry():
    token = issue_chat_token(SECRET, "author-a", ttl_s=100, now=1000.0)
    assert verify_chat_token(token, SECRET, "author-a", now=1099.0) is not None


def test_token_expires_after_ttl():
    token = issue_chat_token(SECRET, "author-a", ttl_s=100, now=1000.0)
    assert verify_chat_token(token, SECRET, "author-a", now=1101.0) is None


def test_token_rejected_with_wrong_secret():
    token = issue_chat_token(SECRET, "author-a", ttl_s=3600, now=1000.0)
    assert verify_chat_token(token, "a-different-secret", "author-a", now=1000.0) is None


def test_token_rejected_when_presented_by_a_different_author_f3():
    """F3's core fix: a token solved by one client cannot be replayed by a different
    IP/subnet -- verified by presenting the SAME token but a DIFFERENT author_id."""
    token = issue_chat_token(SECRET, "author-a", ttl_s=3600, now=1000.0)
    assert verify_chat_token(token, SECRET, "author-b", now=1000.0) is None


def test_token_rejected_when_tampered():
    token = issue_chat_token(SECRET, "author-a", ttl_s=3600, now=1000.0)
    author, iat, exp, _sig = token.split(".")
    tampered = f"{author}.{iat}.{int(exp) + 999999}.{'0' * 64}"
    assert verify_chat_token(tampered, SECRET, "author-a", now=1000.0) is None


def test_token_rejected_when_author_field_is_tampered_without_resigning():
    """Swapping just the author_id field (without a matching signature) must fail -- otherwise
    an attacker could claim any identity by editing the token's plaintext prefix."""
    token = issue_chat_token(SECRET, "author-a", ttl_s=3600, now=1000.0)
    _author, iat, exp, sig = token.split(".")
    forged = f"author-b.{iat}.{exp}.{sig}"
    assert verify_chat_token(forged, SECRET, "author-b", now=1000.0) is None


def test_verify_rejects_malformed_tokens():
    for bad in ("", "no-dot-here", "a.b.c", "a.b.c.d.e", "author.notanumber.deadbeef.sig"):
        assert verify_chat_token(bad, SECRET, "author-a") is None


def test_verify_rejects_empty_token():
    assert verify_chat_token("", SECRET, "author-a") is None


def test_verify_returns_claims_with_issued_at_for_session_age_checks():
    token = issue_chat_token(SECRET, "author-a", ttl_s=3600, now=1000.0)
    claims = verify_chat_token(token, SECRET, "author-a", now=1000.0)
    assert claims is not None
    assert claims.author_id == "author-a"
    assert claims.issued_at == 1000
    assert claims.expires_at == 4600
