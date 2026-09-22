"""chat_admin.py: the operator CLI (docs/LIVE.md "Live chat"). Each test points `LIVE_DB` at a
throwaway file (same convention as tests/test_live_api.py's `client` fixture) and calls `main()`
directly, asserting on stdout (`capsys`) and on the store's resulting state.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = tmp_path / "live.sqlite"
    monkeypatch.setenv("LIVE_DB", str(path))

    import bioreservoir.live.config as live_config

    importlib.reload(live_config)
    return path


@pytest.fixture
def live_store(db_path):
    from bioreservoir.live.store import LiveStore

    s = LiveStore(path=db_path)
    yield s
    s.close()


def test_list_prints_recent_messages(db_path, live_store, capsys):
    from bioreservoir.live import chat_admin

    live_store.chat_send("fly-1", "hello", ip_hash="h1")
    live_store.chat_send("fly-2", "world", ip_hash="h2")
    chat_admin.main(["list"])
    out = capsys.readouterr().out
    assert "hello" in out
    assert "world" in out
    assert "2 message(s)" in out


def test_list_excludes_deleted_by_default(db_path, live_store, capsys):
    from bioreservoir.live import chat_admin

    id1 = live_store.chat_send("fly-1", "will be deleted", ip_hash="h1")
    live_store.chat_delete(id1)
    chat_admin.main(["list"])
    out = capsys.readouterr().out
    assert "will be deleted" not in out
    assert "0 message(s)" in out


def test_list_all_includes_deleted(db_path, live_store, capsys):
    from bioreservoir.live import chat_admin

    id1 = live_store.chat_send("fly-1", "will be deleted", ip_hash="h1")
    live_store.chat_delete(id1)
    chat_admin.main(["list", "--all"])
    out = capsys.readouterr().out
    assert "will be deleted" in out
    assert "deleted:admin" in out


def test_list_reported_shows_only_auto_hidden(db_path, live_store, capsys):
    from bioreservoir.live import chat_admin

    id1 = live_store.chat_send("fly-1", "auto-hidden", ip_hash="h1")
    id2 = live_store.chat_send("fly-2", "admin-deleted", ip_hash="h2")
    live_store.chat_auto_hide(id1)
    live_store.chat_delete(id2)
    chat_admin.main(["list", "--reported"])
    out = capsys.readouterr().out
    assert "auto-hidden" in out
    assert "admin-deleted" not in out


def test_delete_marks_a_message_deleted(db_path, live_store, capsys):
    from bioreservoir.live import chat_admin

    id1 = live_store.chat_send("fly-1", "bye", ip_hash="h1")
    chat_admin.main(["delete", str(id1)])
    assert "deleted" in capsys.readouterr().out
    assert live_store.chat_get(id1)["deleted"] == 1


def test_delete_reports_not_found_for_unknown_id(db_path, live_store, capsys):
    from bioreservoir.live import chat_admin

    chat_admin.main(["delete", "999999"])
    assert "not found" in capsys.readouterr().out


def test_restore_un_hides_a_message(db_path, live_store, capsys):
    from bioreservoir.live import chat_admin

    id1 = live_store.chat_send("fly-1", "back again", ip_hash="h1")
    live_store.chat_auto_hide(id1)
    chat_admin.main(["restore", str(id1)])
    assert "restored" in capsys.readouterr().out
    assert live_store.chat_get(id1)["deleted"] == 0


def test_clear_soft_deletes_every_active_message(db_path, live_store, capsys):
    from bioreservoir.live import chat_admin

    live_store.chat_send("fly-1", "one", ip_hash="h1")
    live_store.chat_send("fly-2", "two", ip_hash="h2")
    chat_admin.main(["clear"])
    assert "cleared 2 message(s)" in capsys.readouterr().out
    assert live_store.chat_recent(limit=10) == []


def test_ban_marks_a_client_hash_banned(db_path, live_store, capsys):
    from bioreservoir.live import chat_admin

    chat_admin.main(["ban", "some-hash"])
    assert "banned some-hash" in capsys.readouterr().out
    assert live_store.chat_is_banned("some-hash") is True


def test_slowmode_updates_the_runtime_interval(db_path, live_store, capsys):
    from bioreservoir.live import chat_admin

    chat_admin.main(["slowmode", "10"])
    assert "10" in capsys.readouterr().out
    assert live_store.chat_state()["slowmode_s"] == pytest.approx(10.0)


def test_off_and_on_toggle_the_runtime_kill_switch(db_path, live_store, capsys):
    from bioreservoir.live import chat_admin

    chat_admin.main(["off"])
    assert live_store.chat_state()["enabled"] is False
    capsys.readouterr()
    chat_admin.main(["on"])
    assert live_store.chat_state()["enabled"] is True


def test_missing_command_exits_nonzero():
    from bioreservoir.live import chat_admin

    with pytest.raises(SystemExit):
        chat_admin.main([])


# -- F4 (2026-09-18 security review): ban-author + author id printed in list ---------------------


def test_list_prints_the_author_id(db_path, live_store, capsys):
    from bioreservoir.live import chat_admin

    live_store.chat_send("fly-1", "hello", ip_hash="some-author-id")
    chat_admin.main(["list"])
    out = capsys.readouterr().out
    assert "author=some-author-id" in out


def test_ban_author_bans_the_message_poster(db_path, live_store, capsys):
    from bioreservoir.live import chat_admin

    id1 = live_store.chat_send("fly-1", "troll message", ip_hash="troll-author-id")
    chat_admin.main(["ban-author", str(id1)])
    out = capsys.readouterr().out
    assert "troll-author-id" in out
    assert live_store.chat_is_banned("troll-author-id") is True


def test_ban_author_reports_message_not_found(db_path, live_store, capsys):
    from bioreservoir.live import chat_admin

    chat_admin.main(["ban-author", "999999"])
    assert "not found" in capsys.readouterr().out


def test_ban_author_handles_purged_identity(db_path, live_store, capsys):
    """F4: identity linkage is blanked after the retention window -- ban-author on such a message
    must not crash or silently ban an empty string."""
    from bioreservoir.live import chat_admin

    id1 = live_store.chat_send("fly-1", "old message", ip_hash="")
    chat_admin.main(["ban-author", str(id1)])
    out = capsys.readouterr().out
    assert "purged" in out.lower() or "cannot" in out.lower()
    assert live_store.chat_is_banned("") is False
