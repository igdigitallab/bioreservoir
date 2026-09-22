"""Operator CLI for the live chat (docs/LIVE.md "Live chat" section):

    python -m bioreservoir.live.chat_admin list [--limit N] [--all] [--reported]
    python -m bioreservoir.live.chat_admin delete <id>
    python -m bioreservoir.live.chat_admin restore <id>
    python -m bioreservoir.live.chat_admin clear
    python -m bioreservoir.live.chat_admin ban <client_hash>
    python -m bioreservoir.live.chat_admin ban-author <message_id>
    python -m bioreservoir.live.chat_admin slowmode <seconds>
    python -m bioreservoir.live.chat_admin off
    python -m bioreservoir.live.chat_admin on

Talks directly to the shared `LIVE_DB` SQLite file (`store.LiveStore`, WAL + busy-timeout, same
convention as the queue side) -- no IPC with the running API process. A `delete`/`clear`/`restore`
takes effect on the API process's next `GET /api/events` poll (`events.ChatEventTracker` notices
the changed row and broadcasts `chat_delete`/`chat` accordingly); `off`/`on`/`slowmode` take effect
on the next poll too, broadcast as `chat_state`.

`<client_hash>` throughout is `chat.author_id(ip, CHAT_ID_PEPPER)` (2026-09-18 security review
F4) -- a STABLE identity, not the daily-rotating one `hash_ip` uses for the question queue, so a
ban survives the UTC day boundary. `list` prints it (previously it didn't, making `ban
<client_hash>` unusable without raw sqlite access); `ban-author <message_id>` looks it up from one
message for the common case of "ban whoever posted this" without needing to copy-paste the hash.
"""

from __future__ import annotations

import argparse
import sys

from bioreservoir.live.store import LiveStore


def _print_row(row, store: LiveStore) -> None:
    status = f"deleted:{row['deleted_reason']}" if row["deleted"] else "active"
    author = row["ip_hash"] or "(purged)"
    # R1 (round-2 security review): a message that crossed the report threshold but was NOT
    # auto-hidden (per-reporter or site-wide cap reached) stays active with an elevated report
    # count -- this is its "operator review queue", so `list` must surface the
    # count even for active, never-hidden rows, not just already-auto-hidden ones.
    reports = store.chat_report_count(row["id"])
    reports_note = f"  reports={reports}" if reports else ""
    print(
        f"[{row['id']:>6}] ({status}) {row['created_at']}  author={author}{reports_note}  "
        f"{row['nickname']}: {row['text']}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bioreservoir.live.chat_admin")
    sub = parser.add_subparsers(dest="command", required=True)

    list_p = sub.add_parser("list", help="print recent chat messages")
    list_p.add_argument("--limit", type=int, default=100)
    list_p.add_argument("--all", action="store_true", help="include deleted messages")
    list_p.add_argument("--reported", action="store_true", help="only auto-hidden (reported) messages")

    delete_p = sub.add_parser("delete", help="soft-delete one message")
    delete_p.add_argument("id", type=int)

    restore_p = sub.add_parser("restore", help="un-hide a deleted/auto-hidden message")
    restore_p.add_argument("id", type=int)

    sub.add_parser("clear", help="soft-delete every currently-active message")

    ban_p = sub.add_parser("ban", help="ban a hashed client from posting")
    ban_p.add_argument("client_hash")

    ban_author_p = sub.add_parser("ban-author", help="ban whoever posted this message id")
    ban_author_p.add_argument("message_id", type=int)

    slowmode_p = sub.add_parser("slowmode", help="set the runtime per-client minimum interval")
    slowmode_p.add_argument("seconds", type=float)

    sub.add_parser("off", help="pause chat at runtime, no redeploy (LIVE_CHAT_ENABLED stays the hard default)")
    sub.add_parser("on", help="resume chat at runtime")

    args = parser.parse_args(argv)

    with LiveStore() as s:
        if args.command == "list":
            rows = s.chat_reported(limit=args.limit) if args.reported else s.chat_recent(
                limit=args.limit, include_deleted=args.all
            )
            for row in rows:
                _print_row(row, s)
            print(f"{len(rows)} message(s)")
        elif args.command == "delete":
            print("deleted" if s.chat_delete(args.id) else "not found or already deleted")
        elif args.command == "restore":
            print("restored" if s.chat_restore(args.id) else "not found or not deleted")
        elif args.command == "clear":
            print(f"cleared {s.chat_clear()} message(s)")
        elif args.command == "ban":
            s.chat_ban(args.client_hash)
            print(f"banned {args.client_hash}")
        elif args.command == "ban-author":
            row = s.chat_get(args.message_id)
            if row is None:
                print("message not found")
            elif not row["ip_hash"]:
                print("author identity already purged for this message (see retention policy) -- cannot ban")
            else:
                s.chat_ban(row["ip_hash"])
                print(f"banned {row['ip_hash']} (author of message {args.message_id})")
        elif args.command == "slowmode":
            s.chat_set_slowmode(args.seconds)
            print(f"slow mode set to {args.seconds}s")
        elif args.command == "off":
            s.chat_set_enabled(False)
            print("chat paused (runtime) -- LIVE_CHAT_ENABLED env var is unchanged")
        elif args.command == "on":
            s.chat_set_enabled(True)
            print("chat resumed (runtime)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
