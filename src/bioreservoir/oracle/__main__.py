"""`python -m bioreservoir.oracle <run|score> ...` — see experiments/001-fly-oracle/RUNNING.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bioreservoir.oracle.config import CONFIG_YAML, load_config
from bioreservoir.oracle.ledger import stamp_sha256sums, update_sha256sums
from bioreservoir.oracle.questions import load_questions


def _cmd_run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    question_ids = args.question or [q.id for q in load_questions()]
    brains = tuple(args.brain) if args.brain else ("malecns", "banc")

    if args.dry_run:
        from bioreservoir.oracle.plan import dry_run_report

        report = dry_run_report(question_ids, config, brains=brains, n_parallel=args.workers)
        print(json.dumps(report, indent=2))
        return

    from bioreservoir.oracle.runner import run_all

    result = run_all(question_ids, config, brains=brains, workers=args.workers, resume=not args.no_resume)
    print(json.dumps(result, indent=2))

    from bioreservoir.oracle.ledger import Ledger

    ledger = Ledger()
    export_path = ledger.export_predictions()
    ledger.close()
    if export_path:
        print(f"exported new predictions to {export_path}")
        if args.stamp:
            proof = stamp_sha256sums()
            print(f"ots stamp: {proof.stdout}")
    else:
        print("no new predictions to export")


def _cmd_score(args: argparse.Namespace) -> None:
    from bioreservoir.oracle.ledger import Ledger
    from bioreservoir.oracle.score import build_report, write_scoreboard_json

    config = load_config(args.config)
    ledger = Ledger()
    report = build_report(ledger, cluster_by=config.scoring.cluster_by)
    ledger.close()

    out = write_scoreboard_json(report)
    print(f"wrote {out}")
    print(json.dumps(report, indent=2, default=str))

    if args.stamp:
        update_sha256sums()
        proof = stamp_sha256sums()
        print(f"ots stamp: {proof.stdout}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m bioreservoir.oracle")
    parser.add_argument("--config", type=str, default=str(CONFIG_YAML))
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run the work plan (resumable)")
    run_p.add_argument("--question", action="append", help="repeatable; default: every question in questions.yaml")
    run_p.add_argument("--brain", action="append", choices=["malecns", "banc"], help="repeatable; default: both brains")
    run_p.add_argument("--workers", type=int, default=3)
    run_p.add_argument("--dry-run", action="store_true", help="print the work plan and CPU-hour estimate, run nothing")
    run_p.add_argument("--no-resume", action="store_true", help="rerun even combinations the ledger already has 'done'")
    run_p.add_argument(
        "--stamp",
        action="store_true",
        help="also OTS-stamp predictions/SHA256SUMS after export (opentimestamps-client, 'stamp' extra) "
        "— NOT run by CI or any default invocation, task brief: 'implement but do not run it'",
    )
    run_p.set_defaults(func=_cmd_run)

    score_p = sub.add_parser("score", help="score resolved predictions and write site/data/scoreboard.json")
    score_p.add_argument("--stamp", action="store_true", help="also OTS-stamp predictions/SHA256SUMS — see 'run --stamp'")
    score_p.set_defaults(func=_cmd_score)

    args = parser.parse_args(argv)
    args.config = Path(args.config)
    args.func(args)


if __name__ == "__main__":
    main()
    sys.exit(0)
