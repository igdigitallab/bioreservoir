"""`GET /api/stats` — aggregate statistics over every answered question in the live store (task
brief: "no fake values; zeros are fine"). Pure function over plain dicts (already-parsed
`answer_json` rows + their `created_at`), no SQLite access here — `store.LiveStore.all_answered`
owns the query, this module owns only the aggregation, so it is independently unit-testable.

Legacy rows with no `lab` object (should not exist once this ships, but defensive anyway) simply
contribute 0 to every `lab`-derived number rather than raising — the same "zeros are fine" rule
the task brief states for an empty store.
"""

from __future__ import annotations

from bioreservoir.live import states as states_mod


def compute_stats(answered_rows: list[dict], earliest_created_at: str | None) -> dict:
    """`answered_rows`: `[{"answer": <Answer dict>, "created_at": iso8601}, ...]`, one per
    `status='answered'` row currently in the store, oldest first (order does not matter here,
    every statistic below is order-independent)."""
    answered = len(answered_rows)
    yes = sum(1 for r in answered_rows if r["answer"].get("answer") == "yes")
    no = sum(1 for r in answered_rows if r["answer"].get("answer") == "no")

    total_spikes = 0
    simulated_ms = 0.0
    neuron_updates = 0
    abs_biases: list[float] = []
    active_fractions: list[float] = []
    state_counts = dict.fromkeys(states_mod.STATE_NAMES, 0)

    for r in answered_rows:
        a = r["answer"]
        sim_ms = float(a.get("sim_ms") or 0.0)
        n_trials = int(a.get("n_trials") or 0)
        simulated_ms += sim_ms * n_trials
        abs_biases.append(abs(float(a.get("lateral_bias") or 0.0)))

        lab = a.get("lab")
        if lab:
            total_spikes += int(lab.get("total_spikes") or 0)
            active_fractions.append(float(lab.get("active_fraction") or 0.0))
            provenance = lab.get("provenance") or {}
            n_neurons = provenance.get("n_neurons")
            dt_ms = provenance.get("dt_ms")
            if n_neurons and dt_ms:
                steps = sim_ms / dt_ms
                neuron_updates += round(n_neurons * n_trials * steps)

        states_dict = a.get("states") or {}
        for name in states_mod.STATE_NAMES:
            if states_dict.get(name) is not None:
                state_counts[name] += 1

    by_question: dict[str, list[str]] = {}
    for r in answered_rows:
        a = r["answer"]
        by_question.setdefault(a.get("question", ""), []).append(a.get("answer"))
    repeated_questions = sum(1 for answers in by_question.values() if len(answers) >= 2)
    identical_answers = sum(
        1 for answers in by_question.values() if len(answers) >= 2 and len(set(answers)) == 1
    )

    return {
        "answered": answered,
        "yes": yes,
        "no": no,
        "total_spikes": total_spikes,
        "simulated_ms": simulated_ms,
        "neuron_updates": neuron_updates,
        "mean_abs_corrected_bias": (sum(abs_biases) / len(abs_biases)) if abs_biases else 0.0,
        "mean_active_fraction": (sum(active_fractions) / len(active_fractions)) if active_fractions else 0.0,
        "state_counts": state_counts,
        "repeat_consistency": {
            "repeated_questions": repeated_questions,
            "identical_answers": identical_answers,
        },
        "since": earliest_created_at,
    }
