"""Live fly page: a public, moderated question -> real-brain-answer service.

See `docs/LIVE.md` for the architecture. Submodules:

    config.py         env vars, paths, tunables.
    moderation.py      question filtering pipeline (docs/LIVE.md "Moderation policy").
    store.py            SQLite queue + answer store.
    atlas.py             neuron_id -> atlas-index mapping (frontend-produced files).
    states.py             named-population "behavioural state" readouts + live-states.yaml.
    frames.py             per-25ms-bin active-neuron-index encoding for the share/stream visual.
    pipeline.py            pure question -> Answer computation, no Brian2 import at module load.
    worker.py               `python -m bioreservoir.live.worker` — the real-brain loop.
    validate_states.py       `python -m bioreservoir.live.validate_states` — state calibration.
    card.py                    PNG share-card + `/a/{id}` OpenGraph HTML renderer.
    api.py                     FastAPI app (the process `uvicorn` serves).
"""
