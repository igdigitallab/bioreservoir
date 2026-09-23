# Live fly page

Public, moderated "ask the fly" service. A visitor submits a yes/no question -> moderation -> queue -> the real male
brain (MaleCNS) answers it -> share card. Format B ("fly's take"), entertainment, not a forecast —
distinct from experiment 001's scored pipeline (`bioreservoir.oracle`), which it reuses
(`oracle.readout`/`handedness`/`side_mapping`, `experiments/001-fly-oracle/config.yaml`'s encoder
settings) rather than duplicating.

## Architecture

Two processes, one SQLite file (`LIVE_DB`):

- **API** (`bioreservoir.live.api`, FastAPI/`uvicorn`): `POST /api/ask` (moderate -> enqueue),
  `GET /api/answers/{id}`, `GET /api/now`, `GET /api/feed`, `GET /api/events` (SSE),
  `GET /api/card/{id}.png` (share image), `GET /a/{id}` (OpenGraph HTML), `GET /api/stats`,
  `GET /api/validation`, `GET /healthz`.
- **Worker** (`bioreservoir.live.worker`, `python -m bioreservoir.live.worker`): loads the male
  brain ONCE, polls the queue FIFO, runs `LIVE_N_TRIALS` x `trial.duration_ms` trials per question,
  writes the answer back. Must run inside the memory/CPU cage described in
  `experiments/001-fly-oracle/RUNNING.md` — the simulation will starve any host it runs on
  unmetered.

`bioreservoir.live.pipeline.compute_answer` is the pure question -> Answer boundary between them
(no Brian2 import), so it and everything upstream of a real `LIFNetwork` call is unit-testable.

## API contract (v2, 2026-09-19: built for a crowd that mostly waits)

A visitor watches the fly answer OTHER people's questions while their own waits in line, so every
public surface carries a lean **summary** and the heavy answer is fetched once, from a cacheable
URL.

`AnswerSummary` (`live/summary.py`), used by `/api/feed`, `/api/now` and the SSE `answered` event:
`{id, question, answer, embargoed, yes_side, lateral_bias, turn_strength, answered_at, likes}` — no spike
data (a full answer is ~150 KB, a summary a few hundred bytes). For an election question before
`ELECTION_CARD_EMBARGO_UNTIL`, `embargoed` is true and every verdict field is `null` on these
public surfaces; `GET /api/answers/{id}` still returns the full answer (the asker needs it) and
flags `embargoed` so the share page can hide the verdict from everyone else.

`turn_strength` (`verdict.answer_turn_strength`) = `|corrected bias| / 0.1375`, clipped to 0..1 —
the share of the turn the same brain makes when one antenna's Johnston's-organ neurons are driven
(+0.118 / −0.157, `docs/MODEL.md` §Calibration). It replaces the old `confidence` on every visible
surface: that scale started at 50 %, so every real answer read as a medium-strength turn.

| Route | Returns | `Cache-Control` |
|---|---|---|
| `GET /api/now` | `{thinking: {id, question, started_at, embargoed, yes_side} \| null, last: AnswerSummary \| null, queue_length, claimed_total, avg_cycle_s}` | `public, max-age=0, s-maxage=2` |
| `GET /api/feed?limit=20` | `{queue_length, recent: AnswerSummary[]}` | `public, max-age=0, s-maxage=3` |
| `GET /api/questions?sort=recent\|top&limit=20&offset=0` | `{sort, total, has_more, items: AnswerSummary[]}` — the archive of answered questions, newest or most liked first. Nothing per-visitor in it, so every tab shares one cached page | `public, max-age=0, s-maxage=10` |
| `POST /api/answers/{id}/like` | `{id, likes, liked}` — toggles this client's like. One like per (question, client) is the `live_question_likes` primary key, so a double tap takes the like back instead of counting twice | `no-store` |
| `GET /api/answers/{id}` | queued: `{id, question, status, position, claimed_total, avg_cycle_s, yes_side}`; thinking: `+started_at`; answered: `{…, embargoed, answer: <full Answer>}` | answered: `public, max-age=31536000, immutable`; otherwise `no-store` |
| `POST /api/ask` | queued: `{id, status, position, claimed_total, avg_cycle_s, joined}`; repeat: `{id, status:"answered", repeat:true}`; rejected: `{status, reason, message}` (422) | — |
| `GET /api/chat?before_id=&limit=` | `{messages, state, has_more}` | — |
| `GET /api/stats`, `GET /api/validation` | unchanged | `s-maxage=10` / `s-maxage=300` |

**SSE `GET /api/events`.** One `events.Broadcaster` per process polls the store, serialises each
event once and fans it out to per-subscriber bounded queues (oldest event dropped for a subscriber
that falls behind). Events: `thinking`, `answered` (summary), `queue`
`{queue_length, claimed_total, avg_cycle_s}` (on change, and at least every 30 s), plus the chat's
`chat`/`chat_delete`/`chat_state`. There is no per-question `queued` broadcast any more. A client
computes its live place in line without polling:
`max(1, position_at_ask − (claimed_total_now − claimed_total_at_ask))`, and an ETA from
`avg_cycle_s` (median started→answered of the last 20 answers, floor 5 s, fallback 60 s).

**Ask dedupe.** `POST /api/ask` runs Turnstile and the per-IP limits for every ask, then looks the
normalized wording (`pipeline.question_key`) up: an answered twin returns at once (`repeat` — same
words, same seeds, same answer; the UI labels it "asked before"), a queued/thinking twin is joined
(`joined`, no second simulation and no second LLM call), a content-rejected twin is rejected with
the same reason. A unique partial index on `question_key` for `queued`/`thinking` rows makes two
simultaneous identical asks safe. Because a deduped ask creates no row, the per-IP limit also
counts attempts in memory (`live/attempts.py`).

**Moderation under load.** The LLM gate is capped at a few hundred calls a minute. When it
rate-limits or times out (`moderation.LLM_BUSY`), the question is NOT rejected: it joins the line
with `needs_llm`, and the worker classifies it right before simulating (one call per answered
question). A rejection there shows up on the asker's own `GET /api/answers/{id}`; nothing about it
is ever broadcast.

## Env vars (names only — set via your deployment's secret store or environment, never commit values)

| Var | Purpose |
|---|---|
| `LIVE_DB` | queue/answer SQLite path (default `data/live/live.sqlite`) |
| `LIVE_ENV` | `dev` skips Turnstile/LLM-gate when their vars are unset; anything else fails closed |
| `LIVE_RATE_PER_MINUTE`, `LIVE_RATE_PER_DAY`, `LIVE_QUEUE_CAP_PER_IP` | moderation step 3 |
| `TURNSTILE_SECRET` | Cloudflare Turnstile server secret |
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | OpenAI-compatible moderation classifier (fleet's LiteLLM gateway, model group `auto-json`) |
| `LIVE_SIDE_BASE`, `LIVE_POLL_INTERVAL_S`, `LIVE_SSE_HEARTBEAT_S` | tuning, see `live/config.py` |
| `LIVE_B0_CACHE` | local handedness-baseline cache path (worker fallback) |
| `BIORESERVOIR_CODE_SHA` | fallback for `provenance.code_sha` when `oracle.ledger.git_sha()` finds no `.git` (the deployed container's case — see infra `stacks/bioreservoir-live/Dockerfile`'s build ARG, set by `deploy.sh` to its verified `SOURCE_PIN`) |
| `LIVE_CHAT_ENABLED` | hard, deploy-time chat kill switch (default on; `0` -> chat routes 503, frontend hides the chat entirely). See "Live chat" below for the separate runtime-adjustable switch |
| `CHAT_SESSION_SECRET` | HMAC secret for chat session tokens. **Generate with `openssl rand -hex 32`**. ⚠️ **No default value** (2026-09-18 security review F8) — if unset, ALL chat routes return 503 and it's logged once at API startup. Set via the deploy environment's secret store |
| `CHAT_ID_PEPPER` | HMAC pepper for the stable chat author identity (bans/reports/handles/token-binding — see "Live chat" below). **Generate with `openssl rand -hex 32`**. ⚠️ **No default value** — same 503-and-log-once behaviour as `CHAT_SESSION_SECRET` if unset |
| `CHAT_LLM_API_KEY` | Chat's OWN LiteLLM virtual key — **never** falls back to `LLM_API_KEY` (the question pipeline's key), so chat abuse exhausting its own gateway budget cannot take `/api/ask` down too. ⚠️ **No default value** — same 503-and-log-once behaviour if unset. Same `LLM_BASE_URL`/`LLM_MODEL` gateway, different key |
| `CHAT_TOKEN_TTL_S` | chat session token lifetime in seconds (default `3600`) |
| `CHAT_DEFAULT_SLOWMODE_S` | seed value only, for the FIRST time a given `LIVE_DB` file is created (default `4.0`); after that, `chat_admin slowmode <seconds>` owns the live value |
| `CHAT_RATE_PER_10MIN` | per-client chat *attempts* (accepted, rejected, or failed — see F1/F2 below) allowed in a rolling 10-minute window (default `15`) |
| `CHAT_REPORT_THRESHOLD` | distinct eligible reporters that auto-hide a message pending operator review (default `5`, raised from `3` per the security review) |
| `CHAT_REPORT_MIN_SESSION_AGE_S` | a report only counts toward the threshold once the reporter's session token is at least this old (default `600` = 10 min) |
| `CHAT_REPORTS_PER_10MIN` | round-2 R1: per-author cap on report ACTIONS that can trigger an auto-hide in a 10-minute window (default `5`) — a report beyond this is still recorded, just doesn't hide |
| `CHAT_AUTO_HIDE_PER_10MIN` | round-2 R1: site-wide cap on how many messages can be auto-hidden in a 10-minute window, regardless of how many distinct reporter identities are involved (default `10`) |
| `CHAT_LLM_MAX_PER_MINUTE` | site-wide circuit breaker: max chat LLM classification calls per minute, independent of any one client (default `200`, raised from `30` per round-2 R2 — the old default tripped under ~35 ordinary concurrent chatters) |
| `CHAT_LLM_PER_AUTHOR_PER_MINUTE` | round-2 R2: per-author LLM-classification quota, checked BEFORE the site-wide breaker, so a handful of identities can't keep it tripped for everyone (default `6`) |

## Run locally

```bash
uv sync --extra sim --extra encode --extra live
export LIVE_ENV=dev  # skips Turnstile/LLM if those vars are unset
uv run uvicorn bioreservoir.live.api:app --reload
uv run python -m bioreservoir.live.worker   # separate process, needs the CPU cage in prod
```

## Moderation policy (`live/moderation.py`)

Order: normalize the text (zero-width characters dropped, smart apostrophes and the full-width
question mark made plain, whitespace collapsed; what is stored and simulated is the normalized text)
-> length <=140 -> yes/no heuristic -> per-IP rate limit/queue cap -> Turnstile -> deterministic
rule blocklists (`voting_procedure` is a **hard block**, wins over everything including the LLM;
election OUTCOME questions are allowed; the profanity list for questions allows words that are also
names or topics, `QUESTION_PROFANITY_ALLOWED`, and leaves them to the LLM) -> LLM classifier
fallback, prompted to default to ok and block only real harm. LLM busy, erroring or unparsable: the
question is queued with `needs_llm` and the worker classifies it before simulating; if that fails
too it is rejected with a plain "could not check" message. Not configured outside `LIVE_ENV=dev`:
fails closed (`spam`). Nothing is simulated without a verdict. Rejections are logged without raw IPs
(`store.py`'s daily-random-salt IP hash, never a raw IP on disk).

## Data retention

Rejected questions: text blanked after 7 days (`store.purge_expired`, called periodically by the
worker loop), row kept for aggregate stats. Answered questions: kept indefinitely (site content).

## Behavioural states

`experiments/001-fly-oracle/live-states.yaml` gates which of appetite/fear/backoff/courtship/
arousal are published (`validated: true`) vs. `null`. Only `arousal` starts validated (a fraction,
0..1 by construction). Run `python -m bioreservoir.live.validate_states` (cage-only, run manually)
to propose the rest after real trials exist.

## Atlas (frontend-owned)

`site/public/data/atlas/{meta.json,neuron_ids.u64}` do not exist in this repo yet. `live/atlas.py`
is written against the documented format and tested with a fixture; the worker degrades to
`frames: null` (no per-neuron visual) until the frontend agent generates them.

## `Answer.lab` — proof this is a real simulation, not a game

Operator rule (2026-09-18): every number in `lab` comes from the actual trial run or a committed
file — nothing decorative. Built by `live/lab.py` (pure) from `live/pipeline.py`'s already-run
`LiveTrial`s and a `LabContext` (graph metadata + annotation lookups, built once at worker startup
by `worker.build_resources`). Fields:

| Field | Meaning |
|---|---|
| `trials` | one `{seed, spikes_left, spikes_right, bias}` per trial — the actual Poisson seed and readout spike counts |
| `b0`, `corrected_bias`, `yes_side` | the handedness correction and per-question coin that produced `answer`/`lateral_bias` |
| `stimulated` | how many neurons were driven, per side and per raw dataset modality (`class`/`cell_class` column) |
| `active_neurons`, `active_fraction`, `total_spikes` | whole-network activity for this answer, raw (not normalized like `states.arousal`) |
| `readout_latency_ms` | mean, over trials that fired one, of the first `descending_all` spike after stimulus onset (`null` if none did) |
| `top_cell_types` | top 12 non-stimulated cell types by spike count, with `n_neurons`/`spikes`/`rate_hz` |
| `raster` | <=300 neurons (every readout neuron + a seeded sample of trial-0's active ones), `atlas_indices` (`-1` if no atlas yet), `cell_types`, `spikes_b64` (little-endian uint16 `(row, 0.1ms-tick)` pairs, trial 0 only) |
| `provenance` | brain/graph/model identity, `code_sha`, `config_hash`, and the exact `reproduce` CLI command for this answer |

## The game — "which one is the real fly?" (removed from the site, 2026-09-19)

> The game was removed from the live page on 2026-09-19 (`LIVE_GAME=0`, the frontend code
> is gone). The backend is still described below because the flag still exists; nothing here runs
> with the game off.

Seven external critics on the public-launch virality review found the honest problem: the real
brain answers near 50/50, and the random-graph control *looks* more confident. Instead of hiding
that, every answer now runs THREE contenders under identical conditions and lets the visitor guess
which is real (`live/game.py`, `site/src/pages/game.ts`).

| Contender | What it is | Own b0? |
|---|---|---|
| `real` | the live page's existing MaleCNS `LIFNetwork` — unchanged | yes, `(malecns, real)` |
| `random_graph` | Erdos-Renyi control, same neuron/edge count (`oracle.controls.erdos_renyi_like`), the EXACT cached graph the sealed batch scores against (`config.yaml`'s `seed.er_base`, `data/processed/malecns-min<N>-er-seed<er_base>/`) | yes, `(malecns, er)` |
| `no_brain` | `oracle.controls.no_brain_baseline` — the readout rule applied to the encoder's own left/right input rates, no connectome at all | yes, `(malecns, no_brain)` |

All three share the SAME encoded question (`id_to_dense` and neuron ordering are identical for
`real`/`random_graph` — the ER graph is a permutation of the real graph's own arrays, built from
the same `bench.graph_to_arrays` call), the SAME per-trial Poisson seeds (`worker.live_trial_seed`,
called identically for the real trials and the ER submission), and the SAME per-question yes-side
coin (`side_mapping.left_is_yes_for`, computed once in `pipeline.compute_answer` and passed into
`game.build_game`) — but each contender's raw bias is corrected against its OWN b0, never the real
brain's, exactly mirroring how `oracle.runner` scores the sealed batch's three graph/no-brain
conditions.

`Answer.game` (absent — not `null` — on every answer computed before this shipped, or whenever
`worker.answer_question` is called with no `er_pool`):
```json
{"order": ["random_graph", "real", "no_brain"],
 "contenders": {
   "real": {"answer": "yes", "corrected_bias": 0.114, "decisiveness": 0.69},
   "random_graph": {"answer": "no", "corrected_bias": -0.05, "decisiveness": 0.58},
   "no_brain": {"answer": "yes", "corrected_bias": 0.02, "decisiveness": 0.53}}}
```
`order[i]` is slot `"ABC"[i]` — deterministically shuffled per question (`game.contender_order`,
seeded only by the normalized question text, its own seed base distinct from every other coin in
this pipeline) so asking the same wording twice gives the same order, but there is no cross-
question pattern a repeat asker could learn. `decisiveness` is `verdict.confidence_from_bias`
applied to that contender's own corrected bias — the SAME formula/scale the real answer's
top-level `confidence` already used, so the three bars are comparable.

**Concurrency (`game.ErContenderPool`).** The random graph needs a real Brian2 simulation, which
would double per-question wall time if run after the real brain's own trials. Its `LIFNetwork`
lives in a SEPARATE OS process, spawned once at worker startup
(`ProcessPoolExecutor(max_workers=1)`) and reused across every question via the same
store/restore pattern `LIFNetwork` itself uses. `worker.answer_question` submits the ER trials
(`er_pool.submit_trials(...)`, non-blocking) BEFORE running the real brain's own trial loop, then
calls `.result()` on the returned future after that loop finishes — the two Brian2 runs overlap
instead of adding up. `no_brain` needs no simulation at all (a few numpy ops on the already-
computed encoded rates), so it costs nothing extra.

**b0 for `random_graph`/`no_brain`.** Same order as the real brain's own b0
(`worker.get_or_compute_b0`, generalized with a `condition` kwarg): main ledger (read-only) first,
then the local JSON cache, then computed from scratch and cached. `random_graph`'s reference-
sentence trials run against the ER pool (`game.compute_er_b0_via_pool`, blocking — this only runs
once, at worker startup, so nothing needs to overlap it with); `no_brain`'s (`worker.
compute_no_brain_b0_locally`) never touch Brian2 at all, so it is cheap regardless. **Verified
2026-09-19 against the sealed batch's own ledger:** `(malecns, er, <current config_hash>)` and
`(malecns, no_brain, <current config_hash>)` both already have all 24 reference runs `done` — the
same ledger the deploy already copies in for the real brain's b0 (this doc's env var table,
`MAIN_LEDGER_PATH`) also satisfies these two, so a fresh worker start hits the fast (ledger) path
for all three conditions and never falls into the expensive from-scratch computation. Confirm this
still holds after any future re-run of the sealed batch or any `config.yaml` change (a new
`config_hash` needs fresh reference runs for `er`/`no_brain` too, not just `real`).

**Measured cost (2026-09-19, in the CPU/memory cage, one real `LIFNetwork` + one real ER
`LIFNetwork` running one real question concurrently — see the cage rule in
`experiments/001-fly-oracle/RUNNING.md`, never run outside it):**

| | measured |
|---|---|
| main worker process peak RSS (real net + encoder + one answer) | 1703 MB |
| ER subprocess peak RSS (`RUSAGE_CHILDREN` after `shutdown()`) | 1899 MB |
| combined peak (both at their own peak, worst case, same container) | **3601 MB** |
| one `answer_question` call, real trials + ER collected concurrently | 43.4 s (3x250ms trials) |

The live worker container's current `mem_limit` is 3 GB — below the measured 3601 MB combined
peak. **Recommend raising it to at least 4.5 GB** (measured peak + ~25% headroom for the FastAPI
process's own overhead, atlas/annotation lookups, and growth over a long-running process); this is
an `infra` repo change, out of this repo's scope. The 43.4 s per-question figure ran on a shared,
possibly contended dev host (this repo's own CPU benchmark for one network alone is ~7 s/250ms-
trial, i.e. ~21s for 3 trials, so the concurrent design's overlap did work — the ER collection
added ~0s beyond the real trials' own duration — but the real trials themselves ran slower than
the single-network baseline under contention); re-measure once deployed to the target host before
trusting a hard latency number, and consider raising the worker container from 3 to 4 CPUs if it
is still much slower than the real-only baseline.

**Deploy data paths.** The live deploy already rsyncs `data/processed/malecns-min5/` (this doc's
existing convention). It should ALSO sync `data/processed/malecns-min5-er-seed<er_base>/` (the
cached ER control graph — already present from the sealed batch, `er_base=2` in the committed
`config.yaml`) so worker startup does not have to rebuild it (cheap either way — same seed
reproduces the identical graph — but shipping the cache avoids paying that cost on every restart).

**`POST /api/guess`** `{id, pick: "A"|"B"|"C"|"skip"}` -> `{correct: bool, real: "A"|"B"|"C",
counted: bool, pick: "A"|"B"|"C"|"skip", repeat: bool}`. One guess per (answer id, client), keyed
by `LiveStore.hash_ip` (the question queue's own daily-salted hash — NOT chat's stable author_id;
a guess is a one-off honesty stat, not a moderation/ban target). A repeat call for an id this
client already guessed returns the SAME result the first call got — `pick` is the STORED pick, not
necessarily this call's own, and `repeat: true` — ignoring whatever `pick` the repeat call sent,
idempotent, not a re-score (a repeat is answered BEFORE the rate limit below, so a legitimate
reload can never be 429'd). 404 if the id does not exist, is not yet answered, or has no `game`
(legacy answers). `pick == "skip"` (round-2 review R2(b)) is recorded too, as a NEVER-`counted`
guess — it still consumes the (answer, client) slot, so a later real guess attempt on the same id
(e.g. via the share link, with the answer now known from having skipped) hits the SAME idempotent
"repeat" path above instead of a fresh, countable one.

`counted` (2026-09-19 logic review, "the site-wide statistic can be farmed"; tightened in round 2):
anyone can guess — share links, the public feed included — but a guess only counts toward
`GET /api/stats`'s `guesses: {total, correct}` (`LiveStore.guess_stats`, WHERE `counted = 1`;
merged in by `api.py`, not part of `stats.compute_stats`'s pure aggregation since guesses live in
their own table) if ALL of:
- the question's OWN asker is guessing (`live_questions.ip_hash`, set once at ask time, compared
  against the guesser's current `hash_ip` — both use the daily-rotating salt, so this is exact only
  within the same UTC day, an accepted limitation right at the midnight boundary);
- within `config.GUESS_COUNT_WINDOW_S` (15 min) of the answer;
- this answer is the FIRST answered row for its normalized question text
  (`api._is_first_answer_for_question`, backed by `store.answered_questions_before` — round-2 R2(a):
  without this, the example chips or any repeated wording let an asker's own, prompt guess count
  even though an earlier asker's identical reveal was already public).

Every guess is still stored and scored either way (`live_guesses.counted` is 0 for an uncounted
one, or for a skip) — nothing is silently dropped, only excluded from the published percentage.
`POST /api/guess` is itself rate-limited per client (`LiveStore.chat_reserve_attempt`, a
`"guess:{ip_hash}"`-namespaced key on the same `live_chat_attempts` table chat's own report/
LLM-quota namespacing already established) — `config.GUESS_SLOWMODE_S`/`GUESS_RATE_WINDOW_S`/
`GUESS_RATE_PER_WINDOW`, 429 past the limit, checked after the idempotent-repeat lookup.

**Share card (`GET /api/card/{id}.png`).** A game answer's card shows all three verdicts in A/B/C
order and marks which was real (`card._game_summary_line`) — unlike the live page's own suspense,
this card is generated for an already-settled answer being shared after the fact, so there is
nothing left to spoil. The election embargo still wins: an embargoed card returns before reaching
the game-summary line at all, so a noisy election "call" can never leak through the game reveal
either.

### Reliability (2026-09-19 adversarial logic review, two rounds)

An independent review before public ship found and this repo fixed, across two rounds (round 2
specifically re-checked round 1's fixes with real subprocesses, not mocks, and found two of them
incomplete):

- **The real verdict leaked before the guess.** The recent-answers feed row and the lab-stats
  yes/no counter used to update immediately on the `answered` SSE event, even for the visitor's
  own pending, game-bearing question — a feed line right under the anonymous A/B/C game card
  showing the real brain's own YES/NO. `site/src/pages/game.ts`'s `shouldRevealImmediately(answer,
  pendingId)` now gates both updates: held until the visitor guesses or skips for their own
  game-bearing question, unchanged (immediate) for every other case. Covered by
  `game.test.ts`'s `shouldRevealImmediately` suite and verified with a real headless-browser
  click-through (feed/stats stayed at zero pre-guess, updated correctly post-guess). Round 2 found
  the SAME leak via a second path: `labStats`'s own 30s poll moved the yes/no counter mid-
  deliberation regardless of this gate. `LabStats.pause()`/`resume()` (new) hold that poll too,
  toggled by `home.ts` alongside the SSE gate — see "R3" below.
- **The published percentage could be farmed.** Round 1: only the asker's own prompt guess counts
  (`counted`, `POST /api/guess`); the endpoint is rate-limited. Round 2 found this still countable
  with the answer already effectively known, two ways: (a) the example chips and any popular/
  repeated wording — the SAME question answered twice (by design) let a second asker's own,
  prompt, otherwise-legitimate guess count even though an earlier asker's identical reveal was
  already public in the feed; (b) skip revealed the answer without recording anything, so the
  asker could skip, then open the share link on their own id and get a fresh, countable game with
  the answer already known. Fixed:
  - `counted` now ALSO requires this answer to be the FIRST answered row for its normalized
    question text (`api._is_first_answer_for_question`, backed by
    `store.answered_questions_before`) — closes the example-chip/repeat path regardless of who
    asked or when.
  - `pick == "skip"` now calls `POST /api/guess` too and is recorded as a NEVER-`counted` guess
    for that (answer, client) — consumes the slot, so a later real guess on the same id (share
    link included) is the idempotent "repeat" path, not a fresh countable one.
  - The disclosure line now reads "Counted: each asker's first guess on a question nobody asked
    before, made within 15 minutes of the answer."
- **A dead or hung ER subprocess used to break every later question silently.** Round 1:
  `game.ErContenderUnavailable` raised on an already-broken pool or a
  `er_future.result(timeout=game.expected_er_timeout_s(...))` timeout (3x expected wall time,
  floor 30s); `worker.run_loop` marks the question failed, then `raise SystemExit(1)` so the
  container restarts. Round 2 found this incomplete: on a TIMEOUT (as opposed to a pool already
  reported broken), the child is still RUNNING, not dead — `shutdown(wait=False, ...)` alone does
  not stop `concurrent.futures`' own `atexit` hook (`_python_exit`) from joining the executor's
  manager thread forever, which waits on that still-running work item. `SystemExit(1)` was raised
  but the process never actually exited (reproduced on Python 3.12 and 3.13; the SAME thing
  happens to a warm-up timeout at startup, leaving the ~1.9 GB child alive for the whole process
  lifetime). Fixed: `game.ErContenderPool.kill()` — `multiprocessing.Process.kill()` (SIGKILL) on
  every live child, then a brief `join()`, then `shutdown()` — called BEFORE `raise SystemExit(1)`
  in `worker.run_loop`'s handler AND in `worker.build_resources`'s except branch (verified fix
  exits in a few seconds; without it, the same script hangs past a 15s `timeout`). `worker.
  build_resources` still does NOT crash-loop on a startup failure (missing/unwritable ER cache —
  the `data/processed` mount is `:ro` in production — or a broken/killed warm-up): it disables the
  game for that process's lifetime instead, since restarting fixes neither a missing cache file
  nor a bad mount. `game.ErContenderPool.warm_up()` still forces the subprocess up front so this
  is caught immediately, not on whoever asks first.
- **Missing control b0 used to silently default to 0.0.** `worker.answer_question` now checks
  `random_graph_b0`/`no_brain_b0` are both present before running the game at all; if either is
  missing it logs an error and serves the plain real-brain answer (`Answer.game` absent) instead
  of scoring an uncorrected control.
- **A repeat guess used to mark the NEW pick, not the stored one.** Fixed by `pick`/`repeat` in
  the `POST /api/guess` response (above); `game.ts` renders `res.pick`, and
  `game.shouldRecordLocally(response)` (`= !response.repeat`) stops the visitor's own local tally
  from double-counting a replayed guess.
- **Worker liveness.** `worker.touch_heartbeat()` writes `config.WORKER_HEARTBEAT_PATH` (default
  `data/live/worker.heartbeat`, override with `LIVE_WORKER_HEARTBEAT`) on every `run_loop`
  iteration (idle poll or a finished question) and once right after `build_resources()` returns —
  proof the main loop is actually alive, not just that the process exists. Round 2 confirmed the
  path matches the production compose mount and that the infra repo's healthcheck has since been
  switched to it (owned there, not in this repo).

**R3 (round 2, low): the lab-stats 30s poll defeated the F1 gate on its own.** See above —
`LabStats.pause()`/`resume()`, toggled alongside the SSE-driven gate in `home.ts`.

**R4 (round 2, low): `/stream` shows every answer immediately, ungated, by design.** It is a
passive broadcast screen (OBS, a physical display), not itself a place a visitor plays the
guessing game — documented with a comment in `stream.ts`, not gated. Deliberate cross-tab/
cross-device peeking (watching `/stream` on a second screen while playing on the main site) is out
of scope.

Residual risk, not fixed here: the ER pool uses Python's default `fork` start method, which the
review reproduced a `DeprecationWarning` for ("this process is multi-threaded, use of fork() may
lead to deadlocks in the child") — `ProcessPoolExecutor(mp_context=get_context("forkserver"))` (or
`"spawn"`) would remove that hazard and was flagged as a candidate follow-up, not applied here
since it changes subprocess startup semantics (picklability of initializer args, cold-start cost)
beyond what this review round's fixes required. `kill()` makes a hang recoverable either way
(the container restarts instead of wedging), which is the actual fix this round asked for.

## Reproducing an answer

```bash
uv run python -m bioreservoir.live.reproduce --id <id> --question "<exact text>"
```

Recomputes the SAME `Answer` from scratch (`live/reproduce.py`) — every seed is derived only from
`--id` and `config.yaml`'s committed seeds, so a matching `code_sha`/`config_hash` guarantees a
matching result. Cage-only (builds a real `LIFNetwork`); the CLI itself is a thin wrapper over
`worker.answer_question`, tested for determinism with a fake network on a tiny synthetic trial set.
Recomputes all three of the game's contenders too (`resources["er_pool"]`, built the same way
`worker.build_resources` builds it) — `Answer.game`, if the original answer had one, reproduces
byte-for-byte the same as everything else.

## `GET /api/stats`

Aggregates every `answered` row in `LIVE_DB` (`live/stats.py`, pure over plain dicts): counts,
`total_spikes`/`simulated_ms`/`neuron_updates` (neurons x trials x `sim_ms/dt_ms` steps), mean
`|corrected_bias|`/`active_fraction`, per-state publish counts, and `repeat_consistency` (does the
same question text asked more than once get the same `answer`). Empty store -> all zeros, never
fabricated numbers.

## `GET /api/validation`

Assembled only from committed files (`live/validation.py`): brain passport (neuron/connection/
synapse counts, both brains, read from the harmonized graph), `docs/MODEL.md`'s calibration
numbers read verbatim from `experiments/001-fly-oracle/calibration/*.json`, a computed male-passed/
female-failed lateral-validation verdict from those same numbers, `(malecns, real)` handedness b0
read-only from the main ledger, and the OpenTimestamps stamp manifests (`stamps/*.sha256[.ots]`,
filenames + raw lines, never reformatted). Cached on first request per process (graph/calibration
files do not change without a redeploy).

## Live chat

Moderated, livestream-style public chat bolted onto the same app/store/SSE stream as the rest of
this page -- not a second service, not a second database. Backend: `live/chat.py` (identity,
session tokens, anonymous handles, nickname-impersonation checks), `live/moderation.py`'s
chat-specific policy section (`CHAT_REASON_CODES`, `moderate_chat_text`, `check_nickname`,
`check_election_logistics`, ...), `live/store.py`'s `live_chat_*` tables, and `live/events.py`'s
`ChatEventTracker`. Frontend: `site/src/pages/chat.ts` (`mountChat`), styled by
`site/src/chat.css`; mounted interactively on the home page and read-only (dark) on `/stream`.

Hardened 2026-09-18 against a security review (`F1`-`F8` referenced below).

**Routes.**

| Route | Notes |
|---|---|
| `POST /api/chat` | `{text, nickname?, turnstile_token?, chat_token?}` -> `{status:"ok", id, nickname, text, created_at, chat_token?}` or `{status:"rejected", reason, message, chat_token?}` (422). `chat_token` rides along on EVERY response once issued, success or reject, so a content-rejected first message still leaves the client authenticated for its retry |
| `GET /api/chat?limit=100` | `{messages: [...], state: {enabled, slowmode_s}}` -- non-deleted, oldest-first. `limit` is clamped to `[1, 200]` |
| `POST /api/chat/{id}/report` | `{chat_token}` -> `{status:"ok", id, reports, auto_hidden}`. One report per (message, report-group -- see below); a report only moves the count if the reporter is eligible (F3, below). The Nth eligible distinct reporter-group (`CHAT_REPORT_THRESHOLD`) auto-hides the message the same way an operator delete does, and is logged (`chat auto-hide id=... reports=...`) |

All three 503 when `LIVE_CHAT_ENABLED=0` (hard kill switch) **or** when any of the three required
chat secrets is missing (F8, below) -- checked once at API startup, logged once, cached on
`app.state.chat_secrets_ok`. `GET`/`POST /api/chat/{id}/report` otherwise ignore the SOFT runtime
pause (`chat_admin off`) -- only `POST /api/chat` rejects new messages with reason `chat_paused`
while soft-paused, so the frontend can keep polling/listening for the flip back to enabled instead
of losing the connection.

**Identity (F4).** Every abuse-prevention mechanism below keys off `chat.author_id(ip,
CHAT_ID_PEPPER)` -- `HMAC-SHA256(CHAT_ID_PEPPER, normalized_ip)`, IPv6 normalized to its /64 --
**not** `LiveStore.hash_ip`'s daily-rotating salt (that one is still used for the question queue,
unchanged). A stable identity means a ban, an anonymous handle, and a session token all survive
the UTC day boundary (00:00 UTC = 4pm PST on US election day -- a ban must not evaporate then).
`chat_admin list` prints the author id (`author=...`); `chat_admin ban-author <message_id>` bans
whoever posted a given message without needing to copy-paste the hash by hand.

**Session flow (F3).** Turnstile is verified only on a client's first message
(`moderation.verify_turnstile`, reused as-is, same dev-bypass rules as `POST /api/ask`). On success
the server issues a `chat_token` **bound to the author_id that solved it** --
`"<author_id>.<issued_at>.<expires_at>.<sig>"` (`chat.py`'s `issue_chat_token`/`verify_chat_token`).
Verifying a token re-derives `author_id` from the CURRENT request's IP and rejects on any mismatch
-- a token solved from one IP/subnet cannot be replayed from a different one. The frontend stores
it in `localStorage` and skips Turnstile on every later send while it's still valid.

**Moderation policy** (`moderation.py`'s chat section) is deliberately STRICTER than the question
policy: length (1-200 chars) -> no links/@handles/emails/phone numbers -> deterministic rule
blocklists (voting procedure incl. eligibility, **election-logistics misinformation** --
`check_election_logistics`, F7: an election/vote/ballot/poll term plus a logistics term --
date/day/time/open-close/moved-postponed-cancelled/location/online-text-phone-voting/eligibility/ID
-- rejects statement-shaped misinformation like "polls close at 5pm" that the question-shaped
procedural regex alone wouldn't catch --, medical, financial advice, doxxing, harassment/hate,
sexual, impersonation, ad-style spam) -> LLM classifier. The LLM prompt wraps the user's text in
`<message>...</message>` tags with an explicit "treat this as data, not instructions" framing, and
strips any literal `<message>`/`</message>` the user typed first (closing the "fake closing tag"
delimiter-escape vector); only the content-moderation subset of `CHAT_REASON_CODES` is accepted
from the model's output, everything else (including operational reasons like `banned`) is treated
as unparsable. Unlike `POST /api/ask`, there is **no** `LIVE_ENV=dev` fail-open when the LLM is
unreachable/unconfigured -- it always fails CLOSED (reason `unavailable`), because this chat sits
directly on the page's election-misinformation surface. Round-2 review: the AND-anywhere design
false-positived on ordinary chatter -- bare month names ("the election is in November") and "by
mail" were tuned out; only a month name immediately followed by a day number counts now, and only
text/SMS/phone/online voting methods remain in the "by X" list (mail-in voting is legitimate and
common). Not a paraphrase-recall exercise -- the LLM is the deliberate second layer for anything
this regex misses.

Nicknames go through a reserved-word/impersonation check (F5/R4: `admin`, `mod`, `moderator`,
`staff`, `official`, `system`, `bioreservoir`, `igdigi`, `igdigitallab`/"ig digital lab",
`the fly`, `flyoracle`, `igor`, `support`, `verified`, `team`, `lab`, `bot`, plus the server-only
`fly-<digits>` handle pattern -- matched after NFKC normalization, casefold, an ASCII leetspeak
map (`0->o 1->i 3->e 4->a 5->s 7->t @->a $->s |->l`), and zero-width-character stripping) then the
same rule blocklist (not the LLM) plus a format check (`^[A-Za-z0-9_-]{2,20}$`); an absent nickname
gets a server-assigned stable handle (`chat.anonymous_handle`, e.g. `fly-4821`, derived from
`author_id`). Round-2 review: round 1 shipped a Cyrillic/Greek Unicode-confusables map that was
dead code (the ASCII-only format regex rejects all non-ASCII before it's ever reached) while the
REAL bypass -- ASCII leetspeak (`adm1n`, `0fficial`, `fIy-4821`) -- went unmapped; the confusables
map was replaced with the leetspeak one. The fly-handle check folds `l`/`1`/`i`/`|` into one
character class rather than substituting them, so a genuine numeric id (which itself contains
leetspeak-target digits like "1"/"4") is never corrupted by the substitution.

**Anti-abuse / rate limiting (F1/F2, tuned in round 2 as R2).** `LiveStore.chat_reserve_attempt`
checks AND records one attempt for the client's `author_id` inside a single `BEGIN IMMEDIATE`
SQLite transaction, called BEFORE any `await` (Turnstile verification, the LLM call) -- this
closes a check-then-act race a concurrent burst could otherwise use to bypass slow-mode and the
10-minute cap entirely (a same-connection or even same-process reproduction isn't enough to rule
this out; the fix is transactional so it also holds under true multi-connection/multi-process
concurrency, confirmed with real OS processes in round 2). EVERY attempt counts -- accepted,
rejected, or failed downstream -- not just accepted messages, so a client can't dodge the limiter
by sending content it knows will be rejected. The SAME atomic mechanism is reused, namespaced,
for two more independent budgets: `f"report:{author_id}"` (R1, see report/auto-hide below) and
`f"llm:{author_id}"` (R2, a smaller PER-AUTHOR LLM quota -- default `CHAT_LLM_PER_AUTHOR_PER_MINUTE`
-- checked before the site-wide breaker). The site-wide breaker itself
(`LiveStore.chat_reserve_llm_call`, `CHAT_LLM_MAX_PER_MINUTE`/min, independent of any one client's
identity) gates the LLM step, rejecting with `chat_busy` once tripped -- round 2 found the
original 30/min default tripped under ~35 ordinary concurrent chatters, so it was raised to 200
(the LiteLLM virtual key's own budget is the real spend backstop). Duplicate-message suppression
(same trimmed text twice in a row from the same client). `chat_admin ban <client_hash>` /
`ban-author <message_id>` block an author outright, checked on both `POST /api/chat` and
`POST /api/chat/{id}/report` (a banned client cannot report either).

**Report/auto-hide (F3, hardened in round 2 as R1).** Two gates, at different strictness: (a) a
banned author, or (b) a `chat_token` not bound to the CURRENT request's author_id, gets an
outright 422 rejection (reasons `banned`/`captcha`) — reporting itself is refused. Past that, the
report is always accepted (`{"status":"ok", ...}`) but only moves the auto-hide tally if the
reporter (c) has held their session for at least `CHAT_REPORT_MIN_SESSION_AGE_S` (default 10 min)
AND (d) has posted >=1 accepted message themselves — an ineligible-but-valid report gives no hint
about which of (c)/(d) failed. Distinct counted reporters are deduped by `chat.report_group_id` --
an IPv4 /24 or IPv6 /64, coarser than `author_id` (whose IPv4 side is the full address) -- so a
handful of addresses in the same subnet can't multiply their vote.

Round 1's rules alone still let a small number of PATIENT, eligible identities hide an unlimited
number of messages, five requests apiece, forever -- reproduced 20/20 in round 2. R1's fix is two
independent caps, both consulted ONLY once a report actually crosses `CHAT_REPORT_THRESHOLD` (an
ordinary non-crossing report never spends either budget): a per-reporter quota
(`CHAT_REPORTS_PER_10MIN`, reused through the same atomic `chat_reserve_attempt` path, namespaced
`f"report:{author_id}"`) and a site-wide cap on how many auto-hides can happen in one window
(`CHAT_AUTO_HIDE_PER_10MIN`, `LiveStore.chat_auto_hide_count_recent`). The report is ALWAYS
recorded regardless of either cap -- only the HIDE is withheld -- so a message that hits a cap
still shows an elevated `reports` count in `chat_admin list`'s output for manual operator review,
it just isn't auto-hidden. Once a hide does go
through, the message gets a soft-delete tagged `deleted_reason='reported'` (vs. `'admin'` for an
operator's own delete) -- both broadcast identically as `chat_delete` over SSE, no special-cased
event type -- and it's logged. `chat_admin list --reported` is the auto-hidden-specifically
review queue; `chat_admin restore <id>` un-hides a message AND clears its report tally, so the
exact same reporters can't instantly re-trigger auto-hide the moment it's restored.

**Runtime-adjustable slow-mode + soft kill switch** (`live_chat_state`, a single-row table) — no
redeploy needed:

```bash
python -m bioreservoir.live.chat_admin slowmode 8   # 1 message / 8s per client
python -m bioreservoir.live.chat_admin off          # pause chat (LIVE_CHAT_ENABLED env stays the hard default)
python -m bioreservoir.live.chat_admin on
```

Every connected SSE client gets the new `{enabled, slowmode_s}` immediately as a `chat_state`
event (`ChatEventTracker` emits it on the first poll and again on every change), so the frontend
can show/hide the chat and update its slow-mode note live, no page reload.

**Operator CLI** (`python -m bioreservoir.live.chat_admin`, talks directly to `LIVE_DB`, no IPC
with the running API process -- takes effect on its next SSE poll):

```bash
python -m bioreservoir.live.chat_admin list [--limit N] [--all] [--reported]
python -m bioreservoir.live.chat_admin delete <id>
python -m bioreservoir.live.chat_admin restore <id>
python -m bioreservoir.live.chat_admin clear
python -m bioreservoir.live.chat_admin ban <client_hash>
python -m bioreservoir.live.chat_admin ban-author <message_id>
python -m bioreservoir.live.chat_admin slowmode <seconds>
python -m bioreservoir.live.chat_admin off|on
```

**SSE events** (same `/api/events` stream, no second endpoint): `chat` (a new message, or a
restored one reappearing -- same shape either way, so the frontend needs no fourth event type),
`chat_delete` (`{id}`, admin-deleted or auto-hidden), `chat_state` (`{enabled, slowmode_s}`). A
fresh connection seeds its deletion/restore cursors to "now" (F6) -- it never replays history from
before it connected, even after a large `chat_admin clear`.

**Data retention (F4 + the 7-day rule below).** Messages are never hard-deleted, only soft
(`deleted`/`deleted_at`/`deleted_reason`). `LiveStore.chat_purge_expired`, called from the same
periodic hook `worker.py` already uses for rejected-question purging (every 100 processed
questions -- an approximation; chat traffic is independent of the question queue, same limitation
`purge_expired` already has):
1. blanks `text` on soft-deleted messages older than `REJECTED_RETENTION_DAYS` (7, same as
   rejected questions),
2. blanks the `ip_hash` (author_id) linkage on ANY message older than 30 days, **unless** that
   author is currently banned (keeps a ban enforceable/auditable without holding every past
   visitor's identity forever), and
3. (round-2 review) hard-deletes rows from `live_chat_attempts`/`live_chat_llm_calls` older than
   1 day -- both tables only ever need to be queried within a 10-minute window (the widest
   rate-limit window in use), and were found unbounded/never purged in round 2.

**Frontend captcha recovery (R3, round 2).** A "captcha" rejection almost always means the
visitor's IP changed (wifi -> cellular is the common case, and mobile is the election-night
audience), not an attack -- F3's token binding makes that legitimate case indistinguishable from
a stale/forged token at the server. Before this fix the frontend kept the stale token in
`localStorage` (still "likely valid" by its own un-checked expiry) and never re-mounted Turnstile
once ANY valid-looking token existed at page load, so every send failed silently until the
~1-hour TTL expired. `chat.ts`'s `captchaRecoveryAction` (pure, vitest-covered) now drives: on
`reason === "captcha"`, drop the stored token and re-show/re-mount Turnstile so the visitor can
re-verify and keep chatting immediately.
