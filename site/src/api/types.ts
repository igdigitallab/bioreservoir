// API contract types, mirroring the backend's live fly page service (`bioreservoir.live.api`,
// see docs/LIVE.md's "API contract"). This file is the single source of truth the frontend
// codes against.

// Matches content/copy.json's ask.rejected_messages keys exactly (the copywriter's file is the
// source of truth for which reasons exist) — coordinate with backend before adding a new one.
export type RejectReason =
  | "not_yes_no"
  | "voting_procedure"
  | "medical"
  | "financial_advice"
  | "private_person"
  | "violence_or_hate"
  | "sexual"
  | "spam"
  | "too_long"
  | "rate_limited"
  | "captcha";

export interface AskResponseQueued {
  id: string;
  status: "queued";
  /** 1-based place in line at the moment of asking. */
  position: number;
  /** Questions the worker had taken so far when this position was computed; the live position is
   * `livePosition(position, claimed_total, <latest claimed_total>)` (queue.ts). */
  claimed_total: number;
  /** Median seconds per answer lately (worker time, started→answered). */
  avg_cycle_s: number;
  /** True when the same words were already waiting: this ask joined that run, no second one. */
  joined?: boolean;
}

/** The same words were answered before: same seeds, same YES-side coin, so the identical answer.
 * The UI says "asked before"; it is never presented as a new live run. */
export interface AskResponseRepeat {
  id: string;
  status: "answered";
  repeat: true;
}

export interface AskResponseRejected {
  status: "rejected";
  reason: RejectReason;
  message: string;
}

export type AskResponse = AskResponseQueued | AskResponseRepeat | AskResponseRejected;

/** `GET /api/answers/{id}` (api.py `get_answer`): a question row, whose `answer` is present only
 * once it has been answered. Not an Answer itself — client.getAnswer() unwraps it. */
export interface AnswerLookup {
  id: string | number;
  question: string;
  status: "queued" | "thinking" | "answered" | "rejected";
  answer?: Answer;
  /** queued only */
  position?: number;
  claimed_total?: number;
  avg_cycle_s?: number;
  /** thinking only */
  started_at?: string;
  /** queued/thinking only: the side a YES would face — see ThinkingItem.yes_side. */
  yes_side?: "left" | "right";
  /** answered only: an election question before Nov 4 (show its verdict to the asker only). */
  embargoed?: boolean;
  /** rejected only */
  reason?: string;
  message?: string;
}

/** Everything the public surfaces (feed, SSE `answered`, `/api/now`) carry about an answer: no
 * spike data. `embargoed` = an election question before Nov 4: its verdict fields are null on
 * every public surface (the asker still gets the full answer from `/api/answers/{id}`). */
export interface AnswerSummary {
  id: string;
  question: string;
  answer: "yes" | "no" | null;
  embargoed: boolean;
  yes_side: "left" | "right" | null;
  lateral_bias: number | null;
  turn_strength: number | null;
  answered_at: string;
  /** How many visitors liked this question. Absent on responses from a server older than the
   * likes feature; the UI treats that as zero. */
  likes?: number;
}

/** `GET /api/questions`: one page of the answered archive, newest or most-liked first. */
export interface QuestionsPage {
  sort: "recent" | "top";
  total: number;
  has_more: boolean;
  items: AnswerSummary[];
}

/** `POST /api/answers/{id}/like`: the new count, and whether this client now likes it. */
export interface LikeResponse {
  id: number;
  likes: number;
  liked: boolean;
}

/** The question the worker is simulating right now. */
export interface ThinkingItem {
  id: string;
  question: string;
  started_at: string;
  embargoed: boolean;
  /** Which physical side means YES for this wording (api.py `pipeline.yes_side_for`). Fixed by
   * the text alone, so the stage can label both options before the answer exists. */
  yes_side?: "left" | "right";
}

export interface QueueTick {
  queue_length: number;
  /** Monotonic count of questions the worker has taken, ever. */
  claimed_total: number;
  avg_cycle_s: number;
}

/** `GET /api/now`: what a freshly loaded page shows on the live stage. */
export interface NowResponse extends QueueTick {
  thinking: ThinkingItem | null;
  last: AnswerSummary | null;
}

export interface BehaviourStates {
  /** MN9 proboscis extension drive, null if the readout did not respond, see docs/LIVE.md's "Behavioural states". */
  appetite: number | null;
  /** DNp01 giant-fibre escape drive. */
  fear: number | null;
  /** MDN backward-walking drive. */
  backoff: number | null;
  /** P1/pC1 + pIP10 courtship drive — male brain only, always null on the female. */
  courtship: number | null;
  /** Overall network activity, always present when an answer exists. */
  arousal: number | null;
}

export interface AnswerFrames {
  /** Always 25 in the real API; kept as `number` (not a `25` literal) so decodeFrames can be
   * unit-tested with other bin counts without a cast, see frames.test.ts. */
  bin_ms: number;
  /** Always 10 in the real API — see the previous field's comment. */
  n_bins: number;
  /** One base64 string per bin: little-endian uint32 indices into the atlas neuron_ids order. */
  active_b64: string[];
}

export interface LabTrial {
  seed: number;
  spikes_left: number;
  spikes_right: number;
  /** Raw (L-R)/(L+R) for this one trial, before the handedness correction. */
  bias: number;
}

export interface LabStimulated {
  total: number;
  left: number;
  right: number;
  by_modality: Record<string, number>;
}

export interface LabTopCellType {
  cell_type: string;
  super_class: string;
  n_neurons: number;
  spikes: number;
  rate_hz: number;
}

export interface LabRaster {
  /** Atlas row index per raster row, or -1 if that neuron has no plottable soma position. */
  atlas_indices: number[];
  cell_types: string[];
  /** Little-endian uint16 pairs (row, time_0.1ms) for trial 0 only — see raster.ts. */
  spikes_b64: string;
}

export interface LabProvenance {
  brain: string;
  n_neurons: number;
  n_connections: number;
  n_synapses: number;
  min_syn: number;
  model: string;
  dt_ms: number;
  sim_ms: number;
  code_sha: string;
  config_hash: string;
  /** Copy-pasteable CLI command that reproduces this exact answer. `null` while the repository
   * is still private (`lab.REPO_URL` unset) — the panel then shows the pinning hashes and says
   * the code is not published yet, instead of a `git clone` that 404s. */
  reproduce: string | null;
}

/** The "prove it's real" readout — see docs/MODEL.md for the biology behind each field
 * (handedness correction = reference.yaml's b0, per-question yes_side = the side-mapping
 * decision documented in experiments/001-fly-oracle/README.md's "Handedness and side mapping"). */
export interface AnswerLab {
  trials: LabTrial[];
  /** Mean lateral bias over the 24 neutral reference sentences for this brain (handedness). */
  b0: number;
  /** Mean trial bias minus b0 — the number the yes/no call is actually made from. */
  corrected_bias: number;
  /** Which physical side (left/right descending pool) counts as YES for this specific question
   * (per-question seeded mapping, not a global left=yes rule — see fly-handedness memory). */
  yes_side: "left" | "right";
  stimulated: LabStimulated;
  active_neurons: number;
  active_fraction: number;
  total_spikes: number;
  readout_latency_ms: number | null;
  top_cell_types: LabTopCellType[];
  raster: LabRaster;
  provenance: LabProvenance;
}

export interface Answer {
  id: string;
  question: string;
  answer: "yes" | "no";
  /** In [0, 1]. */
  confidence: number;
  /** Signed lateral descending-neuron bias (L-R)/(L+R) that produced the yes/no call. */
  lateral_bias: number;
  states: BehaviourStates;
  frames: AnswerFrames;
  brain: "malecns";
  sim_ms: number;
  n_trials: number;
  answered_at: string;
  lab: AnswerLab;
  /** |corrected bias| as a fraction (0..1) of the turn a one-antenna sound cue (Johnston's organ
   * calibration, docs/MODEL.md) produces. Optional only for rows older than the field. */
  turn_strength?: number;
}

export interface StatsStateCounts {
  appetite: number;
  fear: number;
  backoff: number;
  courtship: number;
  arousal: number;
}

export interface Stats {
  answered: number;
  yes: number;
  no: number;
  total_spikes: number;
  simulated_ms: number;
  neuron_updates: number;
  mean_abs_corrected_bias: number;
  mean_active_fraction: number;
  state_counts: StatsStateCounts;
  repeat_consistency: {
    repeated_questions: number;
    identical_answers: number;
  };
  /** ISO timestamp the counters have been accumulating since. */
  since: string;
}

// This block mirrors the REAL shape returned by `bioreservoir.live.validation.build_validation()`
// (checked against a live `GET /api/validation` in dev mode, 2026-09-18) — it is keyed per
// dataset (`malecns`/`banc`), not a single "passport", because `/api/validation` documents both
// brains, not just the one the live page answers with (see validation.py's module docstring).

export interface BrainPassportEntry {
  dataset: string;
  n_neurons: number;
  n_connections: number;
  n_synapses: number;
  min_syn: number;
}

/** One side's `descending_all` lateral bias over the calibration trials driving that side. */
export interface BiasStat {
  mean: number;
  std: number;
  n: number;
}

/** `passed` is computed server-side (`validation.lateral_validation_verdict`), not asserted —
 * left-driven bias positive AND right-driven bias negative AND their gap > 0.05. `null` when no
 * calibration file is committed yet for that dataset. */
export interface LateralValidationEntry {
  passed: boolean;
  jo_left_driven_descending_bias: BiasStat;
  jo_right_driven_descending_bias: BiasStat;
  left_vs_right_bias_gap: number;
}

/** `(malecns, real)` handedness b0 read-only from the main ledger's `done` reference runs under
 * the CURRENT config hash — `b0: null` until that group has reference runs (see docs/LIVE.md
 * "Handedness b0"). Per-answer b0 lives on each `Answer.lab.b0` instead; this is the same number,
 * shown once here for the "how is handedness corrected" explainer. */
export interface HandednessB0 {
  brain: string;
  condition: string;
  config_hash: string;
  b0: number | null;
  n_reference_sentences: number;
}

/** One committed OpenTimestamps manifest: filename, its raw lines verbatim (comments + `hash
 * path` rows), and its `.ots` proof filename if one exists next to it. */
export interface ValidationStamp {
  manifest_file: string;
  lines: string[];
  ots_file: string | null;
}

export interface Validation {
  model: string;
  /** Keyed by dataset (`"malecns"` | `"banc"`). */
  brain_passport: Record<string, BrainPassportEntry>;
  /** Keyed by dataset. The raw dose-response/bitter-suppression trial dumps also live in the real
   * payload under this key (`live/validation.py`'s `calibration_summary`) but are not typed or
   * rendered here — the page shows the server-computed `lateral_validation` verdict instead of
   * re-deriving a pass/fail threshold from raw trials client-side. */
  calibration?: unknown;
  /** Keyed by dataset. */
  lateral_validation: Record<string, LateralValidationEntry | null>;
  handedness_b0: HandednessB0;
  stamps: ValidationStamp[];
}

export interface Feed {
  queue_length: number;
  recent: AnswerSummary[];
}

// -- live chat: moderated livestream-style chat on the live page (docs/LIVE.md "Live chat") -----

export interface ChatMessage {
  id: number;
  nickname: string;
  text: string;
  created_at: string;
}

/** Runtime-adjustable slow-mode interval + soft kill switch (`chat_admin slowmode`/`off`/`on`) --
 * distinct from the hard `LIVE_CHAT_ENABLED` env var, which is checked by a 503 on every chat
 * route instead of this field (see client.ts's `getChat`). */
export interface ChatState {
  enabled: boolean;
  slowmode_s: number;
}

export interface ChatGetResponse {
  /** Oldest first. */
  messages: ChatMessage[];
  state: ChatState;
  /** More history exists before the oldest message returned (`GET /api/chat?before_id=`). */
  has_more?: boolean;
}

export type ChatReason =
  | "empty"
  | "too_long"
  | "contains_contact_info"
  | "voting_procedure"
  | "medical"
  | "financial_advice"
  | "private_person"
  | "harassment_or_hate"
  | "sexual"
  | "spam"
  | "impersonation"
  | "rate_limited"
  | "duplicate"
  | "captcha"
  | "unavailable"
  | "nickname_invalid"
  | "banned"
  | "chat_paused";

export interface ChatPostOk {
  status: "ok";
  id: number;
  nickname: string;
  text: string;
  created_at: string;
  /** Only present right after a Turnstile-verified first message — see client.ts's postChat. */
  chat_token?: string;
}

export interface ChatPostRejected {
  status: "rejected";
  reason: ChatReason;
  message: string;
  chat_token?: string;
}

export type ChatPostResponse = ChatPostOk | ChatPostRejected;

export interface ChatReportResponse {
  status: "ok";
  id: number;
  reports: number;
  auto_hidden: boolean;
}

export type ServerEvent =
  | { type: "thinking"; data: ThinkingItem }
  | { type: "answered"; data: AnswerSummary }
  | { type: "queue"; data: QueueTick }
  /** Synthetic (client.ts): the stream came back after a drop, so events may have been missed. */
  | { type: "reconnected"; data: null }
  | { type: "chat"; data: ChatMessage }
  | { type: "chat_delete"; data: { id: number } }
  | { type: "chat_state"; data: ChatState };
