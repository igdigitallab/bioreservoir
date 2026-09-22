// Dev-only fixture server: patches window.fetch and window.EventSource so `npm run dev` with
// VITE_USE_MOCK=1 works with no backend running. Never imported in a production build — see the
// `import.meta.env.DEV` guard in api/client.ts, which dead-code-eliminates this whole module.
//
// Numbers here fall into two groups: (1) plausible per-trial spike counts, generated randomly
// because this is a *fixture* for a value the real backend computes fresh per question, and
// (2) whole-brain facts (neuron/synapse counts, the calibration results, the two real
// OpenTimestamps stamp files) copied verbatim from docs/MODEL.md, docs/DATA.md and
// experiments/001-fly-oracle/stamps/ — real, committed numbers, not invented for this fixture.
import type {
  Answer,
  AnswerLookup,
  AnswerSummary,
  AskResponse,
  ChatGetResponse,
  ChatMessage,
  ChatPostResponse,
  ChatReportResponse,
  Feed,
  LabTopCellType,
  NowResponse,
  ServerEvent,
  Stats,
  Validation,
} from "../api/types";

const MOCK_ATLAS_N = 140024; // matches scripts/export_atlas.py's current output; approximate is fine, dev-only.
const SIM_MS = 250;
const N_TRIALS = 3;
const N_NEURONS = 165122; // docs/DATA.md, MaleCNS Traced count.

function randomBin(count: number): string {
  const bytes = new Uint8Array(count * 4);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < count; i++) {
    view.setUint32(i * 4, Math.floor(Math.random() * MOCK_ATLAS_N), true);
  }
  let binary = "";
  bytes.forEach((b) => (binary += String.fromCharCode(b)));
  return btoa(binary);
}

// Real MaleCNS cell-type names (docs/DATA.md "Key cell types") used so the fixture's lab panel
// looks like the real thing rather than "type_1"/"type_2" placeholders.
const REAL_CELL_TYPES: Array<{ cell_type: string; super_class: string }> = [
  { cell_type: "DNp01", super_class: "descending_neuron" },
  { cell_type: "MN9", super_class: "cb_motor" },
  { cell_type: "MDN", super_class: "descending_neuron" },
  { cell_type: "pC1", super_class: "cb_intrinsic" },
  { cell_type: "pIP10", super_class: "descending_neuron" },
  { cell_type: "DNa01", super_class: "descending_neuron" },
  { cell_type: "DNa02", super_class: "descending_neuron" },
];

function mockTopCellTypes(): LabTopCellType[] {
  return REAL_CELL_TYPES.slice(0, 3 + Math.floor(Math.random() * 3)).map((ct) => {
    const n = ct.cell_type === "MN9" || ct.cell_type === "DNp01" ? 2 : ct.cell_type === "MDN" ? 4 : 2;
    const spikes = Math.floor(Math.random() * 400);
    return {
      cell_type: ct.cell_type,
      super_class: ct.super_class,
      n_neurons: n,
      spikes,
      rate_hz: (spikes / n) / (SIM_MS / 1000),
    };
  });
}

function mockRasterAndSpikes(): { atlas_indices: number[]; cell_types: string[]; spikes_b64: string } {
  const rowCount = 24 + Math.floor(Math.random() * 24);
  const atlas_indices: number[] = [];
  const cell_types: string[] = [];
  const pairs: Array<[number, number]> = [];
  for (let row = 0; row < rowCount; row++) {
    atlas_indices.push(Math.floor(Math.random() * MOCK_ATLAS_N));
    const ct = REAL_CELL_TYPES[row % REAL_CELL_TYPES.length]!;
    cell_types.push(ct.cell_type);
    const spikeCount = 1 + Math.floor(Math.random() * 8);
    for (let s = 0; s < spikeCount; s++) {
      pairs.push([row, Math.floor(Math.random() * SIM_MS * 10)]); // time in units of 0.1ms
    }
  }
  const bytes = new Uint8Array(pairs.length * 4);
  const view = new DataView(bytes.buffer);
  pairs.forEach(([row, t], i) => {
    view.setUint16(i * 4, row, true);
    view.setUint16(i * 4 + 2, t, true);
  });
  let binary = "";
  bytes.forEach((b) => (binary += String.fromCharCode(b)));
  return { atlas_indices, cell_types, spikes_b64: btoa(binary) };
}

function mockAnswer(id: string, question: string): Answer {
  const isMale = true; // dev fixture only ever mocks the male brain (the one this page ships with).
  const trials = Array.from({ length: N_TRIALS }, (_, i) => {
    const left = 60 + Math.floor(Math.random() * 80);
    const right = 60 + Math.floor(Math.random() * 80);
    return {
      seed: 1000 + i,
      spikes_left: left,
      spikes_right: right,
      bias: (left - right) / (left + right),
    };
  });
  const b0 = 0.02;
  const meanRawBias = trials.reduce((s, t) => s + t.bias, 0) / trials.length;
  const correctedBias = meanRawBias - b0;
  // yes_side is an independent per-question coin (oracle.side_mapping.left_is_yes_for), not
  // derived from the answer — the answer is derived FROM it and corrected_bias, the same
  // direction the real backend computes in (pipeline.compute_answer ->
  // oracle.handedness.apply_correction -> oracle.readout.probability_from_bias:
  // p_yes = (1+bias)/2 if yes_side=="left" else (1-bias)/2, so answer="yes" iff
  // (yes_side=="left" && bias>=0) || (yes_side=="right" && bias<=0)). Previously this fixture
  // picked `yes` first and set yes_side to match it, independent of corrected_bias's actual
  // sign — inconsistent with the real backend and caught during live-page deploy reconciliation.
  const yesSide: "left" | "right" = mockYesSide(question);
  const yes = yesSide === "left" ? correctedBias >= 0 : correctedBias <= 0;
  const activeFraction = 0.04 + Math.random() * 0.06;
  const activeNeurons = Math.round(N_NEURONS * activeFraction);

  // Stimulated population: left == right by construction (docs/MODEL.md's input regime is
  // balanced per side unless a question deliberately drives one side), total = left + right,
  // and per side falls in the documented 300-1000-neuron range (docs/MODEL.md), so
  // total is 600-2000 -- internally consistent, not independently-random fields.
  const stimulatedPerSide = 300 + Math.floor(Math.random() * 700);
  const stimulatedTotal = stimulatedPerSide * 2;
  const modalityShare = { olfactory: 0.3, gustatory: 0.1, visual: 0.35, mechanosensory: 0.25 };
  const modalityKeys = Object.keys(modalityShare) as Array<keyof typeof modalityShare>;
  const byModality: Record<string, number> = {};
  let assigned = 0;
  modalityKeys.forEach((k, i) => {
    if (i === modalityKeys.length - 1) {
      byModality[k] = stimulatedTotal - assigned; // remainder, so the parts always sum exactly.
    } else {
      const v = Math.round(stimulatedTotal * modalityShare[k]);
      byModality[k] = v;
      assigned += v;
    }
  });

  const codeSha = "devmock0"; // fixture placeholder — real backend fills the actual git commit.
  const configHash = "devmockconfig0";
  // Shape matches the real backend's live-answer replay CLI (python -m
  // bioreservoir.live.reproduce), not the batch oracle.run/sim.calibrate commands documented in
  // experiments/001-fly-oracle/RUNNING.md — values are fixture placeholders, the real backend
  // fills real ones.
  const reproduce = `uv run python -m bioreservoir.live.reproduce --brain malecns --seed ${trials[0]!.seed} --config-hash ${configHash} --question-id ${id}`;

  const confidence = 0.55 + Math.random() * 0.4;

  return {
    id,
    question,
    answer: yes ? "yes" : "no",
    confidence,
    lateral_bias: correctedBias,
    turn_strength: Math.min(1, Math.abs(correctedBias) / 0.1375),
    states: {
      appetite: Math.random() > 0.5 ? Math.random() : null,
      fear: Math.random() > 0.7 ? Math.random() : null,
      backoff: Math.random() > 0.7 ? Math.random() : null,
      courtship: isMale && Math.random() > 0.6 ? Math.random() : null,
      arousal: Math.random(),
    },
    frames: {
      bin_ms: 25,
      n_bins: 10,
      active_b64: Array.from({ length: 10 }, () => randomBin(200 + Math.floor(Math.random() * 300))),
    },
    brain: "malecns",
    sim_ms: SIM_MS,
    n_trials: N_TRIALS,
    answered_at: new Date().toISOString(),
    lab: {
      trials,
      b0,
      corrected_bias: correctedBias,
      yes_side: yesSide,
      stimulated: {
        total: stimulatedTotal,
        left: stimulatedPerSide,
        right: stimulatedPerSide,
        by_modality: byModality,
      },
      active_neurons: activeNeurons,
      active_fraction: activeFraction,
      total_spikes: Math.round(activeNeurons * (1 + Math.random() * 3)),
      readout_latency_ms: Math.random() > 0.3 ? 40 + Math.random() * 60 : null,
      top_cell_types: mockTopCellTypes(),
      raster: mockRasterAndSpikes(),
      provenance: {
        brain: "malecns",
        n_neurons: N_NEURONS,
        n_connections: 6235682,
        n_synapses: 89731551,
        min_syn: 5,
        model: "shiu-lif-cython",
        dt_ms: 0.1,
        sim_ms: SIM_MS,
        code_sha: codeSha,
        config_hash: configHash,
        reproduce,
      },
    },
  };
}

const store = new Map<string, Answer>();
const recent: Answer[] = [];
const listeners = new Set<(event: ServerEvent) => void>();
const since = new Date().toISOString();

// -- a simulated queue with other visitors, so the live stage has something to show in dev. The
// worker takes one question every WORKER_MS (the real one takes ~60 s); fake visitors ask now and
// then. Same contract as the backend (api.py): claimed_total, avg_cycle_s, dedupe by question key.
const WORKER_MS = 9000;
const VISITOR_QUESTIONS = [
  "Will it snow in Denver this week?",
  "Is pineapple good on pizza?",
  "Will the Lakers win tonight?",
  "Should I learn the violin?",
  "Will the Senate flip in the midterms?",
  "Is there life on Europa?",
  "Will my cat forgive me?",
];
interface MockRow { id: string; question: string; key: string; status: "queued" | "thinking" | "answered"; started_at?: string }
const rows = new Map<string, MockRow>();
const queue: string[] = [];
let nextId = 100;
let claimedTotal = 0;
let thinking: MockRow | undefined;

const questionKey = (q: string) => q.normalize("NFKC").toLowerCase().replace(/\s+/g, " ").trim().replace(/[ ?!.]+$/, "");

/** The backend's `pipeline.yes_side_for` in miniature: the YES side is fixed by the wording
 * alone, so `thinking`/`queued` payloads can carry it and the stage labels never swap sides when
 * the answer lands. */
function mockYesSide(question: string): "left" | "right" {
  let hash = 0;
  for (const ch of questionKey(question)) hash = (Math.imul(hash, 31) + ch.charCodeAt(0)) >>> 0;
  return hash % 2 === 0 ? "left" : "right";
}
const isElection = (q: string) => /\b(senate|house|midterm|election|vote|trump|harris|governor)\b/i.test(q);

function summary(a: Answer): AnswerSummary {
  const embargoed = isElection(a.question);
  return {
    id: a.id,
    question: a.question,
    answer: embargoed ? null : a.answer,
    embargoed,
    yes_side: embargoed ? null : a.lab.yes_side,
    lateral_bias: embargoed ? null : a.lateral_bias,
    turn_strength: embargoed ? null : (a.turn_strength ?? null),
    answered_at: a.answered_at,
    likes: likes.get(String(a.id)) ?? 0,
  };
}

/** Likes given in this dev session: id -> count, plus which ones this "client" gave, so the
 * toggle behaves like the real endpoint (one like per client, tapping twice takes it back). */
const likes = new Map<string, number>();
const likedHere = new Set<string>();

function handleLike(id: string): { id: number; likes: number; liked: boolean } | undefined {
  const row = rows.get(id);
  if (!row || row.status !== "answered") return undefined;
  const liked = !likedHere.has(id);
  if (liked) likedHere.add(id);
  else likedHere.delete(id);
  const next = Math.max(0, (likes.get(id) ?? 0) + (liked ? 1 : -1));
  likes.set(id, next);
  return { id: Number(id), likes: next, liked };
}

function handleQuestions(sort: string, limit: number, offset: number) {
  const all = [...store.values()].filter((a) => rows.get(String(a.id))?.status === "answered").map(summary);
  all.sort((a, b) =>
    sort === "top" ? (b.likes ?? 0) - (a.likes ?? 0) || Number(b.id) - Number(a.id) : Number(b.id) - Number(a.id),
  );
  const page = all.slice(offset, offset + limit);
  return { sort, total: all.length, has_more: offset + page.length < all.length, items: page };
}

function tick() {
  return { queue_length: queue.length, claimed_total: claimedTotal, avg_cycle_s: WORKER_MS / 1000 };
}

function enqueueRow(question: string): MockRow {
  const row: MockRow = { id: String(nextId++), question, key: questionKey(question), status: "queued" };
  rows.set(row.id, row);
  queue.push(row.id);
  emit({ type: "queue", data: tick() });
  return row;
}

function workerStep() {
  if (thinking) {
    const row = thinking;
    const answer = mockAnswer(row.id, row.question);
    store.set(row.id, answer);
    recent.unshift(answer);
    row.status = "answered";
    thinking = undefined;
    emit({ type: "answered", data: summary(answer) });
  }
  const nextRowId = queue.shift();
  if (nextRowId) {
    const row = rows.get(nextRowId)!;
    row.status = "thinking";
    row.started_at = new Date().toISOString();
    thinking = row;
    claimedTotal += 1;
    emit({ type: "thinking", data: { id: row.id, question: row.question, started_at: row.started_at, embargoed: isElection(row.question), yes_side: mockYesSide(row.question) } });
  }
  emit({ type: "queue", data: tick() });
}
/** Started by installMock(), never at import: tests import client.ts (and so this module) in node. */
function startMockQueue() {
  window.setInterval(workerStep, WORKER_MS);
  window.setInterval(() => {
    if (queue.length < 6) enqueueRow(VISITOR_QUESTIONS[Math.floor(Math.random() * VISITOR_QUESTIONS.length)]!);
  }, 7000);
  // Some history, so the feed and the replay-on-load have something to show.
  for (const q of VISITOR_QUESTIONS.slice(0, 3)) {
    const row = enqueueRow(q);
    queue.pop();
    row.status = "answered";
    const answer = mockAnswer(row.id, q);
    store.set(row.id, answer);
    recent.unshift(answer);
  }
  // A long chat history, so scrolling back through it can be tried in dev.
  for (let i = 0; i < 160; i++) {
    chatMessages.push({ id: chatNextId++, nickname: `fly-${1000 + i}`, text: `Earlier message ${i + 1}`, created_at: new Date(Date.now() - (160 - i) * 60000).toISOString() });
  }
}

function emit(event: ServerEvent) {
  listeners.forEach((l) => l(event));
}

class MockEventSource {
  static readonly OPEN = 1;
  readyState = 1;
  constructor(_url: string) {
    // eslint-disable-next-line no-console
    console.info("[mock] EventSource connected (dev fixture, not the real backend)");
  }
  addEventListener(type: string, handler: (evt: MessageEvent) => void) {
    listeners.add((event) => {
      if (event.type === type) handler({ data: JSON.stringify(event.data) } as MessageEvent);
    });
  }
  removeEventListener() {
    /* fixture: listeners are cleared with the page, no-op is fine for dev */
  }
  close() {
    /* no real connection to tear down */
  }
}

async function handleAsk(body: string): Promise<AskResponse> {
  const { question } = JSON.parse(body) as { question: string };
  if (question.length > 140) {
    return { status: "rejected", reason: "too_long", message: "Keep it under 140 characters." };
  }
  const key = questionKey(question);
  const twin = [...rows.values()].reverse().find((r) => r.key === key);
  if (twin?.status === "answered") return { id: twin.id, status: "answered", repeat: true };
  if (twin) {
    const position = twin.status === "thinking" ? 1 : queue.indexOf(twin.id) + 1;
    return { id: twin.id, status: "queued", position, claimed_total: claimedTotal, avg_cycle_s: WORKER_MS / 1000, joined: true };
  }
  const row = enqueueRow(question);
  return { id: row.id, status: "queued", position: queue.length, claimed_total: claimedTotal, avg_cycle_s: WORKER_MS / 1000, joined: false };
}

function handleLookup(id: string): AnswerLookup | undefined {
  const row = rows.get(id);
  if (!row) return undefined;
  if (row.status === "answered") return { id, question: row.question, status: "answered", embargoed: isElection(row.question), answer: store.get(id) };
  if (row.status === "thinking") return { id, question: row.question, status: "thinking", started_at: row.started_at, yes_side: mockYesSide(row.question) };
  return { id, question: row.question, status: "queued", position: queue.indexOf(id) + 1, claimed_total: claimedTotal, avg_cycle_s: WORKER_MS / 1000, yes_side: mockYesSide(row.question) };
}

function handleNow(): NowResponse {
  const last = recent[0];
  return {
    ...tick(),
    thinking: thinking ? { id: thinking.id, question: thinking.question, started_at: thinking.started_at!, embargoed: isElection(thinking.question), yes_side: mockYesSide(thinking.question) } : null,
    last: last ? summary(last) : null,
  };
}

function computeStats(): Stats {
  const answered = recent.length;
  const yes = recent.filter((a) => a.answer === "yes").length;
  const state_counts: Stats["state_counts"] = {
    appetite: recent.filter((a) => a.states.appetite !== null).length,
    fear: recent.filter((a) => a.states.fear !== null).length,
    backoff: recent.filter((a) => a.states.backoff !== null).length,
    courtship: recent.filter((a) => a.states.courtship !== null).length,
    arousal: recent.filter((a) => a.states.arousal !== null).length,
  };
  const totalSpikes = recent.reduce((s, a) => s + a.lab.total_spikes, 0);
  const simulatedMs = answered * N_TRIALS * SIM_MS;
  // Questions asked more than once, tracked by exact normalized text (dev fixture only).
  const byQuestion = new Map<string, Answer[]>();
  for (const a of recent) {
    const key = a.question.trim().toLowerCase();
    byQuestion.set(key, [...(byQuestion.get(key) ?? []), a]);
  }
  let repeated = 0;
  let identical = 0;
  for (const answers of byQuestion.values()) {
    if (answers.length > 1) {
      repeated += 1;
      if (answers.every((a) => a.answer === answers[0]!.answer)) identical += 1;
    }
  }
  return {
    answered,
    yes,
    no: answered - yes,
    total_spikes: totalSpikes,
    simulated_ms: simulatedMs,
    neuron_updates: Math.round((simulatedMs / 0.1) * N_NEURONS),
    mean_abs_corrected_bias:
      answered > 0 ? recent.reduce((s, a) => s + Math.abs(a.lab.corrected_bias), 0) / answered : NaN,
    mean_active_fraction:
      answered > 0 ? recent.reduce((s, a) => s + a.lab.active_fraction, 0) / answered : NaN,
    state_counts,
    repeat_consistency: { repeated_questions: repeated, identical_answers: identical },
    since,
  };
}

// Real, committed numbers (docs/MODEL.md "Calibration") — not fixture-random, because the real
// backend's /api/validation answer would be the same static report either way. Shape matches the
// REAL `GET /api/validation` response (checked live in dev mode 2026-09-18), see api/types.ts.
const VALIDATION: Validation = {
  model: "LIF, Shiu et al. 2024",
  brain_passport: {
    malecns: { dataset: "malecns", n_neurons: N_NEURONS, n_connections: 6235682, n_synapses: 89731551, min_syn: 5 },
    banc: { dataset: "banc", n_neurons: 114456, n_connections: 1395876, n_synapses: 17903013, min_syn: 5 },
  },
  lateral_validation: {
    malecns: {
      passed: true,
      jo_left_driven_descending_bias: { mean: 0.118, std: 0.002, n: 5 },
      jo_right_driven_descending_bias: { mean: -0.157, std: 0.005, n: 5 },
      left_vs_right_bias_gap: 0.275,
    },
    banc: {
      passed: false,
      jo_left_driven_descending_bias: { mean: -1.0, std: 0.0, n: 5 },
      jo_right_driven_descending_bias: { mean: -1.0, std: 0.0, n: 5 },
      left_vs_right_bias_gap: 0.0,
    },
  },
  handedness_b0: {
    brain: "malecns",
    condition: "real",
    config_hash: "devmockconfig0",
    b0: null, // per-question/per-brain, not a single constant — real backend fills this in.
    n_reference_sentences: 24,
  },
  stamps: [
    {
      manifest_file: "2026-09-18-preregistration.sha256",
      lines: Array.from({ length: 6 }, (_, i) => `line ${i + 1}`),
      ots_file: "2026-09-18-preregistration.sha256.ots",
    },
    {
      manifest_file: "2026-09-18-pre-batch.sha256",
      lines: Array.from({ length: 9 }, (_, i) => `line ${i + 1}`),
      ots_file: "2026-09-18-pre-batch.sha256.ots",
    },
  ],
};

// -- live chat (dev fixture only -- no moderation, no real Turnstile/HMAC, see chat.py/api.py for
// the real rules this stands in for) -------------------------------------------------------------

const chatMessages: ChatMessage[] = [];
let chatNextId = 1;
const chatReportsByMessage = new Map<number, Set<string>>();
const chatState = { enabled: true, slowmode_s: 4 };
const MOCK_CHAT_TOKEN = "mock-chat-token";

function handleChatGet(beforeId?: number): ChatGetResponse {
  const pool = beforeId === undefined ? chatMessages : chatMessages.filter((m) => m.id < beforeId);
  const limit = beforeId === undefined ? 100 : 50;
  return { messages: pool.slice(-limit), state: { ...chatState }, has_more: pool.length > limit };
}


function handleChatPost(body: string): ChatPostResponse {
  const { text, nickname } = JSON.parse(body) as { text?: string; nickname?: string };
  const trimmed = (text ?? "").trim();
  if (trimmed.length === 0 || trimmed.length > 200) {
    return { status: "rejected", reason: "too_long", message: "Keep it to 200 characters or fewer." };
  }
  const message: ChatMessage = {
    id: chatNextId++,
    nickname: nickname?.trim() || `fly-${1000 + Math.floor(Math.random() * 9000)}`,
    text: trimmed,
    created_at: new Date().toISOString(),
  };
  chatMessages.push(message);
  emit({ type: "chat", data: message });
  return { status: "ok", ...message, chat_token: MOCK_CHAT_TOKEN };
}

function handleChatReport(id: number): ChatReportResponse {
  const reporters = chatReportsByMessage.get(id) ?? new Set<string>();
  reporters.add(`mock-reporter-${reporters.size}`);
  chatReportsByMessage.set(id, reporters);
  const autoHidden = reporters.size >= 3;
  if (autoHidden) {
    const idx = chatMessages.findIndex((m) => m.id === id);
    if (idx >= 0) chatMessages.splice(idx, 1);
    emit({ type: "chat_delete", data: { id } });
  }
  return { status: "ok", id, reports: reporters.size, auto_hidden: autoHidden };
}

export function installMock(): void {
  startMockQueue();
  const realFetch = window.fetch.bind(window);
  window.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = typeof input === "string" ? input : input.toString();
    if (url.includes("/api/ask") && init?.method === "POST") {
      const data = await handleAsk(String(init.body));
      return new Response(JSON.stringify(data), { status: data.status === "rejected" ? 422 : 202 });
    }
    if (url.match(/\/api\/chat\/\d+\/report/) && init?.method === "POST") {
      const id = Number(url.match(/\/api\/chat\/(\d+)\/report/)![1]);
      return new Response(JSON.stringify(handleChatReport(id)), { status: 200 });
    }
    if (url.includes("/api/chat") && init?.method === "POST") {
      const data = handleChatPost(String(init.body));
      return new Response(JSON.stringify(data), { status: data.status === "rejected" ? 422 : 200 });
    }
    if (url.includes("/api/chat")) {
      const before = new URL(url, window.location.origin).searchParams.get("before_id");
      return new Response(JSON.stringify(handleChatGet(before === null ? undefined : Number(before))), { status: 200 });
    }
    if (url.includes("/api/now")) {
      return new Response(JSON.stringify(handleNow()), { status: 200 });
    }
    if (url.includes("/api/feed")) {
      const feed: Feed = { queue_length: queue.length, recent: recent.slice(0, 20).map(summary) };
      return new Response(JSON.stringify(feed), { status: 200 });
    }
    const likeMatch = url.match(/\/api\/answers\/([^/?]+)\/like/);
    if (likeMatch && init?.method === "POST") {
      const data = handleLike(likeMatch[1]!);
      if (!data) return new Response("not found", { status: 404 });
      return new Response(JSON.stringify(data), { status: 200 });
    }
    if (url.includes("/api/questions")) {
      const params = new URL(url, window.location.origin).searchParams;
      const data = handleQuestions(
        params.get("sort") === "top" ? "top" : "recent",
        Number(params.get("limit") ?? 20),
        Number(params.get("offset") ?? 0),
      );
      return new Response(JSON.stringify(data), { status: 200 });
    }
    if (url.includes("/api/stats")) {
      return new Response(JSON.stringify(computeStats()), { status: 200 });
    }
    if (url.includes("/api/validation")) {
      return new Response(JSON.stringify(VALIDATION), { status: 200 });
    }
    const answerMatch = url.match(/\/api\/answers\/([^/?]+)/);
    if (answerMatch) {
      // Same row shape as the real endpoint (api.py get_answer), not a bare Answer: the fixture
      // returning the bare Answer is what hid the prod share-link bug until 2026-09-19.
      const row = handleLookup(answerMatch[1]!);
      if (!row) return new Response("not found", { status: 404 });
      return new Response(JSON.stringify(row), { status: 200 });
    }
    return realFetch(input, init);
  };
  // @ts-expect-error -- dev-only fixture stands in for the real EventSource constructor.
  window.EventSource = MockEventSource;
}
