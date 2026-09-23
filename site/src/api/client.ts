// Real API client for the contract defined in `site/src/api/types.ts` (see docs/LIVE.md).
// Named SSE events ("queued" | "thinking" | "answered") are read with EventSource.addEventListener
// per event name — https://developer.mozilla.org/en-US/docs/Web/API/EventSource (a message with
// an explicit `event:` field only fires listeners registered for that name, not `onmessage`).
import type {
  Answer,
  AnswerLookup,
  AskResponse,
  ChatGetResponse,
  ChatPostResponse,
  ChatReportResponse,
  Feed,
  LikeResponse,
  NowResponse,
  QuestionsPage,
  ServerEvent,
  Stats,
  Validation,
} from "./types";
import { installMock } from "../mock/mockApi";

const BASE = import.meta.env.VITE_API_BASE ?? "";

// Dev-only mock: import.meta.env.DEV is statically replaced with `false` in production builds
// (https://vite.dev/guide/env-and-mode.html), so this whole branch — and the mock module it
// imports — is dead-code-eliminated from the shipped bundle.
if (import.meta.env.DEV && import.meta.env.VITE_USE_MOCK === "1") {
  installMock();
}

async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok && res.status !== 422) {
    throw new Error(`API error ${res.status}: ${await res.text()}`);
  }
  return res.json() as Promise<T>;
}

export function askQuestion(question: string, turnstileToken: string): Promise<AskResponse> {
  return fetch(`${BASE}/api/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, turnstile_token: turnstileToken }),
  }).then((res) => asJson<AskResponse>(res));
}

/** The Answer inside a lookup row; throws for a question that is still queued or was rejected. */
export function unwrapAnswerLookup(body: AnswerLookup): Answer {
  if (body.status !== "answered" || !body.answer) throw new Error(`answer ${body.id} is ${body.status}`);
  return body.answer;
}

/** The question row as it stands: queued (with its place in line), thinking, or answered. */
export function getAnswerLookup(id: string): Promise<AnswerLookup> {
  return fetch(`${BASE}/api/answers/${encodeURIComponent(id)}`).then((res) => asJson<AnswerLookup>(res));
}

export function getAnswer(id: string): Promise<Answer> {
  return getAnswerLookup(id).then(unwrapAnswerLookup);
}

/** What the live stage shows on page load: the question being simulated now, the last answer,
 * and the queue. Edge-cached for ~2 s, so a crowd reloading costs the origin almost nothing. */
export function getNow(): Promise<NowResponse> {
  return fetch(`${BASE}/api/now`).then((res) => asJson<NowResponse>(res));
}

export function getFeed(limit = 20): Promise<Feed> {
  return fetch(`${BASE}/api/feed?limit=${limit}`).then((res) => asJson<Feed>(res));
}

/** One page of the answered archive. Carries nothing per-visitor, so every tab asking for the
 * same page gets the same cached response. */
export function getQuestions(sort: "recent" | "top" = "recent", limit = 20, offset = 0): Promise<QuestionsPage> {
  return fetch(`${BASE}/api/questions?sort=${sort}&limit=${limit}&offset=${offset}`).then((res) =>
    asJson<QuestionsPage>(res),
  );
}

/** Like a question, or take the like back — the same call toggles. */
export function likeAnswer(id: string): Promise<LikeResponse> {
  return fetch(`${BASE}/api/answers/${encodeURIComponent(id)}/like`, { method: "POST" }).then((res) =>
    asJson<LikeResponse>(res),
  );
}

/** Lab-wide counters for the "Lab stats" section — see api/types.ts's Stats doc comments. */
export function getStats(): Promise<Stats> {
  return fetch(`${BASE}/api/stats`).then((res) => asJson<Stats>(res));
}

/** Brain passport + calibration checks + handedness + OpenTimestamps stamps, for "Validation". */
export function getValidation(): Promise<Validation> {
  return fetch(`${BASE}/api/validation`).then((res) => asJson<Validation>(res));
}

export function shareUrl(id: string): string {
  const pub = import.meta.env.VITE_PUBLIC_URL ?? window.location.origin;
  return `${pub}/a/${id}`;
}

export function cardImageUrl(id: string): string {
  return `${BASE}/api/card/${id}.png`;
}

// -- live chat --------------------------------------------------------------------------------

/** `null` means the hard kill switch is off (`LIVE_CHAT_ENABLED=0`, 503) -- callers hide the
 * whole chat component in that case, see pages/chat.ts's mountChat. A SOFT pause
 * (`chat_state.enabled === false`) is a normal 200 response instead, so the caller can keep
 * listening for the `chat_state` SSE event and react live without a reload. */
export function getChat(limit = 100, beforeId?: number): Promise<ChatGetResponse | null> {
  const before = beforeId === undefined ? "" : `&before_id=${beforeId}`;
  return fetch(`${BASE}/api/chat?limit=${limit}${before}`).then((res) => {
    if (res.status === 503) return null;
    return asJson<ChatGetResponse>(res);
  });
}

export function postChat(
  text: string,
  opts: { nickname?: string; turnstileToken?: string; chatToken?: string } = {},
): Promise<ChatPostResponse> {
  return fetch(`${BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text,
      nickname: opts.nickname ?? "",
      turnstile_token: opts.turnstileToken ?? "",
      chat_token: opts.chatToken ?? "",
    }),
  }).then((res) => asJson<ChatPostResponse>(res));
}

export function reportChat(id: number, chatToken: string): Promise<ChatReportResponse> {
  return fetch(`${BASE}/api/chat/${encodeURIComponent(String(id))}/report`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chat_token: chatToken }),
  }).then((res) => asJson<ChatReportResponse>(res));
}

export type ServerEventHandler = (event: ServerEvent) => void;

const SERVER_EVENT_NAMES: Exclude<ServerEvent["type"], "reconnected">[] = ["thinking", "answered", "queue", "chat", "chat_delete", "chat_state"];

// A SINGLE shared EventSource for the whole page, refcounted across every `subscribeEvents()`
// caller (home.ts's own question-flow subscription, chat.ts's chat subscription, both mounted on
// the same page) -- one physical SSE connection instead of one per caller. The public API
// (`subscribeEvents(onEvent): unsubscribe`) is unchanged; callers don't know or care that the
// underlying connection is shared. The last unsubscribe closes it; the next subscribe after that
// opens a fresh one.
let sharedSource: EventSource | null = null;
let sharedRefCount = 0;

function ensureSharedSource(): EventSource {
  if (!sharedSource) {
    sharedSource = new EventSource(`${BASE}/api/events`);
  }
  return sharedSource;
}

/** Subscribe to /api/events. Returns an unsubscribe function. */
export function subscribeEvents(onEvent: ServerEventHandler): () => void {
  const source = ensureSharedSource();
  sharedRefCount += 1;
  const listeners = SERVER_EVENT_NAMES.map((name) => {
    const listener = (evt: MessageEvent<string>) => {
      const data = JSON.parse(evt.data);
      onEvent({ type: name, data } as ServerEvent);
    };
    source.addEventListener(name, listener);
    return { name, listener };
  });
  // EventSource reconnects by itself after a drop, but whatever happened meanwhile is lost: every
  // "open" after the first tells the subscriber to re-read the state it cares about.
  let opened = source.readyState === EventSource.OPEN;
  const onOpen = () => {
    if (opened) onEvent({ type: "reconnected", data: null });
    opened = true;
  };
  source.addEventListener("open", onOpen);
  let unsubscribed = false;
  return () => {
    if (unsubscribed) return; // idempotent -- a double-unsubscribe must not double-decrement
    unsubscribed = true;
    listeners.forEach(({ name, listener }) => source.removeEventListener(name, listener));
    source.removeEventListener("open", onOpen);
    sharedRefCount = Math.max(0, sharedRefCount - 1);
    if (sharedRefCount === 0 && sharedSource) {
      sharedSource.close();
      sharedSource = null;
    }
  };
}
