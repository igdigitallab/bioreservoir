// Moderated, livestream-style public chat for the live fly page (docs/LIVE.md "Live chat"). Two mount modes:
// interactive (home.ts, the default) and read-only (stream.ts's OBS-captured dark view, which is
// only ever looked at, never touched — same convention as the rest of that page, see stream.ts's
// own module docstring).
//
// Backend contract: POST/GET /api/chat, POST /api/chat/{id}/report (api.py), broadcast over the
// SAME /api/events SSE stream as three new event types -- "chat", "chat_delete", "chat_state"
// (api/client.ts's subscribeEvents already lists them). See docs/LIVE.md's "Live chat" section.
import "../chat.css";
import { h, clear } from "../dom";
import { getChat, postChat, reportChat, subscribeEvents } from "../api/client";
import { mountTurnstile, type TurnstileHandle } from "../turnstile";
import { localStore } from "../storage";
import type { ChatMessage, ChatState, ServerEvent } from "../api/types";

export const MAX_CHAT_LEN = 200;

const NICKNAME_KEY = "bioreservoir_chat_nickname";
const TOKEN_KEY = "bioreservoir_chat_token";


// -- pure helpers (unit-tested in chat.test.ts) --------------------------------------------------

/** Compact relative time for a livestream-style row ("now" / "5s" / "12m" / "3h" / "2d") — never
 * a full date/time, which would be too wide for a compact chat row. */
export function formatRelativeTime(iso: string, now: Date = new Date()): string {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const diffS = Math.max(0, Math.floor((now.getTime() - then) / 1000));
  if (diffS < 5) return "now";
  if (diffS < 60) return `${diffS}s`;
  const diffM = Math.floor(diffS / 60);
  if (diffM < 60) return `${diffM}m`;
  const diffH = Math.floor(diffM / 60);
  if (diffH < 24) return `${diffH}h`;
  return `${Math.floor(diffH / 24)}d`;
}

/** Whether a scrollable element is close enough to its bottom edge to count as "stuck" -- while
 * stuck, new messages auto-scroll into view; once the visitor scrolls up, new messages instead
 * increment the "new messages" pill until they scroll back down or click it. */
export function isStuckToBottom(
  el: { scrollTop: number; scrollHeight: number; clientHeight: number },
  thresholdPx = 48,
): boolean {
  return el.scrollHeight - el.scrollTop - el.clientHeight <= thresholdPx;
}

export function isValidNicknameFormat(nickname: string): boolean {
  return /^[A-Za-z0-9_-]{2,20}$/.test(nickname);
}

export type ChatTextValidation = { ok: true } | { ok: false; reason: "empty" | "too_long" };

/** Client-side mirror of moderation.py's check_chat_length -- a pure UX shortcut (instant
 * feedback, no round-trip for the obvious cases), never the source of truth: the server re-checks
 * everything regardless, including this. */
export function validateChatText(text: string, maxLen = MAX_CHAT_LEN): ChatTextValidation {
  const trimmed = text.trim();
  if (trimmed.length === 0) return { ok: false, reason: "empty" };
  if (trimmed.length > maxLen) return { ok: false, reason: "too_long" };
  return { ok: true };
}

/** A chat_token is `"<author_id>.<issued_at>.<expires_at>.<sig>"` (chat.py's issue_chat_token,
 * hardened 2026-09-18 to bind the token to the author_id that solved Turnstile -- security review
 * F3). The timestamps are plain text by design (only the signature is secret), so the client can
 * skip showing Turnstile again for an already-valid stored token without asking the server first.
 * This is a UX hint only: the server re-derives author_id from the CURRENT request and verifies
 * the signature for real on every request that uses the token. */
export function isTokenLikelyValid(token: string | null | undefined, nowMs: number = Date.now()): boolean {
  if (!token) return false;
  const parts = token.split(".");
  if (parts.length !== 4) return false;
  const exp = Number(parts[2]);
  return Number.isFinite(exp) && exp * 1000 > nowMs;
}

export interface CaptchaRecoveryAction {
  clearToken: boolean;
  showTurnstile: boolean;
}

/** R3 (round-2 security review): a "captcha" rejection means the token/author binding no longer
 * matches the CURRENT request (F3) -- the most common real cause is the visitor's IP changing
 * (wifi -> cellular), not an attack. Before this fix the stale token stayed in localStorage
 * (still "likely valid" by its own expiry) and Turnstile was never re-mounted once a valid-looking
 * token existed at init, so every send failed silently until the ~1h TTL expired -- effectively
 * locking a visitor out of chat for an hour after any network change. Any OTHER rejection reason
 * leaves the existing session alone; this is a pure decision, wired into the real DOM/localStorage
 * side effects by `handleSend` in `mountChat` below. */
export function captchaRecoveryAction(reason: string): CaptchaRecoveryAction {
  if (reason !== "captcha") return { clearToken: false, showTurnstile: false };
  return { clearToken: true, showTurnstile: true };
}

// -- component ------------------------------------------------------------------------------------

export interface ChatHandle {
  el: HTMLElement;
  dispose(): void;
}

export function mountChat(opts: { readOnly?: boolean } = {}): ChatHandle {
  const readOnly = opts.readOnly ?? false;

  const el = h("div", { class: `chat-panel panel${readOnly ? " chat-dark" : ""}` });
  const messagesEl = h("div", { class: "chat-messages", role: "log", "aria-live": "polite" });
  const jumpPill = h("button", { class: "chat-jump-pill", type: "button", hidden: "" }, "");
  const pausedBanner = h("p", { class: "chat-paused-banner", hidden: "" }, "Chat is paused.");

  el.appendChild(h("div", { class: "chat-header" }, [h("h3", { class: "label-mono" }, "Live chat")]));
  const messagesWrap = h("div", { class: "chat-messages-wrap" }, [messagesEl, jumpPill]);
  el.appendChild(messagesWrap);
  el.appendChild(pausedBanner);

  // -- composer (interactive mode only) -----------------------------------------------------------
  let nicknameEl: HTMLInputElement | undefined;
  let textEl: HTMLTextAreaElement | undefined;
  let charCount: HTMLElement | undefined;
  let sendBtn: HTMLButtonElement | undefined;
  let rejectionEl: HTMLElement | undefined;
  let slowmodeNote: HTMLElement | undefined;
  let turnstileMount: HTMLElement | undefined;

  if (!readOnly) {
    nicknameEl = h("input", {
      class: "chat-nickname",
      type: "text",
      maxlength: "20",
      placeholder: "Your name or nickname",
      "aria-label": "Nickname",
    }) as HTMLInputElement;
    nicknameEl.value = localStore()?.getItem(NICKNAME_KEY) ?? "";

    textEl = h("textarea", {
      class: "chat-text",
      rows: "1",
      maxlength: String(MAX_CHAT_LEN * 2), // generous hard stop; validateChatText owns the real limit
      placeholder: "Say something…",
      "aria-label": "Chat message",
    }) as HTMLTextAreaElement;
    charCount = h("span", { class: "chat-char-count label-mono" }, `0/${MAX_CHAT_LEN}`);
    sendBtn = h("button", { class: "btn chat-send", type: "button", disabled: "" }, "Send") as HTMLButtonElement;
    rejectionEl = h("div", { class: "chat-rejection", "aria-live": "assertive" }, "");
    slowmodeNote = h("p", { class: "chat-slowmode-note label-mono", hidden: "" }, "");
    turnstileMount = h("div", { class: "chat-turnstile-mount" });

    el.appendChild(
      h("div", { class: "chat-composer" }, [
        nicknameEl,
        turnstileMount,
        h("div", { class: "chat-input-row" }, [textEl, sendBtn]),
        h("div", { class: "chat-input-meta" }, [charCount, slowmodeNote]),
        rejectionEl,
      ]),
    );
  }

  // -- state -----------------------------------------------------------------------------------
  const renderedIds = new Set<number>();
  let pendingNewCount = 0;
  let chatToken: string | null = localStore()?.getItem(TOKEN_KEY) ?? null;
  let currentState: ChatState = { enabled: true, slowmode_s: 4 };
  let turnstileHandle: TurnstileHandle | undefined;
  let unsubscribe: (() => void) | undefined;
  let disposed = false;
  // History paging (2026-09-19: the chat must keep its history and scroll back through
  // it): scrolling near the top loads the page of messages before the oldest one shown.
  let oldestId: number | undefined;
  let hasMore = false;
  let loadingOlder = false;
  const olderNote = h("p", { class: "chat-older", hidden: "" }, "Loading earlier messages…");

  function loadOlder() {
    if (!hasMore || loadingOlder || oldestId === undefined || disposed) return;
    loadingOlder = true;
    olderNote.hidden = false;
    getChat(50, oldestId)
      .then((resp) => {
        if (disposed || !resp) return;
        const before = messagesEl.scrollHeight;
        const frag = document.createDocumentFragment();
        for (const msg of resp.messages) {
          if (renderedIds.has(msg.id)) continue;
          renderedIds.add(msg.id);
          frag.appendChild(buildRow(msg));
        }
        olderNote.after(frag);
        // Keep the message the visitor was looking at where it was.
        messagesEl.scrollTop += messagesEl.scrollHeight - before;
        if (resp.messages.length) oldestId = Math.min(oldestId ?? Infinity, ...resp.messages.map((m) => m.id));
        hasMore = !!resp.has_more && resp.messages.length > 0;
      })
      .catch((err) => console.error("chat: older page failed", err))
      .finally(() => {
        loadingOlder = false;
        olderNote.hidden = true;
      });
  }

  function showLoading() {
    clear(messagesEl);
    messagesEl.appendChild(h("p", { class: "chat-empty-state" }, "Loading chat…"));
  }

  function showEmptyIfNoMessages() {
    if (renderedIds.size > 0) return;
    clear(messagesEl);
    messagesEl.appendChild(olderNote);
    messagesEl.appendChild(h("p", { class: "chat-empty-state" }, readOnly ? "Waiting for chat…" : "No messages yet — say hi!"));
  }

  function clearEmptyState() {
    messagesEl.querySelector(".chat-empty-state")?.remove();
  }

  function updateJumpPill() {
    jumpPill.hidden = pendingNewCount === 0;
    jumpPill.textContent = pendingNewCount === 1 ? "↓ 1 new message" : `↓ ${pendingNewCount} new messages`;
  }

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
    pendingNewCount = 0;
    updateJumpPill();
  }

  function buildRow(msg: ChatMessage): HTMLElement {
    const row = h("div", { class: "chat-row", "data-msg-id": String(msg.id) }, [
      h("span", { class: "chat-row-nick" }, msg.nickname),
      h("span", { class: "chat-row-text" }, msg.text),
      h("span", { class: "chat-row-time label-mono" }, formatRelativeTime(msg.created_at)),
    ]);
    if (!readOnly) {
      const reportBtn = h(
        "button",
        { class: "chat-report-btn", type: "button", "aria-label": `Report message from ${msg.nickname}` },
        "report",
      ) as HTMLButtonElement;
      reportBtn.addEventListener("click", () => void onReport(msg.id, reportBtn));
      row.appendChild(reportBtn);
    }
    return row;
  }

  function appendMessage(msg: ChatMessage) {
    if (renderedIds.has(msg.id)) return;
    renderedIds.add(msg.id);
    clearEmptyState();
    if (readOnly) {
      messagesEl.appendChild(buildRow(msg));
      messagesEl.scrollTop = messagesEl.scrollHeight; // an OBS capture is never scrolled by hand
      return;
    }
    const stuck = isStuckToBottom(messagesEl);
    messagesEl.appendChild(buildRow(msg));
    if (stuck) scrollToBottom();
    else {
      pendingNewCount += 1;
      updateJumpPill();
    }
  }

  function removeMessage(id: number) {
    renderedIds.delete(id);
    messagesEl.querySelector(`[data-msg-id="${id}"]`)?.remove();
    showEmptyIfNoMessages();
  }

  async function onReport(id: number, btn: HTMLButtonElement) {
    if (!isTokenLikelyValid(chatToken)) return; // nothing to authenticate the report with yet
    btn.disabled = true;
    try {
      await reportChat(id, chatToken as string);
      btn.textContent = "reported";
    } catch (err) {
      console.error("chat: report failed", err);
      btn.disabled = false;
    }
  }

  /** Either an already-valid stored session token, or a freshly-solved Turnstile widget --
   * mirrors the server's own "Turnstile verified on the first message only" rule so
   * Send stays disabled until one of the two is true. */
  function hasAuth(): boolean {
    return isTokenLikelyValid(chatToken) || Boolean(turnstileHandle?.getToken());
  }

  function canComposeRightNow(): boolean {
    return currentState.enabled && (textEl?.value.trim().length ?? 0) > 0 && hasAuth();
  }

  function applyState(state: ChatState) {
    currentState = state;
    pausedBanner.hidden = state.enabled;
    if (readOnly) return;
    if (textEl) textEl.disabled = !state.enabled;
    if (sendBtn) sendBtn.disabled = !canComposeRightNow();
    if (slowmodeNote) {
      const show = state.enabled && state.slowmode_s > 0;
      slowmodeNote.hidden = !show;
      slowmodeNote.textContent = show ? `slow mode: 1 message / ${state.slowmode_s}s` : "";
    }
  }

  function showRejection(message: string) {
    if (rejectionEl) rejectionEl.textContent = message;
  }

  async function handleSend() {
    if (!textEl || !sendBtn) return;
    const validation = validateChatText(textEl.value);
    if (!validation.ok) {
      showRejection(validation.reason === "empty" ? "Say something first." : `Keep it to ${MAX_CHAT_LEN} characters or fewer.`);
      return;
    }
    const nickname = nicknameEl?.value.trim() ?? "";
    if (nickname && !isValidNicknameFormat(nickname)) {
      showRejection("Nicknames must be 2-20 characters: letters, numbers, _ or - only.");
      return;
    }
    if (!currentState.enabled) {
      showRejection("Chat is paused.");
      return;
    }
    // The Enter-key path bypasses the Send button's disabled state, so re-check auth here too
    // (not just in canComposeRightNow, which only gates the button).
    if (!hasAuth()) {
      recoverExpiredAuth();
      showRejection("Still verifying you're human — try again in a moment.");
      return;
    }

    showRejection("");
    sendBtn.disabled = true;
    const hasToken = isTokenLikelyValid(chatToken);
    const text = textEl.value;
    const resp = await postChat(text, {
      nickname,
      turnstileToken: hasToken ? undefined : turnstileHandle?.getToken(),
      chatToken: hasToken ? (chatToken as string) : undefined,
    }).catch((err) => {
      showRejection(String(err));
      return undefined;
    });
    sendBtn.disabled = !canComposeRightNow();
    if (!resp) return;

    if (resp.chat_token) {
      chatToken = resp.chat_token;
      localStore()?.setItem(TOKEN_KEY, chatToken);
      if (turnstileMount) turnstileMount.hidden = true;
      // The widget's token is single-use and now spent; forget it so an expired session token
      // later reads as "no auth" and recoverExpiredAuth() can bring a fresh challenge back.
      turnstileHandle = undefined;
    }

    if (resp.status === "rejected") {
      showRejection(resp.message);
      const recovery = captchaRecoveryAction(resp.reason);
      if (recovery.clearToken) {
        // R3: the token the server just rejected is stale (IP/author mismatch, or expired) --
        // drop it everywhere so canComposeRightNow()/hasAuth() stop treating it as valid.
        chatToken = null;
        localStore()?.removeItem?.(TOKEN_KEY);
        turnstileHandle = undefined;
      }
      if (recovery.showTurnstile) showTurnstileChallenge();
      else turnstileHandle?.reset();
      return;
    }

    if (nickname) localStore()?.setItem(NICKNAME_KEY, nickname);
    textEl.value = "";
    updateCharCount();
    appendMessage({ id: resp.id, nickname: resp.nickname, text: resp.text, created_at: resp.created_at });
  }

  /** A stored session token can expire (1 h TTL) while the tab stays open, and nothing on screen
   * would ever ask for a new one: the visitor sat on "Still verifying" until a reload (security
   * review round 3). Bring the challenge back once, as soon as they try to write again.
   * `turnstileMount.hidden` doubles as the "no challenge showing" flag, so this never re-mounts a
   * widget that is already up. */
  function recoverExpiredAuth() {
    if (turnstileMount?.hidden && !isTokenLikelyValid(chatToken) && !turnstileHandle) {
      showTurnstileChallenge();
    }
  }

  function updateCharCount() {
    if (!textEl || !charCount) return;
    recoverExpiredAuth();
    const len = textEl.value.trim().length;
    charCount.textContent = `${len}/${MAX_CHAT_LEN}`;
    charCount.classList.toggle("chat-char-count-over", len > MAX_CHAT_LEN);
    if (sendBtn) sendBtn.disabled = !canComposeRightNow();
  }

  /** Mounts (or re-mounts) the Turnstile widget into `turnstileMount` -- used both at init (no
   * valid stored token yet) and, since R3, after a "captcha" rejection to let the visitor recover
   * without reloading the page. `clear()` first so a re-mount doesn't stack a second widget on
   * top of a stale one still sitting in the container. */
  function showTurnstileChallenge() {
    if (!turnstileMount) return;
    clear(turnstileMount);
    turnstileMount.hidden = false;
    // `handle` is assigned by the time a real visitor could ever solve the widget (render() is
    // synchronous; onToken only fires after user interaction) -- so by the time onToken runs,
    // `turnstileHandle` already holds the real handle and `.getToken()` returns the solved token.
    // This callback only needs to re-evaluate the Send button's disabled state.
    mountTurnstile(turnstileMount, () => {
      if (sendBtn) sendBtn.disabled = !canComposeRightNow();
    })
      .then((handle) => {
        turnstileHandle = handle;
      })
      .catch((err) => console.error("chat: turnstile failed to load", err));
  }

  if (!readOnly && textEl && sendBtn) {
    textEl.addEventListener("input", updateCharCount);
    textEl.addEventListener("keydown", (event) => {
      if (event.key !== "Enter") return;
      event.preventDefault(); // Shift+Enter is deliberately not a newline -- compact single-line rows
      if (!event.shiftKey) void handleSend();
    });
    sendBtn.addEventListener("click", () => void handleSend());

    if (!isTokenLikelyValid(chatToken) && turnstileMount) {
      showTurnstileChallenge();
    } else {
      if (turnstileMount) turnstileMount.hidden = true;
      sendBtn.disabled = !canComposeRightNow();
    }
  }

  messagesEl.addEventListener("scroll", () => {
    if (messagesEl.scrollTop < 80) loadOlder();
    if (!readOnly && isStuckToBottom(messagesEl)) {
      pendingNewCount = 0;
      updateJumpPill();
    }
  });
  jumpPill.addEventListener("click", scrollToBottom);

  function handleServerEvent(event: ServerEvent) {
    if (event.type === "chat") appendMessage(event.data);
    else if (event.type === "chat_delete") removeMessage(event.data.id);
    else if (event.type === "chat_state") applyState(event.data);
  }

  showLoading();
  getChat()
    .then((resp) => {
      if (disposed) return;
      if (resp === null) {
        // Hard kill switch (LIVE_CHAT_ENABLED=0) -- hide the whole component, no SSE subscription.
        el.hidden = true;
        return;
      }
      clear(messagesEl);
      messagesEl.appendChild(olderNote);
      hasMore = !!resp.has_more;
      if (resp.messages.length) oldestId = Math.min(...resp.messages.map((m) => m.id));
      // Bulk initial history, not "new" messages -- render directly instead of going through
      // appendMessage's stuck-to-bottom/pill accounting (which would otherwise judge each row
      // against a scroll position that hasn't caught up yet mid-loop, and misreport a pile of
      // "N new messages" for a view nobody has scrolled away from yet).
      for (const msg of resp.messages) {
        if (renderedIds.has(msg.id)) continue;
        renderedIds.add(msg.id);
        messagesEl.appendChild(buildRow(msg));
      }
      scrollToBottom();
      showEmptyIfNoMessages();
      applyState(resp.state);
      unsubscribe = subscribeEvents(handleServerEvent);
    })
    .catch((err) => {
      console.error("chat: initial load failed", err);
      if (disposed) return;
      clear(messagesEl);
      messagesEl.appendChild(h("p", { class: "chat-empty-state" }, "Couldn't load chat — reload to try again."));
    });

  return {
    el,
    dispose() {
      disposed = true;
      unsubscribe?.();
    },
  };
}
