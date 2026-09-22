import { describe, expect, it } from "vitest";
import {
  captchaRecoveryAction,
  formatRelativeTime,
  isStuckToBottom,
  isTokenLikelyValid,
  isValidNicknameFormat,
  MAX_CHAT_LEN,
  validateChatText,
} from "./chat";

describe("formatRelativeTime", () => {
  const now = new Date("2026-09-18T12:00:00Z");

  it("renders very recent times as 'now'", () => {
    expect(formatRelativeTime("2026-09-18T11:59:57Z", now)).toBe("now");
  });

  it("renders seconds", () => {
    expect(formatRelativeTime("2026-09-18T11:59:30Z", now)).toBe("30s");
  });

  it("renders minutes", () => {
    expect(formatRelativeTime("2026-09-18T11:55:00Z", now)).toBe("5m");
  });

  it("renders hours", () => {
    expect(formatRelativeTime("2026-09-18T09:00:00Z", now)).toBe("3h");
  });

  it("renders days", () => {
    expect(formatRelativeTime("2026-09-15T12:00:00Z", now)).toBe("3d");
  });

  it("never returns a negative duration for a clock-skewed future timestamp", () => {
    expect(formatRelativeTime("2026-09-18T12:05:00Z", now)).toBe("now");
  });

  it("returns an empty string for an unparsable timestamp", () => {
    expect(formatRelativeTime("not-a-date", now)).toBe("");
  });
});

describe("isStuckToBottom", () => {
  it("is true when scrolled exactly to the bottom", () => {
    expect(isStuckToBottom({ scrollTop: 200, scrollHeight: 300, clientHeight: 100 })).toBe(true);
  });

  it("is true within the threshold of the bottom", () => {
    expect(isStuckToBottom({ scrollTop: 170, scrollHeight: 300, clientHeight: 100 }, 48)).toBe(true);
  });

  it("is false once scrolled further up than the threshold", () => {
    expect(isStuckToBottom({ scrollTop: 100, scrollHeight: 300, clientHeight: 100 }, 48)).toBe(false);
  });

  it("is true when content doesn't overflow at all", () => {
    expect(isStuckToBottom({ scrollTop: 0, scrollHeight: 80, clientHeight: 100 })).toBe(true);
  });
});

describe("isValidNicknameFormat", () => {
  it("accepts 2-20 char alphanumerics, underscore and hyphen", () => {
    expect(isValidNicknameFormat("ab")).toBe(true);
    expect(isValidNicknameFormat("fly_fan-99")).toBe(true);
    expect(isValidNicknameFormat("a".repeat(20))).toBe(true);
  });

  it("rejects too short or too long", () => {
    expect(isValidNicknameFormat("a")).toBe(false);
    expect(isValidNicknameFormat("a".repeat(21))).toBe(false);
  });

  it("rejects disallowed characters", () => {
    expect(isValidNicknameFormat("bad name")).toBe(false);
    expect(isValidNicknameFormat("bad@name")).toBe(false);
    expect(isValidNicknameFormat("bad.name")).toBe(false);
  });
});

describe("validateChatText", () => {
  it("accepts ordinary text", () => {
    expect(validateChatText("hello there")).toEqual({ ok: true });
  });

  it("rejects empty (and whitespace-only) text", () => {
    expect(validateChatText("")).toEqual({ ok: false, reason: "empty" });
    expect(validateChatText("   ")).toEqual({ ok: false, reason: "empty" });
  });

  it("rejects text over the limit", () => {
    expect(validateChatText("a".repeat(MAX_CHAT_LEN + 1))).toEqual({ ok: false, reason: "too_long" });
  });

  it("accepts text at exactly the limit", () => {
    expect(validateChatText("a".repeat(MAX_CHAT_LEN))).toEqual({ ok: true });
  });

  it("ignores leading/trailing whitespace when measuring length", () => {
    const padded = `  ${"a".repeat(MAX_CHAT_LEN)}  `;
    expect(validateChatText(padded)).toEqual({ ok: true });
  });
});

describe("isTokenLikelyValid", () => {
  // Token shape (chat.py's issue_chat_token, hardened 2026-09-18 for security review F3):
  // "<author_id>.<issued_at>.<expires_at>.<sig>" -- 4 dot-separated fields, expiry is the 3rd.
  it("is false for a missing token", () => {
    expect(isTokenLikelyValid(undefined)).toBe(false);
    expect(isTokenLikelyValid(null)).toBe(false);
    expect(isTokenLikelyValid("")).toBe(false);
  });

  it("is true for a token whose expiry is in the future", () => {
    const exp = Math.floor(Date.now() / 1000) + 3600;
    const token = `abc123.1000.${exp}.deadbeef`;
    expect(isTokenLikelyValid(token)).toBe(true);
  });

  it("is false for an expired token", () => {
    const exp = Math.floor(Date.now() / 1000) - 10;
    const token = `abc123.1000.${exp}.deadbeef`;
    expect(isTokenLikelyValid(token)).toBe(false);
  });

  it("is false for a malformed token", () => {
    expect(isTokenLikelyValid("not-a-token")).toBe(false);
  });

  it("is false for the old 2-field token shape (pre-F3 format)", () => {
    expect(isTokenLikelyValid(`${Math.floor(Date.now() / 1000) + 3600}.deadbeef`)).toBe(false);
  });

  it("accepts an explicit now for deterministic testing", () => {
    expect(isTokenLikelyValid("abc123.500.1000.sig", 999_000)).toBe(true);
    expect(isTokenLikelyValid("abc123.500.1000.sig", 1_000_000)).toBe(false);
  });
});

describe("captchaRecoveryAction", () => {
  // R3 (round-2 security review): a "captcha" rejection means the stored token no longer matches
  // this request's author (most commonly an IP change, e.g. wifi -> cellular) -- the state
  // transition must clear the stale token AND re-show Turnstile, or the visitor is locked out of
  // chat for up to the token's ~1h TTL.
  it("clears the token and shows Turnstile on a captcha rejection", () => {
    expect(captchaRecoveryAction("captcha")).toEqual({ clearToken: true, showTurnstile: true });
  });

  it("leaves the session alone for every other rejection reason", () => {
    for (const reason of [
      "rate_limited",
      "duplicate",
      "chat_busy",
      "chat_paused",
      "banned",
      "voting_procedure",
      "spam",
      "too_long",
      "empty",
      "nickname_invalid",
      "impersonation",
    ]) {
      expect(captchaRecoveryAction(reason)).toEqual({ clearToken: false, showTurnstile: false });
    }
  });
});
