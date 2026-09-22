import { describe, expect, it } from "vitest";
import { parseRoute } from "./router";

describe("parseRoute", () => {
  it("maps / with no query to home", () => {
    expect(parseRoute("/")).toEqual({ name: "home" });
    expect(parseRoute("")).toEqual({ name: "home" });
  });

  // The server's OG share page (GET /a/{id}, card.py) meta-refreshes real visitors to
  // /?a={id} — this is the deep link a shared answer actually arrives on.
  it("maps /?a=<id> to the share route, same as the path-based /a/:id route", () => {
    expect(parseRoute("/", "?a=4")).toEqual({ name: "share", id: "4" });
  });

  it("ignores other query params on /", () => {
    expect(parseRoute("/", "?utm_source=twitter")).toEqual({ name: "home" });
  });

  it("still maps the path-based /a/:id route (client-generated share links)", () => {
    expect(parseRoute("/a/4")).toEqual({ name: "share", id: "4" });
    expect(parseRoute("/a/4/")).toEqual({ name: "share", id: "4" });
  });

  it("decodes a URL-encoded id on the path route", () => {
    expect(parseRoute("/a/abc%20def")).toEqual({ name: "share", id: "abc def" });
  });

  it("does not decode the id from the query route (URLSearchParams already decodes it)", () => {
    expect(parseRoute("/", "?a=abc%20def")).toEqual({ name: "share", id: "abc def" });
  });

  it("routes both legal documents, with or without a trailing slash", () => {
    expect(parseRoute("/terms")).toEqual({ name: "legal", doc: "terms" });
    expect(parseRoute("/terms/")).toEqual({ name: "legal", doc: "terms" });
    expect(parseRoute("/privacy")).toEqual({ name: "legal", doc: "privacy" });
    expect(parseRoute("/privacy/")).toEqual({ name: "legal", doc: "privacy" });
  });

  it("sends the retired /scoreboard path to the live page, not to a dead end", () => {
    expect(parseRoute("/scoreboard")).toEqual({ name: "home" });
  });
  it("maps /stream", () => {
    expect(parseRoute("/stream")).toEqual({ name: "stream" });
  });

  it("falls back to not-found for an unknown path", () => {
    expect(parseRoute("/nope")).toEqual({ name: "not-found" });
  });
});
