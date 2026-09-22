import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { localStore } from "./storage";

const SRC = new URL("./", import.meta.url).pathname;

function tsFilesUnder(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) return tsFilesUnder(full);
    return entry.name.endsWith(".ts") ? [full] : [];
  });
}

describe("the one guarded door to localStorage", () => {
  const globals = globalThis as { localStorage?: unknown };

  afterEach(() => {
    delete globals.localStorage;
  });

  it("hands back undefined when reading localStorage throws, instead of throwing", () => {
    // Safari private browsing and "block all cookies" throw on the property ACCESS itself.
    Object.defineProperty(globals, "localStorage", {
      configurable: true,
      get() {
        throw new DOMException("The operation is insecure.", "SecurityError");
      },
    });
    expect(() => localStore()).not.toThrow();
    expect(localStore()).toBeUndefined();
  });

  it("hands back the real store when it is available", () => {
    const fake = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
    Object.defineProperty(globals, "localStorage", { configurable: true, value: fake });
    expect(localStore()).toBe(fake);
  });

  it("is the only place in src/ that touches localStorage", () => {
    // The regression this guards (2026-09-21): chat.ts read `localStorage.getItem(...)` directly,
    // and home.ts mounted the chat BEFORE appending the page's sections and footer — so on a
    // phone with storage blocked the whole lower half of the live page silently vanished. Two
    // other modules already had a private copy of the guard; the next one written must not have
    // to remember. Comments are allowed to say the word; code is not.
    const offenders: string[] = [];
    for (const file of tsFilesUnder(SRC)) {
      if (file.endsWith("/storage.ts") || file.endsWith("/storage.test.ts")) continue;
      const code = readFileSync(file, "utf8")
        .replace(/\/\*[\s\S]*?\*\//g, "")
        .replace(/(^|[^:])\/\/.*$/gm, "$1");
      if (/\blocalStorage\b/.test(code)) offenders.push(file.slice(SRC.length));
    }
    expect(offenders).toEqual([]);
  });
});
