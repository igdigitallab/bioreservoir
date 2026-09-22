// One guarded door to `localStorage` for the whole site.
//
// Safari private browsing, an embedded webview and "block all cookies" make `localStorage` throw
// on ACCESS, not only on write, so `localStorage.getItem(...)` is a throwing expression on a real
// visitor's phone. queue.ts and questions.ts each carried their own copy of this guard and were
// safe; chat.ts was written later without one, and because home.ts mounted the chat before it
// appended the page's sections and footer, that single throw deleted the entire lower half of the
// live page — How it works, Brain validation, About the lab, and the Terms/Privacy links with it.
//
// Hence one module instead of three copies, and `storage.test.ts` fails if any module under src/
// reaches for `localStorage` directly again.

export interface KeyValueStore {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  /** Optional: most callers only read and write, and their test doubles model just those two. */
  removeItem?(key: string): void;
}

/** The browser's localStorage, or `undefined` where reading it is blocked or it does not exist. */
export function localStore(): KeyValueStore | undefined {
  try {
    return typeof localStorage === "undefined" ? undefined : localStorage;
  } catch {
    return undefined;
  }
}
