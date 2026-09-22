// Typed access to content/legal.json — the Terms of Use and the Privacy Policy, kept in content/
// for the same reason copy.json is (see content.ts): editorial text never lives in TS.
//
// Both documents describe what the service ACTUALLY does. When the data handling changes — a new
// third party, a different retention window, a cookie — the matching section here changes in the
// same commit, or the policy becomes a lie.
import raw from "../content/legal.json";

export interface LegalSection {
  /** Stable anchor for sections a test binds to, so renumbering the heading cannot silently
   * disarm the check. Only "retention" uses one today
   * (tests/test_legal_retention_matches_code.py). Not rendered. */
  id?: string;
  h: string;
  /** Paragraphs rendered BEFORE the list. */
  p?: string[];
  list?: string[];
  /** Paragraphs rendered AFTER the list — for the sections where the list is the content and the
   * prose is its coda (Terms §5, Privacy §3 and §4), which read backwards when everything prose
   * was forced above the list. */
  after?: string[];
}

export interface LegalDoc {
  route: string;
  title: string;
  /** Human-readable date shown as "Last updated". */
  updated: string;
  intro: string;
  sections: LegalSection[];
}

export interface Legal {
  terms: LegalDoc;
  privacy: LegalDoc;
}

export const legal = raw as Legal;
