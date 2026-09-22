// Formatting helpers enforcing the one hard rule for every number on this page: it must come
// from the API or a committed doc, never be invented for decoration. A missing/null/NaN value
// renders as "—", never a placeholder number, a zero standing in for "unknown", or a fabricated
// figure.
const DASH = "—";

export function fmtNum(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) return DASH;
  return value.toLocaleString("en-US", { maximumFractionDigits: digits, minimumFractionDigits: 0 });
}

export function fmtPct(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return DASH;
  return `${(value * 100).toFixed(digits)}%`;
}

export function fmtSigned(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return DASH;
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(digits)}`;
}

export function fmtMs(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return DASH;
  return `${fmtNum(value, 1)} ms`;
}

export function fmtStr(value: string | null | undefined): string {
  if (value === null || value === undefined || value.length === 0) return DASH;
  return value;
}

/** ISO timestamp (Answer.answered_at) -> local HH:MM, for the recent-answers list — real data
 * from the API, not a fabricated relative time ("2m ago" would need a ticking re-render this
 * page doesn't otherwise need). Invalid/missing input renders "—", same convention as every
 * other fmt* helper here. */
export function fmtTime(value: string | null | undefined): string {
  if (value === null || value === undefined || value.length === 0) return DASH;
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return DASH;
  return d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });
}

export { DASH };
