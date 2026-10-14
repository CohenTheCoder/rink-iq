export interface DemoCard { grade: string; score: number | null; strengths: string[]; weaknesses: string[]; minutes: number | null }
export interface DemoPlayer {
  id: number; team: string; goalie: boolean; ref: boolean; name: string; has_name: boolean;
  top_mph: number | null; avg_mph: number | null; distance_ft: number | null; seconds: number | null;
  card: DemoCard | null;
  /** [t, x, y, speed, facing x, facing y] */
  pts: (number | null)[][];
}
export interface DemoEvent { t: number; type: string; player: number; team: string; speed_mph: number | null; x: number | null; y: number | null }
export interface DemoData {
  title: string; note: string; duration: number; calibrated: boolean;
  teams: Record<string, { code: string; name: string }>;
  stars: number[];
  players: DemoPlayer[];
  puck: number[][];
  events: DemoEvent[];
  review: { player: number; start: number; end: number; reason: string }[];
  box_score: Record<string, Record<string, number | null>>;
}
export interface Sample { x: number; y: number; speed: number | null; fx: number | null; fy: number | null }

const clamp = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi);

/** Position of a player at time t, interpolated between tracked frames. null while they aren't being tracked. */
export function sample(p: DemoPlayer, t: number): Sample | null {
  const a = p.pts;
  if (!a.length) return null;
  if (t < (a[0][0] as number) - 0.05 || t > (a[a.length - 1][0] as number) + 0.05) return null;
  let lo = 0, hi = a.length - 1;
  while (hi - lo > 1) {
    const m = (lo + hi) >> 1;
    if ((a[m][0] as number) <= t) lo = m; else hi = m;
  }
  const A = a[lo], B = a[hi];
  const span = (B[0] as number) - (A[0] as number);
  if (span > 0.3 && t > (A[0] as number) + 0.15 && t < (B[0] as number) - 0.15) return null; // a gap in the track
  const k = span > 1e-6 ? clamp((t - (A[0] as number)) / span, 0, 1) : 0;
  const mix = (i: number) => (A[i] == null || B[i] == null ? (A[i] ?? B[i]) : (A[i] as number) + ((B[i] as number) - (A[i] as number)) * k);
  return { x: mix(1) as number, y: mix(2) as number, speed: mix(3), fx: mix(4), fy: mix(5) };
}

export function puckAt(puck: number[][], t: number): { x: number; y: number } | null {
  let best: number[] | null = null, bd = 0.2;
  for (const p of puck) {
    const d = Math.abs(p[0] - t);
    if (d < bd) { bd = d; best = p; }
  }
  return best ? { x: best[1], y: best[2] } : null;
}

export const TEAM_COLOR: Record<string, string> = { A: "#2f6fd6", B: "#d0393e", REF: "#9aa7b4" };

/** Short on-canvas label: "#63" from "#63 Z. Ostapchuk", "A-LD" stays whole, "ID 107" becomes "107". */
export function shortLabel(name: string): string {
  if (name.startsWith("#")) return name.split(" ")[0];
  if (name.startsWith("ID ")) return name.slice(3);
  return name;
}
