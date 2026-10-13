import { createClient, type SupabaseClient } from "@supabase/supabase-js";
import { NHL_TEAMS } from "./nhl";
import type { JobOptions } from "./types";

export const VIDEO_BUCKET = "videos";
export const RESULT_BUCKET = "results";
export const ALLOWED_EXT = ["mp4", "mov", "m4v", "webm", "mkv", "avi"];
export const MAX_UPLOAD_MB = Number(process.env.NEXT_PUBLIC_MAX_UPLOAD_MB || 50);
export const MAX_ACTIVE_JOBS = Number(process.env.MAX_ACTIVE_JOBS || 8);
export const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export function configured(): { ok: boolean; missing: string[] } {
  const need = ["NEXT_PUBLIC_SUPABASE_URL", "NEXT_PUBLIC_SUPABASE_ANON_KEY", "SUPABASE_SERVICE_ROLE_KEY"];
  const missing = need.filter((k) => !process.env[k]);
  return { ok: missing.length === 0, missing };
}

let client: SupabaseClient | null = null;
/** Server-only client with the SECRET key. It bypasses row-level security, so it must never reach the browser. */
export function db(): SupabaseClient {
  if (!client) {
    client = createClient(process.env.NEXT_PUBLIC_SUPABASE_URL!, process.env.SUPABASE_SERVICE_ROLE_KEY!, {
      auth: { persistSession: false, autoRefreshToken: false },
    });
  }
  return client;
}

/** Keep letters, digits, dot, dash, underscore. Never trust a file name from a browser. */
export function safeName(name: string): string {
  const base = name.split(/[\\/]/).pop() || "video.mp4";
  const cleaned = base.replace(/[^A-Za-z0-9._-]+/g, "_").replace(/^\.+/, "").slice(-80);
  return cleaned || "video.mp4";
}

export function extOf(name: string): string {
  return (name.split(".").pop() || "").toLowerCase();
}

function num(v: unknown, def: number, lo: number, hi: number): number {
  const n = Number(v);
  return Number.isFinite(n) ? Math.min(Math.max(n, lo), hi) : def;
}

/** Whitelist and bound the options a browser sends. Anything unknown is dropped. */
export function cleanOptions(raw: unknown): JobOptions {
  const o = (raw && typeof raw === "object" ? raw : {}) as Record<string, unknown>;
  const team = (v: unknown) => (typeof v === "string" && v.toUpperCase() in NHL_TEAMS ? v.toUpperCase() : "");
  const season = typeof o.season === "string" && /^\d{8}$/.test(o.season) ? o.season : "20252026";
  const quality = o.quality === "balanced" || o.quality === "detailed" ? o.quality : "fast";
  return {
    start_s: num(o.start_s, 0, 0, 6 * 3600),
    seconds: num(o.seconds, 20, 2, 3 * 3600),
    quality,
    render: o.render !== false,
    team_a: team(o.team_a),
    team_b: team(o.team_b),
    season,
  };
}

export const json = (data: unknown, status = 200) =>
  new Response(JSON.stringify(data), { status, headers: { "content-type": "application/json", "cache-control": "no-store" } });

export const fail = (message: string, status = 400) => json({ error: message }, status);
