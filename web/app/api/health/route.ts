import { configured, db } from "@/lib/server";
import { json } from "@/lib/server";

export const dynamic = "force-dynamic";

/** Setup check: which settings are present, and can we reach the database and both buckets? No secrets are returned. */
export async function GET() {
  const cfg = configured();
  if (!cfg.ok) return json({ ok: false, step: "env", missing: cfg.missing });
  const supa = db();
  const jobs = await supa.from("jobs").select("id", { count: "exact", head: true });
  if (jobs.error) {
    return json({ ok: false, step: "database", hint: "Run supabase/migrations/001_init.sql in the Supabase SQL editor.", detail: jobs.error.message });
  }
  const buckets = await supa.storage.listBuckets();
  const have = new Set((buckets.data || []).map((b) => b.id));
  const missing = ["videos", "results"].filter((b) => !have.has(b));
  if (missing.length) return json({ ok: false, step: "storage", missing, hint: "The SQL file creates these buckets; re-run it." });
  return json({ ok: true, jobs: jobs.count ?? 0, password_set: Boolean(process.env.RINKIQ_PASSWORD) });
}
