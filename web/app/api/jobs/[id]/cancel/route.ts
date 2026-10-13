import { UUID_RE, configured, db, fail, json } from "@/lib/server";

export const dynamic = "force-dynamic";
type Ctx = { params: Promise<{ id: string }> };

/** Stop a job that hasn't finished. A running worker notices on its next progress update and stops. */
export async function POST(_req: Request, { params }: Ctx) {
  const { id } = await params;
  if (!UUID_RE.test(id)) return fail("Bad job id.", 400);
  if (!configured().ok) return fail("Supabase is not configured.", 503);
  const upd = await db()
    .from("jobs")
    .update({ status: "canceled", message: "Canceled", finished_at: new Date().toISOString() })
    .eq("id", id)
    .in("status", ["uploading", "queued", "running"])
    .select("id")
    .maybeSingle();
  if (upd.error) return fail(upd.error.message, 500);
  if (!upd.data) return fail("That job already finished.", 409);
  return json({ ok: true });
}
