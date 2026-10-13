import { RESULT_BUCKET, UUID_RE, VIDEO_BUCKET, configured, db, fail, json } from "@/lib/server";
import type { JobResult } from "@/lib/types";

export const dynamic = "force-dynamic";
type Ctx = { params: Promise<{ id: string }> };

/** One job, plus short-lived download links once it is done. */
export async function GET(_req: Request, { params }: Ctx) {
  const { id } = await params;
  if (!UUID_RE.test(id)) return fail("Bad job id.", 400);
  if (!configured().ok) return fail("Supabase is not configured.", 503);
  const supa = db();
  const { data: job, error } = await supa.from("jobs").select("*").eq("id", id).maybeSingle();
  if (error) return fail(error.message, 500);
  if (!job) return fail("No such job.", 404);

  const urls: Record<string, string> = {};
  if (job.status === "done" && job.result) {
    const files = (job.result as JobResult).files || {};
    await Promise.all(
      Object.entries(files).map(async ([key, name]) => {
        const s = await supa.storage.from(RESULT_BUCKET).createSignedUrl(`${id}/${name}`, 3600);
        if (s.data?.signedUrl) urls[key] = s.data.signedUrl;
      }),
    );
  }
  return json({ job: { ...job, urls } });
}

/** Delete a job and every file that belongs to it. */
export async function DELETE(_req: Request, { params }: Ctx) {
  const { id } = await params;
  if (!UUID_RE.test(id)) return fail("Bad job id.", 400);
  if (!configured().ok) return fail("Supabase is not configured.", 503);
  const supa = db();
  const { data: job } = await supa.from("jobs").select("video_path, status").eq("id", id).maybeSingle();
  if (!job) return fail("No such job.", 404);
  if (job.status === "running") return fail("Cancel the job first, then delete it.", 409);
  await supa.storage.from(VIDEO_BUCKET).remove([job.video_path]);
  await supa.storage.from(RESULT_BUCKET).remove(["broadcast.mp4", "report.html", "stats.zip"].map((f) => `${id}/${f}`));
  const del = await supa.from("jobs").delete().eq("id", id);
  if (del.error) return fail(del.error.message, 500);
  return json({ ok: true });
}
