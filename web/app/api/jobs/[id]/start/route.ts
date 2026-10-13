import { UUID_RE, VIDEO_BUCKET, configured, db, fail, json } from "@/lib/server";

export const dynamic = "force-dynamic";
type Ctx = { params: Promise<{ id: string }> };

/** Step 2 of an upload: the browser says "finished". Check the file really arrived, then put the job in the queue. */
export async function POST(_req: Request, { params }: Ctx) {
  const { id } = await params;
  if (!UUID_RE.test(id)) return fail("Bad job id.", 400);
  if (!configured().ok) return fail("Supabase is not configured.", 503);
  const supa = db();
  const { data: job } = await supa.from("jobs").select("video_path, status").eq("id", id).maybeSingle();
  if (!job) return fail("No such job.", 404);
  if (job.status !== "uploading") return fail(`This job is already ${job.status}.`, 409);

  const dir = job.video_path.split("/")[0];
  const listed = await supa.storage.from(VIDEO_BUCKET).list(dir);
  const file = listed.data?.find((f) => `${dir}/${f.name}` === job.video_path);
  if (!file || !(Number(file.metadata?.size) > 0)) {
    return fail("The upload didn't arrive. Please try again.", 400);
  }
  const upd = await supa
    .from("jobs")
    .update({ status: "queued", message: "Waiting for a worker", progress: 0 })
    .eq("id", id)
    .eq("status", "uploading")
    .select("id, status")
    .maybeSingle();
  if (upd.error) return fail(upd.error.message, 500);
  return json({ ok: true, job: upd.data });
}
