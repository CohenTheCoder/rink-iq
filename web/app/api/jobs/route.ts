import {
  ALLOWED_EXT, MAX_ACTIVE_JOBS, MAX_UPLOAD_MB, VIDEO_BUCKET, cleanOptions, configured, db, extOf, fail, json, safeName,
} from "@/lib/server";

export const dynamic = "force-dynamic";

/** Recent jobs, newest first. */
export async function GET() {
  if (!configured().ok) return fail("Supabase is not configured.", 503);
  const { data, error } = await db()
    .from("jobs")
    .select("id, filename, status, progress, message, created_at, finished_at")
    .order("created_at", { ascending: false })
    .limit(25);
  if (error) return fail(error.message, 500);
  return json({ jobs: data });
}

/**
 * Step 1 of an upload: validate, create the job row, and hand the browser a one-time signed upload URL so the
 * video goes straight to Supabase Storage (it never passes through Vercel, which caps request bodies at ~4.5 MB).
 */
export async function POST(req: Request) {
  if (!configured().ok) return fail("Supabase is not configured.", 503);
  let body: { filename?: unknown; size?: unknown; options?: unknown };
  try {
    body = await req.json();
  } catch {
    return fail("Send JSON: { filename, size, options }.");
  }
  const filename = typeof body.filename === "string" ? body.filename : "";
  const size = Number(body.size);
  if (!filename || !ALLOWED_EXT.includes(extOf(filename))) {
    return fail(`Use a video file (${ALLOWED_EXT.join(", ")}).`);
  }
  if (!Number.isFinite(size) || size <= 0) return fail("Missing file size.");
  if (size > MAX_UPLOAD_MB * 1024 * 1024) {
    return fail(`That file is ${(size / 1048576).toFixed(0)} MB; the limit is ${MAX_UPLOAD_MB} MB. Trim or compress it first.`, 413);
  }

  const supa = db();
  const active = await supa.from("jobs").select("id", { count: "exact", head: true }).in("status", ["uploading", "queued", "running"]);
  if ((active.count ?? 0) >= MAX_ACTIVE_JOBS) {
    return fail("The queue is full right now. Try again once a video finishes.", 429);
  }

  const id = crypto.randomUUID();
  const path = `${id}/${safeName(filename)}`;
  const options = cleanOptions(body.options);
  const ins = await supa.from("jobs").insert({
    id, filename: safeName(filename), video_path: path, size_bytes: Math.round(size), options, status: "uploading",
  });
  if (ins.error) return fail(ins.error.message, 500);

  const signed = await supa.storage.from(VIDEO_BUCKET).createSignedUploadUrl(path);
  if (signed.error || !signed.data) {
    await supa.from("jobs").delete().eq("id", id);
    return fail(signed.error?.message || "Could not create an upload link.", 500);
  }
  return json({
    id,
    uploadUrl: signed.data.signedUrl,
    apiKey: process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY, // public key: required by Supabase's gateway, grants nothing on its own
  });
}
