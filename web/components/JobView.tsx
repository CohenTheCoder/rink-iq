"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import type { Job } from "@/lib/types";
import Results from "./Results";

const LABEL: Record<string, string> = {
  uploading: "Uploading", queued: "In the queue", running: "On the ice", done: "Final", failed: "Scratched", canceled: "Canceled",
};
const ACTIVE = new Set(["uploading", "queued", "running"]);

export default function JobView({ id }: { id: string }) {
  const router = useRouter();
  const [job, setJob] = useState<Job | null>(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    const load = async () => {
      let again = true;
      try {
        const r = await fetch(`/api/jobs/${id}`);
        const d = await r.json();
        if (!alive) return;
        if (!r.ok) { setErr(d.error || "Could not load this job."); again = r.status !== 404; }
        else { setJob(d.job); setErr(""); again = ACTIVE.has(d.job.status); }
      } catch { /* keep polling */ }
      if (alive && again) timer = setTimeout(load, 2000);
    };
    load();
    return () => { alive = false; clearTimeout(timer); };
  }, [id]);

  async function act(path: string, method: string) {
    setBusy(true);
    try {
      const r = await fetch(path, { method });
      if (!r.ok) setErr((await r.json()).error || "That didn't work.");
      else if (method === "DELETE") { router.push("/"); return; }
      else window.location.reload();
    } finally { setBusy(false); }
  }

  if (!job) return err ? <div className="notice bad">{err}</div> : <p className="hint">Loading…</p>;
  const waitedLong = job.status === "queued" && Date.now() - new Date(job.created_at).getTime() > 30000;
  return (
    <>
      <div className="crumbs">
        <Link href="/">← All games</Link>
        <span className={`chip ${job.status}`}>{LABEL[job.status]}</span>
      </div>
      <section className="board">
        <h1 style={{ fontSize: "clamp(1.4rem,4vw,2.2rem)", wordBreak: "break-all" }}>{job.filename}</h1>
        <div className="hint">Uploaded {new Date(job.created_at).toLocaleString()}{job.size_bytes ? ` · ${(job.size_bytes / 1048576).toFixed(1)} MB` : ""}</div>

        {ACTIVE.has(job.status) && (
          <div style={{ marginTop: 18 }}>
            <div className="row" style={{ justifyContent: "space-between" }}>
              <span className="led" style={{ fontSize: "2.4rem" }}>{String(Math.round(job.progress * 100)).padStart(2, "0")}%</span>
              <span>{job.message}</span>
            </div>
            <div className="bar" style={{ marginTop: 8 }}><i style={{ width: `${Math.max(job.progress * 100, job.status === "queued" ? 3 : 1)}%` }} /></div>
            {waitedLong && (
              <div className="notice">
                Still waiting for a worker. The analysis runs on a computer with the RinkIQ worker running
                (<code>python worker/worker.py</code>). Start it and this job will be picked up automatically.
              </div>
            )}
            <div className="row" style={{ marginTop: 14 }}>
              <button className="btn ghost" disabled={busy} onClick={() => act(`/api/jobs/${id}/cancel`, "POST")}>Cancel</button>
            </div>
          </div>
        )}
        {job.status === "failed" && <div className="notice bad"><strong>This video couldn’t be analysed.</strong><div>{job.error}</div></div>}
        {job.status === "canceled" && <div className="notice">Canceled.</div>}
        {(job.status === "failed" || job.status === "canceled" || job.status === "done") && (
          <div className="row" style={{ marginTop: 14 }}>
            <button className="btn ghost" disabled={busy} onClick={() => confirm("Delete this video and its results?") && act(`/api/jobs/${id}`, "DELETE")}>Delete</button>
          </div>
        )}
      </section>
      {err && <div className="notice bad">{err}</div>}
      {job.status === "done" && <Results job={job} />}
    </>
  );
}
