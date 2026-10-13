"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import type { JobSummary } from "@/lib/types";

const LABEL: Record<string, string> = {
  uploading: "Uploading", queued: "In the queue", running: "On the ice", done: "Final", failed: "Scratched", canceled: "Canceled",
};

export default function JobList() {
  const [jobs, setJobs] = useState<JobSummary[] | null>(null);
  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    const load = async () => {
      try {
        const r = await fetch("/api/jobs");
        const d = await r.json();
        if (alive && r.ok) setJobs(d.jobs);
      } catch { /* try again on the next tick */ }
      if (alive) timer = setTimeout(load, 4000);
    };
    load();
    return () => { alive = false; clearTimeout(timer); };
  }, []);

  if (!jobs || jobs.length === 0) return null;
  return (
    <>
      <h2>Game sheet</h2>
      <div className="panel sheet" style={{ padding: 0 }}>
        {jobs.map((j) => (
          <Link key={j.id} href={`/jobs/${j.id}`} className="game">
            <div>
              <div className="name">{j.filename}</div>
              <div className="sub">
                {new Date(j.created_at).toLocaleString()} · {j.status === "running" ? `${Math.round(j.progress * 100)}% · ${j.message ?? ""}` : j.message ?? ""}
              </div>
            </div>
            <span className={`chip ${j.status}`}>{LABEL[j.status] ?? j.status}</span>
          </Link>
        ))}
      </div>
    </>
  );
}
