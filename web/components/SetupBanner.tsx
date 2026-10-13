"use client";

import { useEffect, useState } from "react";

interface Health { ok: boolean; step?: string; missing?: string[]; hint?: string; detail?: string; password_set?: boolean }

/** Tells the owner exactly which setup step is still missing. Shows nothing once everything works. */
export default function SetupBanner() {
  const [h, setH] = useState<Health | null>(null);
  useEffect(() => {
    fetch("/api/health").then((r) => r.json()).then(setH).catch(() => setH({ ok: false, step: "network" }));
  }, []);
  if (!h || h.ok) return null;
  return (
    <div className="notice bad" style={{ marginBottom: 16 }}>
      <strong>Setup isn’t finished.</strong>{" "}
      {h.step === "env" && <>Missing settings: {h.missing?.map((m) => <code key={m}>{m}</code>).reduce<React.ReactNode[]>((a, c, i) => (i ? [...a, " ", c] : [c]), [])}. Run <code>scripts/setup_online.sh</code> or add them in Vercel → Settings → Environment Variables, then redeploy.</>}
      {h.step === "database" && <>The database tables are missing. {h.hint}</>}
      {h.step === "storage" && <>Storage buckets missing: {h.missing?.join(", ")}. {h.hint}</>}
      {h.step === "network" && <>Couldn’t reach the server.</>}
      {h.detail && <div className="hint">{h.detail}</div>}
    </div>
  );
}
