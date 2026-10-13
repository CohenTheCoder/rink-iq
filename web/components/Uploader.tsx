"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { NHL_TEAMS } from "@/lib/nhl";

const MAX_MB = Number(process.env.NEXT_PUBLIC_MAX_UPLOAD_MB || 50);
const QUALITY = [
  { id: "fast", label: "Fast", hint: "10 fps, no pose. Best for long clips." },
  { id: "balanced", label: "Balanced", hint: "15 fps + body orientation." },
  { id: "detailed", label: "Detailed", hint: "30 fps + body orientation. Slowest." },
] as const;

/** PUT the file straight to Supabase Storage with a real progress bar (fetch() can't report upload progress). */
function putFile(url: string, apiKey: string | undefined, file: File, onProgress: (f: number) => void): Promise<void> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", url);
    if (apiKey) {
      xhr.setRequestHeader("apikey", apiKey);
      if (apiKey.split(".").length === 3) xhr.setRequestHeader("Authorization", `Bearer ${apiKey}`);
    }
    xhr.setRequestHeader("content-type", file.type || "video/mp4");
    xhr.setRequestHeader("cache-control", "max-age=3600");
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => (xhr.status < 300 ? resolve() : reject(new Error(`Upload failed (${xhr.status}): ${xhr.responseText.slice(0, 200)}`)));
    xhr.onerror = () => reject(new Error("Network error during upload."));
    xhr.send(file);
  });
}

export default function Uploader() {
  const router = useRouter();
  const input = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [over, setOver] = useState(false);
  const [seconds, setSeconds] = useState(20);
  const [start, setStart] = useState(0);
  const [quality, setQuality] = useState<"fast" | "balanced" | "detailed">("fast");
  const [teamA, setTeamA] = useState("");
  const [teamB, setTeamB] = useState("");
  const [render, setRender] = useState(true);
  const [busy, setBusy] = useState(false);
  const [pct, setPct] = useState(0);
  const [err, setErr] = useState("");

  const pick = (f?: File | null) => {
    setErr("");
    if (!f) return;
    if (f.size > MAX_MB * 1024 * 1024) {
      setErr(`That file is ${(f.size / 1048576).toFixed(0)} MB. The limit is ${MAX_MB} MB, so trim or compress it first (for example with HandBrake, or cut a shorter clip).`);
      return;
    }
    setFile(f);
  };

  async function submit() {
    if (!file) return;
    setBusy(true);
    setErr("");
    setPct(0);
    try {
      const init = await fetch("/api/jobs", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          filename: file.name, size: file.size,
          options: { seconds, start_s: start, quality, render, team_a: teamA, team_b: teamB },
        }),
      });
      const created = await init.json();
      if (!init.ok) throw new Error(created.error || "Could not start the upload.");
      await putFile(created.uploadUrl, created.apiKey, file, setPct);
      const started = await fetch(`/api/jobs/${created.id}/start`, { method: "POST" });
      if (!started.ok) throw new Error((await started.json()).error || "Could not queue the job.");
      router.push(`/jobs/${created.id}`);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  }

  return (
    <div>
      <div
        className={`drop ${over ? "over" : ""}`}
        onClick={() => !busy && input.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); if (!busy) pick(e.dataTransfer.files[0]); }}
      >
        <h3>{file ? "Ready to go" : "Drop game film here"}</h3>
        {file ? (
          <div className="file">{file.name} · {(file.size / 1048576).toFixed(1)} MB</div>
        ) : (
          <p>or click to choose · mp4, mov, m4v, webm, mkv, avi · up to {MAX_MB} MB · elevated camera (TV broadcast, rink camera)</p>
        )}
        <input ref={input} type="file" hidden accept="video/*,.mkv" onChange={(e) => pick(e.target.files?.[0])} />
      </div>

      <div className="panel">
        <h3>Game settings</h3>
        <div className="grid">
          <label className="f">Seconds to analyse
            <input type="number" min={2} max={3600} value={seconds} onChange={(e) => setSeconds(Number(e.target.value))} />
          </label>
          <label className="f">Starting at (s)
            <input type="number" min={0} value={start} onChange={(e) => setStart(Number(e.target.value))} />
          </label>
          <div>
            <label className="f" htmlFor="q">Speed vs. detail</label>
            <div className="seg" id="q">
              {QUALITY.map((q) => (
                <button key={q.id} type="button" className={quality === q.id ? "on" : ""} onClick={() => setQuality(q.id)} title={q.hint}>{q.label}</button>
              ))}
            </div>
          </div>
          <label className="f">Team A (dark jerseys)
            <select value={teamA} onChange={(e) => setTeamA(e.target.value)}>
              <option value="">(unknown)</option>
              {Object.entries(NHL_TEAMS).map(([k, v]) => <option key={k} value={k}>{k} · {v}</option>)}
            </select>
          </label>
          <label className="f">Team B (light jerseys)
            <select value={teamB} onChange={(e) => setTeamB(e.target.value)}>
              <option value="">(unknown)</option>
              {Object.entries(NHL_TEAMS).map(([k, v]) => <option key={k} value={k}>{k} · {v}</option>)}
            </select>
          </label>
          <label className="f check">
            <input type="checkbox" checked={render} onChange={(e) => setRender(e.target.checked)} /> Render the broadcast-style video
          </label>
        </div>
        <p className="hint">Picking the teams turns jersey numbers into player names (NHL rosters). Longer stretches take longer: roughly the clip length × 30 at "Fast" on a laptop CPU.</p>
      </div>

      {busy && (
        <div className="panel">
          <div className="row"><span className="led">{Math.round(pct * 100)}%</span><span>Uploading…</span></div>
          <div className="bar" style={{ marginTop: 8 }}><i style={{ width: `${pct * 100}%` }} /></div>
        </div>
      )}
      {err && <div className="notice bad">{err}</div>}
      <div className="row" style={{ marginTop: 18 }}>
        <button className="btn" disabled={!file || busy} onClick={submit}>{busy ? "Uploading…" : "Analyze"}</button>
        {file && !busy && <button className="btn ghost" onClick={() => setFile(null)}>Clear</button>}
      </div>
    </div>
  );
}
