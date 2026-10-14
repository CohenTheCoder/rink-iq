"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { drawRink, RINK_L, RINK_W } from "@/lib/rinkDraw";
import { puckAt, sample, shortLabel, TEAM_COLOR, type DemoData, type DemoPlayer } from "@/lib/demo";

const DATASETS = [
  { key: "simulated", label: "Scripted game", sub: "the answer is known" },
  { key: "sample", label: "Real NHL clip", sub: "10 s of broadcast film" },
] as const;
const SPEEDS = [0.25, 0.5, 1, 2];
const STAR = ["⭐⭐⭐", "⭐⭐", "⭐"];
const STAR_NAME = ["First star", "Second star", "Third star"];

function Sparkline({ p, t, duration, onSeek }: { p: DemoPlayer; t: number; duration: number; onSeek: (t: number) => void }) {
  const pts = p.pts.filter((r) => r[3] != null);
  if (pts.length < 2) return null;
  const maxV = Math.max(25, ...pts.map((r) => r[3] as number));
  const X = (tt: number) => (tt / duration) * 300;
  const Y = (v: number) => 62 - (v / maxV) * 56;
  const d = pts.map((r, i) => `${i ? "L" : "M"}${X(r[0] as number).toFixed(1)},${Y(r[3] as number).toFixed(1)}`).join("");
  return (
    <svg viewBox="0 0 300 70" className="spark" onClick={(e) => {
      const b = e.currentTarget.getBoundingClientRect();
      onSeek(((e.clientX - b.left) / b.width) * duration);
    }}>
      <line x1="0" x2="300" y1={Y(15)} y2={Y(15)} stroke="rgba(255,255,255,.25)" strokeDasharray="3 3" />
      <text x="2" y={Y(15) - 3} fontSize="9" fill="#93a4bd">15 mph</text>
      <path d={d} fill="none" stroke={TEAM_COLOR[p.team] ?? "#fff"} strokeWidth="2" />
      <line x1={X(t)} x2={X(t)} y1="0" y2="70" stroke="#ffb81c" strokeWidth="1.5" />
    </svg>
  );
}

export default function Demo() {
  const [key, setKey] = useState<(typeof DATASETS)[number]["key"]>("simulated");
  const [data, setData] = useState<DemoData | null>(null);
  const [err, setErr] = useState("");
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [sel, setSel] = useState<number | null>(null);
  const [trails, setTrails] = useState(true);
  const [facing, setFacing] = useState(true);
  const [labels, setLabels] = useState(true);

  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const live = useRef({ t: 0, playing: true, speed: 1, sel: null as number | null, trails: true, facing: true, labels: true, data: null as DemoData | null });
  const hits = useRef<{ id: number; px: number; py: number }[]>([]);
  Object.assign(live.current, { playing, speed, sel, trails, facing, labels, data });

  useEffect(() => {
    let alive = true;
    setData(null); setErr(""); setSel(null); live.current.t = 0; setT(0);
    fetch(`/demo-data/${key}.json`).then((r) => r.json()).then((d) => alive && setData(d)).catch(() => alive && setErr("Couldn’t load the demo data."));
    return () => { alive = false; };
  }, [key]);

  const byId = useMemo(() => new Map((data?.players ?? []).map((p) => [p.id, p])), [data]);
  const nameOf = useCallback((id: number) => byId.get(id)?.name ?? `ID ${id}`, [byId]);

  const draw = useCallback(() => {
    const c = canvas.current, w = wrap.current, d = live.current.data;
    if (!c || !w || !d) return;
    const cssW = w.clientWidth;
    const m = Math.max(10, cssW * 0.012);
    const cssH = 2 * m + (RINK_W / RINK_L) * (cssW - 2 * m);
    const dpr = window.devicePixelRatio || 1;
    if (c.width !== Math.round(cssW * dpr) || c.height !== Math.round(cssH * dpr)) {
      c.width = Math.round(cssW * dpr); c.height = Math.round(cssH * dpr);
      c.style.height = `${cssH}px`;
    }
    const ctx = c.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const { X, Y, s } = drawRink(ctx, cssW, cssH);
    const { t: now, sel: selected } = live.current;
    const R = Math.max(6, s * 2.1);

    const vis = d.players.map((p) => ({ p, s: sample(p, now) })).filter((v) => v.s);
    const skaters = vis.filter((v) => !v.p.goalie && !v.p.ref);
    const fastest = skaters.reduce<typeof vis[number] | null>((b, v) => ((v.s!.speed ?? 0) > (b?.s?.speed ?? 0) ? v : b), null);
    hits.current = [];

    if (live.current.trails) {
      for (const { p } of vis) {
        ctx.beginPath();
        let started = false;
        for (let k = 10; k >= 0; k--) {
          const sm = sample(p, now - k * 0.1);
          if (!sm) { started = false; continue; }
          if (!started) { ctx.moveTo(X(sm.x), Y(sm.y)); started = true; } else ctx.lineTo(X(sm.x), Y(sm.y));
        }
        ctx.strokeStyle = TEAM_COLOR[p.team] ?? "#888";
        ctx.globalAlpha = p.id === selected ? 0.85 : 0.4;
        ctx.lineWidth = p.id === selected ? 4 : 2.5;
        ctx.lineCap = "round";
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    }
    for (const { p, s: sm } of vis) {
      const px = X(sm!.x), py = Y(sm!.y), color = TEAM_COLOR[p.team] ?? "#888";
      hits.current.push({ id: p.id, px, py });
      if (live.current.facing && sm!.fx != null && sm!.fy != null && !p.ref) {
        ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(px + sm!.fx * R * 2.4, py - sm!.fy * R * 2.4);
        ctx.strokeStyle = "#f2a900"; ctx.lineWidth = 2.5; ctx.stroke();
      }
      if (fastest && fastest.p.id === p.id && (sm!.speed ?? 0) > 12) {
        ctx.beginPath(); ctx.arc(px, py, R + 5, 0, Math.PI * 2); ctx.strokeStyle = "#ffb81c"; ctx.lineWidth = 3; ctx.stroke();
      }
      ctx.beginPath(); ctx.arc(px, py, p.id === selected ? R * 1.25 : R, 0, Math.PI * 2);
      ctx.fillStyle = color; ctx.fill();
      ctx.lineWidth = p.goalie ? 3 : 2; ctx.strokeStyle = p.goalie ? "#0a1220" : "#fff"; ctx.stroke();
      if (p.id === selected) {
        ctx.beginPath(); ctx.arc(px, py, R * 1.25 + 6 + 2 * Math.sin(performance.now() / 150), 0, Math.PI * 2);
        ctx.strokeStyle = "#ffb81c"; ctx.lineWidth = 2.5; ctx.stroke();
      }
      if (live.current.labels || p.id === selected) {
        const text = p.id === selected ? p.name + (sm!.speed != null ? `  ${sm!.speed.toFixed(0)} mph` : "") : shortLabel(p.name);
        ctx.font = `600 ${Math.max(10, s * 2.4)}px var(--f-body), sans-serif`;
        const tw = ctx.measureText(text).width;
        ctx.fillStyle = "rgba(10,18,32,.82)"; ctx.fillRect(px - tw / 2 - 4, py - R - 20, tw + 8, 17);
        ctx.fillStyle = "#fff"; ctx.textAlign = "center"; ctx.fillText(text, px, py - R - 7);
      }
    }
    const pk = puckAt(d.puck, now);
    if (pk) {
      ctx.beginPath(); ctx.arc(X(pk.x), Y(pk.y), Math.max(4, s * 1.1), 0, Math.PI * 2);
      ctx.fillStyle = "#111"; ctx.fill(); ctx.strokeStyle = "#ffe97a"; ctx.lineWidth = 2; ctx.stroke();
    }
    const ev = d.events.find((e) => now - e.t >= 0 && now - e.t < 1.4);
    if (ev) {
      const label = ev.type === "shot" ? `SHOT  ${ev.speed_mph?.toFixed(0) ?? ""} MPH` : ev.type.toUpperCase();
      const sub = `${nameOf(ev.player)}`;
      ctx.font = `800 ${Math.max(18, s * 5)}px var(--f-display), sans-serif`;
      const w2 = Math.max(ctx.measureText(label).width, 120) + 40;
      ctx.globalAlpha = Math.min(1, (1.4 - (now - ev.t)) * 2);
      ctx.fillStyle = ev.type === "shot" ? "#c8102e" : ev.type === "pass" ? "#0b3d91" : "#444";
      ctx.beginPath(); ctx.roundRect(cssW / 2 - w2 / 2, 12, w2, 52, 10); ctx.fill();
      ctx.fillStyle = "#fff"; ctx.textAlign = "center"; ctx.fillText(label, cssW / 2, 40);
      ctx.font = `500 ${Math.max(11, s * 2.4)}px var(--f-body), sans-serif`; ctx.fillText(sub, cssW / 2, 57);
      ctx.globalAlpha = 1;
    }
  }, [nameOf]);

  // animation loop: advances time, draws every frame, updates React state ~15×/s
  useEffect(() => {
    let raf = 0, last = performance.now(), lastUi = 0;
    const tick = (now: number) => {
      const L = live.current;
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      if (L.data && L.playing) {
        L.t += dt * L.speed;
        if (L.t > L.data.duration) L.t = 0;
      }
      draw();
      if (now - lastUi > 66) { lastUi = now; setT(L.t); }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [draw]);

  const seek = (v: number) => { live.current.t = Math.max(0, Math.min(v, data?.duration ?? 0)); setT(live.current.t); };
  const click = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const b = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - b.left, y = e.clientY - b.top;
    let best: { id: number; d: number } | null = null;
    for (const h of hits.current) {
      const d = Math.hypot(h.px - x, h.py - y);
      if (d < 22 && (!best || d < best.d)) best = { id: h.id, d };
    }
    setSel(best ? best.id : null);
  };

  const selected = sel != null ? byId.get(sel) ?? null : null;
  const ranked = useMemo(
    () => (data?.players ?? []).filter((p) => p.card).sort((a, b) => (b.card!.score ?? 0) - (a.card!.score ?? 0)),
    [data],
  );
  const myReview = (data?.review ?? []).filter((r) => r.player === sel);

  return (
    <>
      <div className="seg" style={{ maxWidth: 520 }}>
        {DATASETS.map((d) => (
          <button key={d.key} className={key === d.key ? "on" : ""} onClick={() => setKey(d.key)}>
            {d.label} <span style={{ opacity: 0.7, fontSize: "0.8em" }}>· {d.sub}</span>
          </button>
        ))}
      </div>
      {data && <p className="hint" style={{ maxWidth: 780 }}>{data.note}</p>}
      {err && <div className="notice bad">{err}</div>}

      <div className="demo-grid">
        <div>
          <div ref={wrap} className="demo-canvas">
            <canvas ref={canvas} onClick={click} style={{ width: "100%", display: "block", cursor: "pointer" }} />
            {!data && !err && <div className="hint" style={{ padding: 20 }}>Loading…</div>}
          </div>
          <div className="demo-controls">
            <button className="btn blue" style={{ padding: "6px 16px", fontSize: "1.1rem" }} onClick={() => setPlaying(!playing)}>{playing ? "❚❚ Pause" : "▶ Play"}</button>
            <div className="scrub">
              <input type="range" min={0} max={data?.duration ?? 1} step={0.01} value={t} onChange={(e) => seek(Number(e.target.value))} aria-label="Time" />
              {data?.events.map((e, i) => (
                <i key={i} title={`${e.type} ${e.speed_mph ?? ""}`} className={e.type} style={{ left: `${(e.t / data.duration) * 100}%` }} />
              ))}
            </div>
            <span className="led" style={{ minWidth: 76, textAlign: "right" }}>{t.toFixed(1)}s</span>
            <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))} style={{ width: 80, marginTop: 0 }} aria-label="Speed">
              {SPEEDS.map((s) => <option key={s} value={s}>{s}×</option>)}
            </select>
          </div>
          <div className="row" style={{ marginTop: 6, gap: 18 }}>
            <label className="check" style={{ marginTop: 0, paddingTop: 0 }}><input type="checkbox" checked={trails} onChange={(e) => setTrails(e.target.checked)} /> trails</label>
            <label className="check" style={{ marginTop: 0, paddingTop: 0 }}><input type="checkbox" checked={facing} onChange={(e) => setFacing(e.target.checked)} /> body-facing arrows</label>
            <label className="check" style={{ marginTop: 0, paddingTop: 0 }}><input type="checkbox" checked={labels} onChange={(e) => setLabels(e.target.checked)} /> labels</label>
            {data && Object.entries(data.teams).map(([k, v]) => (
              <span key={k} className="hint" style={{ margin: 0 }}><span className="dotkey" style={{ background: TEAM_COLOR[k] }} /> {v.name}</span>
            ))}
            <span className="hint" style={{ margin: 0 }}><span className="dotkey" style={{ background: "transparent", border: "2px solid #ffb81c" }} /> fastest skater</span>
          </div>
        </div>

        <aside className="demo-side">
          {selected ? (
            <div className="star" style={{ borderTopColor: TEAM_COLOR[selected.team] }}>
              {selected.card && <span className="g">{selected.card.grade}</span>}
              <div className="s">{data?.stars.includes(selected.id) ? STAR[data.stars.indexOf(selected.id)] + " " + STAR_NAME[data.stars.indexOf(selected.id)] : selected.ref ? "Referee" : selected.goalie ? "Goalie" : "Skater"}</div>
              <h3>{selected.name}</h3>
              <div className="meta">{data?.teams[selected.team]?.name ?? `Team ${selected.team}`} · tracked {selected.seconds?.toFixed(1) ?? "–"} s</div>
              <div className="mini">
                <div><b>{selected.top_mph?.toFixed(1) ?? "–"}</b><span>top mph</span></div>
                <div><b>{selected.avg_mph?.toFixed(1) ?? "–"}</b><span>avg mph</span></div>
                <div><b>{selected.distance_ft?.toFixed(0) ?? "–"}</b><span>feet skated</span></div>
              </div>
              <Sparkline p={selected} t={t} duration={data!.duration} onSeek={seek} />
              {selected.card && (
                <ul>
                  {selected.card.strengths.map((s) => <li key={s}>✅ {s}</li>)}
                  {selected.card.weaknesses.map((w) => <li key={w}>⚠️ {w}</li>)}
                </ul>
              )}
              {myReview.length > 0 && (
                <div style={{ marginTop: 8 }}>
                  <b>Film review</b>
                  {myReview.map((r, i) => (
                    <button key={i} className="reviewbtn" onClick={() => { seek(r.start); setPlaying(true); }}>
                      ▶ {r.start.toFixed(1)}–{r.end.toFixed(1)} s: {r.reason}
                    </button>
                  ))}
                </div>
              )}
              <button className="btn ghost" style={{ marginTop: 10, color: "#0a1220", borderColor: "#aab" }} onClick={() => setSel(null)}>← all players</button>
            </div>
          ) : (
            <div className="panel" style={{ marginTop: 0 }}>
              <h3>Click a player on the ice</h3>
              <p className="hint" style={{ marginTop: 0 }}>…or pick from the leaderboard. Grades compare each skater with the others in this clip.</p>
              {ranked.map((p, i) => (
                <button key={p.id} className="lb" onClick={() => setSel(p.id)}>
                  <span className="dotkey" style={{ background: TEAM_COLOR[p.team] }} />
                  <span className="nm">{p.name}{data?.stars.includes(p.id) ? " " + STAR[data.stars.indexOf(p.id)] : ""}</span>
                  <span className="g">{p.card!.grade}</span>
                  <span className="sc">{i + 1}</span>
                </button>
              ))}
            </div>
          )}
        </aside>
      </div>

      {data && Object.keys(data.box_score).length > 0 && (
        <>
          <h2>Box score</h2>
          <div className="panel" style={{ padding: 6 }}>
            <table>
              <thead><tr><th />{Object.keys(data.box_score).map((k) => <th key={k}>{k}</th>)}</tr></thead>
              <tbody>
                {Object.keys(Object.values(data.box_score)[0]).map((m) => (
                  <tr key={m}><td>{m}</td>{Object.keys(data.box_score).map((k) => <td key={k} className="num">{data.box_score[k][m] ?? "–"}</td>)}</tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </>
  );
}
