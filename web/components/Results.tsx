import type { Card, Job } from "@/lib/types";

const STARS = ["⭐⭐⭐ First star", "⭐⭐ Second star", "⭐ Third star"];
const fmt = (v: number | null | undefined, d = 1) => (v == null ? "–" : v.toFixed(d));

function Stars({ cards }: { cards: Card[] }) {
  return (
    <div className="stars">
      {cards.map((c, i) => (
        <div key={c.track_id} className={`star team-${c.team.toLowerCase()}`}>
          <span className="g">{c.grade}</span>
          <div className="s">{STARS[i]}</div>
          <h3>{c.name}</h3>
          <div className="meta">Team {c.team} · {c.score.toFixed(0)}/100 · {c.minutes.toFixed(1)} min tracked</div>
          <ul>{(c.strengths.length ? c.strengths : ["Solid all-round shift"]).map((s) => <li key={s}>✅ {s}</li>)}</ul>
        </div>
      ))}
    </div>
  );
}

export default function Results({ job }: { job: Job }) {
  const r = job.result;
  if (!r) return null;
  const t = r.tiles;
  const teams = Object.keys(r.teams);
  const metrics = teams.length ? Object.keys(r.teams[teams[0]]) : [];
  const urls = job.urls ?? {};

  return (
    <>
      <div className="tiles">
        <div className="tile"><div className="k">Skaters tracked</div><div className="v led">{t.skaters}</div></div>
        <div className="tile"><div className="k">Players named</div><div className="v led">{t.named}</div><div className="s">{t.numbers_read} numbers read</div></div>
        <div className="tile"><div className="k">Top speed</div><div className="v led">{fmt(t.top_speed_mph)}<small> mph</small></div><div className="s">{t.fastest ?? ""}</div></div>
        <div className="tile"><div className="k">Shots · passes</div><div className="v led">{t.shots} · {t.passes}</div></div>
        <div className="tile"><div className="k">Mode</div><div className="v led" style={{ fontSize: "1.25rem", paddingTop: 8 }}>{r.calibration.mode}</div></div>
      </div>
      {r.calibration.auto ? (
        <p className="hint">Rink found automatically at {r.calibration.auto.seconds.toFixed(1)} s from {r.calibration.auto.markings} markings and {r.calibration.auto.lines} painted lines (fit error {r.calibration.auto.fit_error_ft.toFixed(1)} ft).</p>
      ) : (
        <div className="notice">The rink couldn’t be found automatically, so this run measures speed only (positions, zones and positioning stats need the rink). A clip with faceoff dots and the blue/center lines in view works best.</div>
      )}

      {urls.video && (
        <>
          <h2>Broadcast cut</h2>
          <video src={urls.video} controls playsInline preload="metadata" />
        </>
      )}
      <div className="row" style={{ marginTop: 12 }}>
        {urls.report && <a className="btn blue" href={urls.report} target="_blank" rel="noreferrer">Game report</a>}
        {urls.stats && <a className="btn ghost" href={urls.stats}>All stats (CSV zip)</a>}
        {urls.video && <a className="btn ghost" href={urls.video} download>Download video</a>}
      </div>

      {r.stars.length > 0 && (
        <>
          <h2>Three stars</h2>
          <Stars cards={r.stars} />
        </>
      )}

      {teams.length > 0 && (
        <>
          <h2>Box score</h2>
          <div className="panel" style={{ padding: 6 }}>
            <table>
              <thead><tr><th /> {teams.map((x) => <th key={x}>{x}</th>)}</tr></thead>
              <tbody>
                {metrics.map((m) => (
                  <tr key={m}><td>{m}</td>{teams.map((x) => <td key={x} className="num">{r.teams[x][m] ?? "–"}</td>)}</tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {r.cards.length > 0 && (
        <details>
          <summary>Every report card ({r.cards.length})</summary>
          <div className="cards">
            {r.cards.map((c) => (
              <div key={c.track_id} className={`card ${c.team === "B" ? "b" : ""}`}>
                <div className="top"><strong>{c.name}</strong><span className="g">{c.grade}</span></div>
                <div className="hint" style={{ marginTop: 0 }}>Team {c.team} · {c.minutes.toFixed(1)} min</div>
                <ul>
                  {c.strengths.map((s) => <li key={s}>✅ {s}</li>)}
                  {c.weaknesses.map((w) => <li key={w}>⚠️ {w}</li>)}
                </ul>
              </div>
            ))}
          </div>
        </details>
      )}

      {r.events.length > 0 && (
        <details>
          <summary>Play log ({r.events.length})</summary>
          <div className="panel" style={{ padding: 6 }}>
            <table>
              <thead><tr><th>Time</th><th>Play</th><th>Player</th><th>Speed</th></tr></thead>
              <tbody>
                {r.events.map((e, i) => (
                  <tr key={i}>
                    <td className="num">{e.t.toFixed(1)}s</td>
                    <td style={{ textAlign: "right", textTransform: "uppercase" }}>{e.type}</td>
                    <td style={{ textAlign: "right" }}>{e.player}{e.to ? ` → ${e.to}` : ""}</td>
                    <td className="num">{e.speed_mph == null ? "–" : `${e.speed_mph.toFixed(0)} mph`}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}

      {r.camera_shots.length > 1 && (
        <details>
          <summary>Camera shots ({r.camera_shots.length})</summary>
          <div className="panel" style={{ padding: 6 }}>
            <table>
              <thead><tr><th>From</th><th>To</th><th>Result</th></tr></thead>
              <tbody>
                {r.camera_shots.map((s, i) => (
                  <tr key={i}><td className="num">{s.start_s.toFixed(1)}s</td><td className="num">{s.end_s.toFixed(1)}s</td><td style={{ textAlign: "right" }}>{s.status}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </>
  );
}
