"""Turn the pre-analysed NHL sample into one small JSON file for the website's interactive demo page.

    python scripts/export_demo_data.py        # → web/public/demo-data/{sample,simulated}.json

The file holds only *derived* data: each skater's smoothed rink position, speed and facing over time, the puck,
the detected plays and the report cards. It contains no video or images, so the demo page can be public.
Names come from the public NHL roster API (jersey number → player); offline, players show as numbers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rinkiq import identity, metrics, pipeline  # noqa: E402
from rinkiq.metrics import is_goalie  # noqa: E402
from rinkiq.report import report_cards, team_comparison  # noqa: E402

OUT = ROOT / "web" / "public" / "demo-data" / "sample.json"  # siblings are written next to it
TEAMS = {"A": {"code": "SJS", "name": "San Jose Sharks"}, "B": {"code": "MTL", "name": "Montréal Canadiens"}}


def r(x, d=1):
    return None if x is None or not np.isfinite(x) else round(float(x), d)


def build(title: str, players, puck, meta, teams: dict, names: dict | None = None, note: str = "") -> dict:
    res = metrics.analyze(players, puck, meta)
    tr, summ = res["tracks"], res["summary"].set_index("track_id")
    if names is None:
        try:
            rosters = {k: identity.nhl_roster(v["code"], "20252026") for k, v in teams.items()}
        except Exception as e:  # noqa: BLE001
            print("roster lookup failed, using numbers only:", e)
            rosters = {}
        team_of = tr.drop_duplicates("track_id").set_index("track_id")["team"].to_dict()
        names = identity.assign_names(meta.get("numbers", {}), team_of, rosters)
    cards = {c.track_id: c for c in report_cards(res["summary"], res["review"], names)}
    stars = [c.track_id for c in list(cards.values())[:3]]

    out_players = []
    for tid, g in tr.sort_values("t").groupby("track_id"):
        if len(g) < 15 and tid not in names:  # drop tiny track fragments
            continue
        label = g["label"].iloc[0]
        team = g["team"].iloc[0]
        s = summ.loc[tid] if tid in summ.index else None
        card = cards.get(tid)
        out_players.append({
            "id": int(tid), "team": team, "goalie": bool(is_goalie(label)), "ref": team == "REF",
            "name": identity.label(tid, names), "has_name": tid in names and " " in names[tid],
            "top_mph": None if s is None else r(s["top_speed_mph"]),
            "avg_mph": None if s is None else r(s["avg_speed_mph"]),
            "distance_ft": None if s is None else r(s["distance_ft"], 0),
            "seconds": None if s is None else r(s["seconds"]),
            "card": None if card is None else {
                "grade": card.grade, "score": r(card.score, 0), "strengths": card.strengths,
                "weaknesses": card.weaknesses, "minutes": r(card.minutes, 2)},
            # t, x, y, speed, facing x, facing y
            "pts": [[r(row.t, 2), r(row.sx), r(row.sy), r(row.speed_mph), r(row.face_x, 2), r(row.face_y, 2)]
                    for row in g.itertuples()],
        })
    keep = {p["id"] for p in out_players}
    ev, pk, rv = res["events"], res["puck"], res["review"]
    teams_tbl = team_comparison(res["summary"], ev)
    return {
        "title": title, "note": note,
        "duration": r(float(tr["t"].max()), 2),
        "teams": teams,
        "calibrated": bool(meta.get("calibrated")),
        "stars": [i for i in stars if i in keep],
        "players": out_players,
        "puck": [[r(row.t, 2), r(row.x), r(row.y)] for row in pk.itertuples() if np.isfinite(row.x)],
        "events": [] if ev.empty else [
            {"t": r(row.t, 2), "type": row.type, "player": int(row.player), "team": row.team,
             "speed_mph": r(row.speed_mph), "x": r(row.x), "y": r(row.y)} for row in ev.itertuples()],
        "review": [] if rv.empty else [
            {"player": int(row.track_id), "start": r(row.start_s, 2), "end": r(row.end_s, 2), "reason": row.reason}
            for row in rv.itertuples() if int(row.track_id) in keep],
        "box_score": {c: {k: (None if v is None or (isinstance(v, float) and not np.isfinite(v)) else v)
                          for k, v in teams_tbl[c].items()} for c in teams_tbl.columns} if not teams_tbl.empty else {},
    }


def save(name: str, data: dict) -> None:
    out = OUT.with_name(f"{name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":"), default=float))
    print(f"wrote {out.name} ({out.stat().st_size / 1024:.0f} KB): {len(data['players'])} players, "
          f"{len(data['puck'])} puck points, {len(data['events'])} plays, {len(data['review'])} review moments")


def main():
    from rinkiq import simulate

    p, k, m = pipeline.load_saved_run(ROOT / "samples" / "sample_run")
    save("sample", build("Sharks vs Canadiens · 10 s of real NHL broadcast film", p, k, m, TEAMS,
                         note="Real film, tracked automatically. Names come from jersey numbers read by a model."))
    p, k, m = simulate.simulate_game()
    save("simulated", build("Scripted game · the answer is known", p, k, m,
                            {"A": {"code": "", "name": "Blue"}, "B": {"code": "", "name": "Red"}}, names=m["names"],
                            note="A simulated 20 s sequence: breakout, an 88 mph one-timer, a counter-attack. "
                                 "Red's left winger (B-LW) cheats up ice on purpose. Does the analysis catch it?"))


if __name__ == "__main__":
    main()
