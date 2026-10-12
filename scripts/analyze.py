"""Command-line version of the whole pipeline, built for long videos and full games.

    python scripts/analyze.py --demo                                   # synthetic game, no video
    python scripts/analyze.py film.mp4 --seconds 30                    # speed-only mode, first 30 s
    python scripts/analyze.py game.mp4 --points points.json \\
        --job outputs/game1 --preset long --nhl SJS,MTL --season 20252026 --render

* --job DIR    resumable: results are saved in chunks; run the same command again to continue
* --points     JSON list of [px, py, "landmark name"] (names: rinkiq/rink.py LANDMARKS) → rink feet
* --nhl A,B    NHL team codes for team A (dark jerseys) and team B → player names from jersey numbers
* --roster-a / --roster-b  text files with one "63 Firstname Lastname" per line (any league)
* --render     also draw the broadcast-style highlight video (outputs/…/broadcast.mp4)

Writes CSVs (summary, events, review, shape, tracks, puck), report.html and, optionally, the video.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rinkiq import identity, metrics, pipeline, simulate  # noqa: E402
from rinkiq.render import render_broadcast, render_rink_replay  # noqa: E402
from rinkiq.report import html_report, report_cards, team_comparison  # noqa: E402
from rinkiq.rink import Calibration  # noqa: E402

PRESETS = {  # analysed fps, pose, number reading interval
    "long": dict(fps=10, pose=False, numbers=10),
    "balanced": dict(fps=30, pose=True, numbers=8),
    "detailed": dict(fps=30, pose=True, numbers=4),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?")
    ap.add_argument("--demo", action="store_true", help="analyse the synthetic game instead of a video")
    ap.add_argument("--points", help="JSON list of [px, py, landmark] for rink calibration")
    ap.add_argument("--reference-frame", type=int, default=0)
    ap.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    ap.add_argument("--end", type=float, help="end time in seconds")
    ap.add_argument("--seconds", type=float, help="shortcut: analyse this many seconds from --start")
    ap.add_argument("--preset", choices=PRESETS, default="balanced")
    ap.add_argument("--job", help="folder for resumable chunked results")
    ap.add_argument("--nhl", help="NHL team codes for team A,B, e.g. SJS,MTL")
    ap.add_argument("--season", default="20252026")
    ap.add_argument("--roster-a")
    ap.add_argument("--roster-b")
    ap.add_argument("--render", action="store_true", help="also render the broadcast-style video")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    out = Path(args.job or args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.demo:
        players, puck, meta = simulate.simulate_game()
    elif args.video:
        cal = None
        if args.points:
            pts = json.loads(Path(args.points).read_text())
            cal = Calibration.from_pairs([((p[0], p[1]), p[2]) for p in pts])
            print(f"Calibration error: {cal.reprojection_error_ft():.1f} ft")
        info = pipeline.video_info(args.video)
        p = PRESETS[args.preset]
        end = args.end if args.end is not None else (args.start + args.seconds if args.seconds else None)
        cfg = pipeline.RunConfig(frame_stride=max(1, round(info["fps"] / p["fps"])), use_pose=p["pose"],
                                 number_every=p["numbers"], start_s=args.start, end_s=end)
        players, puck, meta = pipeline.run_video(
            args.video, calibration=cal, reference_frame=args.reference_frame, config=cfg, job_dir=args.job,
            progress=lambda f, m: print(f"\r{f:6.1%}  {m}        ", end="", flush=True))
        print()
        for s in meta["shots"]:
            print(f"  shot {s['start_frame'] / meta['fps']:7.1f}s → {s['end_frame'] / meta['fps']:7.1f}s  {s['status']}")
    else:
        ap.error("give a video path or --demo")

    res = metrics.analyze(players, puck, meta)
    rosters = {}
    if args.nhl:
        a, b = args.nhl.split(",")
        rosters = {"A": identity.nhl_roster(a.strip().upper(), args.season),
                   "B": identity.nhl_roster(b.strip().upper(), args.season)}
    for team, path in (("A", args.roster_a), ("B", args.roster_b)):
        if path:
            rosters[team] = identity.parse_roster(Path(path).read_text())
    team_of = res["tracks"].drop_duplicates("track_id").set_index("track_id")["team"].to_dict()
    names = meta.get("names") or identity.assign_names(meta.get("numbers", {}), team_of, rosters)

    for key in ("summary", "events", "review", "shape", "tracks", "puck"):
        if not res[key].empty:
            df = res[key].copy()
            if "track_id" in df:
                df.insert(1, "player", df["track_id"].map(lambda t: identity.label(t, names)))
            df.to_csv(out / f"{key}.csv", index=False)
    cards = report_cards(res["summary"], res["review"], names)
    (out / "report.html").write_text(html_report(cards, team_comparison(res["summary"], res["events"])))

    print("\n⭐ Three stars")
    for i, c in enumerate(cards[:3], 1):
        print(f"  {i}. {c.name} (Team {c.team}, {c.grade}): {'; '.join(c.strengths[:2])}")
    print(f"\nEvents: {res['events']['type'].value_counts().to_dict() if not res['events'].empty else 'none'}")
    if args.render:
        video = out / "broadcast.mp4"
        if args.demo:
            render_rink_replay(res, meta, str(video), names)
        else:
            render_broadcast(args.video, res, puck, meta, str(video), names,
                             progress=lambda f, m: print(f"\r{m}", end="", flush=True))
            print()
        print(f"Video: {video.resolve()}")
    print(f"Report + CSVs: {out.resolve()}")


if __name__ == "__main__":
    main()
