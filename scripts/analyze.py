"""Command-line version of the whole pipeline.

    python scripts/analyze.py film.mp4                       # speed-only mode
    python scripts/analyze.py film.mp4 --points points.json  # calibrated (rink feet)
    python scripts/analyze.py --demo                         # synthetic game, no video

points.json = [[px, py, "Neutral dot · left · far"], ...]  (landmark names: rinkiq/rink.py LANDMARKS)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rinkiq import metrics, pipeline, simulate  # noqa: E402
from rinkiq.rink import Calibration  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?")
    ap.add_argument("--demo", action="store_true", help="analyse the synthetic game instead of a video")
    ap.add_argument("--points", help="JSON list of [px, py, landmark] for rink calibration")
    ap.add_argument("--reference-frame", type=int, default=0)
    ap.add_argument("--seconds", type=float, help="only analyse the first N seconds")
    ap.add_argument("--no-pose", action="store_true")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    if args.demo:
        players, puck, meta = simulate.simulate_game()
    elif args.video:
        cal = None
        if args.points:
            pts = json.loads(Path(args.points).read_text())
            cal = Calibration.from_pairs([((p[0], p[1]), p[2]) for p in pts])
            print(f"Calibration error: {cal.reprojection_error_ft():.1f} ft")
        info = pipeline.video_info(args.video)
        cfg = pipeline.RunConfig(use_pose=not args.no_pose,
                                 max_frames=int(args.seconds * info["fps"]) if args.seconds else None)
        players, puck, meta = pipeline.run_video(
            args.video, calibration=cal, reference_frame=args.reference_frame, config=cfg,
            progress=lambda f, m: print(f"\r{m}", end="", flush=True))
        print()
    else:
        ap.error("give a video path or --demo")

    res = metrics.analyze(players, puck, meta)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for key in ("summary", "events", "review", "shape", "tracks", "puck"):
        if not res[key].empty:
            res[key].to_csv(out / f"{key}.csv", index=False)
    cols = [c for c in ["track_id", "label", "skate_score", "top_speed_mph", "avg_speed_mph", "distance_ft",
                        "in_position_pct", "shots", "hardest_shot_mph"] if c in res["summary"]]
    print(res["summary"][res["summary"]["seconds"] >= 1][cols].round(1).to_string(index=False))
    print(f"\nEvents:\n{res['events'].round(1).to_string(index=False) if not res['events'].empty else 'none'}")
    print(f"\nCSV files written to {out.resolve()}")


if __name__ == "__main__":
    main()
