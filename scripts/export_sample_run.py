"""Save the sample clip's analysis as small CSV/JSON files so the app can load it instantly.

    python scripts/make_readme_assets.py   # (runs + caches the analysis in outputs/sample_run.pkl)
    python scripts/export_sample_run.py    # → samples/sample_run/
"""
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "samples" / "sample_run"


def main():
    players, puck, meta = pickle.loads((ROOT / "outputs" / "sample_run.pkl").read_bytes())
    OUT.mkdir(parents=True, exist_ok=True)
    players.round(3).to_csv(OUT / "players.csv", index=False)
    puck.round(3).to_csv(OUT / "puck.csv", index=False)
    meta["frames"].round(8).to_csv(OUT / "frames.csv", index=False)
    m = {k: v for k, v in meta.items() if k != "frames"}
    m["numbers"] = {str(k): list(v) for k, v in m.get("numbers", {}).items()}
    m["merged_tracks"] = {str(k): int(v) for k, v in m.get("merged_tracks", {}).items()}
    (OUT / "meta.json").write_text(json.dumps(m, indent=1, default=float))
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
