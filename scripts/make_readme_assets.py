"""Regenerate the images in docs/assets used by the README.

    python scripts/make_readme_assets.py

Runs the full pipeline on the Hugging Face sample clip (cached in outputs/) and
the synthetic demo game, then draws matplotlib versions of the app's views.
"""
import pickle
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rinkiq import identity, metrics, models, pipeline, simulate  # noqa: E402
from rinkiq.draw import annotate  # noqa: E402
from rinkiq.render import render_broadcast, render_rink_replay  # noqa: E402
from rinkiq.rink import Calibration  # noqa: E402

ASSETS = ROOT / "docs" / "assets"
CACHE = ROOT / "outputs" / "sample_run.pkl"
CLIP = "Broadcast clip · neutral zone (MTL vs SJ)"
TEAM = {"A": "#2f6fd6", "B": "#d0393e", "REF": "#8a96a3"}
INK = "#0f1b2d"


def draw_rink(ax):
    ax.add_patch(FancyBboxPatch((0, 0), 200, 85, boxstyle="round,pad=0,rounding_size=28", fc="#f4f8fb",
                                ec="#9aa7b4", lw=1.5))
    for x, c, w in [(11, "#d0393e", 1), (189, "#d0393e", 1), (75, "#2f6fd6", 3), (125, "#2f6fd6", 3),
                    (100, "#d0393e", 3)]:
        ax.plot([x, x], [1.5, 83.5], color=c, lw=w, zorder=1)
    for cx in (31, 169):
        for cy in (20.5, 64.5):
            ax.add_patch(Circle((cx, cy), 15, fill=False, ec="#d0393e", lw=0.8))
            ax.add_patch(Circle((cx, cy), 1, color="#d0393e"))
    for cx in (80, 120):  # neutral-zone dots, 5 ft inside each blue line
        for cy in (20.5, 64.5):
            ax.add_patch(Circle((cx, cy), 1, color="#d0393e"))
    ax.add_patch(Circle((100, 42.5), 15, fill=False, ec="#2f6fd6", lw=0.8))
    ax.set_xlim(-3, 203)
    ax.set_ylim(-3, 88)
    ax.set_aspect("equal")
    ax.axis("off")


def rink_snapshot(ax, res, t, names=None, title=None):
    draw_rink(ax)
    tr = res["tracks"]
    near = tr[(tr["t"] - t).abs() < 0.02]
    for _, r in near.iterrows():
        c = TEAM.get(r["team"], "#888")
        ax.scatter(r["sx"], r["sy"], s=90, color=c, edgecolor="white", lw=1.2, zorder=3)
        if not np.isnan(r.get("face_x", np.nan)):
            ax.annotate("", xy=(r["sx"] + 8 * r["face_x"], r["sy"] + 8 * r["face_y"]), xytext=(r["sx"], r["sy"]),
                        arrowprops=dict(arrowstyle="->", color="#f2a900", lw=1.4), zorder=2)
        label = names.get(r["track_id"], "") if names else ""
        if label:
            ax.text(r["sx"], r["sy"] + 3.2, label, ha="center", fontsize=6.5, color=INK)
    pk = res["puck"]
    p = pk[(pk["t"] - t).abs() < 0.03]
    if len(p):
        ax.scatter(p["x"].iloc[0], p["y"].iloc[0], s=22, color="black", zorder=4)
    if title:
        ax.set_title(title, fontsize=10, color=INK, loc="left")


def _names(res, meta):
    """Sample clip = San Jose (team A, dark) vs Montréal (team B), 2025–26 rosters from the NHL API."""
    try:
        rosters = {"A": identity.nhl_roster("SJS", "20252026"), "B": identity.nhl_roster("MTL", "20252026")}
    except Exception:  # offline → numbers only
        rosters = {}
    team_of = res["tracks"].drop_duplicates("track_id").set_index("track_id")["team"].to_dict()
    return identity.assign_names(meta.get("numbers", {}), team_of, rosters)


def _gif(mp4: Path, gif: Path, width=640, fps=10, seconds=None):
    import imageio.v2 as imageio

    cap = cv2.VideoCapture(str(mp4))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(src_fps / fps))
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok or (seconds and i > seconds * src_fps):
            break
        if i % step == 0:
            f = cv2.resize(f, (width, int(f.shape[0] * width / f.shape[1])), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        i += 1
    imageio.mimsave(gif, frames, duration=1 / fps, loop=0)


def hero():
    if CACHE.exists():
        players, puck, meta = pickle.loads(CACHE.read_bytes())
        if "frames" not in meta:  # cached by an older version → re-run
            CACHE.unlink()
            return hero()
    else:
        clip = str(models.download_sample_clip(CLIP))
        cal = Calibration.from_pairs([((x, y), n) for x, y, n in models.SAMPLE_CALIBRATION[CLIP]])
        players, puck, meta = pipeline.run_video(clip, calibration=cal,
                                                 progress=lambda f, m: print(f"\r{m}", end="", flush=True))
        CACHE.parent.mkdir(exist_ok=True)
        CACHE.write_bytes(pickle.dumps((players, puck, meta)))
    res = metrics.analyze(players, puck, meta)
    names = _names(res, meta)
    clip = str(models.download_sample_clip(CLIP))
    tr = res["tracks"]
    fno = int(sorted(tr["frame"].unique())[len(tr["frame"].unique()) // 2])
    rows = tr[tr["frame"] == fno]
    img = annotate(pipeline.read_frame(clip, fno), rows, names=names)
    out = ROOT / "outputs" / "broadcast.mp4"
    render_broadcast(clip, res, puck, meta, str(out), names, start_s=2.0, end_s=8.0, out_width=960)
    _gif(out, ASSETS / "broadcast.gif", width=560, fps=8, seconds=5)
    print("names:", names)
    cv2.imwrite(str(ASSETS / "hero_frame.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    fig, ax = plt.subplots(figsize=(6, 2.9), dpi=170)
    rink_snapshot(ax, res, float(rows["t"].iloc[0]), {i: identity.label(i, names) for i in rows["track_id"]})
    fig.tight_layout()
    fig.savefig(ASSETS / "hero_rink.png", transparent=False, facecolor="white")
    plt.close(fig)
    return res


def demo_figures():
    players, puck, meta = simulate.simulate_game()
    res = metrics.analyze(players, puck, meta)
    names = meta["names"]
    out = ROOT / "outputs" / "replay.mp4"
    out.parent.mkdir(exist_ok=True)
    render_rink_replay(res, meta, str(out), names)
    import imageio.v2 as imageio  # 8–12 s: zone entry → one-timer → rebound

    cap, frames, i = cv2.VideoCapture(str(out)), [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if 240 <= i <= 360 and i % 3 == 0:
            frames.append(cv2.cvtColor(cv2.resize(f, (720, 338), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
        i += 1
    imageio.mimsave(ASSETS / "replay.gif", frames, duration=0.1, loop=0)
    s = res["summary"]
    s = s[s["team"].isin(["A", "B"]) & ~s["label"].str.startswith("goalie")]

    # Speed traces
    fig, ax = plt.subplots(figsize=(7, 2.8), dpi=170)
    for tid in s.sort_values("top_speed_mph", ascending=False)["track_id"].head(3):
        g = res["tracks"][res["tracks"]["track_id"] == tid]
        ax.plot(g["t"], g["speed_mph"], lw=1.6, label=names[tid],
                color=TEAM[g["team"].iloc[0]], alpha=0.95 if names[tid].startswith("A") else 0.75)
    ax.axhline(15, ls=":", color="#8a96a3")
    ax.text(0.2, 15.6, "high intensity", fontsize=7, color="#8a96a3")
    ax.set_xlabel("seconds")
    ax.set_ylabel("mph")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(ASSETS / "speed.png", facecolor="white")
    plt.close(fig)

    # Shot moment + out-of-position moment
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.7), dpi=170)
    rink_snapshot(axes[0], res, 10.6, names, "t = 10.6 s · A-LD one-timer (87 mph)")
    rv = res["review"]
    cheat = rv[rv["track_id"] == meta["truth"]["cheater"]].iloc[0]
    rink_snapshot(axes[1], res, float(cheat["start_s"]), names, f"t = {cheat['start_s']:.1f} s · B-LW flagged: "
                  f"{cheat['reason']}")
    g = res["tracks"]
    h = g[(g["track_id"] == cheat["track_id"]) & ((g["t"] - cheat["start_s"]).abs() < 0.02)]
    axes[1].scatter(h["sx"], h["sy"], s=600, facecolor="none", edgecolor="#f2a900", lw=2, zorder=5)
    fig.tight_layout()
    fig.savefig(ASSETS / "moments.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    demo_figures()
    hero()
    print("wrote", sorted(p.name for p in ASSETS.iterdir()))
