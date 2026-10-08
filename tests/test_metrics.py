"""The analytics layer checked against a simulated game with known answers."""
import numpy as np
import pytest

from rinkiq.metrics import analyze, clean_track
from rinkiq.simulate import simulate_game


@pytest.fixture(scope="module")
def game():
    players, puck, meta = simulate_game()
    return analyze(players, puck, meta), meta


def test_all_passes_found(game):
    res, meta = game
    assert (res["events"]["type"] == "pass").sum() == meta["truth"]["passes"]


def test_shot_speeds_within_5_percent(game):
    res, meta = game
    shots = res["events"][res["events"]["type"] == "shot"]["speed_mph"].to_numpy()
    truth = np.array(meta["truth"]["shots"], dtype=float)
    assert len(shots) == len(truth)
    assert np.all(np.abs(shots - truth) / truth < 0.05)


def test_top_speeds_close_to_truth(game):
    res, meta = game
    s = res["summary"].set_index("track_id")
    for tid, true_mph in meta["truth"]["top_speed_mph"].items():
        assert abs(s.loc[tid, "top_speed_mph"] - true_mph) < 1.5, tid


def test_cheater_is_flagged(game):
    res, meta = game
    assert meta["truth"]["cheater"] in set(res["review"]["track_id"])


def test_attack_direction_from_goalies(game):
    res, _ = game
    assert res["attack"] == {"A": True, "B": False}


def test_backward_skating_only_for_defence(game):
    res, meta = game
    s = res["summary"].assign(name=lambda d: d["track_id"].map(meta["names"]))
    forwards = s[s["name"].str.contains("-(?:C|LW|RW)$")]
    defence = s[s["name"].str.contains("-(?:LD|RD)$")]
    assert forwards["backward_pct"].max() < 5
    assert defence["backward_pct"].mean() > 10


def test_clean_track_drops_teleports_and_splits_id_switches():
    t = np.arange(20) / 30
    x = np.linspace(0, 10, 20)
    x[5] += 40                  # one-frame glitch → dropped
    x[12:] += 100               # persistent jump → new segment
    seg = clean_track(t, x, np.zeros(20))
    assert seg[5] == -1
    assert seg[0] == seg[11] and seg[-1] > seg[0]
