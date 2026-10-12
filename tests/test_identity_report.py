from rinkiq.identity import merge_tracks, parse_roster, short, vote_numbers
from rinkiq.metrics import analyze
from rinkiq.report import html_report, report_cards, team_comparison
from rinkiq.simulate import simulate_game


def test_vote_prefers_consistent_two_digit_number():
    reads = [(1, "63", 0.8), (1, "3", 0.85), (1, "63", 0.7), (2, "9", 0.9)]
    out = vote_numbers(reads)
    assert out[1][0] == "63"
    assert 2 not in out  # a single reading isn't enough


def test_parse_roster_formats():
    r = parse_roster("63 Zack Ostapchuk\n#9, Dmitry Orlov\n30 - Yaroslav Askarov\nnot a line")
    assert r == {"63": "Zack Ostapchuk", "9": "Dmitry Orlov", "30": "Yaroslav Askarov"}
    assert short("Zack Ostapchuk") == "Z. Ostapchuk"


def test_merge_tracks_joins_non_overlapping_pieces():
    players, _, _ = simulate_game(seconds=4)
    p = players.copy()
    # split player 1 into two pieces with different ids
    late = (p["track_id"] == 1) & (p["t"] > 2)
    p.loc[late, "track_id"] = 99
    merged, remap = merge_tracks(p, {1: ("63", 0.9), 99: ("63", 0.8)})
    assert remap == {99: 1}
    assert 99 not in set(merged["track_id"])


def test_report_cards_and_html():
    players, puck, meta = simulate_game()
    res = analyze(players, puck, meta)
    cards = report_cards(res["summary"], res["review"], meta["names"])
    assert len(cards) == 10
    assert all(c.grade in {"A+", "A", "B+", "B", "C+", "C", "D"} for c in cards)
    assert any(c.strengths for c in cards) and any(c.weaknesses for c in cards)
    cheater = next(c for c in cards if c.track_id == meta["truth"]["cheater"])
    assert any("out-of-position" in w or "Out of position" in w for w in cheater.weaknesses)
    page = html_report(cards, team_comparison(res["summary"], res["events"]))
    assert "Three stars" in page and cards[0].name in page
