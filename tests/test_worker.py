"""The online worker, end to end against a stand-in Supabase (the analysis itself is stubbed: it has its own tests)."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from fake_supabase import FakeSupabase  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rinkiq_worker", ROOT / "worker" / "worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)

from rinkiq.job import Analysis, JobCanceled  # noqa: E402


@pytest.fixture()
def fake(monkeypatch, tmp_path):
    with FakeSupabase() as f:
        monkeypatch.setenv("SUPABASE_URL", f.url)
        monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "a.b.c")
        monkeypatch.delenv("RINKIQ_MAX_SECONDS", raising=False)
        yield f


def queue_job(fake, **extra):
    """What the website does: a job row plus an uploaded video."""
    job = fake.state.insert({"filename": "game.mp4", "video_path": "J1/game.mp4", "status": "queued",
                             "options": {"seconds": 5, "team_a": "SJS"}, **extra})
    fake.state.objects[("videos", job["video_path"])] = b"fake video bytes"
    return job


def run_once():
    return worker.main(["--once", "--purge-days", "0", "--env", "/nonexistent"])


def stub_ok(video, options, workdir, progress):
    progress(0.2, "tracking")
    progress(0.8, "rendering")
    out = Path(workdir)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("broadcast.mp4", b"MP4"), ("report.html", b"<html>"), ("stats.zip", b"ZIP")):
        (out / name).write_bytes(data)
    return Analysis({"tiles": {"skaters": 3}, "files": {"video": "broadcast.mp4"}},
                    {"video": out / "broadcast.mp4", "report": out / "report.html", "stats": out / "stats.zip"})


def test_happy_path_uploads_results_and_marks_done(fake, monkeypatch):
    job = queue_job(fake)
    seen = {}

    def stub(video, options, workdir, progress):
        seen["bytes"] = Path(video).read_bytes()
        seen["seconds"], seen["team_a"] = options.seconds, options.team_a
        return stub_ok(video, options, workdir, progress)

    monkeypatch.setattr(worker, "run_analysis", stub)
    assert run_once() == 0
    row = fake.state.jobs[job["id"]]
    assert row["status"] == "done" and row["progress"] == 1 and row["result"]["tiles"]["skaters"] == 3
    assert seen == {"bytes": b"fake video bytes", "seconds": 5.0, "team_a": "SJS"}  # downloaded + options parsed
    assert fake.state.objects[("results", f"{job['id']}/broadcast.mp4")] == b"MP4"
    assert fake.state.objects[("results", f"{job['id']}/stats.zip")] == b"ZIP"
    assert row["worker"] and row["attempts"] == 1 and row["finished_at"]


def test_failure_is_recorded_and_worker_survives(fake, monkeypatch):
    job = queue_job(fake)
    monkeypatch.setattr(worker, "run_analysis", lambda *a, **k: (_ for _ in ()).throw(ValueError("No skaters were detected.")))
    assert run_once() == 0
    row = fake.state.jobs[job["id"]]
    assert row["status"] == "failed" and row["error"] == "No skaters were detected."


def test_unexpected_error_message_is_labelled(fake, monkeypatch):
    job = queue_job(fake)
    monkeypatch.setattr(worker, "run_analysis", lambda *a, **k: 1 / 0)
    run_once()
    assert fake.state.jobs[job["id"]]["error"].startswith("Unexpected error: ZeroDivisionError")


def test_cancel_from_the_website_stops_the_job(fake, monkeypatch):
    job = queue_job(fake)
    monkeypatch.setattr(worker, "PROGRESS_EVERY_S", 0)  # don't throttle: every update checks for a cancel

    def stub(video, options, workdir, progress):
        progress(0.1, "tracking")
        fake.state.jobs[job["id"]]["status"] = "canceled"   # the user pressed Cancel
        progress(0.2, "tracking more")                       # next update notices
        raise AssertionError("worker should have stopped")

    monkeypatch.setattr(worker, "run_analysis", stub)
    run_once()
    assert fake.state.jobs[job["id"]]["status"] == "canceled"
    assert not any(k[0] == "results" for k in fake.state.objects)


def test_no_jobs_exits_cleanly(fake):
    assert run_once() == 0


def test_two_workers_never_take_the_same_job(fake):
    queue_job(fake)
    a = worker.Supa(fake.url, "k").claim_job("w1")
    b = worker.Supa(fake.url, "k").claim_job("w2")
    assert a is not None and b is None


def test_stale_running_job_is_picked_up_again(fake):
    job = queue_job(fake)
    fake.state.jobs[job["id"]].update(status="running", worker="dead", attempts=1,
                                      heartbeat_at="2020-01-01T00:00:00+00:00")
    got = worker.Supa(fake.url, "k").claim_job("w2", stale_s=60)
    assert got and got["worker"] == "w2" and got["attempts"] == 2


def test_job_that_keeps_killing_workers_is_failed(fake):
    job = queue_job(fake)
    fake.state.jobs[job["id"]].update(status="running", attempts=3, heartbeat_at="2020-01-01T00:00:00+00:00")
    assert worker.Supa(fake.url, "k").claim_job("w2", stale_s=60) is None
    assert fake.state.jobs[job["id"]]["status"] == "failed"


def test_purge_removes_old_jobs_and_files(fake):
    old = queue_job(fake)
    fake.state.jobs[old["id"]]["created_at"] = "2020-01-01T00:00:00+00:00"
    fake.state.objects[("results", f"{old['id']}/report.html")] = b"x"
    fresh = fake.state.insert({"filename": "new.mp4", "video_path": "J2/new.mp4", "status": "done"})
    assert worker.purge_old(worker.Supa(fake.url, "k"), days=14) == 1
    assert old["id"] not in fake.state.jobs and fresh["id"] in fake.state.jobs
    assert ("videos", "J1/game.mp4") not in fake.state.objects
    assert ("results", f"{old['id']}/report.html") not in fake.state.objects


def test_new_style_secret_key_is_sent_only_as_apikey():
    s = worker.Supa("http://x", "sb_secret_abc")
    assert s.headers == {"apikey": "sb_secret_abc"}
    assert "Authorization" in worker.Supa("http://x", "aaa.bbb.ccc").headers


def test_env_file_parsing(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text('# c\nSUPABASE_URL="http://h"\nFOO = bar \n')
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    worker.load_env(p)
    assert __import__("os").environ["SUPABASE_URL"] == "http://h"


def test_options_are_bounded():
    from rinkiq.job import parse_options
    o = parse_options({"seconds": 99999, "start_s": -5, "quality": "ultra", "team_a": "sjs'; drop", "season": "x" * 99},
                      max_seconds=60)
    assert o.seconds == 3 * 3600 and o.start_s == 0 and o.quality == "fast" and o.team_a == "SJS" and len(o.season) == 8
    assert o.max_seconds == 60
