"""RinkIQ worker: takes uploaded videos off the Supabase queue and runs the full pipeline on them.

    python worker/worker.py            # run forever (Ctrl-C to stop; a job in flight is put back in the queue)
    python worker/worker.py --once     # handle at most one job, then exit (handy for testing)

Setup: copy worker/.env.example to worker/.env and fill in SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.
The service-role key is a SECRET (it bypasses all access rules). It lives only on the machine running this
worker and in Vercel's server-side environment, never in the browser and never in git.

Talks to Supabase over its REST API with plain `requests`: no extra dependencies to install.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rinkiq.job import JobCanceled, parse_options, run_analysis  # noqa: E402

VIDEO_BUCKET, RESULT_BUCKET = "videos", "results"
HEARTBEAT_EVERY_S = 30
PROGRESS_EVERY_S = 1.5
CONTENT_TYPES = {".mp4": "video/mp4", ".html": "text/html; charset=utf-8", ".zip": "application/zip",
                 ".json": "application/json"}


def load_env(path: Path) -> None:
    """Minimal .env reader (KEY=value per line). Real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class Supa:
    """The few Supabase calls the worker needs: Postgres via PostgREST, files via Storage."""

    def __init__(self, url: str, key: str, timeout: float = 60.0):
        self.url = url.rstrip("/")
        self.timeout = timeout
        # Legacy keys are JWTs and go in both headers. The newer `sb_secret_…` keys are not JWTs and
        # must only be sent as `apikey`.
        self.headers = {"apikey": key}
        if key.count(".") == 2:
            self.headers["Authorization"] = f"Bearer {key}"

    def _req(self, method: str, path: str, **kw) -> requests.Response:
        headers = {**self.headers, **kw.pop("headers", {})}
        r = requests.request(method, f"{self.url}{path}", headers=headers, timeout=kw.pop("timeout", self.timeout), **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"Supabase {method} {path.split('?')[0]} → {r.status_code}: {r.text[:300]}")
        return r

    # --- Postgres
    def claim_job(self, worker: str, stale_s: int = 300) -> dict | None:
        rows = self._req("POST", "/rest/v1/rpc/claim_job", json={"p_worker": worker, "p_stale_seconds": stale_s}).json()
        return rows[0] if rows else None

    def patch_job(self, job_id: str, fields: dict, *, where: dict | None = None) -> list[dict]:
        """Update a job. `where` adds filters (e.g. only while still running). Returns the updated rows."""
        q = "&".join([f"id=eq.{job_id}"] + [f"{k}=eq.{v}" for k, v in (where or {}).items()])
        return self._req("PATCH", f"/rest/v1/jobs?{q}", json=fields,
                         headers={"Prefer": "return=representation", "Content-Type": "application/json"}).json()

    def old_jobs(self, older_than_iso: str) -> list[dict]:
        return self._req("GET", f"/rest/v1/jobs?select=id,video_path&created_at=lt.{older_than_iso}&limit=50").json()

    def delete_job(self, job_id: str) -> None:
        self._req("DELETE", f"/rest/v1/jobs?id=eq.{job_id}")

    # --- Storage
    def download(self, bucket: str, path: str, dest: Path) -> int:
        with self._req("GET", f"/storage/v1/object/{bucket}/{path}", stream=True, timeout=600) as r:
            n = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
                    n += len(chunk)
        return n

    def upload(self, bucket: str, path: str, file: Path) -> None:
        ctype = CONTENT_TYPES.get(file.suffix.lower()) or mimetypes.guess_type(file.name)[0] or "application/octet-stream"
        with open(file, "rb") as f:
            self._req("POST", f"/storage/v1/object/{bucket}/{path}", data=f, timeout=900,
                      headers={"Content-Type": ctype, "x-upsert": "true", "cache-control": "max-age=3600"})

    def remove(self, bucket: str, paths: list[str]) -> None:
        if paths:
            self._req("DELETE", f"/storage/v1/object/{bucket}", json={"prefixes": paths})


# ------------------------------------------------------------------ one job
def friendly_error(e: Exception) -> str:
    if isinstance(e, ValueError):
        return str(e)[:500]
    return f"Unexpected error: {type(e).__name__}: {e}"[:500]


def process(supa: Supa, job: dict, worker: str, max_seconds: float | None) -> None:
    job_id = job["id"]
    options = parse_options(job.get("options"), max_seconds)
    work = Path(tempfile.mkdtemp(prefix=f"rinkiq_{job_id[:8]}_"))
    stop_beat = threading.Event()
    last = {"t": 0.0, "msg": ""}

    def beat():  # keeps the job "alive" during long silent stretches (model downloads, encoding)
        while not stop_beat.wait(HEARTBEAT_EVERY_S):
            try:
                supa.patch_job(job_id, {"heartbeat_at": now_iso()}, where={"status": "running", "worker": worker})
            except Exception:  # noqa: BLE001
                pass

    def progress(frac: float, msg: str) -> None:
        t = time.time()
        if t - last["t"] < PROGRESS_EVERY_S and msg == last["msg"]:
            return
        last.update(t=t, msg=msg)
        rows = supa.patch_job(job_id, {"progress": round(min(max(frac, 0.0), 0.99), 3), "message": msg[:200],
                                       "heartbeat_at": now_iso()}, where={"status": "running", "worker": worker})
        if not rows:  # canceled from the website (or taken over by another worker)
            raise JobCanceled()

    threading.Thread(target=beat, daemon=True).start()
    try:
        video = work / Path(job["video_path"]).name
        progress(0.0, "Downloading the video…")
        supa.download(VIDEO_BUCKET, job["video_path"], video)
        analysis = run_analysis(video, options, work / "out", progress)
        for name, path in analysis.files.items():
            supa.upload(RESULT_BUCKET, f"{job_id}/{path.name}", path)
        supa.patch_job(job_id, {"status": "done", "progress": 1, "message": "Done", "result": analysis.result,
                                "finished_at": now_iso()}, where={"status": "running", "worker": worker})
        log(f"job {job_id[:8]} done ({job.get('filename')})")
    except JobCanceled:
        log(f"job {job_id[:8]} canceled")
    except Exception as e:  # noqa: BLE001 — one bad video must never stop the worker
        traceback.print_exc()
        try:
            supa.patch_job(job_id, {"status": "failed", "error": friendly_error(e), "message": "Failed",
                                    "finished_at": now_iso()}, where={"worker": worker})
        except Exception:  # noqa: BLE001
            pass
    finally:
        stop_beat.set()
        shutil.rmtree(work, ignore_errors=True)


def release(supa: Supa, job_id: str, worker: str) -> None:
    """Ctrl-C while a job was running: put it back so another run picks it up."""
    try:
        supa.patch_job(job_id, {"status": "queued", "message": "Worker stopped, waiting to retry"},
                       where={"status": "running", "worker": worker})
    except Exception:  # noqa: BLE001
        pass


# ------------------------------------------------------------------ housekeeping
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def purge_old(supa: Supa, days: int) -> int:
    """Delete jobs (and their files) older than `days`, so storage doesn't fill up."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    n = 0
    for j in supa.old_jobs(cutoff):
        try:
            supa.remove(VIDEO_BUCKET, [j["video_path"]])
            supa.remove(RESULT_BUCKET, [f"{j['id']}/{f}" for f in ("broadcast.mp4", "report.html", "stats.zip")])
            supa.delete_job(j["id"])
            n += 1
        except Exception as e:  # noqa: BLE001
            log(f"could not purge {j['id'][:8]}: {e}")
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true", help="handle at most one job, then exit")
    ap.add_argument("--poll", type=float, default=3.0, help="seconds between checks for new work")
    ap.add_argument("--purge-days", type=int, default=14, help="delete jobs older than this (0 = never)")
    ap.add_argument("--env", default=str(Path(__file__).with_name(".env")))
    args = ap.parse_args(argv)

    load_env(Path(args.env))
    url, key = os.environ.get("SUPABASE_URL", ""), os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        print("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (see worker/.env.example).", file=sys.stderr)
        return 2
    max_seconds = float(os.environ.get("RINKIQ_MAX_SECONDS", "0")) or None
    worker = f"{socket.gethostname()}-{os.getpid()}"
    supa = Supa(url, key)
    log(f"worker {worker} connected to {url}" + (f" (cap {max_seconds:.0f}s per video)" if max_seconds else ""))

    last_purge = 0.0
    try:
        while True:
            if args.purge_days and time.time() - last_purge > 3600:
                last_purge = time.time()
                try:
                    n = purge_old(supa, args.purge_days)
                    if n:
                        log(f"purged {n} old job(s)")
                except Exception as e:  # noqa: BLE001
                    log(f"purge skipped: {e}")
            try:
                job = supa.claim_job(worker)
            except Exception as e:  # noqa: BLE001 — network blip: wait and retry
                log(f"could not reach Supabase: {e}")
                job = None
                time.sleep(10)
            if job:
                log(f"job {job['id'][:8]}: {job.get('filename')} (attempt {job.get('attempts')})")
                try:
                    process(supa, job, worker, max_seconds)
                except KeyboardInterrupt:
                    release(supa, job["id"], worker)
                    raise
                if args.once:
                    return 0
            elif args.once:
                log("no queued jobs")
                return 0
            else:
                time.sleep(args.poll)
    except KeyboardInterrupt:
        log("stopped")
        return 0


if __name__ == "__main__":
    sys.exit(main())
