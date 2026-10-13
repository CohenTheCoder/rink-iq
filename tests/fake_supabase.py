"""A tiny stand-in for the parts of Supabase that RinkIQ's website and worker use, for tests.

    python tests/fake_supabase.py --port 54321     # then point SUPABASE_URL at http://127.0.0.1:54321

It speaks the same REST protocol as the real thing (PostgREST + Storage), following the supabase-js source:
    /rest/v1/jobs            GET / POST / PATCH / DELETE, with eq./in./lt. filters, select, order, limit,
                             HEAD + Prefer: count=exact, Accept: application/vnd.pgrst.object+json
    /rest/v1/rpc/claim_job   the same logic as the SQL function
    /storage/v1/...          signed upload URLs, signed download URLs, object get/post/delete, list, buckets

It is NOT a database: it proves that our code makes the calls Supabase expects, not that the SQL in
supabase/migrations behaves. (The SQL is checked for syntax only; the first real run happens in your project.)
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import uuid
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

BUCKETS = {"videos", "results"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class State:
    def __init__(self):
        self.lock = threading.RLock()
        self.jobs: dict[str, dict] = {}
        self.objects: dict[tuple[str, str], bytes] = {}
        self.upload_tokens: dict[str, tuple[str, str]] = {}
        self.sign_tokens: dict[str, tuple[str, str]] = {}
        self.log: list[str] = []

    # --- jobs
    def insert(self, row: dict) -> dict:
        full = {
            "id": str(uuid.uuid4()), "created_at": now(), "updated_at": now(), "status": "uploading", "filename": "",
            "video_path": "", "size_bytes": None, "options": {}, "progress": 0, "message": None, "result": None,
            "error": None, "worker": None, "attempts": 0, "started_at": None, "finished_at": None, "heartbeat_at": None,
        }
        full.update(row)
        if full["status"] not in ("uploading", "queued", "running", "done", "failed", "canceled"):
            raise ValueError("violates check constraint")
        self.jobs[full["id"]] = full
        return full

    def claim(self, worker: str, stale: int) -> list[dict]:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=stale)
        def stale_run(j):
            return j["status"] == "running" and j["heartbeat_at"] and datetime.fromisoformat(j["heartbeat_at"]) < cutoff
        for j in self.jobs.values():
            if stale_run(j) and j["attempts"] >= 3:
                j.update(status="failed", error="The worker stopped repeatedly while processing this video.", finished_at=now())
        waiting = sorted((j for j in self.jobs.values() if j["status"] == "queued" or stale_run(j)), key=lambda j: j["created_at"])
        if not waiting:
            return []
        j = waiting[0]
        j.update(status="running", worker=worker, attempts=j["attempts"] + 1, started_at=j["started_at"] or now(),
                 heartbeat_at=now(), message="Picked up by a worker", error=None, updated_at=now())
        return [dict(j)]


def _filters(query: dict[str, list[str]]):
    out = []
    for k, vals in query.items():
        if k in ("select", "order", "limit", "offset", "columns"):
            continue
        for v in vals:
            op, _, arg = v.partition(".")
            out.append((k, op, arg))
    return out


def _match(row: dict, flt) -> bool:
    for col, op, arg in flt:
        val = row.get(col)
        if op == "eq" and str(val) != arg:
            return False
        if op == "in" and str(val) not in [x.strip('"') for x in arg.strip("()").split(",")]:
            return False
        if op == "lt" and not (val is not None and str(val) < arg):
            return False
    return True


class Handler(BaseHTTPRequestHandler):
    state: State
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet
        pass

    # ---- plumbing
    def _send(self, code: int, body=None, headers: dict | None = None, raw: bytes | None = None, ctype="application/json"):
        data = raw if raw is not None else (b"" if body is None else json.dumps(body).encode())
        self.send_response(code)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _json(self):
        b = self._body()
        return json.loads(b) if b else None

    def _need_key(self) -> bool:
        if not (self.headers.get("apikey") or "token=" in self.path):
            self._send(401, {"message": "No API key found in request"})
            return False
        return True

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "authorization, apikey, content-type, x-upsert, cache-control, x-client-info, prefer, accept, range")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self): self._route()
    def do_HEAD(self): self._route()
    def do_POST(self): self._route()
    def do_PUT(self): self._route()
    def do_PATCH(self): self._route()
    def do_DELETE(self): self._route()

    def _route(self):
        u = urlparse(self.path)
        path, q = u.path, parse_qs(u.query)
        S = self.state
        S.log.append(f"{self.command} {path}")
        with S.lock:
            if path.startswith("/rest/v1/"):
                return self._rest(path[len("/rest/v1/"):], q)
            if path.startswith("/storage/v1/"):
                return self._storage(path[len("/storage/v1"):], q)
        self._send(404, {"message": "not found"})

    # ---- PostgREST
    def _rest(self, rel: str, q):
        S = self.state
        if not self._need_key():
            return
        if rel.startswith("rpc/claim_job") and self.command == "POST":
            body = self._json() or {}
            return self._send(200, S.claim(body.get("p_worker", "?"), int(body.get("p_stale_seconds", 300))))
        if rel != "jobs":
            return self._send(404, {"message": "unknown table"})
        flt = _filters(q)
        prefer = self.headers.get("Prefer", "")
        want_obj = "vnd.pgrst.object" in (self.headers.get("Accept") or "")

        if self.command in ("GET", "HEAD"):
            rows = [r for r in S.jobs.values() if _match(r, flt)]
            order = (q.get("order") or [""])[0]
            if order:
                col, _, d = order.partition(".")
                rows.sort(key=lambda r: str(r.get(col)), reverse=(d == "desc"))
            if q.get("limit"):
                rows = rows[: int(q["limit"][0])]
            cols = (q.get("select") or ["*"])[0]
            if cols != "*":
                keep = [c.strip() for c in cols.split(",")]
                rows = [{k: r.get(k) for k in keep} for r in rows]
            total = len([r for r in S.jobs.values() if _match(r, flt)])
            headers = {"Content-Range": f"0-{max(total - 1, 0)}/{total}"} if "count=exact" in prefer else {}
            if self.command == "HEAD":
                return self._send(200, None, headers)
            return self._send(200, rows, headers)

        if self.command == "POST":
            body = self._json()
            try:
                row = S.insert(body)
            except ValueError as e:
                return self._send(400, {"code": "23514", "message": str(e)})
            return self._send(201, [row] if "return=representation" in prefer else None)

        hits = [r for r in S.jobs.values() if _match(r, flt)]
        if self.command == "PATCH":
            patch = self._json() or {}
            if patch.get("status") not in (None, "uploading", "queued", "running", "done", "failed", "canceled"):
                return self._send(400, {"code": "23514", "message": "violates check constraint"})
            for r in hits:
                r.update(patch)
                r["updated_at"] = now()
            out = [dict(r) for r in hits]
            if want_obj and len(out) != 1:
                return self._send(406, {"code": "PGRST116", "details": f"The result contains {len(out)} rows",
                                        "hint": None, "message": "JSON object requested, multiple (or no) rows returned"})
            if "return=representation" not in prefer:
                return self._send(204)
            return self._send(200, out[0] if want_obj else out)
        if self.command == "DELETE":
            for r in hits:
                S.jobs.pop(r["id"], None)
            return self._send(204)
        self._send(405, {"message": "method not allowed"})

    # ---- Storage
    def _storage(self, rel: str, q):
        S = self.state
        if rel == "/bucket" and self.command == "GET":
            if not self._need_key():
                return
            return self._send(200, [{"id": b, "name": b, "public": False} for b in sorted(BUCKETS)])
        m = re.match(r"^/object/upload/sign/([^/]+)/(.+)$", rel)
        if m and self.command == "POST":
            if not self._need_key():
                return
            bucket, path = m.group(1), unquote(m.group(2))
            token = uuid.uuid4().hex
            S.upload_tokens[token] = (bucket, path)
            return self._send(200, {"url": f"/object/upload/sign/{bucket}/{m.group(2)}?token={token}"})
        if m and self.command == "PUT":
            token = (q.get("token") or [""])[0]
            if S.upload_tokens.get(token) != (m.group(1), unquote(m.group(2))):
                return self._send(403, {"message": "invalid token"})
            if not self.headers.get("apikey"):  # the real gateway insists on this
                return self._send(401, {"message": "No API key found in request"})
            S.objects[(m.group(1), unquote(m.group(2)))] = self._body()
            return self._send(200, {"Key": f"{m.group(1)}/{unquote(m.group(2))}"})
        m = re.match(r"^/object/sign/([^/]+)/(.+)$", rel)
        if m and self.command == "POST":
            if not self._need_key():
                return
            bucket, path = m.group(1), unquote(m.group(2))
            if (bucket, path) not in S.objects:
                return self._send(400, {"error": "Object not found", "message": "Object not found", "statusCode": "404"})
            token = uuid.uuid4().hex
            S.sign_tokens[token] = (bucket, path)
            return self._send(200, {"signedURL": f"/object/sign/{bucket}/{m.group(2)}?token={token}"})
        if m and self.command in ("GET", "HEAD"):
            token = (q.get("token") or [""])[0]
            if S.sign_tokens.get(token) != (m.group(1), unquote(m.group(2))):
                return self._send(403, {"message": "invalid token"})
            data = S.objects[(m.group(1), unquote(m.group(2)))]
            ctype = "video/mp4" if path_endswith(m.group(2), ".mp4") else "application/octet-stream"
            return self._send(200, raw=data, ctype=ctype)
        m = re.match(r"^/object/list/([^/]+)$", rel)
        if m and self.command == "POST":
            if not self._need_key():
                return
            prefix = (self._json() or {}).get("prefix", "")
            pre = prefix.rstrip("/") + "/" if prefix else ""
            out = [{"name": p[len(pre):], "metadata": {"size": len(b)}} for (bk, p), b in S.objects.items()
                   if bk == m.group(1) and p.startswith(pre) and "/" not in p[len(pre):]]
            return self._send(200, out)
        m = re.match(r"^/object/([^/]+)$", rel)
        if m and self.command == "DELETE":
            if not self._need_key():
                return
            for p in (self._json() or {}).get("prefixes", []):
                S.objects.pop((m.group(1), p), None)
            return self._send(200, [])
        m = re.match(r"^/object/([^/]+)/(.+)$", rel)
        if m and m.group(1) in BUCKETS:
            if not self._need_key():
                return
            bucket, path = m.group(1), unquote(m.group(2))
            if self.command == "GET":
                if (bucket, path) not in S.objects:
                    return self._send(404, {"error": "not_found", "message": "Object not found"})
                return self._send(200, raw=S.objects[(bucket, path)], ctype="application/octet-stream")
            if self.command == "POST":
                S.objects[(bucket, path)] = self._body()
                return self._send(200, {"Key": f"{bucket}/{path}", "Id": uuid.uuid4().hex})
        self._send(404, {"message": "unhandled storage route", "path": rel})


def path_endswith(p: str, ext: str) -> bool:
    return p.lower().split("?")[0].endswith(ext)


class FakeSupabase:
    """Start/stop in-process: `with FakeSupabase() as fake: fake.url, fake.state`."""

    def __init__(self, port: int = 0):
        self.state = State()
        handler = type("H", (Handler,), {"state": self.state})
        self.server = ThreadingHTTPServer(("127.0.0.1", port), handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self._t = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self.server.shutdown()
        self.server.server_close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=54321)
    a = ap.parse_args()
    with FakeSupabase(a.port) as fake:
        print(f"fake Supabase on {fake.url}", flush=True)
        threading.Event().wait()
