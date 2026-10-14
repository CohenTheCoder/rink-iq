# 🌐 RinkIQ online: Vercel + Supabase

Put RinkIQ on the web so anyone you give the link (and password) to can upload hockey film and get the analysis back.

```
 browser ──upload──▶ Supabase Storage            (private bucket `videos`)
    │                      ▲
    │                      │ download / upload results
    ▼                      │
 Vercel (Next.js site) ──▶ Supabase Postgres ◀── worker (Python, on a computer you control)
   upload · progress · results     job queue          runs the RinkIQ pipeline on each video
```

**Why a separate worker?** The analysis uses PyTorch and ~1 GB of models, far beyond what Vercel's serverless
functions can run. So Vercel only hosts the website; the heavy lifting happens in `worker/worker.py` on any
computer with RinkIQ's Python environment (your laptop, a desktop, a cloud GPU box). Jobs wait in the queue while
the worker is off and get picked up when it starts.

| Piece | Where it lives | What it does |
|---|---|---|
| Website | `web/` → Vercel | Upload with a progress bar, live job page, results (stars, box score, play log, video), **public interactive demo at `/demo`** |
| Database + files | Supabase | `jobs` table (the queue), private `videos` and `results` buckets |
| Worker | `worker/worker.py` | Claims a job, downloads the video, runs `rinkiq.job.run_analysis`, uploads results |

## Set it up (about 10 minutes)

### 1 · Create the Supabase project (free)
1. Go to <https://supabase.com> → **New project**. Pick any name and region, and save the database password it asks for.
2. Open **SQL Editor → New query**, paste the whole of [`supabase/migrations/001_init.sql`](../supabase/migrations/001_init.sql), press **Run**.
   It creates the `jobs` table, the queue function and the two private buckets. It's safe to run again.
3. Open **Project Settings → API**. You need three values: the **Project URL**, the **anon / publishable key** (public)
   and the **service_role / secret key** (**secret**: it bypasses all access rules).

### 2 · Connect the website and the worker
```bash
./scripts/setup_online.sh
```
It asks for those three values (hidden as you type) plus a site password (or generates one), checks the connection,
writes `worker/.env` and `web/.env.local` (git-ignored), sets the variables on Vercel and deploys the site.
Use `--no-vercel` to skip the Vercel part. Prefer the dashboard? Add the same four variables under
Vercel → Project → Settings → Environment Variables (names are in [`web/.env.example`](../web/.env.example)).

### 3 · Run the worker
```bash
pip install -r requirements.txt          # once
python worker/worker.py                  # leave it running
```
Keep the computer awake while it runs (`caffeinate -i python worker/worker.py` on a Mac). `Ctrl-C` is safe: a job in
progress goes back in the queue. Run several workers on several machines and each takes different jobs.

### 4 · Use it
Open your Vercel URL, enter the password (any username), drop in a video and press **Analyze**. The page shows live
progress, then the results. The home page also has a **Setup check** banner that tells you exactly what's missing if
something isn't right.

## Good to know
* **Size limit: 50 MB per video** on Supabase's free plan. Trim or compress longer film first (for example
  `ffmpeg -i game.mp4 -t 120 -vf scale=1280:-2 -an clip.mp4`). On a paid plan, raise the `videos` bucket limit in
  the SQL file and the global upload limit (Storage → Settings), and set `NEXT_PUBLIC_MAX_UPLOAD_MB`.
* **Speed:** about 30× the clip length at "Fast" on a laptop CPU (a 20 s clip ≈ 10 minutes with video rendering).
  A GPU worker is far faster.
* **Cleanup:** the worker deletes jobs and files older than 14 days (`--purge-days`). Each job page has a **Delete** button.
* **Queue protection:** at most 8 jobs can be waiting or running (`MAX_ACTIVE_JOBS`), so a stray visitor can't flood it.

## Security model (read this before sharing the link)
* The whole site sits behind **one shared password** (HTTP Basic auth). In production it **stays locked until
  `RINKIQ_PASSWORD` is set**, so a fresh deploy never exposes uploads by accident.
* The **only public page is `/demo`**, which shows pre-analysed sample data (no video, no uploads).
* The browser never talks to your database. Row-level security is on with no policies, so the public (anon) key can do
  nothing. The server routes use the secret key, and uploads use one-time signed URLs.
* Everyone with the password sees all jobs. It's a shared team tool, not a multi-user product. Per-user accounts
  (Supabase Auth + an owner column and policies) are the natural next step.
* Never commit `worker/.env` or `web/.env.local`; both are git-ignored. If a secret key leaks, rotate it in
  Supabase (Project Settings → API Keys) and re-run `scripts/setup_online.sh`.

## What has and hasn't been tested
Tested: the worker (12 tests) and the website's upload → queue → worker → results loop, both against a stand-in
Supabase server that speaks the same REST protocol (`tests/fake_supabase.py`); the setup script's checks; the SQL parses
as valid PostgreSQL. **Not yet run against a real Supabase project**, so expect the first real run to be the final test.
Report anything odd, and the stand-in server makes it easy to reproduce.
