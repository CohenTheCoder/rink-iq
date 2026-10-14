#!/usr/bin/env bash
# One-time setup for the online version: connects the website (Vercel) and the worker to YOUR Supabase project.
#
#   ./scripts/setup_online.sh              # asks for the values, then configures + deploys
#   ./scripts/setup_online.sh --no-vercel  # only write worker/.env and web/.env.local (skip Vercel)
#
# You paste your keys into THIS terminal (hidden as you type). They go to exactly three places: worker/.env and
# web/.env.local (both git-ignored, chmod 600) and Vercel's encrypted environment variables. Nothing is printed
# or committed. Before running: create a Supabase project and run supabase/migrations/001_init.sql in its SQL editor
# (see docs/ONLINE.md).
#
# Non-interactive use: export SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_ROLE_KEY, RINKIQ_PASSWORD first.
set -euo pipefail
cd "$(dirname "$0")/.."

USE_VERCEL=1
[ "${1:-}" = "--no-vercel" ] && USE_VERCEL=0

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
die() { printf '\n\033[31m%s\033[0m\n' "$*" >&2; exit 1; }

say "RinkIQ online setup"
URL="${SUPABASE_URL:-}"; ANON="${SUPABASE_ANON_KEY:-}"; SVC="${SUPABASE_SERVICE_ROLE_KEY:-}"; PASS="${RINKIQ_PASSWORD:-}"
if [ -z "$URL" ]; then read -r -p "Supabase project URL (https://<ref>.supabase.co): " URL; fi
if [ -z "$ANON" ]; then read -r -s -p "anon / publishable key (public): " ANON; echo; fi
if [ -z "$SVC" ]; then read -r -s -p "service_role / secret key (SECRET): " SVC; echo; fi
if [ -z "$PASS" ]; then
  read -r -s -p "Site password (Enter = generate one): " PASS; echo
  if [ -z "$PASS" ]; then PASS="$(LC_ALL=C tr -dc 'A-HJ-NP-Za-km-z2-9' </dev/urandom | head -c 16)"; GENERATED=1; fi
fi
URL="${URL%/}"

case "$URL" in
  https://*.supabase.co|http://127.0.0.1:*|http://localhost:*) ;;
  *) die "That doesn't look like a Supabase URL. It should be https://<project-ref>.supabase.co (Project Settings → API).";;
esac
[ -n "$ANON" ] && [ -n "$SVC" ] || die "Both keys are required."
[ "$ANON" != "$SVC" ] || die "The anon key and the service_role key must be different. The service key is the SECRET one."

say "Checking the connection…"
HDR=(-H "apikey: $SVC"); case "$SVC" in *.*.*) HDR+=(-H "Authorization: Bearer $SVC");; esac
CODE="$(curl -s -o /tmp/rinkiq_check.json -w '%{http_code}' "${HDR[@]}" "$URL/rest/v1/jobs?select=id&limit=1" || true)"
if [ "$CODE" = "200" ]; then
  echo "✓ Connected, and the jobs table exists."
elif grep -q "PGRST205\|does not exist\|Could not find the table" /tmp/rinkiq_check.json 2>/dev/null; then
  die "Connected, but the database tables are missing. Open Supabase → SQL Editor → paste supabase/migrations/001_init.sql → Run. Then run this script again."
elif [ "$CODE" = "401" ] || [ "$CODE" = "403" ]; then
  die "Supabase rejected the service_role key ($CODE). Re-copy it from Project Settings → API Keys."
else
  die "Couldn't reach $URL (HTTP $CODE). Check the project URL."
fi
rm -f /tmp/rinkiq_check.json

say "Writing worker/.env and web/.env.local (git-ignored)…"
umask 077
printf 'SUPABASE_URL=%s\nSUPABASE_SERVICE_ROLE_KEY=%s\n' "$URL" "$SVC" > worker/.env
printf 'NEXT_PUBLIC_SUPABASE_URL=%s\nNEXT_PUBLIC_SUPABASE_ANON_KEY=%s\nSUPABASE_SERVICE_ROLE_KEY=%s\nRINKIQ_PASSWORD=%s\n' \
  "$URL" "$ANON" "$SVC" "$PASS" > web/.env.local
chmod 600 worker/.env web/.env.local
echo "✓ done"

if [ "$USE_VERCEL" = "1" ]; then
  say "Setting Vercel environment variables…"
  command -v npx >/dev/null || die "Node.js is needed for the Vercel CLI (https://nodejs.org)."
  cd web
  [ -d .vercel ] || npx --yes vercel@latest link --yes --project rinkiq
  setvar() { printf '%s' "$2" | npx --yes vercel@latest env add "$1" production --force >/dev/null && echo "✓ $1"; }
  setvar NEXT_PUBLIC_SUPABASE_URL "$URL"
  setvar NEXT_PUBLIC_SUPABASE_ANON_KEY "$ANON"
  setvar SUPABASE_SERVICE_ROLE_KEY "$SVC"
  setvar RINKIQ_PASSWORD "$PASS"
  say "Deploying to production…"
  npx --yes vercel@latest deploy --prod --yes
  cd ..
fi

say "All set."
if [ "${GENERATED:-0}" = "1" ]; then echo "Your site password (save it, it is not stored anywhere else): $PASS"; fi
echo "Start the worker on a computer that has the RinkIQ Python environment:"
echo "    python worker/worker.py"
echo "Then open your Vercel URL, upload a video, and watch it get picked up."
