# Local Development Runbook

Run commands from the repository root.

For quick UI demos without Docker, use
`docs/superpowers/runbooks/sqlite-smoke.md`.

## Configure the Local Environment

ParaWorks uses one ignored root `.env` file for the backend, Docker Compose,
local Python scripts, and the Next.js frontend. For a new checkout—or to fill
missing local signing secrets without replacing existing provider keys—run:

```powershell
uv run python scripts/bootstrap_local_env.py
```

The bootstrap copies `.env.example` only when `.env` is missing, then generates
independent cryptographic values for the C.5 fingerprint, session, Google OAuth,
Google identity, and deferred Slack state signers. It preserves existing values,
including `OPENAI_API_KEY`, and prints names only—never secret values. This step
is required before the ParaWorks application starts against PostgreSQL because
the durable C.5 key boundary is enforced even while automatic-review rollout is
disabled.

Do not commit `.env` or create a second `frontend/.env.local`. The frontend
reads only `NEXT_PUBLIC_API_BASE_URL` and `NEXT_DIST_DIR` from the root file;
provider keys and other backend-only values are never copied into the frontend
environment. `.env.example` documents optional settings, while defaults not
listed in `.env` continue to come from `backend/app/core/config.py`.

Slack settings remain in the template as a deferred integration section. Leave
them empty until a replacement Slack data source is designed.

## Start Runtime Services

```powershell
docker compose up -d postgres redis minio
```

## Initialize Database Schema

```powershell
uv run alembic upgrade head
uv run python scripts/check_db_schema.py
```

`alembic upgrade head` applies the tracked schema migrations. The schema check
fails loudly when an existing local database is missing a table or column that a
newer branch expects.

## Seed Local Demo Data

```powershell
uv run python -m backend.app.db.init_db
```

This keeps local seed users and optional demo data available. It still has a
`create_all()` fallback for brand-new local databases, but migrations are the
source of truth for schema changes.

## Start Backend

```powershell
uv run uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

## Start Frontend

```powershell
cd frontend
npm.cmd run dev -- --hostname 127.0.0.1 --port 3000
```

## Demo Users

- `X-Demo-User: admin` can see `public`, `internal`, and `restricted` sources.
- `X-Demo-User: viewer` can see `public` and `internal` sources.
