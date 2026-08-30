# PostgreSQL + pgvector Dev Runbook

Use this when validating the production RAG path locally. SQLite smoke remains
the default for fast demos; this runbook is for pgvector write/search behavior.

## Start services

```powershell
docker compose up -d postgres redis
```

If port `127.0.0.1:5432` is unavailable or forbidden, Docker may fail before
Postgres starts. Stop partial containers with:

```powershell
docker compose down
```

Then either free port `5432`, change the compose port mapping locally, or set a
different `DATABASE_URL` that points at an existing Postgres + pgvector instance.
The compose file also supports alternate host ports without editing tracked
files:

```powershell
$env:PARAWORKS_POSTGRES_PORT='5433'
$env:PARAWORKS_REDIS_PORT='56379'
docker compose up -d postgres redis
```

Then point the app at the matching database URL:

```powershell
$env:DATABASE_URL='postgresql+psycopg://paraworks:paraworks@127.0.0.1:5433/paraworks'
```

Or start the app stack with the helper:

```powershell
.\scripts\start-pgvector-dev.ps1
```

If `5432` is already taken, use the helper's port flags:

```powershell
.\scripts\start-pgvector-dev.ps1 -PostgresPort 5433 -RedisPort 56379
```

By default the helper also detects a non-ParaWorks listener on `127.0.0.1:5432`
and falls back to the next available host port, starting at `5433`. This avoids
the common local failure mode where an existing PostgreSQL instance answers on
`5432` or Windows reserves the socket before Docker can bind it.

To prepare only Postgres + Redis without starting the app:

```powershell
.\scripts\start-pgvector-dev.ps1 -SkipApp
```

Default database URL:

```text
postgresql+psycopg://paraworks:paraworks@127.0.0.1:5432/paraworks
```

## Environment

Use the single root `.env` described in `docs/superpowers/runbooks/local-dev.md`.
Run `uv run python scripts/bootstrap_local_env.py` before starting the ParaWorks
application against PostgreSQL; do not overwrite an existing local file or
create a separate frontend env file.

Do not commit `.env`, provider API keys, OAuth tokens, Slack tokens, or any
other secret values.

Required for live embedding writes:

```text
OPENAI_API_KEY=<local secret only>
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EMBEDDING_DIMENSIONS=1536
RAG_USE_PGVECTOR_SEARCH=false
```

`rag_vector_documents.embedding` is intentionally fixed at `vector(1536)` for
the current embedding path. If `OPENAI_EMBEDDING_DIMENSIONS` or the embedding
model is changed, run `uv run python scripts/check_db_schema.py` before writes.
The check compares the configured dimension with the actual PostgreSQL column
type and fails instead of allowing a silent mismatch.

Keep `RAG_USE_PGVECTOR_SEARCH=false` until the index has been populated and
verified. Toggle it only for manual vector retrieval checks.

## Initialize database

```powershell
$env:DATABASE_URL='postgresql+psycopg://paraworks:paraworks@127.0.0.1:5432/paraworks'
uv run alembic upgrade head
uv run python scripts/check_db_schema.py
uv run python -m backend.app.db.init_db
```

`alembic upgrade head` is the source of truth for application schema changes.
`scripts/check_db_schema.py` verifies that an existing database is not missing
new tables or columns after a pull, and also verifies the native pgvector table
and embedding dimension on PostgreSQL. `backend.app.db.init_db` is kept for
local seed users and optional demo data.

The Docker init scripts also create pgvector support and the vector tables for
fresh volumes:

- `docker/postgres/init/001_extensions.sql`
- `docker/postgres/init/002_rag_vector_documents.sql`
- `docker/postgres/init/003_vector_index_states.sql`

The API also calls `PgVectorStore.ensure_schema()` before production writes.
The helper runs the same schema guard explicitly:

```powershell
uv run python scripts/check_db_schema.py --database-url $env:DATABASE_URL
uv run python scripts/check_pgvector_dev.py --database-url $env:DATABASE_URL --ensure-vector-schema
uv run python scripts/check_pgvector_dev.py --database-url $env:DATABASE_URL --expect-app-schema
```

If the check fails with a password error, you are not connected to the expected
ParaWorks database. Use the selected helper port shown in the startup output,
stop the conflicting local Postgres service, or reset the Docker volume intentionally with
`docker compose down -v` when you do not need the local data.

## Safe smoke sequence

First verify the free deterministic preview path:

```powershell
Invoke-RestMethod -Method Post 'http://127.0.0.1:8000/api/v1/integrations/slack/sync'
Invoke-RestMethod -Method Post 'http://127.0.0.1:8000/api/v1/integrations/gmail/sync'
Invoke-RestMethod -Method Post 'http://127.0.0.1:8000/api/v1/rag/reindex/jobs'
Invoke-RestMethod 'http://127.0.0.1:8000/api/v1/rag/indexing/summary'
```

## Celery worker mode

Local smoke and tests default to eager mode:

```text
CELERY_TASK_ALWAYS_EAGER=true
```

That keeps `/api/v1/rag/reindex/jobs` deterministic without requiring a worker.
To validate the real Redis queue path, set eager mode off and run a worker in a
separate terminal:

```powershell
$env:CELERY_TASK_ALWAYS_EAGER='false'
.\scripts\start-celery-worker.ps1
```

On Windows, the worker script uses `--pool=solo`.

With eager mode disabled, `POST /api/v1/rag/reindex/jobs` creates a queued
`SyncJob`, returns immediately, and the worker moves it through
`running -> complete` or `failed`. Poll:

```powershell
Invoke-RestMethod 'http://127.0.0.1:8000/api/v1/rag/reindex/jobs/<job_id>'
Invoke-RestMethod 'http://127.0.0.1:8000/api/v1/rag/indexing/summary'
```

Then, only after setting a local provider key, run the paid write path:

```powershell
Invoke-RestMethod -Method Post 'http://127.0.0.1:8000/api/v1/rag/reindex/jobs?dry_run=false'
```

The response must include:

- `storage_backend=pgvector`
- `incremental=true`
- `indexed_count`
- `skipped_count`
- `saved_embedding_calls`
- `embedding_request_count`
- `embedding_prompt_tokens`

If the database is SQLite, `dry_run=false` must fail with a clear 400 response.
If the provider key is missing, `dry_run=false` must also fail with a clear 400
response.

## Fake embedding integration test

The automated integration test never calls OpenAI. It writes deterministic fake
embeddings into a real pgvector database only when this variable is present:

```powershell
$env:PARAWORKS_PGVECTOR_TEST_DATABASE_URL='postgresql+psycopg://paraworks:paraworks@127.0.0.1:5432/paraworks'
uv run pytest backend/tests/test_pgvector_integration.py -v
```

Without `PARAWORKS_PGVECTOR_TEST_DATABASE_URL`, the test is skipped.
When the helper falls back to an alternate host port, use the matching URL from
the helper output:

```powershell
$env:PARAWORKS_PGVECTOR_TEST_DATABASE_URL='postgresql+psycopg://paraworks:paraworks@127.0.0.1:5433/paraworks'
uv run pytest backend/tests/test_pgvector_integration.py -v
```

## Cost policy

- Never run full-corpus paid re-embedding by default.
- Confirm `skipped_count` and `saved_embedding_calls` before repeated runs.
- Use batch embedding through `embed_many`.
- Keep `OPENAI_EMBEDDING_INPUT_COST_PER_1M_TOKENS` aligned with current provider
  pricing. As of the 2026-05-02 check, OpenAI lists `text-embedding-3-small` at
  `$0.02 / 1M tokens`.
- Keep `RAG_EMBEDDING_MAX_ESTIMATED_COST_USD` low in local/dev environments.
  ParaWorks estimates changed-document embedding input before any paid provider
  call and blocks `dry_run=false` when the estimate exceeds the configured
  budget.
- Keep live provider calls out of tests.
- Record notable cost-related changes in `docs/portfolio-log.md`.
