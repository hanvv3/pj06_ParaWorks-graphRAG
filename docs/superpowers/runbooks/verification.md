# ParaWorks Harness Verification

Run commands from the repository checkout under review.

## Task 2 Verification

```powershell
docker compose config
```

## Full Harness Verification

Requires the frontend workspace under `frontend/`.

```powershell
docker compose config
uv run pytest backend/tests -v
cd frontend
npm.cmd run build
```

For manual runtime checks against the local demo Postgres database, initialize
the application schema before starting Uvicorn:

```powershell
uv run python -m backend.app.db.init_db
```

Expected backend checks:

- health endpoint returns demo mode
- mock connectors expose source evidence
- sync creates pending review items
- approve/reject transitions work
- viewer cannot see restricted source content
- admin can see restricted source content
- SSE job status emits progress and done events

Expected frontend check:

- dashboard, integrations, messages, review, and search routes compile

## SQLite Smoke Verification

Use this when Docker is unavailable but the UI flow needs to be checked:

```powershell
$env:DATABASE_URL="sqlite:///.tmp/paraworks-smoke-verify.db"
uv run python -m backend.app.db.init_db
uv run pytest backend/tests -v
cd frontend
npm.cmd run build
```

For an interactive smoke run, use:

```powershell
.\scripts\demo\start-smoke.ps1
```
