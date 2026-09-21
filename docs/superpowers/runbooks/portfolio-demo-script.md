# ParaWorks Portfolio Demo Script

Updated: 2026-05-03

This is the historical smoke presentation flow, not live GraphRAG acceptance.
For the current final-demo target and prerequisites, use [final-demo.md](final-demo.md).

## Goal

Show ParaWorks as a Korean-first company memory product, not a generic chatbot.
The demo should prove that multi-agent ingestion, human review, approved
knowledge, permission-aware RAG, cost controls, and agent observability work as
one product story.

## Demo Preconditions

- Smoke backend and frontend are running through `scripts/demo/start-smoke.ps1`.
- Backend health returns `demo_mode=true` for local route regression.
- Slack and Google OAuth cards may show configured or reconnect-needed status;
  do not expose secrets.
- Seeded or synced evidence exists for at least one Slack-backed decision or
  history item.
- Paid LLM runs are optional and must use preflight first.

## Story Arc

1. Login and role boundary
   - Open `/login`.
   - Select `admin@paraworks.com`.
   - Explain that the harness now issues httpOnly session and refresh cookies,
     while demo mode keeps account switching convenient.
   - Mention production mode fails closed without a valid session cookie.

2. Integration status
   - Open `/integrations`.
   - Show Slack and Google runtime status cards.
   - Explain delta sync, duplicate skip counts, and why full sync does not mean
     full LLM input.

3. Agent run observability
   - Open `/agent-runs`.
   - Show token usage, estimated cost, cache status, source window strategy, and
     ranked evidence.
   - If using a real provider, run preflight before confirmation and keep the
     paid window small.

4. Human review trust boundary
   - Open `/review`.
   - Inspect a pending item.
   - Open source evidence and show source URL, snippet, permission, confidence,
     and originating AgentRun when available.
   - Approve only items with evidence; request more evidence when confidence or
     support is weak.

5. Approved company memory
   - Open `/knowledge`, then `/decisions`, `/timeline`, and `/history`.
   - Show that approved items become trusted memory records only after review.
   - Explain that LLM output is not directly promoted to official knowledge.

6. Knowledge Map
   - Open `/knowledge-map`.
   - Show memory nodes connected to source evidence nodes.
   - Explain that shared evidence nodes inherit the strictest connected
     permission level.
   - Emphasize that this map is read-only and does not call LLMs or embeddings.

7. Permission-aware RAG
   - Open `/search`.
   - Ask a question about an approved decision or project history.
   - Show citations, retrieval backend disclosure, hidden-match handling, and
     cost policy.
   - Mention that deterministic search is the default smoke path; pgvector is
     enabled only with explicit configuration and provider keys.

8. Portfolio close
   - Open `docs/portfolio-log.md`.
   - Summarize the engineering story:
     three-track ownership, shared contracts, LangGraph orchestration, Review
     Queue HITL, pgvector direction, token-cost guardrails, and Playwright
     regression coverage.

## Cost Script

Use this language during the demo:

- "Sync can collect many source events, but paid LLM input is ranked, deduped,
  and budget-capped."
- "Embedding work is incremental; unchanged content hashes skip provider calls."
- "Status pages and maps are read-only database views and never trigger paid
  model calls."
- "Real provider actions require preflight and explicit confirmation."

## Security Script

Use this language during the demo:

- "Demo mode keeps local account switching fast, but production mode rejects
  missing sessions."
- "Session and refresh cookies are httpOnly; refresh tokens are hashed at rest
  and rotated."
- "Restricted evidence stays restricted through review, knowledge promotion,
  Knowledge Map, and RAG."

## Verification Checklist

Run before recording or presenting:

- Backend tests: `.venv\Scripts\python.exe -m pytest backend\tests -q`
- Frontend lint: `npm.cmd run lint`
- Frontend build: `npm.cmd run build`
- Route regression: `npm.cmd run test:visual -- e2e/page-regression.spec.ts`
- Smoke health: `GET http://127.0.0.1:8000/health`
- Auth smoke: login as `admin@paraworks.com`, then call `/api/v1/auth/me`
  with the same cookie session.

## Remaining Product Gaps

- Add Alembic migrations for production auth tables.
- Add password or OAuth-backed identity verification beyond demo account
  selection.
- Add CSRF and rate-limit protections for cookie-authenticated unsafe methods.
- Add final visual QA screenshots for the portfolio case study.
