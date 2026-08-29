# Task 6 I4/I5/M1 ingestion authority fix report

Date: 2026-08-30 KST

Branch: `codex/task6-fix-ingestion`

Base: `82654be22c53e7dae8ccb1fc8142f0791c77ab91`

Commit: included in `fix: enforce server source authority end to end`; the exact
commit hash is recorded in the controller handoff because a commit cannot contain
its own final hash.

## Scope

- I4: connector parser/chunk/snippet metadata authority at ingestion, persisted
  chunks, vector documents, and Search projection.
- I5: exact server signature/current document pointer/current server parser run
  resolution for Mail/Document preflight.
- M1: byte-for-byte canonical JSON and SHA-256 vectors for Gmail,
  Gmail attachment, Drive, and Calendar.

No Ask/reindex race algorithm, startup/repair recovery, Slack behavior, product
documentation, or live provider/client behavior was changed.

## Root causes

1. `_canonical_source_metadata()` removed signature/pointer keys but preserved
   connector-controlled parser names/statuses, chunk policy hints, snippets,
   hashes, section/page hints, and raw MIME. `persist_parsed_document()` then
   spread this metadata into every server-parsed chunk.
2. RAG indexing copied parser metadata from `DocumentChunk.metadata` instead of
   deriving it from the exact relational current `DocumentParserRun` bound to
   `Document.current_document_version_id`.
3. `canonical_sources.py` read `Source.raw_metadata['content_signature']`, used
   the display `Document.current_version` label, and selected the latest parser
   run. This both rejected new server-signed ingestion and authorized legacy,
   ambiguous, wrong-pointer, or non-server state.
4. The frozen-vector test asserted exact canonical bytes/hash only for Gmail;
   the other three supported source types used substring checks.

## Changes

- Added a shared server parser policy/identity predicate, including normalized
  registry MIME, exact parser/policy/chunk versions, parsed status, and exact
  server content signature.
- Filtered connector parser/chunk/snippet authority before canonical Source and
  server-parsed chunk persistence. Legitimate connector evidence such as
  revision/document labels and operational metadata remains available; MIME is
  stored as the registry-normalized value.
- Rebuilt vector parser metadata from the exact current relational version/run;
  stale or ambiguous state fails closed, and chunk text hashes are recomputed.
- Migrated canonical resolution to `Source.server_content_signature`, the exact
  relational current pointer, one exact server parser run, and its complete,
  contiguous bound chunk set. No raw/latest/display fallback remains.
- Added hostile connector end-to-end coverage through `SourceEvent`, persisted
  Source/chunk, `VectorDocument`, and Search response.
- Added fail-closed resolver regressions for raw-only legacy state, wrong pointer
  with a tempting display-version fallback, ambiguous runs, and non-server runs.
- Added exact hand-derived JSON and SHA-256 literals for Gmail attachment, Drive,
  and Calendar alongside the existing Gmail vector.

## TDD evidence

RED:

- Mail/Document endpoints: `7 failed, 5 passed`; every failure stopped at the
  legacy raw `content_signature` lookup.
- Hostile connector regression: failed because `parser_name` remained in
  `Source.raw_metadata`.
- Resolver regressions: `4 failed`; every case incorrectly returned without
  raising `ReviewWorkflowPreflightError`.

GREEN:

- Mail/Document endpoint file: `12 passed`.
- Canonical resolver file: `12 passed`.
- Signature + ingestion contract slice: `67 passed`.
- RAG indexing/search slice: `37 passed`.
- Final combined focused gate: `170 passed in 6.20s`.
- Focused Ruff `--no-fix`: `All checks passed!`.
- `git diff --check`: exit 0 (only Windows LF-to-CRLF checkout warnings).

All connector/provider tests used repository fakes; no live external API was
called.

## Remaining issues

- Older Task 3/4 Review V2 test fixtures outside this Task 6 gate still construct
  raw-only Source rows and therefore now fail closed when run as a broad legacy
  suite. They need a separately owned fixture migration to create exact server
  signature/pointer/parser/chunk state. Restoring a raw or latest-run fallback
  would violate I5 and is intentionally not used.
- Slack remains on its explicitly deferred legacy-deduplication path and receives
  no C.5 source authority from this change.
