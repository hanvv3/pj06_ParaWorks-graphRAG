# Task 8 Implementation Report

## Result

- Status: COMPLETE
- Base: `ab7d1def904aa973c7eaab3a48ad80e3f09c8e65`
- Commit: `0c88765` — `feat: project rag evidence from canonical rows`
- Live/provider/network/Docker/PostgreSQL calls: 0
- Paid calls/cost: 0 / USD 0

## TDD Evidence

- RED: `uv run pytest backend/tests/test_rag_v2_projection.py -q`
  failed during collection with `ModuleNotFoundError: backend.app.rag.evidence_projection`.
- First GREEN: focused projection suite `12 passed`.
- Canonical integration GREEN: raw and legacy-trusted SQLite resolver projection added; focused suite `13 passed`.
- Adversarial RED/GREEN: forged raw approval-provenance HMAC was initially accepted; the new provenance-drift regression failed, then passed after branch-exact provenance/link-set validation.
- Final focused projection suite: `19 passed in 0.43s`.

## Implemented Contracts

- Stable trusted-first ordering, bounded maximum eight, contiguous `E1..En`, whole-record tail removal, and immutable `EvidenceSlot` values.
- Resolver-owned `CanonicalServingProjection`; public id/URL/snippet/type/permission/parser/version bytes are read from fresh canonical rows rather than model/vector metadata.
- Transaction-bound search and selected-answer projection with immutable fully assembled DTOs.
- Search projects the whole bounded visible set; answer projection emits only a validated selected subset while model-influence finalization preserves every provider-visible slot.
- Pre-generation observations carry no dependency role. Roles are assigned only after exact selected-slot validation.
- Exact separate keyed domains for citation, selected set, ordered search set including empty, prepared influence observation, influence child/set, and hidden membership.
- Binary64 score bits, matched-term order, nullable-key presence, dependency role, slot/order, strictest permission, and capped hidden membership are bound and mutation-tested.
- Generation, readiness, hidden membership, identity/version/content/permission/provenance, selected membership, URL/Unicode/score drift fail closed to a whole empty projection/dependency set.
- Denied identities are accepted only as bounded internal HMAC inputs and are never returned or persisted by the projection DTO.

## Verification

- Direct Task 4-7 affected suite after implementation: `230 passed in 5.23s`.
- All RAG V2/source/trusted tests: `295 passed, 2 skipped, 2061 deselected`.
  The two skips are the existing PostgreSQL-only migration/runtime gates; no PostgreSQL claim is made.
- Post-format focused projection/source/trusted/keyword/pgvector suite: `145 passed in 3.04s`.
- Secret hygiene: `3 passed in 4.90s`.
- Ruff exact six paths: `All checks passed!` with `--no-fix` (direct venv Ruff binary used for the final repeat after a transient uv Windows launcher failure).
- Compile/import passed before formatting; post-format pytest imported and executed all changed modules successfully.
- `git diff --check`: clean; only existing Windows LF-to-CRLF notices.

## Files

- `backend/app/rag/evidence_projection.py` (new)
- `backend/tests/test_rag_v2_projection.py` (new)
- `backend/app/rag/serving_contracts.py`
- `backend/app/rag/source_observations.py`
- `backend/app/rag/trusted_evidence.py`
- `backend/app/rag/retrieval.py`

## Remaining Release Evidence

- Task 8 used SQLite/fakes only as required. PostgreSQL lock linearization and transaction execution remain release-gate evidence; this task does not claim them.

## Review Fix Round 1/5

### Result

- Status: COMPLETE
- Review base: `0c88765c49c23c68b2de6420d07061967f543643`
- Live/provider/network/Docker/PostgreSQL calls: 0 intentional calls
- Paid calls/cost: 0 / USD 0

### TDD and adversarial evidence

- RED began with the new review contracts absent during collection, then covered
  complete-set drop/E2-only/reorder/renumber, child and aggregate one-byte
  mutation, rendered-input/generation/support-mode/branch mutation, recursive
  projection mutation, hidden-count mismatch, strict resolver failure, six-slot
  search, and raw/trusted branch corruption.
- The real ordered two-result search projection golden was captured from a RED
  mismatch and fixed at
  `1ee1340576bbe8a328103cd505f17dab5fc6fb3eb4afb2f4487005663e75ee0c`.
- The final adversarial projection suite is `41 passed in 0.85s`.

### Security corrections

- Introduced immutable `PreparedModelInfluenceSet` authority. Its aggregate HMAC
  binds every contiguous ordered observation, the per-observation child HMAC,
  exact lookup-identity HMAC, corpus and vector generations, readiness identity,
  and rendered model input.
- Finalization accepts exact built-in/dataclass shapes only, authenticates the
  entire original set before validating selected IDs, fresh-resolves every
  original entry exactly once, revalidates exact raw/trusted branch provenance,
  and assigns dependency roles only afterward.
- Dependency and set HMACs bind the fresh identity and derive the strictest
  permission internally. Caller-provided permission authority, reordered sets,
  stale identities, and partial influence sets are rejected.
- Public projection records are recursively frozen mappings without a mutable
  `dict` base class. Mutation APIs, copy aliases, direct dict-base mutation,
  nested sequence mutation, and attribute replacement/deletion are blocked.
- Hidden membership is derived from a transient actual count and the exact
  retained HMAC prefix: `0..20` is uncapped and exact, while `>20` is capped at
  20. No denied IDs are stored or returned.
- Added strict canonical projection resolver paths that distinguish canonical
  ineligibility (whole redaction) from typed infrastructure/read failures
  (exception, no DTO/product HMAC).
- Search projection now has an exact maximum of five results; answer/model
  influence remains bounded to eight contiguous slots.
- Raw and trusted projections require exact branch type, support mode, public
  ID/type, version envelope, provenance, approval, and evidence-link semantics.

### Final verification

- Focused Task 8: `41 passed in 0.85s`.
- Task 4-8 RAG V2/source/trusted: `312 passed, 2 skipped, 10 warnings in 6.50s`.
  The skips are the existing PostgreSQL-only migration gates.
- Adjacent search/permission/pgvector under process-scoped SQLite: `22 passed in 1.81s`.
- Secret hygiene: `3 passed in 5.11s`.
- Ruff exact Task 8 paths with `--no-fix`: `All checks passed!`.
- Compileall and direct imports: PASS.
- `git diff --check`: PASS apart from informational Windows line-ending notices.

### Environment note

- One initial adjacent run inherited the local full-app PostgreSQL configuration
  and could not start that service; the same relevant tests were rerun with
  process-scoped SQLite and passed 22/22.
- Two unrelated quality-suite assertions remain tied to the intentionally absent
  Slack data source and are deferred under the approved Slack-last decision.

## Review Fix Round 2/5

### Result

- Status: COMPLETE
- Review base: `777a067073044559a4cdf4d37bf4e5b73ffb6cba`
- Live/provider/network/Docker/PostgreSQL calls: 0
- Paid calls/cost: 0 / USD 0

### TDD evidence

- First RED: the adversarial suite failed collection because the requested typed
  `CanonicalProjectionTransactionError` contract did not exist.
- Second RED: explicit DTO-to-transport conversion failed collection because
  `projection_record_to_transport` did not exist.
- Intermediate focused run exposed three discriminating failures: changed v2
  goldens, the former caller-authoritative hidden helper, and nested `FrozenDict`
  becoming a tuple through base-class dispatch. Each was corrected before the
  final run.
- Final focused projection suite: `54 passed in 0.95s`.

### Corrections

- Replaced attribute-backed `FrozenDict` with an attribute-free `tuple` subclass
  and exact `Mapping` interface (`__slots__ = ()`). Normal mutation APIs,
  `object.__setattr__`, dict/tuple base operations, copy/deepcopy, nested aliases,
  and pickle round trips cannot mutate committed projection state.
- Added `projection_record_to_transport` as the explicit post-DTO mutable copy
  boundary. Mutating the transport copy leaves the transaction DTO and its HMAC
  bytes unchanged.
- Restored the approved `rag-prepared-model-influence-observation:v1` payload and
  golden `940b88c5...a696`, and the approved
  `rag-model-influence-dependency:v1` payload and selected-role golden
  `885d56f3...02d3`.
- Moved complete prepared-set/readiness/rendered-input authority to
  `rag-prepared-model-influence-set:v2` (golden `c4ba54ec...b3f1`) with child
  observations under an explicit `child:v2` domain. Fresh-lookup set authority is
  now under `rag-model-influence-set:v2` (golden `6021d66c...5f4b`).
- Strengthened raw/trusted cross-field authority: exact branch/envelope/provenance
  types, raw envelope-to-evidence identity/permission/content/citation equality,
  trusted canonical serving ID/type/result equality, and recomputed version,
  approval-provenance, selected-child, and evidence-link-set HMACs.
- Carried ordered canonical evidence-link child HMACs internally on trusted
  resolver DTOs so the projector can recompute the explicit link set rather than
  trusting a caller-supplied aggregate.
- Selected IDs now require exact built-in literal strings, preparation requires
  exactly 1..8 observations, and set/finalizer DTO types remain exact with no
  drop/reorder/renumber/relabel path.
- Hidden membership now accepts only the full transient ordered hidden-member
  HMAC tuple (maximum 50), derives actual/public/capped values and the first 20
  internally, and returns/persists no denied IDs.
- Transaction fencing now captures the exact active SQLAlchemy transaction object
  on entry and rechecks it before/after resolver reads, after validation phases,
  and immediately before every public return. Commit/rollback/close or replacement
  with a new transaction raises the typed transaction failure without returning a
  DTO or product HMAC.

### Final verification

- Focused Task 8: `54 passed in 0.95s`.
- Task 4-8 RAG V2/source/trusted: `325 passed, 2 skipped, 10 warnings in 6.70s`.
  The two skips remain PostgreSQL-only migration gates.
- Adjacent search/permission/pgvector with process-scoped SQLite:
  `22 passed in 1.40s`.
- Secret hygiene: `3 passed in 4.96s`.
- Ruff exact Task 8 paths with `--no-fix`: `All checks passed!`.
- Compileall, direct imports, and `git diff --check`: PASS.
- Existing Slack-data quality failures were not rerun and remain deferred under
  the approved Slack-last decision.
