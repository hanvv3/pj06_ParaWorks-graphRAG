# Task18 F2: proposed exact-provenance legacy integrity contract

Status: APPROVED for implementation, 2026-09-12. The user explicitly approved
this v3 legacy integrity schema/security proposal through the Task18 controller
instruction for fix round 3. The frozen definitions below amend the legacy-only
integrity contract; they do not authorize rollout or claim implementation complete.

PostgreSQL predecessor-trigger integration: the successor must preserve the C.5
append-only guard except for a single NULL-scope staging-to-v3 signing update of
a dependency and its unsigned parent both inserted in the current transaction.
Only signature/scope/role columns may change; existing provenance, references,
owner and ordinal may not. Published rows and historical rows remain immutable.
The exactness trigger retains existing raw and explicit-effect checks and adds
only v3 genuine/pre-provenance current-approved knowledge authority; it must not
infer human approval from missing provenance or admit those rows to D.

## Problem and recommendation

The accepted all-null `legacy_unbound / legacy_v1_only / selected_citation`
child can sign raw current V1 and genuinely pre-provenance legacy knowledge.
It cannot represent the exact selected approval effect of explicit trusted
knowledge. Reducing that knowledge to an all-null child loses revocation
authority when another approval still supports the same provenance. Keeping
future writes on the historical unsigned path instead leaves their answer
bytes mutable without detection. Evidence-derived email with no public
citations is a second uncovered case. The retained independent F2 probe proves
the first failure against newly written content, not merely historical data.

Recommended minimal secure extension: permit existing raw/trusted exact
provenance shapes under `legacy_v1_only`, including selected approval identity
and its exact evidence-reference set; introduce an integrity-only legacy
influence role for evidence not projected as public citations. Reuse existing
columns and reference rows. Do not add a new approval union/table, erase exact
authority, require D promotion eligibility, or grant D eligibility from a
successful legacy integrity check.

## Proposed normative delta

1. Keep `rag_v2` shapes, signatures, roles, resolver eligibility, and linked-run
   requirements unchanged. D serving identity/version fingerprints remain
   mandatory only for their existing D shapes.
2. A newly signed legacy parent uses a versioned legacy-capable schema marker
   (proposed `assistant-dependency-set-hmac:v3`). Existing `:v2` signed legacy
   rows retain their current verifier and all-null contract. Historical null
   markers remain read-only compatibility for pre-existing rows, never a
   fallback for a failed new write or failed signature.
3. For a new `legacy_v1_only` child, allow:
   - `raw_chunk`: the existing exact chunk/version/source/parser/signature and
     policy fields required by `ck_assistant_message_dependency_exact_kind`;
   - `trusted_knowledge`: existing exact knowledge identity, either the selected
     `approval_link_id` and its matching evidence references, or the existing
     `legacy_human_base` / `legacy_source_review_item_id` authority shape;
   - `legacy_unbound`: only the genuinely pre-provenance all-null shape already
     accepted, never a substitute for an explicit approval-dependent snapshot.
   All three keep `serving_identity_hmac`, `serving_version_fingerprint`, and
   D `support_mode` NULL. `legacy_dependency_identity_hmac` is mandatory.
4. `selected_citation` means an exact public citation with its citation HMAC.
   Add `legacy_evidence_influence` only under `legacy_v1_only`; it has no public
   citation and its selected-citation HMAC is NULL. It cannot be used in a D
   envelope or counted as a D model-influence role. Ordered dependencies cover
   every evidence input, not only publicly cited evidence.
5. A legacy evidence-derived parent has a positive dependency count and a
   complete ordered set HMAC. Selected child count equals public citation count.
   Zero selected children is permitted only for an explicitly evidence-derived
   legacy action with empty public citation/projection arrays. The selected
   projection HMAC then authenticates the canonical empty V1 projection; it
   does not disappear. Ordinary non-evidence contact/email actions remain on
   their existing non-RAG path.
6. New legacy identity payloads bind the freshly recomputed V1
   `serving_content_hash` (the payload's named version-fingerprint input), exact
   dependency kind, role, canonical document ID, permission and complete V1
   provenance snapshot. Raw snapshots bind every exact raw authority field.
   Explicit trusted snapshots bind the selected approval ID and its current
   canonical authority fields, plus sorted exact evidence-link IDs/fields.
   Legacy human snapshots bind their exact human/provenance identity. Do not
   union all currently live approvals or substitute a generic eligible object.
7. Proposed versioned domains are `assistant-legacy-dependency-snapshot:v2`,
   `assistant-dependency-child-hmac:v3`, and
   `assistant-dependency-set-hmac:v3`, dispatched by the new parent marker.
   Final domain spelling and canonical payloads require approval and golden
   vectors before coding. Existing v2 domains are never silently reinterpreted.
   Parent content, origin, ordered child set, citation arrays, counts and key
   identity must all agree; missing or mismatched material fails closed.
8. Evidence-derived email integrity must additionally bind the protected action
   payload: action type, `evidence_derived`, recipient order, subject and body.
   Bind its canonical HMAC through the versioned legacy origin payload; an
   extra physical column is not necessary. Mutable operational status/sent
   timestamps are not evidence authority. A protected payload change requires
   an authorized new/current-evidence write, never automatic re-signing of
   modified stored bytes. This does not change send/idempotency policy.

## Concrete implementation impact after approval

### Models and migration

- `backend/app/models/auto_review.py`: update
  `ck_assistant_message_dependency_v2_scope_role`, `_v2_hmacs`, and `_v2_support`
  to express the scope-specific union above. Preserve `_exact_kind` and its
  selected-approval/legacy-human exclusivity. Preserve ordinal/document
  uniqueness and all `AssistantMessageKnowledgeEvidenceRef` composite foreign
  keys binding references to the same selected approval effect.
- `backend/app/models/assistant.py`: allow the new marker only for
  `legacy_trimmed / legacy_evidence`; preserve existing V2 parent shapes and
  linked-run XOR, and require complete content/origin/key/set integrity.
- Add a successor migration to current head `a4d5e6f7b8c9`; do not rewrite
  `e2b3c4d5f6a7` or other applied migrations. Replace relevant CHECKs and update
  `rag_validate_assistant_integrity_for` deferred cross-row validation to bind
  marker/scope/role, counts, contiguous ordinals, selected projection and all
  child signatures. Legacy zero-citation allowance must not relax D parents.
  SQLite batch-rebuild checks and PostgreSQL constraint-trigger definitions
  must encode the same contract. No historical backfill or permission change.
  Downgrade must refuse while new-marker rows exist rather than stripping
  integrity or converting them to unsigned history.

### Writer and reader

- `assistant/legacy_evidence.py`, `evidence_persistence.py`, and `service.py`:
  snapshot current resolver-produced V1 authority, retain exact dependency
  shapes and approval references, then sign every future evidence-backed
  disabled/shadow/non-cutover legacy answer. Remove the explicit-approval and
  empty-citation signing skips only after the new constraints/verifier exist.
  Writer failure aborts the caller-owned transaction; no unsigned fallback,
  extra commit, authority widening or historical-row re-signing.
- `assistant/evidence_reader.py`: dispatch historical/v2/new-legacy verification
  explicitly. For new legacy, recompute current exact V1 raw, selected approval,
  evidence-reference, or legacy-human authority using the same guarantees as
  `_dependency_is_live`. Check exact reference-set equality and selected-effect
  liveness even if another effect remains live. Recompute every new HMAC and
  protected action payload; return only immutable actor-bound snapshots.
- Email/context/capability/send consumers retain those immutable projections;
  protected `evidence_derived` metadata must not be a bypass switch. Audit
  `email_draft_context.py` and send revalidation for the new signed action
  payload. Existing independent non-evidence contact/recipient/email branch
  behavior remains unchanged.
- Do not call a D resolver to manufacture legacy authority. Explicitly test
  that raw/trusted `legacy_v1_only` rows cannot satisfy D serving eligibility.

## Required acceptance tests

- Newly written raw, manual/automatic exact approval, legacy-human-bound and
  genuinely unbound answers are signed across disabled/shadow/non-cutover paths.
- Keep the unchanged selected-effect revocation regression: revoke the chosen
  effect while another effect supports shared provenance; the answer redacts.
  Exercise permission/version/parser/source/content and evidence-set drift.
- Promote the retained F2 new-write content-tamper probe into a tracked test.
  Tamper content, action payload, key/version, child role/identity/hash, refs,
  ordering/counts, delete a child, and partially strip markers: fail closed.
- Evidence-derived email with zero citations gets a complete influence set;
  tamper recipient/subject/body/evidence flag; GET, context and send revalidation
  reject it. Non-evidence email/contact tests retain their prior behavior.
- Exercise real constrained SQLite writes and caller rollback; compile/inspect
  PostgreSQL migration and retain opt-in real-PG constraint tests. Compilation
  alone is not production PostgreSQL/concurrency proof.
- Historical unsigned rows remain readable by their existing rules; no write
  path chooses historical mode for a new evidence-backed output. Existing v2
  legacy and D signatures remain compatible and D exclusion stays explicit.

## Decision requested

Approve or revise this scope-specific union and new legacy marker/role before
implementation. It changes a shared output-integrity schema and trust boundary,
so it is not authorized by the F1/F3 bug-fix round. Until approved, implemented,
verified and independently reviewed, F2 blocks Task18 completion and rollout.

## Frozen implementation contract (2026-09-12 approval)

This section supersedes tentative wording above. No applied migration is edited.
Successor revision: `b5e6f7a8b9c0`, predecessor `a4d5e6f7b8c9`. No new columns.
Every future evidence-backed legacy write uses marker
`assistant-dependency-set-hmac:v3`; v2 and historical null are read compatibility
only. Ordinary non-evidence actions are not signed. Missing dependencies for an
evidence-shaped new write are rejected, never committed as historical data.

### Canonical encoding and payload allowlists

All signatures use existing `keyed_fingerprint` canonical JSON/HMAC-SHA256 and
policy `assistant-evidence:v1`, except existing citation/projection helpers retain
their existing domains/policies. Strings representing stored/user/source bytes
are encoded with existing `exact_utf8_bytes` (no Unicode normalization). In the
provenance and action objects below this transformation applies recursively to
every string value; object keys remain the literal field names. Integers, booleans,
JSON null and list order retain their types. No arbitrary object/string fallback.
All optional fields are included as null, never omitted. Dates are not authority
payload fields; current revocation/status checks remain mandatory independently.

`snapshot` contains exactly these persisted V1 fields: `document_chunk_id`,
`document_version_id`, `source_id`, `parser_run_id`, `current_document_version_id`,
`server_content_signature_schema`, `server_content_signature`,
`parser_policy_version`, `parser_version`, `chunk_policy_version`, `knowledge_type`,
`knowledge_id`, `approval_link_id`, `legacy_human_base`,
`legacy_source_review_item_id`. All are bound, including required NULLs.

`provenance` contains exactly `snapshot`, `approval`, `review`, `evidence_links`.
`approval` is null except explicit trusted, where it has exactly `id`,
`knowledge_type`, `knowledge_id`, `review_item_id`, `security_scope_id`,
`promotion_effect_kind`, `resolution_source`, `claim_fingerprint`,
`permission_level`, `fingerprint_key_version`, `fingerprint_key_material_verifier`,
`active`. `review` is null except bound legacy-human or explicit trusted, where
the current selected review has exactly `id`, `status`, `permission_level`,
`resolution_source`, `candidate_contract_version`, `source_links`,
`source_snippets`. `evidence_links` is empty except explicit trusted: every current
selected-effect link, sorted by integer `id`, has exactly `id`, `approval_link_id`,
`canonical_source_kind`, `canonical_source_id`, `canonical_version_or_signature`,
`evidence_hash`, `fingerprint_key_version`, `fingerprint_key_material_verifier`.
Its IDs must equal the persisted exact approval-bound reference set, not a union.

`assistant-legacy-dependency-snapshot:v2` payload has exactly:
`serving_document_id_bytes`, `dependency_kind`, `dependency_role`,
`effective_permission`, `serving_version_fingerprint` (fresh V1 content hash),
`model_content_hmac`, `canonical_citation_projection_hmac`,
`legacy_public_source_id_bytes`, `legacy_source_links_bytes`,
`legacy_source_snippets_bytes`, `provenance`. Raw identities are compared with
current chunk/source/parser/document values before signing/verifying. Trusted
identities retain exact selected-effect or legacy-human liveness. A genuine
unbound child must resolve to pre-provenance knowledge with no source review or
approval rows; it cannot stand in for raw or selected-approval authority.

`assistant-dependency-child-hmac:v3` retains the exact v2 child payload field set
listed in `sign_legacy_message`/Task17's legacy verifier, but fills raw/trusted IDs,
actual kind/role, and exact sorted evidence IDs. `approval_provenance_hmac` is the
HMAC of `{approval, review}` under `assistant-legacy-approval-provenance:v1` for
explicit trusted only; `evidence_link_set_hmac` is the HMAC of `{evidence_links}`
under `assistant-legacy-evidence-links:v1` for explicit trusted only. Both are NULL
for other legacy branches. `legacy_dependency_identity_hmac` is always nonnull;
D serving identity/version/support columns and payload values remain NULL.

Public citation validation and `rag-v1-evidence-projection:citation:v1` signatures
remain exact, including binary64 score and matched-term order. Assign each public
citation to exactly one distinct dependency by canonical public ID/type/URL/snippet/
permission, preserving public citation order. Unselected dependencies use
`legacy_evidence_influence` and NULL selected citation HMAC. The parent projection
HMAC binds ordered citation HMACs plus its three exact public arrays. Require those
arrays to equal the ordered unique ID/link/snippet projections of public citations;
zero citations means all three arrays empty. Reject missing/duplicate/extra citation
ownership. Dependency ordinals remain resolver order and contiguous from zero.

`assistant-legacy-action-payload:v1` payload has exactly `action_type`,
`evidence_derived`, `email_draft`. `action_type` is string or null;
`evidence_derived` is a strict boolean, default false only if absent;
`email_draft` is null or exactly `{to, subject, body}`, with `to` an ordered string
list and subject/body exact strings. An evidence-derived email requires a complete
nonempty draft; no extra draft keys are accepted. Changing status/sent timestamps
does not alter authority; changing protected bytes requires a new authorized write.

`assistant-legacy-evidence-origin:v2` retains all v1 origin fields and adds exactly
`legacy_action_payload_hmac`, `hidden_match_count`, `permission_notice_bytes`.
Nullable permission/notice are bound; a legacy email's existing NULL public
permission does not waive the per-dependency current actor permission gate.
`assistant-message-content-hmac:v1` keeps its existing exact payload unchanged.
`assistant-dependency-set-hmac:v3` keeps the existing v2 legacy whole-set payload
unchanged, but uses the new domain and ordered v3 child HMACs. Current key/runtime
identity is mandatory. No v3 material may be interpreted under a v2 domain.
If absent on an evidence-backed write, server metadata defaults `agent_name` to
`rag_orchestrator_agent`, `effective_backend` to `deterministic_lexical`, and
`prompt_version` remains null; supplied values are never overwritten. Backend
must be `deterministic_lexical` or `pgvector`. Unknown agent types fail closed.

### Model and cross-row invariants

Parent v3 marker is allowed only for `legacy_trimmed / legacy_evidence`,
`assistant-evidence:v1`, positive dependency count, complete content/key/origin/
set/projection HMACs, NULL linked D run/result and NULL D influence-set HMAC.
Existing D parent markers/XOR remain unchanged. Child legacy scope permits all
three existing exact-kind shapes, selected/influence roles, NULL D columns and
nonnull legacy identity. Exact raw/trusted composite FKs and approval reference
FKs are retained. Explicit v3 trusted rows require both provenance HMACs; other
v3 rows require them NULL. Hashes use lowercase hexadecimal SHA256 text.

V3 selected count equals JSON citation count. Zero selection requires strict
metadata `evidence_derived=true` and all three public arrays empty. Every child
matches parent scope, key and whole-set HMAC, with positive count and contiguous
ordinals. V2 legacy remains selected-only/all-null authority and D exclusions
remain unchanged. PostgreSQL deferred validation covers parent, children and
approval refs. SQLite validates v3 publication and subsequent child/ref mutations
using immediate triggers; the writer stages under an unsigned private parent,
flushes complete children/refs, then publishes the signed parent in the same
transaction. This is staging, never an unsigned commit or fallback. SQLite does
not emulate PostgreSQL deferred timing; both enforce the same published v3 state.
Downgrade refuses while any v3 parent exists; it never strips or backfills markers.

Golden vectors must independently pin canonical JSON/HMAC results for raw,
explicit trusted, legacy-human/unbound, uncited email and ordered whole-set cases.
Tests must also exercise real current-authority drift; a vector alone is not an
authorization or provenance-liveness proof. Approval authorizes this frozen
contract's implementation and verification, not D eligibility or activation.
