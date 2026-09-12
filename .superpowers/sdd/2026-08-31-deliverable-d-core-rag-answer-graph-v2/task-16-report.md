# Task 16 Implementation Report

Date: 2026-09-12

Status: implementation candidate; independent controller review required. This
report does not claim CLEAN or D Core release completion.

## Scope

Implemented only Task16's Assistant delivery algebra, exact recursive public
citation DTO, refusal-only render capability, projection-sensitive GET guard,
write-side guard precedence, scanner pre-user boundary, and cache-isolation
headers. Task17 evidence/content revalidation and Task18 Assistant V2
sole-writer/error cutover remain unimplemented here. Existing legacy and
noncutover Assistant writers/catches remain unchanged after the new pre-user
guards.

## TDD RED

Before production changes, added tests naming the concrete breaks they catch:

- accepting any illegal delivery state/status/body/id/outcome cross-product;
- accepting missing, duplicate, comma-folded, or wrong raw capability headers;
- guarding after conversation/message/provider mutation;
- allowing capability 409 to mask authentication, validation, or owner 404;
- guarding stale, unrelated, legacy-only, or noncontributing GET rows;
- exposing live V2 response bytes to an invalid-capability client;
- accepting null citation URLs or internal citation identity fields;
- treating scanner outage as a post-user safe-failure writer;
- omitting cache isolation headers on capability-dependent outcomes.

Command:

```powershell
.venv-task4-r3-review/Scripts/python.exe -c "from backend.app.core.config import Settings; Settings.model_config['env_file']=None; import pytest; raise SystemExit(pytest.main(['backend/tests/test_assistant_delivery_contract.py','backend/tests/test_assistant_capability.py','-q','-p','no:cacheprovider','--basetemp=.tmp/task16-red-20260912-1']))"
```

Environment was set before imports to `PARAWORKS_DEMO_MODE=true`,
`DATABASE_URL=sqlite:///:memory:`, and
`PARAWORKS_DEMO_DATABASE_URL=sqlite:///:memory:`.

Observed RED: collection failed exactly because
`backend.app.assistant.delivery` and `backend.app.assistant.capability` did not
exist (2 expected missing-feature collection errors, 0 tests executed).
This proves that the feature modules were absent before production work; it is
not claimed as behavioral assertion RED.

## Retrospective Mutation Proof

After GREEN, the following temporary mutations were applied one at a time with
`apply_patch`, tested, and immediately restored. They are honest retrospective
coverage evidence, not relabeled preimplementation RED:

1. Returning `True` unconditionally from the delivery legal-matrix predicate
   caused the cross-combination test to fail with `Failed: DID NOT RAISE
   <class 'ValueError'>` (`1 failed in 0.64s`).
2. Disabling the raw capability mismatch condition caused all missing, wrong,
   duplicate, and comma-folded cases to fail with `Failed: DID NOT RAISE
   <class 'fastapi.exceptions.HTTPException'>` (`4 failed in 0.63s`).
3. Disabling the POST guard caused the real existing-conversation route test
   to return 200, create a user row, run retrieval/generation, and fail its
   expected 409 assertion (`1 failed in 1.00s`). This demonstrates that the
   route test detects the exact guard-before-mutation/provider break.

The exact focused mutation commands selected respectively:

```text
backend/tests/test_assistant_delivery_contract.py::test_assistant_delivery_result_rejects_every_cross_combination
backend/tests/test_assistant_capability.py::test_raw_asgi_render_capability_rejects_missing_duplicate_folded_or_wrong
backend/tests/test_assistant_capability.py::test_existing_conversation_capability_guard_precedes_message_mutation
```

Each used the explicit interpreter/pre-import environment wrapper,
`-p no:cacheprovider`, and a unique `.tmp/task16-mutation-*` basetemp. The
production implementation was restored before the final GREEN gate.

## GREEN Evidence

The same focused command after implementation, with fresh base temp
`.tmp/task16-green-20260912-1`, produced:

```text
33 passed in 2.86s
```

After the authentication and meaningful required-`-k` supplements, the
capability suite produced:

```text
22 passed in 3.18s
```

Task16 plus the complete existing Assistant API suite produced:

```text
65 passed in 8.58s
```

Adjacent Assistant/service/runtime/input contracts produced:

```text
68 passed in 32.15s
```

The required proportional Assistant API selector produced:

```text
3 passed, 26 deselected in 0.39s
```

All pytest invocations used `-p no:cacheprovider`, a fresh `.tmp` basetemp,
the explicit existing interpreter, the three environment variables above, and
`Settings.model_config['env_file']=None` before pytest imported the app.

## Implementation

- `backend/app/assistant/delivery.py`: frozen/slotted validated delivery
  algebra with exact legal matrix and positive non-bool persisted ids.
- `backend/app/assistant/capability.py`: exact raw-ASGI header guard,
  deployment-owned POST requirement query, minimal V2 structural/final-parent
  liveness inspection, and exact summary-contributor selection.
- `backend/app/schemas/assistant.py`: strict Unicode request validation and
  exact recursive `RagCitationResponse` reuse.
- `backend/app/api/v1/assistant.py`: precedence-preserving guard/scanner
  integration and capability-dependent response header isolation.
- `backend/tests/test_assistant_delivery_contract.py`: exhaustive legal and
  cross-combination construction tests.
- `backend/tests/test_assistant_capability.py`: raw headers, precedence,
  mutation/provider absence, row-sensitive GETs, and response headers.
- `backend/tests/test_assistant_api.py`: recursive public citation validation
  and meaningful owner/validation selector names.
- `docs/portfolio-log.md` and
  `docs/superpowers/runbooks/session-handoff.md`: candidate status and next
  ownership boundary.

## Concerns and Boundaries

- The V2 GET inspection intentionally verifies only stable structural markers,
  exact linked final parent/result identity, canned zero-dependency shape, and
  existing evidence liveness. Full HMAC/content/dependency revalidation and
  final redaction projection remain Task17.
- Task18 still owns facade/graph Assistant sole-writer delivery mapping,
  committed safe-failure persistence, terminal-error literal alignment, and
  removal of the enforce-path route/catch second writer.
- Verification is SQLite/fake/provider-free. No live provider, network,
  Docker, paid call, `.env` read, or real PostgreSQL claim was made.
- Final static/compile/diff/credential checks and final fresh combined test
  result are appended below before the scoped local commit.

## Final Verification

Fresh proportional combined gate after all production mutations were restored:

```text
133 passed in 40.76s
```

It covered the two new Task16 suites, complete Assistant API/service/model
suites, the V2 Assistant evidence writer, default RAG runtime, V1 response
contracts, and V2 input preparation.

The required focused Assistant selector was also rerun meaningfully:

```text
3 passed, 26 deselected in 0.39s
```

Final static and hygiene evidence:

```text
Ruff (7 changed Python files): All checks passed!
compileall (4 changed production files): exit 0
git diff --check: exit 0 (line-ending notices only)
credential-pattern scan (all changed files): 0 matches
```

The scoped local commit uses the required message
`feat: enforce assistant rag render capability`; its SHA is reported by the
implementer after Git creates the commit because a commit cannot contain its
own final SHA.

## Independent review fix round 1

The first independent review of `24f470e` was not clean. Four Important
findings were reproduced with real in-memory Assistant HTTP routes before
production changes: capability-dependent authentication, malformed-body, and
unexpected 500 responses lacked the required headers; escaped surrogates
crashed error rendering; GET classification disagreed with the serializer;
and malformed JSON ran before authentication.

The valid behavioral RED command was:

```powershell
$env:PARAWORKS_DEMO_MODE='true'; $env:DATABASE_URL='sqlite:///:memory:'; $env:PARAWORKS_DEMO_DATABASE_URL='sqlite:///:memory:'; .venv-task4-r3-review/Scripts/python.exe -c "from backend.app.core.config import Settings; Settings.model_config['env_file']=None; import pytest; raise SystemExit(pytest.main(['backend/tests/test_assistant_capability.py','-q','-p','no:cacheprovider','--basetemp=.tmp/task16-fix-red-20260912b']))"
```

It produced `10 failed, 22 passed in 6.12s`. Every failure was an expected
contract assertion; the earlier fixture attempt that violated a database
constraint was corrected and is not counted as RED evidence.

The scoped fix resolves the same FastAPI dependency graph before body parsing
only for the two capability-sensitive Assistant POST routes. The marker runs
before authentication, while the endpoint reuses the exact dependency-
overridden DB session, authenticated user, and settings object. A lifecycle
integration test proves one DB acquisition, one authentication call, the same
session/user at persistence, the same settings at both capability decisions,
and generator cleanup after the response. Email-send remains outside this
boundary.

The route boundary now uses ASCII-safe JSON serialization for the unchanged
FastAPI validation detail array, attaches exact cache headers to every
capability-dependent success/error including unexpected 500, and retains the
existing status/body. The minimal V2 structural-parent check moved beside the
existing service liveness decision so guard, serializer, context, and summary
all make one projection decision. Missing/invalid parents redact without V2
bytes. A final pre-candidate spec check corrected hidden-only canned handling:
structurally valid `rag_canned`/`none-v1` rows with bounded hidden metadata are
live, require capability, and retain original content/count/notice in both the
message projection and contributing summary.
Task17 full cryptographic evidence revalidation remains deferred.

That correction had its own behavioral RED before production change:
`test_hidden_only_canned_v2_requires_guard_and_preserves_original_projection`
failed because the missing-capability request returned 200 rather than 409.
After the shared liveness correction it passed `1 passed in 0.38s`.

First focused GREEN was `9 passed, 23 deselected in 1.90s`; the complete
capability suite then passed `32 passed in 5.23s`. Assistant/API/auth regression
was `99 passed in 14.60s`. The final proportional affected gate, including
Task16 contracts, complete Assistant API/service/models/evidence writer,
authentication, direct RAG delivery, V1 response contracts, and V2 input, was:

```text
207 passed in 69.95s
```

Static, compilation, diff, and credential hygiene results are recorded below
after their final fresh run:

```text
Ruff (4 changed Python files): All checks passed!
compileall (3 changed production files): exit 0
git diff --check: exit 0 (line-ending notices only)
credential-pattern scan (all 7 changed files): 0 matches
```

This is a review-fix candidate, not a CLEAN claim. The route boundary uses
FastAPI's dependency solver as a deliberately narrow integration point;
behavioral coverage freezes override reuse, object identity, one DB/auth
lifecycle, cleanup, precedence, and public responses without pinning a private
library version or source text.
