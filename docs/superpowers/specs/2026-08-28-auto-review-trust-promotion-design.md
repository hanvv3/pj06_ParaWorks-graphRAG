# Auto-Review Trust Promotion Design

검토 버전: 1
작성일: 2026-08-28
상태: 사용자 검토 대기

## 1. 결정 요약

ParaWorks는 모든 AI 지식 후보를 사람이 직접 승인하는 구조를 유지하지 않는다.
대신 원본 evidence, 미검토 AI 후보, trusted knowledge를 분리하고, 낮은 위험의
직접 사실만 독립 AI validator와 결정론적 정책을 모두 통과한 경우 자동 승인한다.

이 전달 단위의 이름은 **Deliverable C.5 — Auto-Review Trust Promotion**이다.
Deliverable C Review Queue HITL V2 다음, Deliverable D Retriever/RAG Answer Graph V2
이전에 구현한다.

핵심 결정은 다음과 같다.

1. 원본 메일·문서·캘린더는 AI가 만든 지식이 아니므로 사람 승인 대상이 아니다.
   다만 원본 evidence가 곧 공식 Decision, Timeline, History, Todo가 되는 것은 아니다.
2. AI가 만든 해석은 human approval 또는 versioned auto-review policy를 통과해야만
   trusted knowledge가 된다.
3. 생성 모델은 자신의 결과를 승인하지 않는다. 독립 validator가 근거 충실도를
   structured output으로 보고하고, 최종 권한은 deterministic policy engine이 가진다.
4. 첫 validator는 OpenAI API의 `gpt-5.6-terra`, reasoning effort `medium`이다.
5. 첫 자동 승인 allowlist는 직접 사실인 `timeline_event`와 제한적인
   `history_event`다. `decision_record`, `todo`, `restricted`, 추론·충돌 후보는 사람이
   검토한다.
6. 자동 승인도 기존 locked `ReviewTransitionService`와 exactly-once promotion을
   통과한다. knowledge table을 직접 쓰지 않는다.
7. 자동 승인된 결과는 사람이 취소할 수 있으며, 연결된 knowledge와 vector serving
   document를 정확히 revoke한다.
8. rollout mode는 `disabled`, `shadow`, `enforce`이고 기본값은 `disabled`다.
9. Slack 복구, CDC/streaming, Deliverable D retrieval cutover, Neo4j GraphRAG는 이
   전달 단위에서 제외한다.

이 문서의 정확한 threshold, cost cap, persistence/API/status 추가는 구현 전 검토를
위한 제안 계약이다. 사용자가 이 문서를 승인하면 해당 값을 C.5 구현 계획의 변경
통제 기준으로 고정한다. 승인 전에는 제품 코드나 database schema를 변경하지 않는다.

## 2. 현재 구현과 간극

현재 모든 agent candidate는 confidence가 높아도 `ReviewItem(status='pending_review')`로
저장된다. `ValidationAgent`는 evidence 존재, 필수 payload field, 최소 confidence만
검사하며 entailment, contradiction, version drift를 승인 수준으로 검증하지 않는다.

현재 승인 경계는 다음과 같다.

```text
AI candidate
  -> pending_review
  -> human Review action
  -> ReviewTransitionService row lock
  -> approved
  -> exactly-once knowledge promotion
```

이 구조는 안전하지만 데이터가 계속 쌓이면 사람이 모든 낮은 위험의 직접 사실까지
검토해야 한다. 반대로 모델의 자기 confidence만으로 승인하면 같은 오류가 생성과
승인에 동시에 전파된다.

또한 현재 `reviewer_id`만으로는 사람 승인과 정책 승인을 명확히 구분할 수 없고,
자동 승인 결과를 정확하게 revoke하는 공통 계약도 없다.

## 3. 범위

### 포함

- trust tier와 auto-review public/runtime contract
- deterministic eligibility와 policy engine
- 실제 LangChain `with_structured_output()` validator adapter
- `gpt-5.6-terra` validator model-router role
- validation lease, cache, cost accounting, audit metadata
- 새 immutable Review V2.1 graph의 pre-interrupt auto-review node
- 같은 `ReviewTransitionService`를 사용하는 policy actor
- auto-approved ReviewItem과 promoted knowledge의 정확한 revoke
- candidate별 immutable canonical evidence ref
- raw-chunk index eligibility 동결과 vector revoke tombstone
- shadow/canary/enforce rollout
- 기존 Review 화면 안의 count, badge, filter, revoke UX
- PostgreSQL concurrency/restart와 deterministic SQLite smoke

### 제외

- Decision과 Todo 자동 승인
- restricted evidence 자동 승인
- AI 자동 반려
- raw source를 trusted knowledge로 직접 승격
- Deliverable D `/ask`, `/search`, assistant retrieval cutover
- Neo4j, graph projection, `neo4j-graphrag`, Knowledge Map 변경
- Slack source/agent/fixture 변경
- CDC, outbox, broker, streaming consumer
- live provider를 호출하는 automated test

## 4. 신뢰 계층

### 4.1 Canonical source evidence

원본 Gmail, Drive, Calendar와 parser output이다. source id, immutable version 또는
content signature, permission, URL, snippet을 보존한다. 원본은 별도의 사람 승인 없이
evidence가 될 수 있지만 official knowledge가 아니다.

Deliverable C.5는 현재 RAG index eligibility를 넓히지 않는다. canonical raw evidence와
trusted knowledge를 함께 검색하는 정책은 Deliverable D에서 별도로 구현한다.

### 4.2 Pending AI knowledge

agent가 원본 evidence에서 추출했지만 아직 승인되지 않은 Timeline, History,
Decision, Todo 후보다. Review Queue에는 보이지만 trusted RAG/GraphRAG knowledge로
사용하지 않는다.

### 4.3 Trusted knowledge

다음 중 하나를 만족한 promoted knowledge다.

- authorized human approval
- 이 문서의 auto-review validation과 policy를 통과한 auto-policy approval

사람 승인과 정책 승인은 동일한 knowledge table과 `source_review_item_id` provenance를
사용하되 resolution source를 구분한다.

## 5. 초기 자동 승인 정책

정책 버전은 다음으로 고정한다.

```python
AUTO_REVIEW_POLICY_VERSION = 'auto-review-policy:v1'
AUTO_REVIEW_VALIDATOR_PROMPT_VERSION = 'auto-review-validation:v1'
```

### 5.1 Eligible item

`auto-review-policy:v1`의 제안 초기값은 아래 조건을 모두 만족해야 `auto_approve`를
허용한다.

1. item type이 `timeline_event` 또는 `history_event`다.
2. validator가 claim scope를 `direct_fact`로 반환한다.
3. permission이 `public` 또는 `internal`이다.
4. candidate에 canonical source/version ref와 current content signature가 있다.
5. 모든 substantive field가 하나 이상의 존재하는 evidence slot에 연결된다.
6. validator verdict가 `supported`다.
7. `entailment_score >= 0.98`이다.
8. uncertainty code와 conflict code가 모두 비어 있다.
9. source/version/permission이 모델 호출 전후에 동일하다.
10. policy/model/prompt version이 registry에서 지원된다.
11. cost budget과 bounded input 규칙을 만족한다.
12. candidate generation provider/model/prompt identity가 존재하고 validator의 exact
    provider/model identity와 다르다.

promotion에 실제 사용되는 normalized field만 substantive claim이다.

- `timeline_event`: `title`, `result_summary`
- `history_event`: `title`, `reason`

fallback 전의 원본 `summary`와 기타 operational payload metadata는 중복 claim으로 보내지
않는다. project selection 또는 LLM project routing이 필요한 후보, 알 수 없는 payload
schema, 빈 substantive field는 항상 `human_review`다.

`history_event`는 원문에 직접 적힌 사실만 허용한다. 원인, 의도, 책임, 영향 또는 여러
source를 조합한 해석은 `human_review`다.

### 5.2 Forced human review

다음 중 하나라도 만족하면 Terra를 호출하지 않거나 결과와 무관하게 사람에게 보낸다.

- `decision_record` 또는 `todo`
- `restricted` 또는 unknown permission
- uncertainty reason 존재
- missing/empty source link 또는 snippet
- mutable source의 version/signature 불일치
- 제안, 예정, 조건, 추정, 의견을 확정 사실로 바꿀 가능성
- 기존 trusted knowledge와 conflict 가능성
- validator result가 malformed, partial 또는 unknown-slot
- 지원되지 않는 policy/model/prompt version
- candidate generator identity 누락 또는 validator와 exact same provider/model

### 5.3 Needs more evidence

canonical source가 존재하지만 version이 바뀌었거나 필수 direct evidence가 사라진
후보는 `needs_more_evidence`로 보낸다. foreign/unauthorized source는 존재 여부를
노출하지 않고 auto approval을 금지한다.

### 5.4 Duplicate reaffirmation

동일 normalized claim fingerprint와 동일 target type의 trusted knowledge가 이미 있을
때 새 knowledge row를 만들지 않는다. 기존 canonical knowledge id를 재사용하고 새
evidence provenance만 연결한다.

의미 유사도만으로 중복을 판정하지 않는다. `v1`은 schema-versioned normalized field와
keyed HMAC이 정확히 같은 경우만 reaffirmation으로 인정한다.

trusted conflict preflight는 같은 security scope, item type, project key, normalized title
collision bucket을 두 단계로 조회한다. 먼저 current actor가 볼 수 있는 trusted knowledge
중 exact claim fingerprint 하나만 있으면 `reuse_trusted`, visible bucket이 비어 있으면
다음 단계로 간다. 그 뒤 server-side scope-wide `EXISTS` guard가 inaccessible permission의
collision 존재 여부만 확인한다. hidden collision이 하나라도 있거나 visible fingerprint가
다르거나 lookup이 실패/timeout/ambiguous하면 `human_review`다. hidden row content, id,
permission, count는 읽어 모델에 보내거나 API/audit reason으로 노출하지 않는다. `v1`은
semantic similarity로 “충돌 없음”을 추정하지 않는다.

legacy knowledge의 security scope를 originating workflow에서 정확히 복원할 수 없으면
duplicate reuse 대상이 아니며, 같은 normalized title의 legacy row는 보수적으로 conflict로
취급해 사람이 검토한다.

재확인 ReviewItem도 같은 locked transition service를 통과해 `approved`와
`resolution_source='auto_policy'`를 기록한다. promotion result는 새 row id가 아니라
재사용한 canonical knowledge id와 이 ReviewItem 전용 provenance link를 반환한다.

모든 post-migration approval은 최초 promotion인지 reaffirmation인지와 관계없이 별도의
active approval/evidence link를 가진다. 한 reaffirmation을 revoke하면 그 ReviewItem의
link만 비활성화한다. 같은 canonical knowledge에 다른 active human/auto approval
provenance 또는 migration 이전 human `source_review_item_id`가 남아 있으면 knowledge와
vector document는 계속 serving한다. 마지막 active trusted provenance가 사라질 때만
canonical knowledge와 companion Timeline 및 vector document를 revoke한다.

## 6. 전체 아키텍처

```text
draft_review_candidates_transaction
  -> auto_review_eligibility
  -> canonical_evidence_preflight
  -> validation lease
  -> bounded Terra structured validation
  -> canonical evidence revalidation
  -> deterministic policy evaluation
  -> validation decision persistence
  -> policy Review transition / duplicate reaffirmation
  -> refresh_review_resolution
  -> route_review_boundary
       -> pending exists -> await_human_review [interrupt]
       -> needs_more_evidence -> finalize_needs_more_evidence
       -> all resolved -> finalize_auto_resolved
```

모델 호출 중 DB row/advisory lock을 유지하지 않는다. checkpoint에는 source content,
URL, snippet, model output, validation rationale를 넣지 않는다.

### 6.1 Immutable graph version

기존 `company-memory-review-v2.0` builder, state schema, status count, registry entry는 수정하지
않는다. 이미 생성되거나 pause된 V2.0 thread는 배포 후에도 V2.0으로만 resume한다.

C.5는 새 graph version을 사용한다.

```text
company-memory-review-v2.1-auto-review
```

`AUTO_REVIEW_MODE=disabled`이면 신규 실행도 V2.0을 선택하며 provider call은 0이다.
`shadow` 또는 `enforce`인 신규 실행만 V2.1을 선택한다. 시작 시 선택한 graph version,
mode, policy/prompt/model version, enforce percentage를 workflow request에 immutable하게
저장하고 resume 시 현재 환경 설정으로 바꾸지 않는다. Registry는 V2.0과 V2.1을 동시에
유지하며 알 수 없는 version은 fail closed한다.

운영 rollback은 저장된 mode를 더 강하게 만들 수 없지만 약화할 수 있다. effective mode는
stored mode, current global mode, persisted rollout breaker가 허용하는 mode 중 가장
보수적인 값(`disabled < shadow < enforce`)이다.
따라서 global `disabled`는 pause/resume 중인 V2.1에서도 즉시 zero-call human-review
경로가 되고, 이후 설정을 `enforce`로 올려도 originally-shadow thread가 자동 승인으로
승격되지는 않는다.

## 7. 컴포넌트

### 7.1 `AutoReviewEligibilityService`

모델 호출 전 deterministic pre-check를 수행한다.

- allowlist item type
- exact permission allowlist
- canonical evidence ref/version/signature
- required payload field와 evidence 존재
- uncertainty/high-risk cue
- exact duplicate fingerprint
- policy/model readiness와 cost budget

반환값은 `eligible`, `human_review`, `needs_more_evidence`,
`reuse_trusted` 중 하나와 bounded reason code다.

### 7.2 `AutoReviewValidator`

```python
class AutoReviewValidator(Protocol):
    def validate_many(
        self,
        requests: list[CandidateValidationRequest],
    ) -> list[CandidateValidationResult]: ...
```

production adapter는 실제 LangChain `with_structured_output()`을 사용한다. tests와 local
deterministic smoke는 fake adapter를 사용한다.

### 7.3 `AutoReviewPolicyEngine`

I/O dataclass만 받는 pure function이다. DB, model, network를 호출하지 않는다.

```python
AutoReviewPolicyDecision = Literal[
    'auto_approve',
    'human_review',
    'needs_more_evidence',
    'reuse_trusted',
]
```

AI validator는 승인 권한이 없다. policy engine만 versioned code rule로 위 값을
결정한다.

### 7.4 `AutoReviewResolutionService`

`auto_approve`를 받으면 ReviewItem row를 lock하고 source/version/permission, status,
validation key, policy version을 다시 확인한 뒤 기존 transition service를 호출한다.
knowledge table에 직접 insert하지 않는다. `reuse_trusted`도 별도 우회 update가 아니라
같은 service의 명시적 reaffirmation transition을 사용한다.

public `ReviewAction`에 모델이 호출할 수 있는 새 action을 추가하지 않는다. 대신 locked
service의 internal-only approval directive를 `create_new` 또는
`reuse_existing(expected_type, expected_id, expected_claim_fingerprint)`로 확장한다. human
API는 기존 `approve`와 `create_new`를 사용하고, auto service만 policy decision에 맞는
directive를 전달한다. service는 lock 아래 fingerprint/permission/active status를 다시
검사하고 ReviewItem resolution과 promotion/evidence link를 한 transaction에서 쓴다.

`reuse_existing`은 새 knowledge row를 요구하지 않으며 canonical reused ids와
`promotion_effect='reaffirmed'`를 반환한다. `create_new`는 현재 exactly-once insert/index
규칙을 유지한다. 두 경로 모두 approved ReviewItem replay가 동일 canonical result를
반환해야 하며, 기존 `source_review_item_id` 1:1 provenance를 깨지 않는다.

### 7.5 `AutoReviewValidationStore`

validation claim/lease, persisted result, replay, timeout recovery를 담당한다. 동일
validation key는 한 canonical result만 만든다.

### 7.6 `TrustedKnowledgeEvidenceLinkService`

모든 post-migration human/auto approval effect에 ReviewItem별 active provenance를 만들고,
정확한 duplicate reaffirmation이 기존 knowledge row에 새 evidence provenance를 추가할
때도 사용한다. raw content를 복제하지 않고 canonical source/version ref, evidence HMAC,
permission snapshot, resolution actor type, originating ReviewItem을 연결한다.

### 7.7 `AutoReviewPostAuditService`

enforce promotion ordinal과 stable sample HMAC으로 post-audit 대상을 같은 promotion
transaction에서 만든다. authorized human의 `confirmed` 또는 critical outcome을 기록하고,
critical이면 persistent rollout breaker를 먼저 연 뒤 exact revoke를 수행한다.

### 7.8 `AutoReviewRolloutPolicyService`

security scope와 policy version별 shadow/enforce evidence, enforce ordinal, pending audit,
critical issue를 집계한다. config가 `enforce`여도 breaker가 open이거나 gate가 부족하면
effective mode를 `shadow`로 제한한다. 이 service는 model output이 아니라 persisted human
audit과 deterministic counters만 사용한다.

## 8. Validator input과 output

### 8.1 Input

```python
CandidateSlotId = Annotated[str, Field(pattern=r'^C0[1-8]$')]
EvidenceSlotId = Annotated[str, Field(pattern=r'^E(?:0[1-9]|1[0-2])$')]


@dataclass(frozen=True)
class CandidateValidationRequest:
    candidate_slot_id: CandidateSlotId
    item_type: Literal['timeline_event', 'history_event']
    claims: tuple[ValidationClaimInput, ...]
    evidence_slots: tuple[ValidationEvidenceSlot, ...]


@dataclass(frozen=True)
class ValidationClaimInput:
    field_key: Literal['title', 'result_summary', 'reason']
    text: str


@dataclass(frozen=True)
class ValidationEvidenceSlot:
    slot_id: EvidenceSlotId
    text: str
```

모델에는 request-local slot과 permission-filtered bounded text만 전달한다. canonical row
id, external source id, URL, permission metadata, credential은 전달하지 않는다. claims는
`build_promotion_preview()`가 실제로 promotion할 normalized field에서 server-side로 만들며
모델이 field key를 추가하거나 바꿀 수 없다.

### 8.2 Structured output

```python
UncertaintyCode = Literal[
    'ambiguous_subject',
    'ambiguous_time',
    'conditional_language',
    'partial_evidence',
    'proposal_language',
    'unknown',
]
ConflictCode = Literal[
    'source_contradiction',
    'trusted_claim_conflict',
    'unknown',
]


class FieldValidationResult(BaseModel):
    model_config = ConfigDict(extra='forbid')

    field_key: Literal['title', 'result_summary', 'reason']
    verdict: Literal[
        'supported',
        'partially_supported',
        'unsupported',
        'contradicted',
    ]
    claim_scope: Literal['direct_fact', 'inference', 'decision', 'todo', 'unknown']
    entailment_score: Decimal = Field(
        ge=Decimal('0'),
        le=Decimal('1'),
        max_digits=5,
        decimal_places=4,
    )
    evidence_slot_ids: list[EvidenceSlotId] = Field(max_length=12)


class CandidateValidationResult(BaseModel):
    model_config = ConfigDict(extra='forbid')

    candidate_slot_id: CandidateSlotId
    claim_results: list[FieldValidationResult] = Field(min_length=2, max_length=2)
    uncertainty_codes: list[UncertaintyCode] = Field(max_length=4)
    conflict_codes: list[ConflictCode] = Field(max_length=4)


class CandidateValidationBatchResult(BaseModel):
    model_config = ConfigDict(extra='forbid')

    results: list[CandidateValidationResult] = Field(min_length=1, max_length=8)
```

자유 형식 rationale은 받지 않는다. persisted candidate key/ReviewItem id는 provider에
보내지 않고 call-local `C01`~`C08` alias로 치환한다. request/result candidate slot set과
각 candidate의
expected field key set은 정확히 일치해야 한다. 모든 substantive field가 각각
`supported`, `direct_fact`, non-empty known unique evidence slot, score
`>= Decimal('0.9800')`여야 한다. score는 반올림하거나 epsilon을 적용하지 않으며 정확히
`0.9800`은 통과하고 그보다 작으면 `human_review`다.

missing/duplicate candidate slot, missing/duplicate/unknown field, unknown/duplicate slot,
extra field, invalid decimal, code/list overrun, unrecognized code는 validation failure다. literal
`unknown` code는 schema-valid이지만 항상 `human_review`다. structured
batch 하나라도 malformed면 연결 오류를 피하기 위해 그 batch 전체를 `human_review`로
보내며 attacker-controlled value는 persistence/audit에 기록하지 않는다.

evidence alias는 batch-local `E01`~`E12`이고 batch 전체에서 최대 12개다. candidate당
substantive claim은 정확히 2개, claim text는 2,000자 이하, 모든 claim/evidence text의
serialized provider input은 합계 12,000자 이하다. alias mapping은 call memory에만 두고
checkpoint, API, AuditLog에 넣지 않는다.

## 9. Model policy

### 9.1 Live validator

```text
provider: openai
model: gpt-5.6-terra
reasoning.effort: medium
structured output: required
validator prompt version: auto-review-validation:v1
```

Terra는 candidate generation을 수행하지 않고 validation만 담당한다. provider/model은
model-router boundary로 주입한다.

### 9.2 No silent fallback

Terra가 unavailable, timeout 또는 malformed output이면 Luna, Sol, Gemini로 자동
fallback하지 않는다. 다른 model은 새 golden evaluation과 새 policy version이 필요하다.
실패 후보는 사람 Review Queue로 보낸다.

### 9.3 Offline roles

- `gpt-5.6-sol`: golden-set benchmark와 audited sample 비교
- `gpt-5.6-luna`: 충분한 데이터 이후 별도 policy의 high-volume first-pass 후보

둘 다 `auto-review-policy:v1` enforce path에는 사용하지 않는다.

## 10. Persistence model

### 10.1 Workflow and `ReviewItemEvidenceRef`

V2.1 workflow request persistence에 다음 immutable field를 추가한다.

- `auto_review_mode`: `shadow`, `enforce`
- validator provider/model/reasoning effort
- validator prompt version과 policy version
- `enforce_percentage`
- confirmed extraction/validation/total cost ceiling

새 `ReviewItemEvidenceRef` model과 `review_item_evidence_refs` table은 candidate를 기존
`AgentWorkflowEvidenceRef`에 정확히 연결한다.

- `id`
- `review_item_id`
- `workflow_evidence_ref_id`
- `candidate_slot_ordinal`
- `message_content_fingerprint`
- `created_at`

`(review_item_id, workflow_evidence_ref_id)`와
`(review_item_id, candidate_slot_ordinal)`은 unique다. candidate draft와 같은 transaction에서
만들고 update하지 않는다. source text, URL, snippet은 이 table에 복제하지 않는다.

validator 직전에 이 ref를 canonical source resolver로 다시 읽어 current version,
signature, permission, content fingerprint를 확인하고 request-local slot을 만든다. V2.0 또는
migration 이전 ReviewItem처럼 exact ref가 없는 후보, packet 전체와 candidate ref가
불일치하는 후보는 auto-review eligible이 아니며 사람에게 보낸다.

### 10.2 `AutoReviewValidation`

새 `auto_review_validations` table은 다음 field를 가진다.

- `id`
- `review_item_id`
- `validation_key`, unique
- `evidence_version_hash`
- `candidate_generation_fingerprint`
- `status`: `claimed`, `completed`, `failed`, `expired`
- `validator_provider`
- `validator_model`
- `reasoning_effort`
- `validator_prompt_version`
- `policy_version`
- bounded per-field `claim_results`
- derived `minimum_entailment_score`
- bounded `uncertainty_codes`
- bounded `conflict_codes`
- `policy_decision`
- bounded `policy_reason_codes`
- `input_tokens`, `output_tokens`, `estimated_cost_usd`, `cache_hit`
- `lease_token`, `lease_expires_at`
- `shadow_comparison_status`, nullable
- `shadow_human_resolution`, nullable
- bounded `shadow_exclusion_code`, nullable
- `shadow_compared_at`, nullable
- `created_at`, `completed_at`

raw evidence, URL, snippet, model rationale, provider exception text를 저장하지 않는다.

### 10.3 `ReviewItem` additions

- `resolution_source`: `human`, `auto_policy`, nullable
- `resolution_policy_version`, nullable
- `auto_validation_id`, nullable FK
- `revoked_at`, nullable
- `revoked_by_subject_id`, nullable
- bounded `revocation_reason`, nullable

`revoked`를 ReviewItem/public Review resolution status에 추가한다. V2.1 전용 state/count
schema는 `revoked`를 resolved-but-not-approved로 계산한다. V2.0 graph state/status union과
builder는 그대로 유지하며, revoke endpoint는 `resolution_source='auto_policy'`인 V2.1
item만 허용하므로 기존 paused V2.0 thread에 `revoked`가 나타나지 않는다. 기존 row는
nullable field를 그대로 유지한다. `auto_validation_id`는 같은 ReviewItem을 가리키는
completed canonical validation만 참조할 수 있어야 한다.

auto policy actor는 다음으로 기록한다.

```text
reviewer_id = system:auto-review
resolution_source = auto_policy
resolution_policy_version = auto-review-policy:v1
```

### 10.4 `TrustedKnowledgeEvidenceLink`

정확한 promotion/reaffirmation/revoke를 위해 다음 provenance table을 추가한다.

- `knowledge_type`
- `knowledge_id`
- `review_item_id`
- `security_scope_id`
- `promotion_effect_kind`
- `resolution_source`: `human`, `auto_policy`
- `normalization_schema_version`
- keyed `normalized_claim_fingerprint`
- canonical source kind/id/version-or-signature
- `permission_level`
- keyed `evidence_hash`
- `status`: `active`, `revoked`
- `revoked_at`, nullable
- `created_at`

`(knowledge_type, knowledge_id, review_item_id, promotion_effect_kind)`와
`(knowledge_type, knowledge_id, canonical source ref, evidence_hash, review_item_id)`는
unique다. 최초 human/auto promotion과 companion Timeline도 link를 만들며 source text와
URL을 복제하지 않는다. 기존 human-approved row는 이 migration으로 합성 provenance를
만들지 않고 기존 `source_review_item_id` 자체를 active human base provenance로 간주한다.
기존 human item과 그 knowledge는 C.5 revoke action의 대상이 아니다.

### 10.5 Resolution actor

사람인 것처럼 가짜 `DemoUser`를 만들지 않는다.

```python
@dataclass(frozen=True)
class ReviewResolutionActor:
    subject_id: str
    actor_type: Literal['human', 'auto_policy']
    allowed_permission_levels: tuple[str, ...]
    capabilities: frozenset[
        Literal['human_review', 'auto_review', 'auto_review_rollout_admin']
    ]
    policy_version: str | None = None
```

기존 DemoUser는 기존 RBAC 검사를 통과한 뒤 human adapter로 변환한다. auto-policy actor는
public API body/dependency에서 만들 수 없고 `AutoReviewResolutionService`만 registry에서
구성한다. allowed level은 정확히 `('public', 'internal')`이며 restricted를 포함하지
않는다.

`auto_review_rollout_admin`은 existing admin RBAC를 통과한 human operator adapter에만
부여하고 auto-policy actor에는 부여하지 않는다.

`ReviewTransitionService`와 AuditLog writer는 이 application-owned actor contract를 받도록
좁게 refactor한다. human action은 `human_review`, auto approval/reaffirmation은
`auto_review` capability와 matching policy version을 요구한다. auto actor는 reject,
needs-more-evidence human note, bulk public action을 실행할 수 없다.

### 10.6 `VectorServingTombstone`

마지막 active trusted provenance를 revoke할 때 document id별 tombstone을 같은 PostgreSQL
transaction에 기록한다.

- `document_id`, unique
- `source_review_item_id`
- `reason_code`
- `revoked_at`

C.5 auto approval은 promoted `history_event:{id}` 및 `timeline_event:{id}`만 vector serving
대상으로 만들며 raw `chunk:{id}` eligibility를 새로 만들지 않는다. 기존 source-chunk
indexing 조건은 `approved AND (resolution_source='human' OR resolution_source IS NULL)`로
고정한다. 따라서 auto-policy approval은 trusted promoted knowledge만 index하고 raw source
chunk는 Deliverable D 전까지 현재 human-approved 경계를 유지한다.

vector writer에 `delete_many(document_ids)` contract를 추가한다. revoke transaction은
knowledge status와 tombstone을 먼저 기록하고, pgvector row 및 모든 embedding model의
`VectorIndexState`를 exact document id로 삭제한다. reindexer는 embedding 전과 upsert 직전
새 transaction에서 knowledge `review_status='approved'`와 tombstone 부재를 다시 확인한다.
따라서 reindex가 먼저 끝나면 revoke가 지우고, tombstone이 먼저 생기면 늦은 reindex가
skip하여 revoked content를 되살릴 수 없다. deterministic in-memory store도 같은 delete와
rebuild exclusion contract를 구현한다.

### 10.7 `AutoReviewPostAudit`

enforce에서 자동 승인된 sampled item마다 다음 immutable selection과 human outcome을
저장한다.

- `id`
- `review_item_id`, unique
- `security_scope_id`
- `policy_version`
- `promotion_ordinal`
- `sample_cohort`: `first_50`, `sample_10`, `sample_2`, `manual`
- keyed `sample_fingerprint`
- `status`: `pending`, `completed`, `remediation_required`
- `outcome`: `confirmed`, `incorrect`, `permission_violation`,
  `source_version_violation`, `policy_violation`, nullable
- `remediation_code`: `revoke_failure`, `provenance_incomplete`,
  `vector_delete_failure`, `policy_regression_unverified`, nullable and system-only
- `auditor_subject_id`, nullable
- bounded `audit_reason`, nullable
- `created_at`, `audited_at`

sample row는 promotion과 같은 transaction에서 생성하므로 자동 승인 후 audit queue에서
누락될 수 없다. outcome은 한 번만 확정한다. 잘못 제출한 critical outcome도 덮어쓰지
않으며 별도 operator remediation/audit 절차를 거치기 전에는 breaker를 닫을 수 없다.

shadow의 predicted `auto_approve`는 `AutoReviewValidation`에 bounded comparison status,
human resolution, exclusion code, compared timestamp를 기록한다. human transition 시
evidence version이 그대로인 경우만 completed comparison이며 drift/unresolved item은
분모에서 제외한다.

### 10.8 `AutoReviewRolloutState`

`(security_scope_id, policy_version)` unique row에 다음 deterministic aggregate를 둔다.

- shadow predicted/completed/supported count
- enforce promotion ordinal
- post-audit selected/completed/critical count
- `breaker_open`, bounded `breaker_reason_code`, `breaker_opened_at`
- optimistic `state_version`
- `created_at`, `updated_at`

promotion/comparison/audit는 이 row를 lock하고 replay-safe하게 counter를 갱신한다. breaker는
critical audit가 기록되는 첫 transaction에서 먼저 open/commit한다. 두 번째 transaction이
exact revoke를 수행하고 성공 시 audit를 completed, 실패 시 `remediation_required`로
남긴다. 따라서 revoke failure여도 다음 request부터 effective mode는 즉시 `shadow`다.
breaker는 자동으로 닫히지 않으며, 별도 authorized operator remediation과 bounded audit
reason이 있어야만 닫을 수 있다.

breaker close는 public Review action이 아니라 admin-only rollout service/CLI command다.
`auto_review_rollout_admin` capability, 1~500자 reason, 모든 `remediation_required` item의
해결, 영향 item revoke/adjudication, 관련 regression gate reference를 요구한다. close는
AuditLog에 기록하지만 effective mode는 `shadow`에 남는다. enforce 복귀는 이후 별도 human
운영 결정과 config 변경이 필요하다.

## 11. Transaction, concurrency, cache

### 11.1 Validation key

```text
HMAC(
  candidate_key
  + evidence_version_hash
  + normalized_claim_fingerprint
  + candidate_generation_fingerprint
  + validator_provider/model/reasoning_effort
  + validator_prompt_version
  + policy_version
)
```

canonical JSON, UTF-8 NFC, stable ordering, current fingerprint key version을 사용한다.

### 11.2 Lease sequence

1. 짧은 transaction에서 ReviewItem, immutable candidate evidence refs, exact duplicate,
   trusted collision bucket, existing validation을 확인한다.
2. candidate ref를 canonical resolver로 읽고 eligibility/conflict가 명확한 경우에만
   validation key 단위 lease를 claim하고 commit한다.
3. transaction 밖에서 Terra를 호출한다.
4. 새 transaction에서 lease/status, exact duplicate/conflict result,
   source/version/permission/content fingerprint를 모두 재검증한다.
5. bounded structured validation과 deterministic policy decision을 저장한다.
6. auto approval/reaffirmation이면 ReviewItem과 target knowledge row를 lock하고 같은
   transition/promotion service를 사용한다.

동일 validation의 경쟁자는 canonical completed result를 재사용한다. expired lease만 CAS로
재claim한다.

### 11.3 Bounded work와 cost

- 한 provider call당 후보 최대 8개
- evidence slot 최대 12개
- provider input 최대 12,000자
- estimated batch cost 최대 USD 0.02
- workflow auto-review estimated cost 최대 USD 0.20

cap을 넘은 후보는 사람 Review Queue로 보낸다. cap 변경은 자동 승인 allowlist를
확장하지 않지만 cost-policy version과 regression evidence를 갱신해야 한다.
위 금액은 `gpt-5.6-terra`와 현재 bounded fixture를 위한 제안 초기 cap이며, 문서 승인으로
고정한 뒤 offline token fixture에서 cap 이하임을 검증해야 한다.

### 11.4 Explicit paid-run confirmation

`shadow`도 Terra를 호출하므로 status API, sync polling, page load, dry-run은 provider를
절대 호출하지 않는다. 기존 Integrations의 `검토 후보 미리보기 -> 검토 후보 만들기`
명시적 실행 안에서만 validation을 수행한다.

V2.1 dry-run은 extraction, auto-review upper bound, total estimated tokens/cost, selected mode,
policy/model version을 같은 panel에 표시하고 short-lived HMAC launch confirmation token을
반환한다. token은 input/evidence hash, graph/mode/policy/model, cost ceilings, expiry를 묶는다.
클라이언트는 사용자가 기존 실행 버튼을 눌렀을 때 token을 자동 전송한다. 별도 page,
wizard, 두 번째 확인 click은 추가하지 않는다. preview 이후 값이 바뀌거나 token이
만료되면 start는 `cost_preview_changed`로 fail closed하고 새 preview를 요구한다.

token은 새로운 idempotency namespace가 아니다. 같은 signed preview의 concurrent/retry
start는 기존 security-scope exact-batch ownership과 `client_request_id` 규칙으로 한
workflow에 수렴하고 canonical status를 반환한다. token replay만으로 새 thread나 두 번째
provider call을 만들 수 없다.

## 12. Review Graph integration

새 V2.1 Review graph에만 `run_auto_review`와 `refresh_review_resolution`을 후보 draft 뒤,
review boundary 앞에 추가한다. V2.0 topology는 변경하지 않는다.

```text
draft -> run_auto_review -> refresh -> route
```

- effective `disabled`: auto-review node는 zero-call pass-through
- `shadow`: validation/decision/cost만 저장하고 ReviewItem status는 바꾸지 않음
- `enforce`: stable candidate HMAC으로 enforce percentage에 선택된 eligible result만
  transition하고 나머지는 shadow처럼 pending 유지

route 우선순위는 다음으로 고정한다.

1. `pending_review > 0`: 기존 `interrupt()`로 사람에게 남은 예외를 보낸다.
2. pending이 0이고 `needs_more_evidence > 0`: `finalize_needs_more_evidence`.
3. 나머지 `approved + rejected + revoked == total`: `finalize_auto_resolved` 또는 normal
   completed.

따라서 모든 후보가 auto resolved면 interrupt가 없고, needs-more와 pending이 함께 있으면
pending을 먼저 사람이 처리한다.

`revoked`는 resolved-but-not-approved 상태다. 이미 완료된 workflow를 재개하거나 과거
checkpoint를 수정하지 않으며, live serving eligibility와 현재 Review projection에서만
즉시 제외한다.

V2.1 checkpoint state에는 다섯 status의 bounded count, phase, completed node, allowlisted
error code만 둔다.
validation row id, ReviewItem id, source ref/content, model output을 public summary나
checkpoint에 추가하지 않는다.

## 13. Auto-approval revoke

새 action은 `revoke_auto_approval`이다.

허용 조건:

- actor가 reviewer/admin이고 현재 item permission을 검토할 수 있음
- ReviewItem status가 `approved`
- `resolution_source == 'auto_policy'`
- linked validation/promotion provenance가 완전함

효과:

1. ReviewItem status를 `revoked`로 변경
2. 이 ReviewItem의 approval/evidence link를 `revoked`로 변경
3. 같은 canonical effect에 active trusted provenance가 남았는지 row lock 아래 재계산
4. 남아 있지 않을 때만 linked Decision/History/Timeline/Todo와 companion Timeline의
   `review_status`를 `revoked`로 변경
5. 4번이 실행된 경우에만 exact vector document id와 vector index state를 삭제
6. bounded audit event 기록

row를 물리 삭제하지 않는다. 사람이 승인한 item은 이 action으로 revoke할 수 없다.
동일 revoke 재요청은 canonical result와 `replayed=true`를 반환한다.
공유 knowledge에 active provenance가 남은 replay/response는
`knowledge_remains_trusted=true`를 반환하되 다른 ReviewItem id는 노출하지 않는다.

`revoked`는 V1 terminal 상태이며 같은 ReviewItem을 다시 approve하지 않는다. 복원하려면
새 canonical source version에서 새 candidate를 만들고 사람이 승인해야 한다. reason은
ReviewItem에 bounded하게 저장해 authorized detail에서만 보이며 AuditLog에는 raw reason을
복제하지 않는다. revoke의 knowledge/link/vector/tombstone 변경 중 하나라도 실패하면 같은
transaction 전체를 rollback한다.

waiting workflow에서 revoke는 resolved non-approved 상태로 계산한다. 이미 completed인
workflow의 status를 되돌리지 않지만 trusted serving 결과는 즉시 제외한다.

## 14. API 계약

기존 response field를 제거하거나 이름을 바꾸지 않는다.
이 절의 field, `revoked` status, action endpoint는 additive public contract이며 이 문서의
사용자 승인이 schema/trust-boundary human gate다.

### 14.1 Review item additions

- `resolution_source`
- `resolution_policy_version`
- bounded `auto_review_summary`
  - validator model
  - reasoning effort
  - prompt/policy version
  - supported substantive field count와 minimum entailment score
  - policy reason codes
  - validation timestamp

source content, raw model output, validation key, lease, internal DB id는 노출하지 않는다.

### 14.2 Workflow summary additions

- `auto_review_mode`
- `auto_review_policy_version`
- `auto_review_enforce_percentage`
- `auto_approved_count`
- `human_review_required_count`
- `auto_review_fallback_count`

이 field는 V2.1 status model에만 존재한다. V2.0 dry-run/status Pydantic model과 exact
response shape는 현재 그대로 유지한다. route는 `graph_version` discriminated response
union과 version-specific mapper를 사용하고 frontend도 같은 union으로 narrow한다. count만
반환하고 hidden ReviewItem identity를 노출하지 않는다.

### 14.3 Revoke endpoint

```text
POST /api/v1/review/{review_item_id}/revoke-auto-approval
```

request는 1~500자의 reviewer reason을 요구한다. foreign/inaccessible item은 기존 Review
API의 concealment 규칙을 유지하고, unsupported transition은 bounded 409를 반환한다.

response는 `review_item_id`, `status='revoked'`, `replayed`,
`knowledge_remains_trusted`, `revoked_document_count`만 반환한다. 다른 provenance identity나
document id는 노출하지 않는다.

### 14.4 V2.1 dry-run and launch additions

dry-run `graph_version`은 V2.0/V2.1 union이며 다음 additive field를 가진다.

- `auto_review_mode`
- `auto_review_policy_version`, nullable
- `auto_review_estimated_input_tokens`
- `auto_review_estimated_output_tokens`
- `auto_review_estimated_cost_usd`
- `total_estimated_cost_usd`
- `launch_confirmation_token`, V2.1에서만 non-null

V2.1 run request에는 preview가 반환한 `launch_confirmation_token`이 필수다. V2.0에서는
field를 생략한다. token 불일치/만료는 새 bounded error code `cost_preview_changed`로 409를
반환한다. diagnostic, dry-run, status, resume polling은 어떤 mode에서도 paid provider를
호출하지 않는다.

`cost_preview_changed`를 backend/frontend V2 error-code union에 명시적으로 추가한다.
V2.0 response mapper와 exact-shape regression은 새 field를 받지 않는다.

### 14.5 Post-audit action

```text
POST /api/v1/review/{review_item_id}/auto-review-audit
```

authorized reviewer/admin만 auto-policy-approved item에 `confirmed` 또는 위 human critical outcome
중 하나와 1~500자 reason을 제출할 수 있다. sampled row가 있으면 이를 resolve하고,
unsampled item의 명시적 감사는 `manual` cohort를 만든다. response는 bounded audit status,
`breaker_open`, revoke status만 반환하고 rollout counter, hidden provenance, 다른 item id를
노출하지 않는다. critical outcome은 breaker-first/revoke-second 순서를 사용한다.

## 15. UX

새 페이지, wizard, modal depth를 만들지 않는다.

기존 workflow panel은 다음 count를 표시한다.

```text
자동 승인 N건 | 확인 필요 M건 | 추가 근거 필요 K건
```

기존 Integrations preview의 단일 비용 영역은 후보 생성 비용, 자동 검증 최대 비용, 총 최대
비용을 함께 보여준다. 사용자가 같은 `검토 후보 만들기` 버튼을 누르는 동작이 signed
preview의 명시적 paid-run 확인이며 navigation depth는 늘지 않는다.

Review Queue 기본 목록에는 pending만 남는다. 기존 filter에서 `자동 승인`을 선택하면
감사할 수 있다. 상세 drawer는 `자동 검증 승인` badge, model/policy version, 검증 시각,
bounded reason code와 revoke action을 보여준다.

sampled auto approval은 같은 filter/drawer에서 `감사 필요` badge와 inline audit action을
보인다. 새 page나 wizard를 만들지 않는다. critical audit 뒤 revoke가 실패한 item은
`조치 필요` badge를 유지하고 일반 confirmed 처리로 숨길 수 없다.

Timeline, History, Knowledge 화면은 `사람 승인` 또는 `자동 검증` badge를 표시하고 기존
source evidence drawer를 유지한다.

## 16. Permission, security, privacy

- current actor의 exact allowed permission levels를 사용하며 role에서 재추론하지 않는다.
- source, ReviewItem, knowledge 중 strictest permission을 유지한다.
- unknown permission은 fail closed한다.
- restricted는 `v1` auto approval에서 제외한다.
- provider input 전에 permission filter를 적용한다.
- model result 이후 새 DB session에서 permission과 version을 재확인한다.
- candidate evidence ref가 없거나 canonical workflow ref와 정확히 일치하지 않으면 모델을
  호출하지 않는다.
- trusted collision 조회도 current actor/security scope와 exact permission filter 안에서만
  수행하고 hidden record identity/count를 노출하지 않는다.
- prompt injection text는 evidence data일 뿐 instruction으로 취급하지 않는다.
- raw source, URL, snippet, model rationale, provider error는 checkpoint/audit에 넣지 않는다.
- AuditLog는 actor type/id, review item id, model/policy version, outcome/error code, bounded
  count와 cost만 allowlist로 기록한다.

## 17. Error handling

| 오류 | 동작 |
|---|---|
| validator unavailable/timeout | human review |
| malformed structured output | human review |
| source version changed | needs more evidence |
| permission changed/unknown | auto approval 금지 |
| budget exceeded | human review, provider call 없음 |
| paid preview changed/expired | start 거절, 새 zero-call preview 요구 |
| policy/model version unavailable | human review |
| trusted conflict lookup unavailable/ambiguous | human review, provider call 없음 |
| validation lease lost | result discard, canonical owner 재조회 |
| concurrent human/auto approval | locked transition 한 건만 승리 |
| promotion replay | existing canonical ids 반환 |
| revoke provenance incomplete | fail closed, serving state 변경 없음 |
| vector delete/tombstone failure | revoke transaction rollback |
| reindex races with revoke | final eligibility recheck 또는 tombstone skip |
| critical post-audit outcome | persistent breaker를 먼저 open, exact revoke 시도 |
| post-audit revoke failure | breaker 유지, remediation_required |

C.5 장애는 Review Queue availability를 막지 않는다. 자동화 비율만 낮아진다.

## 18. Golden evaluation

한국어 중심 fixture에 다음을 포함한다.

- direct supported Timeline/History
- proposal vs decision
- planned vs completed
- affirmation vs negation
- conditional/uncertain wording
- date/subject/actor mismatch
- conflicting sources
- source version supersession
- permission loss and unknown permission
- partial support
- high model confidence hard negative
- prompt injection in source text
- duplicate reaffirmation and trusted conflict
- Decision, Todo, restricted forced-human cases

code release gate:

- hard-negative false auto approval 0
- permission/version violation auto approval 0
- duplicate promotion 0
- cross-item revoke 0
- overall auto-approval precision >= 99%
- identical validation key replay is stable
- malformed/unknown output never approves

운영 enforce gate:

- shadow human comparison 최소 500건
- shadow precision >= 99%
- Decision/Todo/restricted auto approval 0
- permission/source-version violation 0
- validator failure false approval 0

recall과 Queue 감소율은 관찰 지표이며 첫 enforce 합격 조건이 아니다.

## 19. Rollout

```text
disabled -> offline eval -> shadow -> 10% canary -> enforce
```

- default는 `disabled`
- canary는 네 번째 mode가 아니다. `enforce` mode의
  `AUTO_REVIEW_ENFORCE_PERCENTAGE=10` 단계이며 stable candidate HMAC으로 선택한다.
- canary에서 선택되지 않은 eligible 후보는 shadow result만 남기고 pending을 유지한다.
- 최초 auto-approved 50건은 human post-audit 100%
- critical error 0건이면 10% sample audit
- 누적 500건 이후 2% sample audit

10% canary에서 최초 50 auto approval은 모두 pending audit row를 만들며 50건 전부
`confirmed`되기 전에는 ordinal 51 자동 승인을 허용하지 않는다. 51~500은 stable HMAC
10%를 감사하고, 누적 500건과 모든 selected audit 완료, critical 0, 감사 precision 99%
이상을 만족한 뒤에만 authorized operator가 신규 workflow의 enforce percentage를 100으로
올릴 수 있다. 그 이후 stable 2% audit를 유지한다. 어떤 stage도 자동 승격하지 않는다.

shadow comparison은 policy가 `auto_approve`로 예측했고 evidence version이 바뀌지 않은
후보를 authorized human이 나중에 처리한 결과만 집계한다. unresolved, drifted, skipped
후보는 precision denominator에 넣지 않는다. shadow precision은 human-approved completed
comparison / all completed comparisons, enforce audit precision은 `confirmed` / all completed
audit outcomes다. enforce percentage 변경은 신규 workflow에만 적용하고 진행 중 thread의
저장된 percentage를 높이지 않는다.

unsupported/contradicted approval, permission/version 위반, revoke 실패, policy 밖 item 승인
중 하나라도 human audit로 기록되면 persistent breaker가 즉시 effective mode를 `shadow`로
강등한다. config restart만으로 breaker를 우회할 수 없다.

## 20. Testing

모든 behavior change는 TDD로 구현한다.

### Unit

- eligibility allowlist/forced-human
- pure policy decisions
- validation key/cache invalidation
- substantive field-to-slot completeness and structured output integrity
- cost budget
- signed paid preview expiry/change
- exact duplicate versus collision-bucket fail-closed behavior
- inaccessible collision existence routes human without identity/count leakage
- post-audit sample ordinal and breaker state machine
- unknown version fail closed
- revoke idempotency

### LangChain

- actual `with_structured_output()` boundary with fake chat model
- malformed/missing/unknown slot rejection
- provider error sanitization
- prompt has no canonical id, URL, permission, credential

### LangGraph

- all auto resolved means no interrupt
- partial auto resolution interrupts only pending remainder
- disabled/shadow/enforce matrix
- V2.0 paused tuple resumes only with unchanged V2.0 topology/schema
- V2.1 `revoked` count and pending-before-needs-more routing
- validation failure becomes human review
- source drift becomes needs more evidence
- checkpoint privacy scan

### PostgreSQL

- concurrent auto/human transition exactly once
- restart and validation cache replay
- immutable candidate-to-canonical evidence refs
- expired lease recovery
- revoke/promotion race
- exact knowledge/vector revoke
- raw chunks remain human-approved-only when resolution source is auto policy
- tombstone versus in-flight reindex cannot resurrect serving content
- shared reaffirmation revoke leaves unrelated active provenance serving
- internal auto actor cannot be forged through a public API
- first 50 audit rows cannot be skipped and critical audit opens breaker before revoke
- post-audit/revoke failure remains remediation-required and enforce stays disabled
- exact generated-id cleanup

### Frontend

- count, badge, filter, drawer metadata
- combined extraction/validation cost preview with the same one-click launch
- sampled audit action and remediation-required state
- authorized revoke
- desktop/mobile no added navigation depth
- raw internal identifiers/errors hidden

Automated tests never call live OpenAI, Gemini, Google, Slack, embedding, OAuth, or other
provider APIs.

## 21. Rollback

- `AUTO_REVIEW_MODE=disabled` stops new validation and transition immediately.
- `shadow` preserves observations without changing trust state.
- existing auto-approved knowledge is not mass-deleted during rollback.
- authorized reviewers can revoke exact auto-approved items.
- critical post-audit opens a persistent breaker that config restart cannot clear.
- model/prompt/policy rollback does not reuse incompatible validation cache.
- failed C.5 rollout does not change Deliverable C human Review flow.

## 22. Deliverable sequencing

```text
Deliverable C.5 — Auto-Review Trust Promotion
  -> Deliverable D — Retriever Port and RAG Answer Graph V2
  -> Deliverable E — Neo4j GraphRAG
  -> Slack recovery last
```

Deliverable D consumes human-approved and policy-approved trusted knowledge and separately
defines how canonical raw evidence participates in RAG. Deliverable E projects trusted
knowledge as graph claims and raw sources only as provenance until a separate reviewed relation
policy exists.

## 23. 성공 기준

- 사람은 낮은 위험의 직접 사실이 아니라 예외·충돌·고위험 후보에 집중한다.
- model confidence alone can never approve a candidate.
- Terra validation과 deterministic policy decision이 분리된다.
- auto approval uses the same locked exactly-once promotion boundary as human approval.
- disabled/shadow failures cannot create trusted knowledge.
- every auto approval is attributable to model, prompt, policy, evidence version and cost.
- incorrect auto approval can be precisely revoked without deleting unrelated knowledge.
- no raw source/model content enters checkpoint or audit.
- public/internal only v1 policy never broadens permissions.
- inaccessible trusted collisions can block automation without leaking identity or count.
- sampled auto approvals have durable human audit outcomes and critical findings demote
  effective mode before remediation.
- Deliverable D and Neo4j can consume trust metadata without changing source agents or API
  routes directly.

## 24. 공식 모델 참고

- [GPT-5.6 Terra](https://developers.openai.com/api/docs/models/gpt-5.6-terra)
- [GPT-5.6 Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna)
- [GPT-5.6 Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol)
- [OpenAI model guidance](https://developers.openai.com/api/docs/guides/latest-model)
