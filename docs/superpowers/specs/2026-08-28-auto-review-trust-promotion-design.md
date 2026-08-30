# Auto-Review Trust Promotion Design

검토 버전: 1
작성일: 2026-08-28
상태: 사용자 승인 — 설계·구현 계획·실행 프로필 확정, 제품 코드 구현 미착수

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
   V2.1 extraction은 정확히 다섯 agent registry만 사용하며 각 선택 agent는 OpenAI
   `gpt-5.4-mini-2026-03-17`, reasoning `none`으로 후보를 0개 또는 1개만 반환한다.
   alias, Azure OpenAI, Gemini, provider/model fallback은 허용하지 않는다.
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

이 문서의 제품 방향, threshold, cost cap, trust/permission boundary와 구현 계획의 exact
schema/profile은 사용자 승인으로 고정되었다. normalized provenance, extraction/validation-call
ledger, collision projection, serving lock, rollout latch, bounded audit projection, source authority,
Assistant evidence dependency 계약도 이 문서와 구현 계획이 함께 변경 통제 기준이다. 제품 코드,
database schema, provider call, rollout enablement는 아직 시작되지 않았으며, 실행 방식 선택과
명시적 구현 승인을 받기 전에는 변경하지 않는다.

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

Slack 경계는 더 좁게 고정한다. C.5는 Slack connector/agent/data를 재구성하지 않는다.
Slack은 V2.0 legacy dedupe 경로만 유지하며 C.5의 server content signature,
`current_document_version_id`, auto-review 또는 trusted-serving authority를 받지 않는다.
기존 Slack row는 별도로 승인된 재동기화 전까지 C.5 serving에서 fail closed한다. 자동화
테스트는 fake repeated-event regression 하나만 추가한다. 승인된 non-Slack 비교 gate의 기존
ten-item 목록에는 새 deselection을 추가하지 않고, full backend comparison에서는 deferred Slack
실패 10개가 그대로 보여야 한다.

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

사람 승인과 정책 승인은 동일한 knowledge table/locked transition을 사용하되 resolution source를
구분한다. C.5-bound approval은 effect별 `TrustedKnowledgeApprovalLink`와 canonical evidence child
links를 반드시 쓰며, migration 이전 human `source_review_item_id`만 implicit legacy base로
보존한다.

## 5. 초기 자동 승인 정책

정책 버전은 다음으로 고정한다.

```python
AUTO_REVIEW_POLICY_VERSION = 'auto-review-policy:v1'
AUTO_REVIEW_VALIDATOR_PROMPT_VERSION = 'auto-review-validation:v2'
AUTO_REVIEW_VALIDATOR_OUTPUT_CONTRACT_VERSION = 'candidate-validation-batch:v1'
AUTO_REVIEW_NORMALIZATION_SCHEMA_VERSION = 'trusted-claim-normalization:v1'
AUTO_REVIEW_TOKEN_ESTIMATOR_VERSION = 'openai-o200k-chat:v1'
AUTO_REVIEW_EXTRACTION_TOKEN_ESTIMATOR_VERSION = 'openai-o200k-extraction:v1'
AUTO_REVIEW_COST_POLICY_VERSION = 'auto-review-cost:v1'
AUTO_REVIEW_EXTRACTION_COST_POLICY_VERSION = 'auto-review-extraction-cost:v1'
AUTO_REVIEW_VALIDATOR_PROVIDER = 'openai'
AUTO_REVIEW_VALIDATOR_MODEL = 'gpt-5.6-terra'
AUTO_REVIEW_VALIDATOR_REASONING_EFFORT = 'medium'
AUTO_REVIEW_EXTRACTION_PROVIDER = 'openai'
AUTO_REVIEW_EXTRACTION_MODEL = 'gpt-5.4-mini-2026-03-17'
AUTO_REVIEW_EXTRACTION_REASONING_EFFORT = 'none'
AUTO_REVIEW_EXTRACTION_ROUTE_VERSION = 'auto-review-extraction-route:v1'
AUTO_REVIEW_EXTRACTION_AGENT_NAMES = (
    'mail_document_agent',
    'timeline_agent',
    'history_agent',
    'decision_record_agent',
    'todo_agent',
)
AUTO_REVIEW_MAX_CANDIDATES_PER_BATCH = 4
AUTO_REVIEW_MAX_CANDIDATES_PER_WORKFLOW = 5
AUTO_REVIEW_MAX_INPUT_TOKENS = 6_000
AUTO_REVIEW_MAX_OUTPUT_TOKENS = 3_072
AUTO_REVIEW_MAX_BATCHES_PER_WORKFLOW = 2
AUTO_REVIEW_MAX_PROVIDER_ATTEMPTS = 1
AUTO_REVIEW_MAX_EXTRACTION_INPUT_CHARS_PER_AGENT = 24_000
AUTO_REVIEW_MAX_EXTRACTION_INPUT_TOKENS_PER_AGENT = 10_000
AUTO_REVIEW_MAX_EXTRACTION_OUTPUT_TOKENS_PER_AGENT = 2_048
AUTO_REVIEW_MAX_EXTRACTION_CANDIDATES_PER_AGENT = 1
AUTO_REVIEW_EXTRACTION_MAX_PROVIDER_ATTEMPTS_PER_AGENT = 1
AUTO_REVIEW_TOKENIZER_ENCODING = 'o200k_base'
AUTO_REVIEW_REPLY_PRIMING_TOKENS = 16
AUTO_REVIEW_FRAMING_SAFETY_TOKENS = 512
AUTO_REVIEW_VALIDATION_LEASE_SECONDS = 120
AUTO_REVIEW_VALIDATION_COMMIT_GRACE_SECONDS = 30
AUTO_REVIEW_EXTRACTION_INPUT_USD_PER_1M = Decimal('0.750000')
AUTO_REVIEW_EXTRACTION_OUTPUT_USD_PER_1M = Decimal('4.500000')
AUTO_REVIEW_VALIDATOR_INPUT_USD_PER_1M = Decimal('2.000000')
AUTO_REVIEW_VALIDATOR_OUTPUT_USD_PER_1M = Decimal('12.000000')
AUTO_REVIEW_MAX_EXTRACTION_CALL_COST_USD = Decimal('0.016716')
AUTO_REVIEW_MAX_EXTRACTION_COST_USD = Decimal('0.083580')
AUTO_REVIEW_MAX_BATCH_COST_USD = Decimal('0.048864')
AUTO_REVIEW_MAX_VALIDATION_COST_USD = Decimal('0.097728')
AUTO_REVIEW_MAX_PROFILE_COST_USD = Decimal('0.181308')
AUTO_REVIEW_MAX_WORKFLOW_COST_USD = Decimal('0.20')
```

선택 agent 수 `N`은 exact five-entry registry의 정렬된 non-empty subset이므로 `1 <= N <= 5`다.
preview는 예측 후보 수가 아니라 strict 0/1 contract를 사용해 extraction을
`N * USD 0.016716`, validation을 `ceil(N / 4) * USD 0.048864`로 reserve한다.
다섯 agent 최대값은 각각 USD 0.083580, USD 0.097728, 합계 USD 0.181308이며
immutable USD 0.20 한도 아래 USD 0.018692를 남긴다. 배포 설정의 price는 registry의
exact six-place Decimal과 같은지 확인하는 값일 뿐 임의 override가 아니다.

### 5.1 Eligible item

`auto-review-policy:v1`의 승인된 초기값은 아래 조건을 모두 만족해야 `auto_approve`를
허용한다.

1. item type이 `timeline_event` 또는 `history_event`다.
2. validator가 claim scope를 `direct_fact`로 반환한다.
3. permission이 `public` 또는 `internal`이다.
4. candidate에 canonical source/version ref와 current content signature가 있다.
5. 모든 substantive field가 하나 이상의 존재하는 evidence slot에 연결된다.
6. validator verdict가 `supported`다.
7. `entailment_score >= Decimal('0.9800')`이다. float 변환, 반올림, epsilon 비교를
   허용하지 않는다.
8. uncertainty code와 conflict code가 모두 비어 있다.
9. source/version/permission이 모델 호출 전후에 동일하다.
10. policy/model/prompt version이 registry에서 지원된다.
11. cost budget과 bounded input 규칙을 만족한다.
12. candidate generation provider/model/prompt identity가 존재하고 validator의 exact
    provider/model identity와 다르다.

generator identity는 `AgentRun`의 exact bounded snapshot으로 보존한다. 기존 `model_name`,
`prompt_version`에 새 `generation_provider`, `generation_route_version`,
`generation_output_contract_version`을 더하며 provider order/route string 또는 mutable metadata를
파싱해 추론하지 않는다. legacy/null/unknown identity는 human-only다.

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
- permission-filtered plaintext가 versioned `credential-scan:v1`의 reviewed high-confidence
  pattern/credential-context entropy rule에 걸림. 이 경우 text를 즉시 폐기하고 validator
  prepare/invoke/cache/trace를 모두 0회로 유지하며 public reason에는 credential 여부를
  노출하지 않는다.

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
zero_call_signed_preview
  -> purpose=extraction safety + one-render/one-attempt ledgers
  -> draft_review_candidates_transaction
  -> auto_review_eligibility
  -> canonical_evidence_preflight
  -> purpose=validation safety + one-render lease/reserve/attempt
  -> bounded Terra structured validation (no DB transaction)
  -> canonical evidence revalidation
  -> deterministic policy evaluation
  -> validation decision persistence
  -> rollout/control_epoch gate + immutable PromotionDecision
  -> policy Review transition / duplicate reaffirmation + selected audit
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
mode, validator/provider/model/reasoning/prompt/output-contract/policy/cost-policy, fingerprint key
version/material verifier, extraction/validation estimator와 framing, exact Decimal price,
per-call token/output/attempt cap, validation batch/lease/grace, requested percentage,
`authorized_percentage_at_launch`, `rollout_authorization_generation`, provider-safety state
version/sorted extraction safety snapshot-set HMAC, rollout `control_epoch`, confirmed
extraction/validation/total ceilings를 workflow
request에 immutable하게 저장한다. resume 시 현재 환경 설정으로 바꾸거나 더 강한
mode/percentage를 부여하지 않는다. Registry는 V2.0과 V2.1을 동시에 유지하며 알 수 없는
version은 fail closed한다.

운영 rollback은 저장된 mode를 더 강하게 만들 수 없지만 약화할 수 있다. effective mode는
stored mode, current global mode, persisted rollout breaker가 허용하는 mode 중 가장
보수적인 값(`disabled < shadow < enforce`)이다. extraction과 validation 각각에 대해
`AutoReviewProviderSafetyState(purpose, provider, model, reasoning_effort)`가 존재하고 closed이며 저장된
estimator/cost-policy/price identity와 일치해야 한다. missing/open/mismatch는 해당 purpose의
provider call을 0회로 만들고 V2.1을 human-only/unavailable로 강등한다.
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
@dataclass(frozen=True)
class PreparedValidationInvocation:
    requests: tuple[CandidateValidationRequest, ...]
    rendered_messages: tuple[BaseMessage, ...]
    serialized_char_count: int
    framed_input_tokens: int
    max_output_tokens: int
    content_hmac: str


class AutoReviewValidator(Protocol):
    def prepare_many(
        self,
        requests: list[CandidateValidationRequest],
    ) -> PreparedValidationInvocation: ...

    def invoke_prepared(
        self,
        invocation: PreparedValidationInvocation,
    ) -> list[CandidateValidationResult]: ...
```

production adapter는 실제 LangChain
`with_structured_output(CandidateValidationBatchResult, method='json_schema', strict=True,
include_raw=True)`을 사용한다. 최종 batch의 alias/messages/native response schema는 정확히 한 번
render하고 그 immutable `PreparedValidationInvocation`의 HMAC/count와 동일한 object를 claim,
attempt admission, 단 한 번의 invoke까지 유지한다. claim 뒤 재-render하거나 raw request를
`invoke_prepared()`에 다시 전달할 수 없다. tests와 local deterministic smoke는 fake adapter를
사용한다.

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

validation call claim/lease, one-attempt marker, exact Decimal reserve/actual charge, child result,
replay, timeout recovery를 담당한다. 동일 batch/validation key는 한 canonical result만 만든다.
`claimed`, `completed`, `failed`만 durable status다. lease expiry는 timestamp condition이며
`expired` terminal status가 아니다. pre-attempt expiry만 reservation을 유지한 채 CAS reclaim할
수 있고, attempt marker 이후 expiry는 reserved charge로 한 번 `failed` 처리할 뿐 provider를
다시 호출하지 않는다.

### 7.6 `TrustedKnowledgeEvidenceLinkService`

모든 post-migration human/auto approval effect에 ReviewItem별 active provenance를 만들고,
정확한 duplicate reaffirmation이 기존 knowledge row에 새 evidence provenance를 추가할
때도 사용한다. raw content를 복제하지 않고 canonical source/version ref, evidence HMAC,
permission snapshot, resolution actor type, originating ReviewItem을 연결한다.

### 7.7 `AutoReviewPostAuditService`

모든 auto approval마다 immutable `AutoReviewPromotionDecision`을 같은 promotion transaction에서
만들고 enforce promotion ordinal과 stable sample HMAC/selection result를 고정한다. 선택된
decision에만 required `AutoReviewPostAudit`을 함께 만든다. unsampled manual audit는 기존
`not_selected` decision을 참조하며 ordinal/selection/counter를 다시 만들지 않는다. authorized
human의 `confirmed` 또는 critical outcome을 기록하고, critical이면 persistent rollout breaker를
먼저 연 뒤 exact revoke를 수행한다.

### 7.8 `AutoReviewRolloutPolicyService`

security scope와 policy version별 shadow/enforce evidence, enforce ordinal, pending audit,
critical issue를 집계한다. config가 `enforce`여도 breaker가 open이거나 gate가 부족하면
effective mode를 `shadow`로 제한한다. 이 service는 model output이 아니라 persisted human
audit과 deterministic counters만 사용한다.

### 7.9 Provider safety와 admin control plane

provider safety는 전역 boolean 하나가 아니라
`(purpose, provider, model, reasoning_effort)` unique state다.
`purpose`는 정확히 `extraction` 또는 `validation`이며 각 row가 authorized estimator/framing,
cost-policy, exact six-place Decimal prices, state version, overrun evidence와 persistent breaker를
독립적으로 가진다. extraction overrun은 extraction row를, validation overrun은 validation row를
열며 어느 경우든 새 V2.1 paid execution은 zero-call로 강등된다. 최초 authorization과 breaker
clear는 restricted local admin service/CLI만 수행하고 public Review route나 request body는 이
권한을 만들 수 없다.

## 8. Validator input과 output

### 8.1 Input

```python
CandidateSlotId = Annotated[str, Field(pattern=r'^C0[1-4]$')]
EvidenceSlotId = Annotated[str, Field(pattern=r'^E(?:0[1-9]|1[0-2])$')]


@dataclass(frozen=True)
class CandidateValidationRequest:
    candidate_slot_id: CandidateSlotId
    item_type: Literal['timeline_event', 'history_event']
    claims: tuple[ValidationClaimInput, ValidationClaimInput]
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
id, external source id, URL, permission metadata, credential은 전달하지 않는다. exact plaintext에
`credential-scan:v1`을 적용한 뒤에만 alias와 `PreparedValidationInvocation`을 만들며, scanner
match는 `sensitive_input_detected`의 zero-call human-only 결과다. claims는
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

    results: list[CandidateValidationResult] = Field(min_length=1, max_length=4)
```

자유 형식 rationale은 받지 않는다. persisted candidate key/ReviewItem id는 provider에
보내지 않고 call-local `C01`~`C04` alias로 치환한다. request/result candidate slot set과
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

provider admission은 JSON-safe native JSON-schema request를 exact NFC sorted compact JSON text
(`ensure_ascii=False`, `allow_nan=False`)로 한 번 canonicalize한다. token count는
`tiktoken.get_encoding('o200k_base').encode(canonical_text)`에 reply-priming 16과 framing-safety
512를 정확히 더하고, HMAC/fixture bytes는 같은 text의 UTF-8 encoding을 사용한다. 최종 framed input은 6,000 tokens,
Responses API의 visible+reasoning을 합친 total output은 3,072 tokens를 넘을 수 없고 character
heuristic으로 대체하지 않는다. workflow는 최대 5 candidates, 2 batches만 허용한다.

## 9. Model policy

### 9.1 Live validator

```text
provider: openai
model: gpt-5.6-terra
reasoning.effort: medium
structured output: required
validator prompt version: auto-review-validation:v2
validator output contract: candidate-validation-batch:v1
Responses API max_output_tokens: 3072
SDK retry: 0
```

Terra는 candidate generation을 수행하지 않고 validation만 담당한다. provider/model은
model-router boundary로 주입한다. signed cost ceiling 한 건은 batch당 provider attempt 한 번만
허용하며 숨은 SDK/app retry를 두지 않는다.

### 9.2 No silent fallback

Terra가 unavailable, timeout 또는 malformed output이면 Luna, Sol, Gemini로 자동
fallback하지 않는다. 다른 model은 새 golden evaluation과 새 policy version이 필요하다.
실패 후보는 사람 Review Queue로 보낸다.

### 9.3 Offline roles

- `gpt-5.6-sol`: golden-set benchmark와 audited sample 비교
- `gpt-5.6-luna`: 충분한 데이터 이후 별도 policy의 high-volume first-pass 후보

둘 다 `auto-review-policy:v1` enforce path에는 사용하지 않는다.

### 9.4 V2.1 extraction boundary

V2.1의 selected live extraction agent마다 `PreparedExtractionInvocation`을 정확히 한 번
render한다. plan은 immutable registry에서만 해석하며 모든 entry가 provider `openai`, model
snapshot `gpt-5.4-mini-2026-03-17`, reasoning `none`, route
`auto-review-extraction-route:v1`, Responses API, `openai-o200k-extraction:v1`, `o200k_base`,
rendered-input HMAC/measured counts, USD 0.750000/M input과 USD 4.500000/M output,
reply-priming 16/framing-safety 512, `max_provider_attempts=1`을 고정한다. alias, Azure OpenAI,
Gemini, mutable route, retry, provider/model fallback은 zero-call unavailable이다. V2.0의 기존
fallback/retry contract만 그대로 유지한다.

registry는 아래 다섯 entry와 그 exact schema만 가진다.

| agent | prompt | output/schema | allowed item type |
|---|---|---|---|
| `mail_document_agent` | `mail-document-extraction:c5-v2` | `mail-document-candidate:c5-v2` / `MailDocumentExtractionResult` | Timeline, History, Decision, Todo union |
| `timeline_agent` | `timeline-extraction:c5-v2` | `timeline-candidate:c5-v2` / `TimelineExtractionResult` | `timeline_event` |
| `history_agent` | `history-extraction:c5-v2` | `history-candidate:c5-v2` / `HistoryExtractionResult` | `history_event` |
| `decision_record_agent` | `decision-record-extraction:c5-v2` | `decision-record-candidate:c5-v2` / `DecisionRecordExtractionResult` | `decision_record` |
| `todo_agent` | `todo-extraction:c5-v2` | `todo-candidate:c5-v2` / `TodoExtractionResult` | `todo` |

모든 schema는 `extra='forbid'`이고 singular envelope
`result_kind: candidate|no_candidate`, `candidate: CandidatePayload|None`, bounded
`no_candidate_reason|None`을 사용한다. cross-field validator는 candidate branch에 candidate를
정확히 하나, no-candidate branch에 candidate를 0개만 허용한다. title 1–160자, summary
1–800자, confidence Decimal 0.0000–1.0000, uncertainty 1–400자, 1–10 field-evidence binding과
prepared input에 실제 존재하는 `S01..S12` slot만 허용한다. Timeline payload는
`result_summary`, History는 `reason`, Decision은 `decision_summary`, Todo는 required
`priority|priority_reason`과 bounded optional `task_summary|assignee|due_date|evidence_reason|
source_type|project_tag`만 허용한다. 각 required/present substantive field는 정확히 한 binding을
가져야 하고 field key는 중복될 수 없다. 하나의 실제 source slot이 여러 다른 field를 직접
뒷받침할 수 있으므로 slot은 field 간 재사용할 수 있지만 prepared input에 없는 slot은 항상
거부한다. 개별 field maximum은 독립 안전 상한이며 모든 maximum의 Cartesian 조합을
유효하다고 보장하지 않는다. 각 output-contract의 final validator는 JSON-mode-safe structure를
만들되 `confidence_score`가 exact Decimal zero이면 signed `-0`도 먼저 positive `Decimal('0')`로
정규화하고, 그 다음 four places로 quantize해 fixed `0.0000` string으로 format한다. 따라서
`-0`, `0`, `0.0000`은 같은 canonical zero이고 `0.98`과 `0.9800`도 같은 canonical value다. complete envelope를 한 exact NFC,
sorted-key compact JSON text(`ensure_ascii=False`, `allow_nan=False`)로 직렬화한 뒤
`o200k_base.encode(canonical_text)`가 2,048을 넘으면 거절한다. 같은 text의 UTF-8 bytes를
HMAC/fixture에 사용한다. frozen Korean/ASCII fixture는 exact 2,048 acceptance, 2,049 rejection,
그리고 개별 field-valid이지만 aggregate budget을 넘는 조합을 함께 검증한다. native Responses
`max_output_tokens=2048`에서 잘린 partial JSON은 복구·축약하지 않고 malformed/human-only로
처리하며 ReviewItem 또는 signed `no_candidate` success marker를 만들지 않는다.

V2.1 extraction의 agent당 hard ceiling은 다음으로 고정한다.

- serialized input characters: 24,000 이하
- framed input tokens: 10,000 이하
- total output tokens: 2,048 이하
- candidates: 0 또는 1
- provider attempts: 정확히 최대 1

각 route는 이 global cap보다 같거나 더 작은 registry-owned cap만 선언할 수 있다. preview와
start는 더 작은 route cap까지 token/HMAC/ceiling에 bind하며, 하나라도 넘으면 잘라서 호출하거나
fallback하지 않고 zero-call unavailable/human path로 보낸다. permission-filtered exact extraction
DTO는 bounded source text/timestamp와 ephemeral `Sxx`/`Mxx`/`Pxx` alias만 포함하며 canonical/
external source id, URL, permission label, participant email, connector metadata, ownership field를
provider에 보내지 않는다. plaintext에도 provider preparation 전에 `credential-scan:v1`을 적용한다. match 시 prepare,
invoke, cache, trace가 모두 0회이고 source bytes/matched credential은 저장·로그하지 않는다.
LangChain tracing/callback/cache는 server-owned per-call boundary에서 비활성화하며 raw
extraction DTO/rendered messages/model output/provider exception은 checkpoint, ledger, cache,
AuditLog에 들어가지 않는다.

extraction은 short E1 reserve -> short E2 attempt marker -> no-transaction provider call -> short E3
finalization 순서다. E3만 call charge, `AgentRun.status='complete'`, exact one-candidate/evidence
bindings 또는 signed `no_candidate` empty-set terminal marker를 함께 commit할 수 있다.
`AutoReviewExtractionCall.status`는 `claimed|completed|failed`이지만 cache/replay는 정확히
`AgentRun.status='complete'`인 run만 재사용하고 claimed/failed placeholder를 성공으로 보지
않는다. completed call은 `result_kind: candidate|no_candidate`, count `1|0`, keyed result-set
HMAC을 저장하며 두 번째 child는 corruption이다. post-call source/permission drift는
output/candidate를 폐기하되 actual 또는 conservative reserve를 정확히 한 번 charge한다. attempt-0
cancellation만 workflow lock 아래 zero-charge terminalize할 수 있고 attempt-1은 candidate 없이
charged failure로만 끝난다.

## 10. Persistence model

### 10.1 Workflow and `ReviewItemEvidenceRef`

V2.1 workflow request persistence에 다음 immutable field를 추가한다.

- `auto_review_mode`: `shadow`, `enforce`
- validator provider/model/reasoning effort
- validator prompt/output-contract version과 policy version
- validation/extraction cost-policy와 token-estimator/encoding identity
- reply-priming/framing-safety constants, exact validation/extraction input/output caps,
  candidates per batch/workflow, validation maximum batches, provider timeout/send-start window/
  attempt lease/commit grace, `max_provider_attempts=1`
- exact six-place Decimal validation/extraction input/output prices
- requested `enforce_percentage`, `authorized_percentage_at_launch`,
  `rollout_authorization_generation`
- validation provider-safety state version, sorted extraction provider-safety snapshot-set HMAC,
  rollout `control_epoch`
- `fingerprint_key_version`, `fingerprint_key_material_verifier`
- selected-agent count, exact prepared extraction plan-set HMAC, fixed
  24,000-character/10,000-framed-input/2,048-total-output/one-candidate extraction hard caps와
  per-agent effective caps, confirmed extraction/validation/total cost ceiling, exact USD 0.20 budget

새 `ReviewItemEvidenceRef` model과 `review_item_evidence_refs` table은 C.5 migration 이후
생성되는 V2.0/V2.1 candidate를 기존 `AgentWorkflowEvidenceRef`에 정확히 연결한다. 이
내부 provenance 추가는 V2.0 public/checkpoint/HMAC shape를 바꾸지 않으며 V2.0 candidate는
graph version 규칙으로 계속 auto-review ineligible이다.

- `id`
- `review_item_id`
- `workflow_thread_id`
- `workflow_evidence_ref_id`
- `candidate_slot_ordinal`
- keyed aggregate `message_content_fingerprint`
- `fingerprint_key_version`, `fingerprint_key_material_verifier`
- `created_at`

`(review_item_id, workflow_evidence_ref_id)`와
`(review_item_id, candidate_slot_ordinal)`은 unique다. approved legacy 이름인
`candidate_slot_ordinal`은 한 ReviewItem 안의 evidence-ref ordinal `1..N`을 뜻하며 각 row가
서로 다른 값을 갖는다. 같은 source version의 여러 cited message는 stable identity로 정렬한
뒤 하나의 aggregate message-set HMAC으로 저장한다. candidate draft와 같은 transaction에서
만들고 update하지 않는다. source text, URL, snippet은 이 table에 복제하지 않는다.

`workflow_thread_id`는 ReviewItem과 `AgentWorkflowEvidenceRef` 양쪽에 named same-workflow
composite FK로 묶는다. bound ref ownership/source/message-HMAC/key identity는 immutable하고,
post-C.5 candidate는 commit 시 같은 workflow의 `AgentRun`, 하나 이상의 ref,
`candidate_contract_version='c5-v1'`을 반드시 가진다. migration은 기존 packet에서 이 binding을
추측해 backfill하지 않는다.

validator 직전에 이 ref를 canonical source resolver로 다시 읽어 current version,
signature, permission, content fingerprint를 확인하고 request-local slot을 만든다. migration
이전 ReviewItem처럼 exact ref가 없는 후보, packet 전체와 candidate ref가 불일치하거나
duplicate snippet을 exact message에 매핑할 수 없는 후보는 auto-review eligible이 아니며
사람에게 보낸다.

### 10.2 Canonical source, parser run, and current-version authority

C.5는 connector가 제공한 display label 또는 signature를 trust authority로 사용하지 않는다.
`Source.connector_content_signature`는 connector diagnostic/dedupe용이고, server가 exact canonical
payload로 계산한 `server_content_signature_schema='server-source-content:v1'`와
`server_content_signature`만 C.5 source authority다. payload는 source type, normalized title/body/
author/participants, source-type registry의 semantic metadata, 그리고 exact semantic timestamp를
포함한다. canonical JSON은 UTF-8 NFC, sorted compact keys, `ensure_ascii=False`, `allow_nan=False`이며
SHA-256 64-hex를 저장한다.

`SourceEvent.semantic_timestamp_raw`는 trailing nullable ingestion field다. Gmail은 exact
`internalDate`, Drive는 exact `modifiedTime`, Calendar는 exact raw `start`를 보존한다. missing,
naive, malformed timestamp를 `datetime.now()`나 connector `updated`로 보충하지 않는다. Calendar
date와 aware instant는 서로 다른 canonical kind이고 offset-equivalent instant만 동일하다.

document parsing은 immutable `DocumentParserRun`에 `parser_policy_version`, parser implementation
version, chunk-policy version, source/server signature를 저장하고 `DocumentChunk.parser_run_id`로
relationally 묶는다. `Document.current_document_version_id`는 같은 document의 exact current row를
가리키는 FK다. 기존 `current_version` 문자열은 display-only이며 serving/validation authority가
아니다. raw chunk와 persisted answer dependency는 current pointer, parser run/policy, server
signature, chunk content hash가 모두 맞을 때만 current다.

ingestion은 하나의
`SourceStateChangeClassification(content_changed, permission_changed, parser_policy_changed,
primary_code)`를 사용한다. 세 boolean이 모두 false일 때만 unchanged다. parser-policy-only 변경은
reparse/rechunk/incremental reindex를 수행하되 extraction/validator 호출은 0회다. permission-only
변경은 visibility/vector metadata를 좁히며 LLM을 다시 부르지 않는다.

backfill/repair는 exact server signature와 parser identity가 하나로 결정되는 row만 채운다.
display label, connector signature, timestamp, transformed body, `MAX(id)`, numeric version
guess로 current pointer를 만들지 않는다. ambiguous/missing row는 fail closed하고 bounded admin
repair/re-sync 대상으로 남긴다. migration은 semantic authority를 추측하지 않는다.

### 10.3 Extraction/validation one-attempt ledgers

`AutoReviewExtractionCall`은 V2.1 live `AgentRun`마다 정확히 한 row이며
`(workflow_thread_id, agent_name)`도 unique다. changed plan HMAC을 제출해 같은 agent를 두 번
호출할 수 없고 call plan은 stored signed plan set membership을 검증한다. same-workflow agent/run,
extraction plan HMAC, provider/model/reasoning/route/prompt/output-contract, estimator/encoding/allowances,
prepared-content HMAC와 24,000 character/10,000 framed-input/2,048 total-output 및 one-candidate
effective route caps, exact Decimal prices, cost/key/provider-safety identities, signed workflow extraction/total
ceilings, `status=claimed|completed|failed`, lease, `max_provider_attempts=1`, attempt marker, reserve, actual-or-conservative charge와
overrun을 저장한다. completed row는 `result_kind=candidate|no_candidate`, count `1|0`, exact keyed
result-set HMAC을 저장하고 두 번째 child를 거절한다. prompt/source/output/provider exception은
저장하지 않는다.

`auto_review_validation_calls`는 한 bounded batch의 유일한 provider attempt/lease/cost ledger다.

- `workflow_thread_id`, keyed `batch_fingerprint`, unique
- `status`: `claimed`, `completed`, `failed`
- `max_provider_attempts=1`
- `provider_attempt_count`: `0`, `1`; `attempt_started_at`, nullable
- candidate count와 reserved input/output/cost ceiling
- actual 또는 conservative charged input/output/cost
- prepared content HMAC, exact serialized character/framed-input count, output cap
- estimator/encoding/framing, exact Decimal price, cost-policy/provider-safety/key identities
- signed extraction/validation/total ceilings와 budget-overrun flag/cost
- `lease_token`, `lease_expires_at`, timestamps

workflow request row를 lock한 뒤 claimed reservation과 completed/failed charge에 next reserve를
더한 validation obligation이 `min(USD 0.097728, stored signed validation ceiling)` 이하이고,
persisted extraction charge와 합친 값이 stored signed total ceiling 이하일 때만 새 call을
claim한다. 최대 2 calls/5 candidates이며 batch ceiling은 exact `Decimal('0.048864')`, workflow
validation ceiling은 exact `Decimal('0.097728')`보다 클 수 없다. extraction 최대 reserve는
`Decimal('0.083580')`, full profile은 `Decimal('0.181308')`, total budget은 `Decimal('0.20')`다.
모든 가격 계산/합/비교는 float 없이 six-place USD
`ROUND_CEILING`을 사용하고 사용자가 sign한 더 작은 ceiling을 global cap으로 대체하지 않는다.
provider usage가 불명확한 실패는 reserved ceiling 전체를 한 번 charge하고 재시도하지 않는다.
lease expiry는 terminal status가 아니다. `provider_attempt_count=0`일 때만 reservation을 유지한
채 CAS reclaim하며, provider 직전 별도 short transaction이 attempt count를 1로 commit한 뒤의
crash/expiry는 provider를 다시 부르거나 reserve를 해제하지 않고 reserved charge의
`failed`/human-review로 한 번 종결한다.

child `auto_review_validations` table은 다음 field를 가진다.

- `id`
- `review_item_id`
- `validation_call_id`
- `validation_key`, unique
- `evidence_version_hash`
- `candidate_generation_fingerprint`
- `status`: `claimed`, `completed`, `failed`
- `validator_provider`
- `validator_model`
- `reasoning_effort`
- `validator_prompt_version`
- `validator_output_contract_version`
- `policy_version`
- `fingerprint_key_version`, `fingerprint_key_material_verifier`
- `cost_policy_version`
- confirmed validation cost ceiling
- bounded per-field `claim_results`
- derived `minimum_entailment_score`
- bounded `uncertainty_codes`
- bounded `conflict_codes`
- `policy_decision`
- bounded `policy_reason_codes`
- deterministically allocated `input_tokens`, `output_tokens`, `estimated_cost_usd`, `cache_hit`
- `shadow_comparison_status`, nullable
- `shadow_human_resolution`, nullable
- bounded `shadow_exclusion_code`, nullable
- `shadow_compared_at`, nullable
- `created_at`, `completed_at`

call total integer token은 validation-key 정렬과 `divmod`로 child에 배분한다. authoritative call
charge를 integer micro-USD로 바꾼 뒤 unquantized weighted share의 floor와 largest-remainder
tie-break(`validation_key`)로 child cost를 배분한다. child cost는 non-negative이고 합은 call
charge와 정확히 같아야 한다. workflow cost는 extraction/validation call row의 final actual 또는
non-final reserve만 정확히 한 번 합산하며 child와 legacy float mirror를 다시 더하지 않는다.
known usage/cost/token cap 또는 workflow ceiling overrun은 actual을 clamp하지 않고 저장하고 해당
purpose의 provider-safety breaker를 같은 transaction에서 연 뒤 모든 child를 human-only로 둔다.
raw evidence, URL, snippet, model rationale, provider exception text를 저장하지 않는다.

### 10.4 `ReviewItem` additions

- `resolution_source`: `human`, `auto_policy`, nullable
- `resolution_policy_version`, nullable
- `auto_validation_id`, nullable FK
- `revoked_at`, nullable
- domain `auto-review-revoke-actor:v1`의 `revoked_by_subject_hmac`, nullable
- `revoked_by_fingerprint_key_version`, nullable
- `revoked_by_fingerprint_key_material_verifier`, nullable
- first-commit `revoke_knowledge_remained_trusted`, `revoke_document_count`, nullable

`revoked`를 ReviewItem/public Review resolution status에 추가한다. V2.1 전용 state/count
schema는 `revoked`를 resolved-but-not-approved로 계산한다. V2.0 graph state/status union과
builder는 그대로 유지하며, revoke endpoint는 `resolution_source='auto_policy'`인 V2.1
item만 허용하므로 기존 paused V2.0 thread에 `revoked`가 나타나지 않는다. 기존 row는
nullable field를 그대로 유지한다. `auto_validation_id`는 named composite FK
`(review_items.id, review_items.auto_validation_id) ->
(auto_review_validations.review_item_id, auto_review_validations.id)`로 같은 ReviewItem의 completed
canonical validation만 참조해야 한다. raw human subject id는 새 C.5 business column에 저장하지
않는다. free-text revoke reason도 ReviewItem에 저장하지 않으며 exact terminal reason code는
immutable `AutoReviewRevocationAssessment`가 소유한다. historical revoke attribution HMAC/key
identity는 rotation 때 재-HMAC하지 않는다.

auto policy actor는 다음으로 기록한다.

```text
reviewer_id = system:auto-review
resolution_source = auto_policy
resolution_policy_version = auto-review-policy:v1
```

### 10.5 Trusted fingerprint and provenance projections

`TrustedKnowledgeFingerprint`는 approved Timeline/History의 hidden-collision `EXISTS`를 위해
다음을 keyed projection으로 저장하고 exact index를 둔다.

- `knowledge_type`, `knowledge_id`, unique
- nullable `security_scope_id`와 `scope_resolution`: `exact`, `legacy_unknown`
- keyed `project_scope_hmac`, `normalized_title_bucket_hmac`
- nullable keyed `normalized_claim_fingerprint`
- `fingerprint_key_version`, `fingerprint_key_material_verifier`, permission/status snapshot,
  timestamps

singleton `AutoReviewRuntimeKeyState`는 key version, raw secret이 아닌 domain-separated
key-material verifier HMAC, generation, ready flag를 저장한다. singleton
`TrustedKnowledgeFingerprintProjectionState`는 projection schema/key version/material verifier,
generation, ready flag, **separate** source/projected active counts, separate 64-lowercase-hex
`source_checksum`/`projected_checksum`, `rebuild_required`, completed time을 저장한다. row digest는
domain `trusted-fingerprint-summary-row:v1`의 big-endian 256-bit HMAC이고 aggregate는 unique row
digest의 modulo `2**256` addition이다. XOR 또는 unspecified whole-table checksum을 사용하지
않는다.

system backfill은 state `ready=false`를 먼저 commit하고 source ReviewItem의 workflow로만 exact
scope를 정하며 나머지는 `legacy_unknown`으로 둔다. promotion/revoke/rebuild finalization은 같은
global projection advisory lock을 사용한다. final transaction의 active-row anti-join,
permission/status/key mismatch, count/checksum이 모두 clean일 때만 ready=true다. preflight도
marker/version/material/ready/count/checksum과 fresh boolean anti-join guard를 확인하므로 projection에
없는 hidden row를 놓치지 않는다. unknown scope row는 title collision을 fail-closed시키지만 exact
reuse하지 않는다. hidden guard는 current scope 또는 legacy-unknown matching bucket을 server-side
`EXISTS`로만 조회하며 id/content/count를 가져오지 않는다.

production keyed mutation의 고정 global lock order는
`AUTO_REVIEW_KEY_GENERATION_LOCK_ID = 1066041229503628369`의 shared generation barrier ->
`AutoReviewRuntimeKeyState FOR SHARE` -> purpose별 provider-safety row(필요한 auto path) ->
`TRUSTED_FINGERPRINT_PROJECTION_LOCK_ID = -2972884933094306491` -> rollout/workflow/ReviewItem/
document locks다. lock id는 secret, version, tenant, document에서 derive하지 않는다.

`AutoReviewRuntimeKeyState.ready`는 key generation/material이 initialized/stable하다는 뜻만
가진다. projection rebuild는 shared generation barrier와 runtime `FOR SHARE`를 유지한 채
projection row만 `ready=false, rebuild_required=true`로 만들며 runtime ready를 내리거나 row-lock을
upgrade하지 않는다. rotation은 global mode disabled에서 generation barrier를 exclusive로 잡고
runtime `FOR UPDATE`, projection lock 순서로 old workers를 기다린 뒤 runtime
version/material/generation을 stable `ready=true`로 advance하고 projection identity만 새 generation의
`ready=false, rebuild_required=true`로 reset한다. 이후 새-generation shared barrier에서 rebuild하고
anti-join/mismatch와 두 summary pair가 모두 일치할 때 projection만 ready로 만든다. old keyed rows나
historical actor/selection HMAC을 re-HMAC하지 않는다.

migration은 secret/environment를 읽지 않는다. startup/admin bootstrap은 PostgreSQL에서 mode와
관계없이 non-placeholder 32 UTF-8 byte 이상 active secret을 요구하고, fixed exclusive barrier
아래 C.5 keyed artifact가 전혀 없을 때만 generation 1/runtime ready와 projection
`ready=false,rebuild_required=true`를 만든다. retained keyed data와 missing/mismatched runtime row는
bounded refusal이며 자동 re-HMAC/repair하지 않는다. disabled SQLite는 process-local guard와
projection/auto readiness false만 허용한다. rotation CLI는 secret manager의 exact
`(current_version,current_secret,next_version,next_secret)` key ring으로 old verifier를 확인하고
다른 next material을 derive한다. global disabled와 exclusive barrier를 요구하고, rebuild 뒤에도
deployment config가 single new active key로 cutover되어 status가 ready를 증명할 때까지 admission은
disabled다.

정확한 promotion/reaffirmation/revoke를 위해 parent `TrustedKnowledgeApprovalLink`와 child
`TrustedKnowledgeEvidenceLink`를 분리한다. parent는 knowledge type/id, ReviewItem,
security scope, effect kind, resolution source, target-specific claim fingerprint, permission,
active/revoked state를 가진다. child는 approval link id와 canonical source kind/id/version,
keyed evidence hash를 가진다. 두 level 모두 keyed value의 fingerprint key version/material
verifier를 snapshot한다. parent 하나에는 child가 하나 이상이어야 하며 replay는 전체
child set이 정확히 일치해야 한다. child 없는 link를 만들지 않는다.

Timeline/History/Decision/Todo primary effect와 companion Timeline은 실제 persisted target
field에서 각자 target-specific fingerprint를 만든다. exact reuse는 Timeline/History에만
허용한다. C.5 migration 이후 생성된 V2.0/V2.1 candidate의 human/auto promotion은 모두
explicit parent/children을 쓴다. migration 이전 unbound human item/knowledge는 기존
`source_review_item_id` 자체를 implicit active human base로 간주하고 C.5 revoke하지 않는다.
source text와 URL은 어느 projection에도 복제하지 않는다.

### 10.6 Resolution actor

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

`ReviewTransitionService`는 이 application-owned actor contract를 받도록 좁게 refactor한다.
기존 generic AuditLog writer와 unrelated DemoUser caller는 그대로 두고 review-only adapter가
system actor를 fixed non-null schema projection으로 기록한다. human action은 `human_review`,
auto approval/reaffirmation은 `auto_review` capability와 matching policy version을 요구한다.
auto actor는 reject, needs-more-evidence human note, bulk public action을 실행할 수 없다.

별도 internal `mark_evidence_stale` transition은 locked canonical resolver가 bound evidence의
version drift/deletion을 증명할 때만 pending item을 `needs_more_evidence`로 바꾼다. client가
호출하거나 note를 전달할 수 없으며 일반 auto actor needs-more 권한으로 사용하지 않는다.

### 10.7 `VectorServingTombstone`

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

canonical source state는 server content signature, parser policy/run, exact current-version pointer,
normalized permission을 포함한다. ingestion은 10.2의
`SourceStateChangeClassification` boolean을 사용하며 same-content permission-only change나
parser-policy-only change를 skipped로 버리지 않는다. permission-only인 경우 `Source`와
current/historical version의 **모든** `DocumentChunk` permission metadata를 갱신하고
parser/extraction 없이 changed-source reconciliation을 요청한다. 기존
`chunk:{id}` vector를 Source-before-document lock order로 narrow한다. content supersession은 raw
indexing을 relational `Document.current_document_version_id`에만 허용하고 old-version chunk vector/index state를
synchronous delete하며 live Source+DocumentVersion SQL guard가 stale physical row를 먼저 제외한다.
PostgreSQL source mutation은 shared generation/runtime 뒤 sorted `Source FOR UPDATE`에서 끝나고,
extraction/validation completion 및 promotion은 같은 prefix 뒤 exact `Source FOR SHARE`를 잡는다.
따라서 current-version/permission check와 trust commit 사이에 source update가 끼어들 수 없다.

`TrustedServingEligibilityService`가 Review/Knowledge/Timeline/Project/Dashboard API projection,
RAG document construction와 agent RAG retrieval, pgvector ranking과 hidden-match accounting 전의
유일한 live predicate다. auto-only
target의 모든 active auto evidence가 현재 canonical Source에 존재하고 exact version/signature와
supported `public|internal` permission을 유지해야 `{eligible=true}`다. absent/deleted,
lookup/database failure, superseded content/version, restricted/unknown permission 또는 mismatch는
reconciliation 완료 전에도 즉시 invisible이다. shared human/auto target은 human provenance가 trust를
유지할 수 있지만 effective permission은 target, ReviewItem, evidence snapshot, resolvable current
sources 중 strictest이며 절대 broaden하지 않는다. pgvector는 knowledge document의 provenance/source
join과 raw chunk의
`document_chunks -> document_versions -> documents.current_document_version_id -> sources`
join을 분리해 둘 다 ranking/hidden count 전에 version과 permission을 검사한다. raw source id나
mismatch reason은 public 결과에 넣지 않는다.

vector writer에 `delete_many(document_ids)`와
`narrow_permissions(document_ids, permission_level)` contract를 추가한다. narrowing은
`public < internal < restricted` 방향으로 retain/narrow만 허용하고 broadening/unknown은 거절하며
embedding provider를 호출하지 않는다. revoke와 reindex는 같은
domain-separated per-document PostgreSQL transaction advisory lock을 sorted document id 순서로
같은 Session/transaction에서 잡는다. revoke는 lock 뒤 knowledge status/tombstone을 기록하고
pgvector row와 모든 `VectorIndexState`를 exact id로 삭제한다. reindex는 provider call을
transaction 밖에서 끝낸 뒤 새 transaction에서 같은 lock을 잡고 approved/no-tombstone을
다시 읽은 뒤 `INSERT ... WHERE NOT EXISTS(tombstone)`로 upsert한다. conditional SQL만으로는
READ COMMITTED race를 막기에 충분하지 않으며 lock과 함께 defense-in-depth로 사용한다.
DB runtime state의 fingerprint-key version 또는 domain-separated key-material verifier HMAC와
process settings가 다르면 serving mutation을 거부한다. version bump를 빠뜨린 secret 변경도
fail closed한다. 모든 serving mutation은 위 shared generation/runtime lock을 commit까지 유지하고,
rotation은 fixed exclusive barrier로 old generation mutation이 끝나기를 기다린다. projection
rebuild 중에도 runtime key state는 stable/ready이고 auto-review만 projection-not-ready로
zero-call 된다.

pgvector search도 ranking/hidden-match count 전에 tombstone row를 제외한다. deterministic
in-memory delete는 DB commit 뒤에만 적용하고 rollback 시 discard하며 restart rebuild는
tombstone을 제외한다. 따라서 어느 interleaving이나 rollback에서도 revoked content가
serving으로 되살아나지 않는다.

source-state commit 뒤 `AutoReviewSourceReconciliationService.reconcile(changed_states)`를
synchronous/idempotent하게 실행하며 CDC/stream을 추가하지 않는다. transaction order는 shared
generation/runtime -> projection -> sorted rollout -> sorted Source -> sorted workflow -> ReviewItem ->
required audit -> approval/evidence links -> target/fingerprint projection and both summary deltas ->
sorted document locks -> tombstone/index/vector다. unchanged content의 stricter supported permission은
ReviewItem effective permission, target/companion, `TrustedKnowledgeFingerprint`, both summary deltas와
physical vector를 같은 transaction에서 좁히고 index-state hash만 갱신한다. immutable evidence/link
snapshot은 historical 값으로 남고 Review/API는 monotonic effective permission을 사용한다. 이후
source가 더 public이 되어도 자동 broaden하지 않는다.
restricted/unknown, absent/deleted, content/version supersession은 server-owned
`source_invalidation` context로 affected auto effect만 revoke하고 human/legacy provenance는
보존한다. last effect에는 normal tombstone/delete를 적용하며 immutable evidence snapshot을 새
source version으로 덮어쓰지 않는다.

source commit과 reconciliation 사이 crash도 live eligibility가 먼저 fail closed하므로 안전하다.
startup은 unlocked bounded stale-id scan(`limit=100`) 한 번만 수행하고 각 id를 위 global order로
reconcile한다. admin CLI가 continuation path다. selected pending audit가 invalidated되면
`outcome=NULL`, internal `source_invalidated_before_audit`, `action_required=false`로 exactly-once
종결하고 모든 precision numerator/denominator에서 제외한다. 그러나 first-50 gate를 충족한 것으로
세지 않는다. `confirmed_mandatory_audit_count < 50`인 동안 다음 eligible promotion을 모두 mandatory
replacement audit로 선택하고, human-confirmed mandatory audit 50건을 채우기 전에는 unsampled/10%
cohort로 넘어가지 않는다. completed/critical/remediation audit은 immutable하다.

`SourceInvalidationRevokeContext`는 이 reconciliation service만 만들 수 있는 별도 narrow gate
bypass다. ordered locks 아래 current absent/restricted/unknown/version mismatch와 affected evidence link,
같은 transaction의 pending-audit invalidation을 증명할 때만 exact permission narrowing/revoke를
허용한다. approve/content access/broad actor permission은 없고 public/human/auto actor가 request,
serialize, forge할 수 없다.

### 10.8 Persisted Assistant evidence dependency and zero-leak serving

Deliverable D의 새 retrieval cutover는 여전히 범위 밖이지만, 현재 존재하는 Search/Ask/Assistant
consumer가 stale 또는 revoked evidence를 재사용하지 않게 하는 안전 repair는 C.5 범위다.
`AssistantMessage.evidence_contract_version`은 정확히 `none-v1|assistant-evidence:v1`이고,
evidence-backed assistant answer는 `assistant-evidence:v1`만 사용한다.

`AssistantMessageEvidenceDependency` parent는 message id, exact dependency count, dependency-set
HMAC, strictest permission, key version/material을 immutable하게 저장한다. ordinal child는 raw 또는
trusted 중 하나다.

- raw dependency는 Source/Document/DocumentVersion/DocumentChunk/DocumentParserRun의 relational id,
  `current_document_version_id`, parser policy/implementation/chunk-policy, server signature,
  chunk content hash, permission snapshot을 bind한다.
- trusted dependency는 knowledge type/id와 content hash, active approval effect 하나, 그 effect의
  모든 `TrustedKnowledgeEvidenceLink` child set을 bind한다. migration 이전 legacy-human base는
  `source_review_item_id`와 human approval이 하나로 증명되는 좁은 경우에만 허용한다.

message와 parent/모든 child는 한 transaction에서 commit한다. expected count와 exact set이 맞지
않으면 message를 저장하지 않으며 answer generation 전에도 incomplete retrieval candidate를
버린다. serving 시 shared `AssistantEvidenceProjectionService`가 message list, conversation
serialization, recent context, derived conversation summary, assistant email draft, RAG orchestrator
입력을 모두 같은 predicate로 검사한다.

current pointer/parser run/server signature/content hash drift, revoke/quarantine, permission mismatch,
missing/extra dependency, lookup/DB failure 중 하나라도 있으면 해당 answer 전체를 bounded
`evidence_unavailable`로 바꾸고 citation/source id/URL/snippet/hidden-match/detail list를 모두 비운다.
부분 문장 또는 surviving citation만 반환하지 않는다. stored conversation summary는 audit/cache
artifact일 뿐 새 prompt authority가 아니며, C.5 이전 unbound evidence message는 audit-only다.

### 10.9 `AutoReviewPromotionDecision`, post-audit, assessment, and correction

모든 enforce auto approval은 approval transaction 안에서 immutable
`AutoReviewPromotionDecision`을 정확히 하나 만든다.

- `id`, `review_item_id`, unique와 same-item composite target
- `(security_scope_id, policy_version)` rollout-state composite FK
- `rollout_authorization_generation`, `promotion_ordinal`
- requested/stored/authorized percentage와 immutable enforce-selection fingerprint
- audit-selection fingerprint
- `selection_result`: `first_50`, `sample_10`, `sample_2`, `not_selected`
- `fingerprint_key_version`, `fingerprint_key_material_verifier`
- timestamps

selection HMAC은 canonical UTF-8/NFC sorted compact JSON에 대해 full HMAC-SHA256 digest를
unsigned big-endian integer로 해석하고 `integer % 100 < percentage`로 판정한다. enforce는
domain `auto-review-enforce-selection:v1`, audit는 `auto-review-audit-selection:v1`을 사용한다.
둘 다 security-scope/workflow-execution HMAC, candidate key, policy, rollout
state/generation/percentage, key version/material을 bind하고 audit identity는 ordinal과 enforce
selection fingerprint까지 bind한다. ordinal 1..50은 `first_50`으로 override한다. runtime/config/key
rotation 뒤 기존 decision을 재-HMAC하거나 sampling을 다시 계산하지 않는다.

선택된 decision에만 같은 transaction에서 required `AutoReviewPostAudit`을 만든다.
`AutoReviewPostAudit`은 scope/policy를 중복 저장하지 않고 immutable decision에서만 derive한다.

- `id`, `review_item_id`, `promotion_decision_id`, 각각 unique 및 same-item composite FK
- `sample_cohort`: `first_50`, `sample_10`, `sample_2`, `manual`
- `status`: `pending`, `completed`, `remediation_required`
- `outcome`: `confirmed`, `incorrect`, `permission_violation`,
  `source_version_violation`, `policy_violation`, nullable
- `system_resolution_code`: exact `source_invalidated_before_audit`, nullable
- `remediation_code`: `revoke_pending`, `revoke_failure`, `provenance_incomplete`,
  `vector_delete_failure`, `policy_regression_unverified`, nullable and system-only
- domain `auto-review-audit-actor:v1`의 `auditor_subject_hmac`, nullable
- `auditor_fingerprint_key_version`, `auditor_fingerprint_key_material_verifier`, nullable
- bounded `audit_reason`, nullable
- `created_at`, `audited_at`

sample row는 promotion과 같은 transaction에서 생성하므로 자동 승인 후 audit queue에서
누락될 수 없다. pending row는 outcome/system code/reason/auditor fields가 모두 null이다.
`status='completed', outcome=NULL`은 server reconciliation이 exact
`system_resolution_code='source_invalidated_before_audit'`를 기록한 경우만 유효하며 human reason/
auditor는 null이다. human `confirmed|critical` completion은 UTF-8 NFC, outer trim 후 1–500자의
immutable `audit_reason`과 auditor key identity를 요구한다. 이유는 access-controlled audit
artifact이며 public Review/Knowledge/API/checkpoint/prompt/normal log로 projection하지 않는다.

outcome은 한 번만 확정한다. 잘못 제출한 critical outcome도 덮어쓰지
않으며 별도 operator remediation/audit 절차를 거치기 전에는 breaker를 닫을 수 없다.
critical outcome transaction은 breaker와 `remediation_required/revoke_pending`을 먼저 commit한다.
다음 idempotent revoke가 성공하면 completed로 바꾸며 crash/restart recovery는 pending
remediation을 재시도한다. recovery는 먼저 unlocked bounded id scan만 수행하고 각 id마다
generation/runtime -> provider safety(필요 시) -> projection -> rollout -> ReviewItem ->
promotion decision/audit -> provenance/knowledge/document 순서로 다시 lock/recheck한다. audit row
lock을 잡은 채 더 앞선 global lock을 획득하지 않는다. outcome/counter는 replay로 바뀌지
않는다. unsampled manual audit는 기존 `not_selected` decision을 참조하며 ordinal/selection/
selected counter를 새로 만들지 않는다.

revoke reason public/internal contract는 아래 enum만 허용하며 free text나 LLM classifier를 쓰지
않는다.

```text
business_withdrawal | incorrect_content | permission_violation |
wrong_source_version | policy_violation
```

`AutoReviewRevocationAssessment`는 review item/promotion decision/reason code, domain-separated
actor HMAC/key identity와 creation time을 insert-once로 저장한다. gate에서 거절된 request는
assessment를 쓰지 않는다. 같은 reason replay는 기존 assessment를 재사용하고 다른 reason은
`revoke_reason_conflict`다. `business_withdrawal`만 existing audit gate를 통과한 뒤 direct exact
revoke를 사용할 수 있다.

나머지 네 quality reason은 `AutoReviewQualityRevokeService`만 처리한다. missing audit이면 기존
promotion decision에서 `manual` audit를 만들고 critical로 완료하며, pending audit이면 critical로
완료한다. 이미 completed/confirmed이면 그 audit row를 절대 수정하지 않고 immutable
`AutoReviewAuditCorrection` child를 append한다. correction은 reason, assessor HMAC/key identity,
created time, exact confirmed parent를 저장하고 derived effective outcome을 critical로 만든다.
같은 scope/policy의 `corrected_critical_count`는 정확히 한 번 증가하며 감소/reset되지 않는다.

quality path는 assessment/audit-or-correction, rollout breaker/generation invalidation과 serving
quarantine를 한 transaction에서 먼저 commit한 뒤 exact revoke를 수행한다. revoke가 실패하거나
crash해도 quarantine와 `remediation_required`는 유지되고 recovery가 재시도한다. physical revoke가
끝나기 전에는 success schema를 반환하지 않고 bounded 409 `remediation_required`를 반환한다.
confirmed-audit correction이 있는 동일 policy는 breaker를 닫아 재사용할 수 없으며 새 reviewed
policy version과 shadow/canary gates가 필요하다.

shadow의 predicted `auto_approve`는 `AutoReviewValidation`에 bounded comparison status,
human resolution, exclusion code, compared timestamp를 기록한다. human transition 시
evidence version이 그대로인 경우만 completed comparison이며 drift/unresolved item은
분모에서 제외한다.

### 10.10 `AutoReviewRolloutState`

`(security_scope_id, policy_version)` unique row에 다음 deterministic aggregate를 둔다.

- shadow predicted/completed/supported count
- enforce promotion ordinal
- post-audit selected/completed/critical count, `confirmed_mandatory_audit_count`,
  `pending_mandatory_audit_count`, `invalidated_before_audit_count`,
  permanent `corrected_critical_count`
- `breaker_open`, bounded `breaker_reason_code`, `breaker_opened_at`
- `max_authorized_percentage`: `0`, `10`, `100`
- authorization generation/time, bounded current regression gate reference
- `last_event_sequence`, `last_event_id` aggregate backpointer
- optimistic `state_version`과 별도 `control_epoch`
- `created_at`, `updated_at`

promotion/comparison/audit는 이 row를 lock하고 replay-safe하게 counter를 갱신한다. metric은
authorization latch를 자동으로 올리지 않는다. shadow gate 후 operator는 0→10만 허용하고,
10% canary 500건/selected audit 완료/critical 0/precision 99% 뒤 별도 authorized action만
10→100을 허용한다. effective percentage는 immutable workflow snapshot, current config,
persistent latch의 최소값이다. breaker는 critical audit 첫 transaction에서 먼저 open/commit하고
latch/effective mode를 zero/shadow로 내린다. 따라서 revoke failure여도 다음 request부터
effective mode는 즉시 `shadow`다. breaker는 자동으로 닫히지 않으며, 별도 authorized operator
remediation과 bounded audit reason이 있어야만 닫을 수 있다.

breaker close는 public Review action이 아니라 admin-only rollout service/CLI command다.
`auto_review_rollout_admin` capability, 1~500자 reason, 모든 `remediation_required` item의
해결, 영향 item revoke/adjudication, 관련 regression gate reference를 요구한다. close는
AuditLog에 기록하지만 authorization latch 0/effective mode `shadow`에 남는다. enforce 복귀는
이후 별도 human authorization generation과 config 변경이 필요하다.
`corrected_critical_count > 0`이면 같은 policy row의 breaker close/reauthorization은 영구
거절하고 새 reviewed policy version을 요구한다.

ordinary metric/audit counter는 `state_version`만 증가시킨다. latch/authorization/breaker/generation
같은 launch-affecting control change만 `state_version`과 `control_epoch`를 함께 증가시킨다.
preview는 noisy `state_version`이 아니라 `control_epoch`와 latch/authorization/breaker snapshot을
bind한다. rollout row가 없으면 read-only sentinel은 control epoch/latch/percentage/generation 0,
breaker false이며 preview가 row를 만들지 않는다. start의 first write만 sentinel row를
ensure/lock하고 signed control fields를 비교한다.

`AutoReviewRolloutControlEvent`는
`percentage_authorized|breaker_opened|breaker_closed|generation_invalidated`만 허용하는
append-only ledger다. event마다 parent scope/policy, gap 없는 sequence, prior/new latch/breaker/
control epoch/authorization generation, bounded reason/gate ref, operator 또는 call-attribution HMAC,
그 event의 key version/material, timestamp를 저장한다. update/delete와 parent delete는 DB guard와
`ON DELETE RESTRICT`로 거절한다. aggregate CAS, event insert, `last_event_id/sequence` 갱신은 한
transaction이다. metric-only update는 event를 만들지 않고 lost CAS/replay도 새 event를 만들지
않는다. key rotation은 과거 event를 re-HMAC하지 않는다.

### 10.11 `AutoReviewProviderSafetyState`

`(purpose, provider, model, reasoning_effort)` unique이며 purpose는 `extraction|validation`이다.
각 row는 authorized cost-policy, estimator/encoding/framing, exact six-place Decimal prices,
optimistic `state_version`, breaker/reason, overrun count/cost/time, regression gate,
authorization/clear time, `last_event_sequence/id` current aggregate만 가진다. mutable shared
operator/call attribution slot은 두지 않는다. row는 명시적인 initial policy authorization 뒤에만
존재한다. overrun breaker는 purpose별로 persistent하고 restart/config change로 닫히지 않는다.
clear는 resolved evidence, CAS, reviewed regression gate와 **다른** cost-policy version/price
snapshot을 요구한다.

`AutoReviewProviderSafetyEvent`는 `initial_authorized|budget_overrun|breaker_cleared` append-only
ledger다. 각 event는 exact parent purpose/provider/model/reasoning, estimator/cost-policy/price
snapshot, prior/new state, gap 없는 sequence/timestamp, 그리고 operator event에는 actor HMAC,
overrun event에는 call HMAC과 그 event의 key version/material을 저장한다. operator event에 call
HMAC, overrun event에 actor HMAC을 넣을 수 없다. update/delete/parent delete/sequence gap/
wrong-parent identity는 DB guard와 `ON DELETE RESTRICT` FK로 거절한다. aggregate update, event
insert, backpointer CAS는 한 transaction이며 replay/lost CAS는 새 event를 만들지 않는다. 과거
event는 key rotation 때 재작성하지 않는다.

## 11. Transaction, concurrency, cache

### 11.1 Validation key

```text
HMAC-SHA256(domain='auto-review-validation-key:v1', canonical_json(
  workflow_execution_identity_hmac
  + security_scope_hmac
  candidate_key
  + evidence_version_hash
  + normalized_claim_fingerprint
  + candidate_generation_fingerprint
  + validator_provider/model/reasoning_effort
  + validator_prompt_version
  + validator_output_contract_version
  + policy_version
  + credential-scan:v1 policy identity
  + fingerprint_key_version/material_verifier
  + token_estimator_version/encoding/reply_priming/framing_safety
  + max_input_tokens/max_output_tokens/max_candidates_per_batch
  + max_batches/max_candidates_per_workflow/max_provider_attempts
  + provider_timeout/send_start_window/attempt_lease/commit_grace
  + exact validator Decimal input/output prices
  + cost_policy_version
  + authorized_percentage_at_launch/rollout_authorization_generation
  + confirmed extraction/validation/total ceilings
)))
```

canonical JSON, UTF-8 NFC, stable ordering, current fingerprint key version/material verifier를
사용한다. `candidate-evidence-state:v1`은 same-workflow canonical ref의 source kind/id/version,
content/message-set HMAC, permission과 key identities를 sorted tuple로 bind한다.
`candidate-generation:v1`은 exact agent/provider/model/reasoning/prompt/route/output-contract와 key identities만
bind하며 mutable metadata를 읽지 않는다. validation-key uniqueness는
`(workflow_thread_id, validation_key)`이고 batch fingerprint는 deterministic partition/aliases,
sorted children, workflow/scope, provider/cost/key/rollout/ceiling snapshot 전체를 추가로 bind한다.
같은 content라도 workflow 또는 scope가 다르면 cross-replay할 수 없다.

### 11.2 Lease sequence

1. short read phase에서 pending ReviewItem/immutable refs를 읽고 workflow owner의 current exact
   permission, canonical source/version/signature/content, exact duplicate/hidden collision을
   preflight한 뒤 모든 ORM/Session/lock을 닫는다.
2. DB transaction 없이 deterministic frame sizer로 candidate를 partition하고 최종 batch마다
   `prepare_many()`를 **정확히 한 번** 호출해 immutable `PreparedValidationInvocation`을 만든다.
3. transaction A는 shared generation/runtime -> exact
   `(purpose='validation', provider='openai', model='gpt-5.6-terra', reasoning='medium')` provider safety -> immutable
   workflow 순서로 lock/revalidate한 뒤 동일 prepared object의 HMAC/count와 signed worst-case
   reserve, batch/child identities를 atomic claim하고 commit한다. replay/busy/mismatch는 object를
   폐기하고 provider를 호출하지 않는다.
4. provider 직전 transaction A3가 같은 leading order와 call/children CAS 아래 owner/source,
   key/mode/breaker, exact purpose/provider/model/reasoning/cost-policy/price,
   prepared HMAC/count/caps를 재확인하고 attempt
   count 0→1, started time, `started_at + stored_lease_seconds`를 commit한다.
5. DB transaction 없이 동일 prepared object를 Terra에 정확히 한 번 보내고 structured batch를
   검증한 뒤 plaintext DTO/alias/messages를 즉시 폐기한다.
6. transaction B는 shared generation/runtime -> validation provider safety -> projection -> rollout
   -> workflow -> validation call/children -> ReviewItem -> promotion decision/audit -> provenance/
   knowledge/document locks 순서를 가진다. owner/source/permission, key/projection readiness,
   duplicate/collision, policy/mode/rollout/budget를 모두 재검증한다.
7. bounded validation, deterministic decision, authoritative call usage/cost와 exact child allocation을
   저장한다. enforce approval, immutable PromotionDecision와 selected audit는 이 같은 locked
   transaction에서만 만든다.

동일 batch 경쟁자는 canonical completed result를 재사용한다. `busy`는 human fallback이나
interrupt가 아니라 같은 `run_auto_review` node의 bounded retry 상태다. pre-attempt expired lease만
CAS reclaim하고 attempt-started call은 provider를 다시 호출하지 않는다. source text는 provider
call 동안 process memory에만 존재하고 persistence, checkpoint, log, cache, AuditLog에 들어가지
않는다.

### 11.3 Bounded work와 cost

- 한 validation provider call당 후보 최대 4개
- evidence slot 최대 12개
- claim당 2,000자, serialized validation input 최대 12,000자
- `openai-o200k-chat:v1` framed validation input 최대 6,000 tokens, visible+reasoning을
  합친 Responses total output 최대 3,072 tokens
- validation 최대 2 batches/5 candidates/workflow, provider attempt 최대 1/batch
- V2.1 extraction agent당 24,000 input chars, 10,000 framed input tokens, canonical/Responses
  total output 2,048 tokens, 후보 0..1, provider attempt 최대 1이며 더 작은 route cap이 우선
- extraction full-cap reserve는 agent당 exact `Decimal('0.016716')`, 다섯 agent 최대
  exact `Decimal('0.083580')`
- validation batch reserve는 exact `Decimal('0.048864')`, 두 batch/workflow 최대
  exact `Decimal('0.097728')`
- full approved profile은 exact `Decimal('0.181308')`, immutable workflow limit는
  exact `Decimal('0.20')`, headroom은 exact `Decimal('0.018692')`
- non-empty selected-agent subset `N`은 `1 <= N <= 5`이며 extraction은
  `N * 0.016716`, validation은 `ceil(N/4) * 0.048864`를 reserve한다. observed/predicted
  zero-candidate 결과로 preview reserve를 할인하지 않는다.

validation candidate/batch/input cap 초과는 기존 pending Review Queue의 human-review 경로다.
반면 extraction의 malformed/truncated response 또는 complete canonical envelope 2,048-token 초과는
그 paid extraction call을 charged `failed`로 종결하고 ReviewItem이나 signed `no_candidate` success
marker를 만들지 않는다. cap 변경은 자동 승인 allowlist를 확장하지 않지만 cost-policy와 해당
prompt/output-contract version, regression evidence를 갱신해야 한다.
위 금액은 승인된 OpenAI `gpt-5.4-mini-2026-03-17`/none extraction과
`gpt-5.6-terra`/medium validation registry의 immutable six-place price/cap identity다. 배포
설정은 exact equality만 확인하며 임의 override가 아니다. 모든 USD reserve/charge/total은
`Decimal`을 six-place `ROUND_CEILING`하며 float로 변환하지 않는다. provider-reported actual이
cap/reserve/ceiling을 넘으면 actual을 보존하고 purpose별 global safety breaker를 열며 승인,
retry, clamp를 모두 금지한다.

### 11.4 Explicit paid-run confirmation

`shadow`도 Terra를 호출하므로 status API, sync polling, page load, dry-run은 provider를
절대 호출하지 않는다. 기존 Integrations의 `검토 후보 미리보기 -> 검토 후보 만들기`
명시적 실행 안에서만 validation을 수행한다.

V2.1 dry-run은 extraction, auto-review upper bound, total estimated tokens/cost, selected mode,
policy/model version을 같은 panel에 표시하고 TTL 600초, 최대 2,048자의 HMAC launch
confirmation token을 반환한다. token은 existing strong agent-runtime fingerprint key와 domain
`auto-review-launch:v1`, base64url compact canonical JSON, HMAC-SHA256, constant-time comparison을
사용한다. 하나의 frozen codec만 다음 exact compact key set을 허용하고 missing/unknown key를
거절한다.

| compact key | exact identity |
|---|---|
| `v,g,m` | launch schema, graph version, configured mode |
| `sh,ah,ph,ih,eh` | keyed scope, actor, server-resolved owner-permission, input, evidence identity |
| `vp,vm,vr,vs,pp,pv,cv,ps` | validator provider/model/reasoning/output-contract/prompt/policy/cost-policy/provider-safety state version |
| `xv,xp,xm,xr,xt,xn` | extraction cost-policy/provider/model snapshot/reasoning/route/selected-agent count |
| `xc,xi,xo,xk` | extraction 24,000-char/10,000-input/2,048-output/one-candidate caps |
| `xe,xen,xrp,xfs,xip,xop` | extraction estimator/encoding/reply-priming/framing-safety/exact prices |
| `ep,es` | sorted selected extraction-plan-set HMAC와 complete extraction provider-safety snapshot-set HMAC |
| `rc,rq,ap,rg` | rollout control epoch/requested percentage/authorized-at-launch percentage/authorization generation |
| `kv,km` | fingerprint key version/material verifier |
| `te,en,rp,fs,it,ot` | validation estimator/encoding/reply-priming/framing-safety/6,000-input/3,072-output caps |
| `mb,mc,mw,ma` | validation 2 batches/4 candidates per batch/5 candidates per workflow/one attempt |
| `pt,sw,ls,cg` | shared provider timeout/send-start window/attempt lease/commit grace |
| `ip,op,bl,ec,vc,tc` | exact validator prices/USD 0.20 budget/extraction·validation·total ceilings |
| `iat,exp` | issuance/expiry |

`ep`는 정렬된 selected-agent names, 각 agent의 exact prompt/output schema, prepared-input
HMAC/count, lower effective caps, one-candidate contract와 timing tuple까지 commit한다. `es`는
각 selected `(purpose, provider, model, reasoning_effort, state_version)`을 commit한다. common
extraction identity는 digest 안에만 숨기지 않고 `x*`에 명시한다. 최대 valid token도 2,048자
이하여야 하며 exact-key-set/missing/unknown/maximal-size fixture를 release gate로 둔다.

signed selected-agent count `N`은 1..5이고 extraction ceiling은 exact
`N * USD 0.016716`, validation ceiling은 exact `ceil(N/4) * USD 0.048864`다. 5-agent
profile은 extraction USD 0.083580 + validation USD 0.097728 = USD 0.181308이다. zero-candidate
예측으로 할인하지 않는다.

dry-run/preview는 row를 쓰거나 provider를 호출하지 않는다. launch 전 token key/material,
permissions/source, purpose별 provider safety, rollout control fields, prepared extraction plan과
모든 output-contract/timing/cap/price/ceiling을 current server state에서 다시 계산한다. `ep`/`es`를
recompute한 뒤 required safety row를 `(purpose, provider, model, reasoning_effort)` 순서로 lock하고
rollout을 ensure/lock한 다음 signed snapshot을 재검증한다. 하나라도 달라지거나 recomputed
obligation이 signed ceiling을 넘으면 thread/provider call 전에
`cost_preview_changed`로 거절한다. rollout counter만 변해 `state_version`이 올라간 경우에는 token을
무효화하지 않는다. 기존 key를 domain-separated하게 재사용하며 shadow/enforce는 known
development placeholder 또는 32 UTF-8 byte 미만 secret으로 시작할 수 없다.
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

V2.1 `draft`의 live extraction은 먼저 signed prepared plan을 one-render하고 purpose=`extraction`
provider-safety -> workflow -> AgentRun/extraction-call 순서의 short reserve/attempt/finalization
transactions 사이에서 같은 ephemeral invocation을 정확히 한 번 호출한다. source/owner permission은
plan 전, attempt marker 직전, provider 결과 commit 전에 다시 해석한다. 24,000 chars/10,000 framed
input/2,048 canonical-total-output 또는 더 작은 route cap, credential scan, exact price/ceiling 중 하나라도 실패하면
provider를 호출하지 않거나 결과를 폐기하고 trusted candidate를 만들지 않는다.

- effective `disabled`: auto-review node는 zero-call pass-through
- `shadow`: validation/decision/cost만 저장하고 ReviewItem status는 바꾸지 않음
- `enforce`: stable candidate HMAC으로 enforce percentage에 선택된 eligible result만
  transition하고 나머지는 shadow처럼 pending 유지

V2.0/V2.1 builder와 V2.1 lifecycle/coordinator는 disabled 또는 provider key 부재에도 항상
register/compile한다. readiness는 신규 shadow/enforce preview/start만 막는다. 이미 저장된
V2.1은 global disabled에서 zero-call로, provider unavailable에서는 bounded human fallback으로
resume하며 V2.0이나 503으로 바뀌지 않는다.

저장된 V2.1의 extraction/validation purpose별 provider-safety identity와 rollout
`control_epoch`는 더 강한 실행 권한으로 갱신하지 않는다. current control/provider state는 실행을
demote할 수만 있다. preview 뒤 control epoch나 purpose별 safety authorization이 바뀌면 새 launch는
thread 생성 전 `cost_preview_changed`; 이미 저장된 thread는 zero-call/human-only로 안전하게
resume한다.

`run_auto_review` state에는 workflow thread id만 두고 runtime dependency가 저장된 owner의
current permission을 매 실행/resume와 provider 전후에 server-side로 다시 해석한다. effective
visibility는 `owner allowed levels ∩ ('public','internal')`이며 auto actor 자체 권한으로
owner보다 넓힐 수 없다. permission context는 checkpoint/token raw field로 저장하지 않는다.

route 우선순위는 다음으로 고정한다.

1. `pending_review > 0`: 기존 `interrupt()`로 사람에게 남은 예외를 보낸다.
2. pending이 0이고 `needs_more_evidence > 0`: `finalize_needs_more_evidence`.
3. 나머지 `approved + rejected + revoked == total`: `finalize_auto_resolved` 또는 normal
   completed.

따라서 모든 후보가 auto resolved면 interrupt가 없고, needs-more와 pending이 함께 있으면
pending을 먼저 사람이 처리한다.

`revoked`는 resolved-but-not-approved 상태다. 과거 checkpoint를 수정하지 않는다. status/resume
reconciliation은 total/resolved가 같고 verified auto-policy ReviewItem의 `approved -> revoked`
delta만 monotonic하게 허용한다. pending/needs-more/rejected/total drift 또는 human item 변화는
fail closed한다. live serving eligibility와 현재 Review projection에서는 즉시 제외한다.

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
- required selected audit이 없거나 `completed/confirmed`임. pending, remediation, critical outcome은
  normal direct revoke를 `audit_required`로 막는다.

normal gate bypass는 이미 finalized된 matching critical audit를 가진 breaker-first recovery
context와 위 `SourceInvalidationRevokeContext` 두 가지 server-owned narrow context뿐이다. 전자는
critical remediation exact revoke만, 후자는 current source mismatch로 affected effect를
invalidate/narrow하는 일만 허용한다. public body/dependency, human reviewer, auto-policy actor는 둘을
construct할 수 없다.

효과:

1. ReviewItem status를 `revoked`로 변경
2. 이 ReviewItem의 approval/evidence link를 `revoked`로 변경
3. 같은 canonical effect에 active trusted provenance가 남았는지 row lock 아래 재계산
4. 남아 있지 않을 때만 linked Decision/History/Timeline/Todo와 companion Timeline의
   `review_status`를 `revoked`로 변경
5. 4번이 실행된 경우에만 exact vector document id와 vector index state를 삭제
6. bounded audit event 기록

post-approval source drift는 reviewer가 발견할 때까지 serving하지 않는다. source
version/signature/content 또는 permission이 바뀐 commit 직후 모든 read path의
`TrustedServingEligibility`가 auto-only stale target을 먼저 제외하거나 stricter permission으로
제한한다. synchronous reconciliation은 permission-only narrowing이면 trust를 유지한 채 target과
vector를 좁히고, restricted/unknown/deleted/superseded이면 exact auto effect를 server-owned
context로 revoke한다. shared human provenance는 보존하며 last provenance만 tombstone/delete한다.
reindex는 embedding 전과 locked upsert 전 같은 live-source predicate를 확인하므로 reconciliation
pending/crash 뒤에도 stale vector를 resurrect하거나 hidden-match count에 포함할 수 없다.

row를 물리 삭제하지 않는다. 사람이 승인한 item은 이 action으로 revoke할 수 없다.
request는 exact `reason_code` enum만 받으며 free text 또는 LLM 분류를 받지 않는다. gate 통과
후 `AutoReviewRevocationAssessment`를 insert-once하고, 동일 reason 재요청은 canonical result와
`replayed=true`, 다른 reason 재요청은 `revoke_reason_conflict`를 반환한다.
공유 knowledge에 active provenance가 남은 replay/response는
`knowledge_remains_trusted=true`를 반환하되 다른 ReviewItem id는 노출하지 않는다.

`revoked`는 V1 terminal 상태이며 같은 ReviewItem을 다시 approve하지 않는다. 복원하려면
새 canonical source version에서 새 candidate를 만들고 사람이 승인해야 한다. direct
`business_withdrawal`의 knowledge/link/vector/tombstone 변경은 한 transaction이므로 실패 시
그 direct revoke transaction을 rollback한다. 네 quality reason은 별도 breaker-first coordinator가
assessment와 audit-or-correction, rollout breaker/generation invalidation, serving quarantine를 먼저
commit한다. 이후 physical revoke가 실패해도 그 안전 상태는 rollback하지 않고
`remediation_required`로 남겨 recovery가 재시도한다. success response는 physical revoke 뒤에만
반환한다.

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
  - prompt/output-contract/policy version
  - supported substantive field count와 minimum entailment score
  - public allowlist `direct_fact_supported|trusted_exact_reaffirmation` only
  - validation timestamp
- bounded nullable `auto_review_audit`
  - `status`: `pending|completed|remediation_required`
  - nullable allowlisted outcome
  - `action_required`

source content, raw model output, validation/promotion/audit key, lease, internal DB id, raw reason,
cohort counter는 노출하지 않는다.
`auto_review_summary`는 same-item completed canonical validation에서만 만들고 hidden-collision,
lookup, permission, registry, budget, drift/failure internal code를 직접 pass-through하지 않는다.
corrupt/unsupported legacy data는 optional summary를 null/fail-closed 처리한다.

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

세 count는 workflow-owned current rows의 disjoint live projection이다.
`auto_approved_count`는 current `approved + resolution_source='auto_policy'`,
`human_review_required_count`는 current `pending_review`, `auto_review_fallback_count`는 current
`needs_more_evidence`다. validator/policy/enforce-not-selected human-review case는 pending count에만,
revoked/rejected/human-approved는 어느 count에도 들어가지 않는다.

### 14.3 Revoke endpoint

```text
POST /api/v1/review/{review_item_id}/revoke-auto-approval
```

request는 strict `reason_code` 하나만 요구한다.

```text
business_withdrawal | incorrect_content | permission_violation |
wrong_source_version | policy_violation
```

free text/note는 받지 않는다. foreign/inaccessible item은 기존 Review API의 concealment
규칙을 유지하고, unsupported transition, audit gate, reason conflict, remediation은 bounded
409 code를 반환한다. quality reason은 breaker/quarantine가 commit된 뒤 physical revoke가
끝나기 전까지 success schema가 아니라 `remediation_required`를 반환한다.

response는 `review_item_id`, `status='revoked'`, `replayed`,
`knowledge_remains_trusted`, `revoked_document_count`만 반환한다. 다른 provenance identity나
document id는 노출하지 않는다.

### 14.4 V2.1 dry-run and launch additions

dry-run `graph_version`은 V2.0/V2.1 union이며 다음 additive field를 가진다.

- `auto_review_mode`
- `auto_review_policy_version`, V2.1에서 non-null
- validator provider/model/reasoning/prompt/output-contract/policy/cost-policy identity
- `auto_review_estimated_input_tokens`
- `auto_review_estimated_output_tokens`
- `auto_review_estimated_cost_usd`
- `total_estimated_input_tokens`
- `total_estimated_output_tokens`
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
unsampled item의 명시적 감사는 기존 immutable `not_selected` PromotionDecision을 참조하는
`manual` cohort만 만든다. 새 ordinal/selection/counter를 만들지 않는다. response는 bounded audit status,
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

- workflow owner의 current exact allowed permission levels를 server-side resolver에서 매
  pre-call/post-call/resume 시 읽으며 `owner ∩ auto-policy allowlist`만 사용한다. client claim이나
  auto actor 고정 권한으로 넓히지 않는다.
- source, ReviewItem, knowledge 중 strictest permission을 유지한다.
- unknown permission은 fail closed한다.
- restricted는 `v1` auto approval에서 제외한다.
- extraction/validation provider input 전에 permission filter와 `credential-scan:v1`을 적용한다.
  scanner match는 plaintext 폐기와 zero prepare/invoke/cache/trace이며 public reason은 generic이다.
- extraction/validation model result 이후 새 DB session에서 permission, source version/signature,
  content fingerprint를 재확인한다.
- candidate evidence ref가 없거나 canonical workflow ref와 정확히 일치하지 않으면 모델을
  호출하지 않는다.
- trusted collision 조회도 current actor/security scope와 exact permission filter 안에서만
  수행하고 hidden record identity/count를 노출하지 않는다.
- prompt injection text는 evidence data일 뿐 instruction으로 취급하지 않는다.
- raw source, URL, snippet, prompt/rendered DTO, model rationale/output, provider error, credential은
  checkpoint, call/validation rows, cache, signed token, AuditLog에 넣지 않는다.
- 새 C.5 business table은 raw human subject id를 저장하지 않는다. revoke, post-audit, rollout/
  provider/key admin attribution은 action-specific versioned HMAC domain과
  fingerprint-key version/material verifier를 함께 저장한다. historical HMAC은 rotation 때
  re-HMAC하지 않고 새 control action만 current key identity를 사용한다.
- 기존 access-controlled AuditLog는 별도 identity audit surface다. review adapter는 actor type과
  governed subject identity, review item id, model/policy version, outcome/error code, bounded count/
  cost만 allowlist로 기록하고 free-form source/model/admin output을 복제하지 않는다.

local admin CLI는 host의 restricted operations identity만 신뢰한다. key CLI는 fixed
`system:local-auto-review-key-admin`, rollout/provider/recovery CLI는 fixed
`system:local-auto-review-rollout-admin`으로 HMAC attribution하며 `--subject`, actor, credential,
secret/verifier/raw-output option을 받지 않는다. secret은 validated `Settings`/CLI-only secret-manager
key ring에서만 읽고 출력하지 않는다. future remote endpoint는 authenticated admin principal을
server dependency에서 받아야 하며 body subject impersonation을 금지한다. CLI output은 aggregate
count/mode/generation/version/readiness/breaker만, exit code는 success `0`, bounded refusal `2`,
readiness/remediation failure `3`으로 고정한다. provider policy authorization/clear는 purpose
`extraction|validation`을 명시적으로 받아 exact
`(purpose, provider, model, reasoning_effort)` row만 mutate한다.

지원 action은 key `status|bootstrap|rebuild|rotate`, rollout/provider
`status|authorize|close-breaker|authorize-initial-provider-policy|clear-provider-breaker|
recover-remediation`과 bounded source-reconciliation recovery뿐이다. key `rotate`는
`expected-version`, `next-version`, 1..500자 reason을 요구하고 secret argument를 금지한다.
rollout mutation은 `expected-state-version`, reason, bounded gate-ref를 요구한다. provider initial/
clear는 `purpose`, provider/model/reasoning, new cost-policy를 반드시 bind하고 clear는 different policy를
요구한다. source/audit recovery는 먼저 unlocked id scan(`limit<=100`)만 한 뒤 canonical global
lock order로 각각 처리한다.

source reconciliation의 executable continuation은
`python -m backend.app.admin.auto_review_source_reconciliation status --limit 100`과
`recover --limit 100`뿐이다. fixed `system:local-auto-review-source-reconciler` attribution을 쓰고
subject/source-id/secret/raw-output option을 금지한다. stale/reconciled/remaining/failure aggregate와
readiness만 출력한다. lifespan은 exactly one bounded batch만 실행하고 남은 work는 CLI로
계속한다.

## 17. Error handling

| 오류 | 동작 |
|---|---|
| validator unavailable/timeout | human review |
| malformed structured output | human review |
| extraction/validation credential scan match | zero-call generic human-only, matched bytes discard |
| extraction route/global hard cap exceeded | zero-call unavailable, truncate/fallback 없음 |
| source version changed | needs more evidence |
| permission changed/unknown | auto approval 금지 |
| budget exceeded | human review, provider call 없음 |
| actual usage/cost exceeds reserve/cap | actual charge 보존, purpose별 provider breaker open, no retry/approval |
| paid preview changed/expired | start 거절, 새 zero-call preview 요구 |
| policy/model version unavailable | human review |
| trusted conflict lookup unavailable/ambiguous | human review, provider call 없음 |
| validation lease expired before attempt | existing reservation을 유지한 CAS reclaim; terminal `expired` row 없음 |
| validation lease expired after attempt | reserved charge로 `failed`, no retry/release |
| concurrent human/auto approval | locked transition 한 건만 승리 |
| promotion replay | existing canonical ids 반환 |
| revoke provenance incomplete | fail closed, serving state 변경 없음 |
| vector delete/tombstone failure | direct business revoke transaction은 rollback; quality path는 committed breaker/quarantine를 유지하고 remediation_required |
| reindex races with revoke | final eligibility recheck 또는 tombstone skip |
| critical post-audit outcome | persistent breaker를 먼저 open, exact revoke 시도 |
| post-audit revoke failure | breaker 유지, remediation_required |
| post-approval source lookup/version/permission drift | serving 즉시 exclude/narrow 후 synchronous exact reconciliation |
| source reconciliation crash | live predicate가 fail closed, bounded startup/admin recovery |

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
- real-looking credential/connector-token/password assignment과 allowlisted fake/example negative
- duplicate reaffirmation and trusted conflict
- Decision, Todo, restricted forced-human cases
- post-approval permission narrowing/loss와 source supersession

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
- 별도 100% authorization 뒤 생성된 full-enforce workflow만 2% sample audit

모든 auto approval은 selected 여부와 관계없이 immutable `AutoReviewPromotionDecision`을 하나
만들고, `first_50|sample_10|sample_2` decision만 required audit row를 함께 만든다.
`not_selected`의 manual audit는 기존 decision identity를 재사용한다. purpose별 provider-safety
row가 missing/open/mismatch이거나 rollout authorization generation이 달라지면 zero-call/shadow로
demote하며 decision/promotion을 만들지 않는다.

10% canary에서 최초 50 auto approval은 모두 pending audit row를 만든다. 정상 cohort의
ordinal 51 진입은 50 mandatory human-confirmed audit이 채워진 뒤에만 가능하다. 다만 source
invalidation으로 pending mandatory audit가 precision-excluded 종결되면 후속 eligible promotion은
50 confirmed를 채울 때까지 ordinal을 가진 mandatory replacement audit로만 허용하며 unsampled/
10% sample이 아니다. 그 뒤 normal canary promotion 500건까지 stable HMAC 10%를 계속 감사하고,
누적 500건과 모든 selected audit 완료, critical 0, 감사 precision 99%
이상을 만족한 뒤에만 authorized operator가 persistent latch를 10→100으로 올리고 신규
workflow의 configured percentage를 100으로 선택할 수 있다. 그 authorization 이후 생성된
stored-100 workflow만 stable 2% audit를 사용한다. 기존 10% workflow는 ordinal 501 이후에도
10% audit이며 어떤 stage도 자동 승격하지 않는다.

shadow comparison은 policy가 `auto_approve`로 예측했고 evidence version이 바뀌지 않은
후보를 authorized human이 나중에 처리한 결과만 집계한다. unresolved, drifted, skipped
후보는 precision denominator에 넣지 않는다. shadow precision은 human-approved completed
comparison / all completed comparisons, enforce audit precision은 `confirmed` / all completed
audit outcomes다. enforce percentage 변경은 신규 workflow에만 적용하고 진행 중 thread의
저장된 percentage를 높이지 않는다.

rollout preview/start는 latch/authorization/breaker의 `control_epoch`만 bind한다. ordinary metric/
audit counters는 `state_version`만 바꾸므로 launch를 불필요하게 무효화하지 않는다. breaker close와
새 authorization은 control epoch/generation을 바꾸며 기존 workflow를 resurrect하지 않는다.

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
- exact Decimal six-place ceiling/reserve/actual/overrun과 one-attempt ledger
- exact five-entry extraction registry, singular 0/1 cardinality, `N * 0.016716` extraction과
  `ceil(N/4) * 0.048864` validation reserve arithmetic
- per-field-valid extraction payload도 canonical complete envelope가 2,048 `o200k_base` tokens를
  넘으면 거절되고 partial/truncated JSON을 복구하지 않음
- signed paid preview expiry/change
- launch token key-material, purpose별 provider-safety, `control_epoch`, estimator/caps/lease/price binding
- exact duplicate versus collision-bucket fail-closed behavior
- inaccessible collision existence routes human without identity/count leakage
- every-promotion decision, selected-only audit, sample ordinal과 breaker state machine
- invalidated mandatory audit exclusion과 50 confirmed까지 mandatory replacement selection
- actor subject HMAC domain/key identity와 no-raw-subject persistence
- runtime/projection key bootstrap, fixed locks, rebuild와 disabled-only rotation
- unknown version fail closed
- revoke idempotency
- exact revoke reason enum, same-reason replay/different-reason conflict, confirmed-audit immutable
  correction과 permanent `corrected_critical_count`

### LangChain

- actual `with_structured_output()` boundary with fake chat model
- final validation batch one-render HMAC/count/send byte equivalence
- validator Responses 6,000 framed-input/3,072 total-output, four-candidate/two-batch bounds와
  `candidate-validation-batch:v1` identity
- V2.1 extraction one-render, 24k chars/10k framed-input/2,048 canonical-total-output
  hard/effective-route caps와 no retry/fallback
- malformed/missing/unknown slot rejection
- provider error sanitization
- prompt has no canonical id, URL, permission, credential
- extraction/validation `credential-scan:v1` match makes prepare/invoke/cache/trace zero calls

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
- pre-attempt-only expired lease recovery와 post-attempt reserved-charge/no-retry terminalization
- revoke/promotion race
- exact knowledge/vector revoke
- raw chunks remain human-approved-only when resolution source is auto policy
- tombstone versus in-flight reindex cannot resurrect serving content
- shared reaffirmation revoke leaves unrelated active provenance serving
- internal auto actor cannot be forged through a public API
- first 50 audit rows cannot be skipped and critical audit opens breaker before revoke
- post-audit/revoke failure remains remediation-required and enforce stays disabled
- recovery uses unlocked bounded id scan before global-order item/audit locking
- purpose별 provider safety authorization/overrun/breaker/CAS and restart persistence
- provider/rollout append-only event sequence, aggregate backpointer CAS, update/delete refusal,
  key-rotation historical attribution preservation
- server signature와 parser run/policy/current-document-version pointer-only authority,
  ambiguous repair fail-close, parser-policy-only zero LLM calls
- Assistant persisted evidence dependency completeness와 list/context/summary/email/RAG 전체
  `evidence_unavailable` zero-leak projection
- permission-only source change is not skipped and narrows API/vector serving without embedding
- absent/restricted/unknown/superseded source is invisible before reconciliation and revokes exact auto effect
- shared human provenance remains trusted at strictest current permission
- bounded startup/admin source reconciliation cannot resurrect stale vector or block audit gate
- unforgeable source-invalidation context, fixed-principal aggregate-only recovery CLI와 limit 100
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
Slack에는 fake repeated-event legacy-dedupe regression 하나만 추가한다. 승인된 non-Slack
비교 gate의 기존 ten-item 목록에 새 deselection을 추가하지 않으며, full backend comparison에서
deferred failure 10개를 그대로 보존한다.

## 21. Rollback

- `AUTO_REVIEW_MODE=disabled` stops new validation and transition immediately.
- `shadow` preserves observations without changing trust state.
- existing auto-approved knowledge is not mass-deleted during rollback.
- authorized reviewers can revoke exact auto-approved items.
- critical post-audit opens a persistent breaker that config restart cannot clear.
- extraction/validation purpose별 cost breaker도 config restart로 닫히지 않으며 reviewed new
  cost-policy와 restricted admin CAS만 clear할 수 있다.
- model/reasoning/prompt/output-contract/policy/cost-policy rollback does not reuse incompatible validation cache.
- key rotation은 disabled-only fixed exclusive barrier와 explicit local key-admin CLI에서만
  수행한다. old historical HMAC은 보존하고 새 runtime generation/projection rebuild가 ready임을
  status로 증명하기 전 admission을 다시 열지 않는다.
- source reconciliation 장애 중에도 live eligibility가 stale auto-only knowledge/vector를
  fail closed하므로 mass delete나 serving bypass로 rollback하지 않는다.
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
- every paid extraction/validation attempt is one-render, one-attempt, exact-Decimal ledgered and bound
  to a purpose-specific provider-safety state and signed ceiling.
- the exact five-route Mini extraction and Terra validation profile recomputes to USD 0.181308 and
  never exceeds the immutable USD 0.20 workflow limit.
- persisted Assistant answers fail closed as a whole when any raw/trusted evidence dependency is
  incomplete, stale, revoked, quarantined, permission-incompatible, or unavailable.
- incorrect auto approval can be precisely revoked without deleting unrelated knowledge.
- post-approval source permission/version drift is excluded or narrowed at read time before synchronous
  reconciliation and can never serve through a stale vector.
- no raw source/model content enters checkpoint or audit.
- public/internal only v1 policy never broadens permissions.
- inaccessible trusted collisions can block automation without leaking identity or count.
- sampled auto approvals have durable human audit outcomes and critical findings demote
  effective mode before remediation.
- every auto approval has one immutable PromotionDecision; selected audits and actor attribution carry
  exact HMAC/key identities without raw C.5 subject ids.
- Deliverable D and Neo4j can consume trust metadata without changing source agents or API
  routes directly.

## 24. 공식 모델 참고

- [GPT-5.4 Mini](https://developers.openai.com/api/docs/models/gpt-5.4-mini)
- [GPT-5.6 Terra](https://developers.openai.com/api/docs/models/gpt-5.6-terra)
- [GPT-5.6 Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna)
- [GPT-5.6 Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol)
- [OpenAI model guidance](https://developers.openai.com/api/docs/guides/latest-model)
