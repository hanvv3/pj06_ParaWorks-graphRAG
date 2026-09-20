# Slack — 합성 데이터 기반 복구 설계

상태: **향후 단계 설계, 미구현** / 2026-09-20 / 코드 점검 `116347d`.
[공통 계약](2026-09-20-remaining-deliverables-design.md) / [구현 계획](../plans/2026-09-20-slack-recovery.md).
순서: **D 기능 기준선 → E GraphRAG → D.1 캐시 → Slack → release readiness**.
Slack은 마지막 기능 단계다. 실제 source 부재는 앞선 기능이나 합성 검증의 진입 조건이 아니다.

## 목표와 현재 한계

기존 실제 Slack 데이터 소스를 사용할 수 없으므로 명시적으로 합성한 대화로 수집부터
검토·승인·검색까지 재현한다. 이를 과거 회사 대화 복구나 live Slack 검증으로 표시하지 않는다.
첫 결과는 비상용 프로토타입의 작은 로컬 데모이며, 실제 source 선택은 이후 별도 결정이다.

`connectors/slack.py`는 thread metadata를 만들지만 현재 권한을 `internal`로 고정한다.
`ingestion/service.py`의 Slack 분기는 legacy dedupe를 유지하며 C.5 current-source authority를
부여하지 않는다. 기존 unsigned Slack 행은 계속 fail-closed다. 아래 흐름은 달성할 목표이지
현재 완성됐다는 주장이 아니다. 이 호환성 차이를 테스트로 드러낸 뒤 기존 정책에 맞춰 연결한다.

## 수집·근거·승인 경계

합성 fixture → `SourceEvent` / connector registry → `sync_connector_events` →
등록된 Slack agent → `ReviewItem(status="pending_review")` → 현재 승인 전이 → indexing/search.

- `backend/app/connectors/base.py`, `registry.py`, `ingestion/sync.py`를 재사용한다.
  agent는 `AgentManifest`/`AgentRegistry` 계약을 유지하며 API에서 별도 수집·승인 경로를 만들지 않는다.
- workspace/channel/message/thread 식별자, timestamp, participants, source URL, 본문 근거를 보존한다.
  본문에서 review의 source snippet을 추적하며 synthetic 표시와 fixture 출처를 유지한다.
- parent/reply와 channel 문맥을 작은 window로 제한한다. 오래된 parent에 붙은 새 reply,
  cursor 경계·동일 메시지 replay·서로 다른 channel의 문맥 혼합을 검증한다.
- 수정·삭제·권한 축소는 현재 source version과 serving 근거에 반영한다. upstream에서 변경을
  알 수 없는 경우 그 한계를 기록하며 감지하지 못한 데이터를 최신 상태라고 주장하지 않는다.
- restricted 입력은 모델 전송 전 필터링하고 결과에 가장 엄격한 권한을 유지한다.
  source 링크·snippet·confidence와 낮은 신뢰도의 불확실성 사유를 보존한다. 근거 없이는 Review 항목을 만들지 않는다.
- 현재 Review/C.5 승인 정책만 trusted promotion을 만든다. fixture가 임의 signature/승인을
  주입하지 않으며 old row의 일괄 trust 변경도 하지 않는다. source authority 확장이나 migration이
  필요하면 변경 범위·기존 행 처리·승인 영향을 명시하고 별도 정책 변경으로 다룬다.
- fetched/created/skipped, agent token/cost, indexed/skipped/saved embedding calls를 기록한다.
  replay는 중복 후보·중복 모델 호출·불필요한 embedding을 만들지 않아야 한다.

## 작은 데모와 검증 결과

합성 fixture는 결정·할 일·대화 맥락을 가진 공개/제한 channel, thread reply,
중복 replay, 수정·삭제·권한 축소를 포함한다. fake client/model과 deterministic indexing으로
수집 → pending 확인 → 권한 있는 검토자의 승인 → 현재 근거 검색 → replay count 비교를 보여준다.
승인 전 후보를 trusted fact로 사용하지 않고, 승인 뒤에도 철회·권한 변경 시 노출을 막는다.
E 관계 근거와 D.1 캐시에도 변경이 전파되는지 확인하며 기능 flag off의 기존 검색을 보존한다.

과거 Slack **10개 실패**는 이력이지 최신 실행 결과가 아니다. 현재 test id·실패 원인을 다시
분류하고 계약 결함/fixture 부적합/환경 부족을 나눈다. skip·삭제·xfail로 숫자를 없애지 않는다.
수정한 항목은 fresh 검증과 이유를 남기고 release manifest 기대값도 근거에 맞춰 갱신한다.

## 오프라인 완료와 실제 연결 준비

자동 테스트는 fake connector/모델, 별도 disposable DB, 외부 호출 0을 사용한다.
로컬 demo mode와 명시적 fake 주입을 사용하고 외부 서비스 자격증명이나 기존 운영 DB에 의존하지 않는다.
SQLite smoke 성공과 PostgreSQL+pgvector/Neo4j 통합 증거를 구분한다. 제품 기본 flag와 live rollout은
바꾸지 않으며 새 flag가 필요하면 구현 착수 시 기존 config에 맞춰 최소한으로 정한다.

완료 보고는 `합성 end-to-end 결과`, `남은 실패`, `실제 Slack 준비 상태`를 따로 기록한다.
실제 연결은 사용자 소유 workspace, 허가된 export, 대체 source 중 선택과 접근·사용 권한 확인 뒤
작은 범위로 검증한다. export/대체 source는 live connector 성공 증거가 아니다.
이 선택 전에도 합성 기능 작업은 완료할 수 있다. 이후 release readiness에서 전체 통합 증거를 모으며,
유료 실행·외부 연결·운영 rollout 승인은 이 문서로 부여되지 않는다.
