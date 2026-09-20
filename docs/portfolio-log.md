# ParaWorks Portfolio Log

## 2026-09-20 — 비상용 프로토타입 우선 순서 승인 반영

- 사용자 결정: 최적화와 GraphRAG 효과 검증을 먼저 진행하며, 기본 안전장치는 유지한다.
- D를 기능 기준선 F-1/F-2와 보류한 정식 release D-0~D-5로 구분했다.
  다음 개발 작업은 기존 graph/API/Assistant의 fake 회귀와 필요한 TDD 보완이다.
- 순서는 `D 기능 기준선 → E → D.1 → Slack → 정식 출시 준비`다.
  runtime이 formal reviewer/runner에 의존하지 않고 D.1도 E의 기술적 선행 조건이 아님을 코드로 확인했다.
- E의 cache-off 관계 검색 비교와 D.1의 cold/warm 생성 비용 비교를 분리했다.
  Slack은 합성 fixture 검증과 실제 source 연동을 구별한 작은 spec/plan을 추가했다.
- 기존 capability P1/NOT CLEAN은 미해결·보류이며 R1 계약 변경은 아직 확정하지 않았다.
  Review Queue·근거·권한·runtime 비용/provider safety는 후속 출시 절차와 함께 미루지 않는다.
- 수정 범위는 문서다. 제품 코드·DB·설정·유료 호출·테스트 실행·원격 push는 하지 않았다.
- 문서 검증: Markdown 14개 범위, 로컬 링크/앵커 48개, D 계획의 테스트 참조 21개
  (기존 20개 + 명시된 신규 1개) 확인 및 `git diff --check` 통과. archive/제품 파일 변경 없음.
  독립 검토에서 활성 계획의 순서·완료 기준·안전 경계 충돌은 발견되지 않았다.

## 2026-09-20 — 남은 단계 문서 재구성

아래는 순서 변경 전 정리 이력이다. 현재 순서와 다음 작업은 위 최신 항목 및 `plan.md`를 따른다.

- 요청: D Core/D.1/E 계획을 프로젝트 규모에 맞추고 반복 입력·과잉 제약·변경 비용을 줄인다.
- 코드 baseline `3f6cb0f`를 읽기 전용 점검하고 D Core, cache/graph, 문서 구조를 병렬 감사했다.
- 현재 로드맵·handoff를 짧게 분리하고 D Core 5개 구현 작업, D.1/E 각 3개 작업으로 정리했다.
- D Core capability의 최신 NOT CLEAN 결과를 반영했다. 앞선 DB 이전 권장만으로는 같은 권한의
  임의 Python 실행을 격리할 수 없다는 한계를 명시했다. 신뢰 모델 변경은 D-0의 미확정 변경안이다.
- D.1은 fresh retrieval 이후 generation 재사용, E는 bounded provenance projection/retrieval을
  초기 범위로 제안했다. 제품 코드·정책 적용·DB·provider 실행·rollout은 하지 않았다.
- 기존 D spec/plan의 경로와 상세 계약을 보존하고 참조 문서로 표시했다.
  긴 실행 이력은 [portfolio archive](superpowers/archive/2026-09-20-portfolio-history.md),
  [handoff archive](superpowers/archive/2026-09-20-session-handoff-history.md),
  [과거 로드맵](superpowers/archive/2026-09-20-product-plan-history.md)에 보존했다.
- 검증 범위는 문서 링크·실존 경로·상태 일관성·diff·제품 파일 미변경 확인이다.
  backend/frontend 테스트를 이번 문서 변경의 fresh GREEN 증거로 재사용하지 않는다.
- 독립 문서 리뷰에서 실행자 session lock 유지, authorization-bootstrap 연결,
  graph seed 선택, 최종 path 재검증의 네 누락을 보완했다.
- 문서 검증: 로컬 링크/앵커 검사, archive 3개 본문 보존 비교, `git diff --check` 통과.
  기본 문맥과 선택한 D spec/plan 전문을 포함한 읽기 분량은 약 2.5만 줄에서 약 740줄로 줄였다.
  이는 줄 수 비교이며 실제 모델 입력 토큰 측정값은 아니다.

## 확인된 이전 코드 상태

| 항목 | 기록 | 해석 |
|---|---|---|
| 코드 | `61eb6d7`, 문서 HEAD `3f6cb0f` | 이번 점검 기준 |
| Task25-B 마지막 회귀 | focused 184, adjacent 361, provider/secret 216 + 1 skip | 과거 기록 |
| 최신 독립 리뷰 | SPEC/CODE NOT CLEAN | memory authority 위조/재사용 미해결 |
| preview | reader unavailable, dispatch 0, authorization false | 실제 runner가 아직 연결되지 않음 |
| PostgreSQL/live release | 잔여 게이트 | D Core 완료로 표시 금지 |

앞으로는 완료한 결과·코드 revision·검증 명령/요약·남은 한계를 한 항목에 기록한다.
중간 진행 로그나 동일 리뷰 내용을 spec/plan/portfolio/handoff 네 곳에 복제하지 않는다.
