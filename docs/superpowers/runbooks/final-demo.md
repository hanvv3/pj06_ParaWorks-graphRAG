# Final demo — 실제 서비스 경로와 준비 상태

2026-09-21 확인. 실행 위치는 저장소 **루트/main**, Python은 루트 `.venv`, 설정은
루트 `.env` 하나다. 이전 `.worktrees/review-hitl-v2-design`에서 실행하지 않는다.

## 목표와 현재 판정

목표는 SQLite/fake 모델 시연이 아니라 실제 PostgreSQL + pgvector + Neo4j + provider를
사용하는 브라우저 시연이다. 데이터는 권한 있는 문서 또는 승인된 합성 데이터여도 되지만,
실제 Slack 연결 성공이나 실제 고객 데이터 검증으로 표현하지 않는다.

**현재는 서버 실행 도구 준비 완료, fully functional GraphRAG final demo는 미완료다.**
E/D.1/S3의 실제 DB + fake 모델 테스트 통과와 브라우저에서 실제 모델까지 연결된
최종 시연은 다르다. 정식 release 미완료만으로 앱 구현이 없다는 뜻은 아니며,
아래에 확인된 구체적 운영 연결 공백이 있다.

| 구성 | 현재 상태 |
|---|---|
| 일반 PostgreSQL 백엔드/프런트엔드 실행 | 루트 `scripts/start.ps1` 지원. DB 스키마·키·계정 준비 필요 |
| 실제 source-agent 추론과 Review | 기존 앱 preflight/유료 실행·pending Review 경로 존재 |
| V2 검색·실제 OpenAI 생성 | `backend/app/agent_runtime/rag_v2_composition.py`에 구성 구현 |
| V2 incremental indexing | 내부 함수/작업 구현. 공개 `/rag/reindex`는 legacy 인덱싱이며 V2 운영 진입점 없음 |
| Neo4j projection 갱신 | 내부 reconcile 구현. 운영용 작업/관리 진입점 없음 |
| 새 paid V2 provider-safety 설정 | committed reviewer-key registry가 비어 있어 지원되는 새 초기화 경로가 거절됨 |
| 답변 캐시 | 구현됨, 선택적/default-off. 위 공백을 해결하지 않음 |

코드 근거: `backend/app/api/v1/rag.py`, `backend/app/rag/reindexing.py`,
`backend/app/tasks/rag_indexing.py`, `backend/app/rag/graph_projection.py`,
`backend/app/admin/rag_provider_safety.py`의 `COMMITTED_PROVIDER_SAFETY_REVIEW_KEYS`.

## 지금 가능한 준비·서버 실행

PowerShell에서 저장소 루트로 이동한다. 과거 환경 선택값이 남아 있다면 현재 터미널에서
`Remove-Item Env:UV_PROJECT_ENVIRONMENT -ErrorAction SilentlyContinue`로 해제한다.

```powershell
uv sync --locked
npm --prefix frontend ci
docker compose up -d postgres
.\scripts\test-provider.ps1
.\scripts\test-services.ps1
```

이미 만든 외부 PostgreSQL을 사용한다면 Compose 명령 대신 그 인스턴스를 켜고 `.env`의
DB 주소를 일치시킨다. 기존 DB·서명 키를 보존한다. 다른 DB로 몰래 전환하거나 볼륨을 지우지 않는다.
Neo4j는 현재 Compose에 포함되지 않는다. final demo에는 별도 인스턴스와 연결정보가 필요하다.

빈 개발 DB를 명시적으로 초기화하는 경우에만 다음을 실행한다. 기존 DB는 먼저 백업하고
마이그레이션 범위를 확인한다. 초기화가 final demo 데이터·인덱스·승인 authority를 만드는 것은 아니다.

```powershell
.\scripts\start.ps1 -InitializeDatabase -SkipApp
.\scripts\test-services.ps1
.\scripts\start.ps1
```

브라우저 `http://localhost:3000`, API health `http://127.0.0.1:8000/health`.
일반 start는 demo seed를 넣지 않으므로 계정을 별도로 준비해야 한다.
서비스 시작 성공은 GraphRAG 활성화나 유료 추론 성공의 증거가 아니다.

```powershell
.\scripts\status.ps1
.\scripts\stop.ps1
.\scripts\restart.ps1
docker compose stop postgres
```

마지막 명령은 Docker DB만 중지하며 데이터를 보존한다. `down -v`는 사용하지 않는다.

## .env 준비

- 기존 `OPENAI_API_KEY`, `DATABASE_URL`, fingerprint/auth 비밀값을 재사용한다.
- 실제 모델·임베딩 단가와 비용 상한을 확인한다. 모델의 현재 단가를 추정해서 넣지 않는다.
- Neo4j의 `RAG_NEO4J_URI`, `RAG_NEO4J_USERNAME`, `RAG_NEO4J_PASSWORD`,
  `RAG_NEO4J_DATABASE`를 해당 인스턴스에 맞게 작성한다.
- 현재는 RAG V2 rollout/graph/cache 기본 disabled를 유지한다. 위 공백을 해결하기 전
  `enforce`나 graph=true를 복사해서 켜는 것은 준비 절차가 아니다.
- source-agent와 이메일 assistant의 모델 사용 스위치는 별개다. 목적에 맞는 consumer만 켠다.

`test-provider.ps1 -Live`는 실제 키를 사용하는 메타데이터 조회 1회일 뿐 생성 검증이 아니다.
이번 정리에서는 실행하지 않았다. 키를 출력하거나 새로 발급하지 않았다.

## Final demo를 완성하는 최소 추가 작업 — 아직 미구현

1. **Provider-safety 초기 설정:** 검토된 reviewer verifier 등록과 기존 서명 승인 초기화
   경로를 사용할 수 있게 한다. 비밀키를 Git에 넣거나 테스트 authority로 대체하지 않는다.
2. **V2 인덱싱 운영 진입점:** 권한 있는 운영자가 기존 V2 incremental job을 실행하고
   indexed/skipped/cost를 확인할 수 있게 한다. legacy reindex를 V2 완료로 표시하지 않는다.
3. **그래프 갱신 운영 진입점:** 정확한 principal/security scope와 current generation에
   대해 projection을 reconcile하고 준비 상태를 확인한다. 무제한 전체 사용자 재구축은 하지 않는다.
4. **실제 브라우저 검증:** 문서 수집 → AI 후보 → 사람 Review → V2 인덱싱 → 그래프 갱신 →
   Assistant 질문 → 실제 provider 사용량·근거 확인. graph-off 비교와 권한 제한도 확인한다.

이 작업에는 기존 운영 진입점·승인 연결의 구현이 필요하다. scripts 폴더 정리만으로
완료되지는 않는다. 별도 짧은 설계/승인 후 구현하며, 정식 release P1/R1을 해결했다고
선언하거나 권한·근거·비용 검증을 해제하지 않는다.

위 조건과 승인된 활성화 절차가 마련된 뒤 목표 경로는 RAG V2 enforce/assistant,
pgvector retrieval + graph enrichment다. 처음에는 cache-off로 관계 검색 효과를 확인하고
그 다음 cache-on으로 반복 호출 절감을 확인한다. 지금 이 문서는 활성화 승인이 아니다.
