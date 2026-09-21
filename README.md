# ParaWorks

한국어 중심 회사 기억 플랫폼입니다. 문서·메일·Slack 근거를 Review Queue에서 검토하고,
LangChain/LangGraph 기반 RAG가 사용자 권한에 맞춰 답변합니다.

PostgreSQL + pgvector가 기준 저장소이며, Neo4j는 관계 검색용 파생 그래프입니다.
GraphRAG와 답변 캐시는 구현되어 있지만 기본 비활성화입니다. 합성 Slack 통합 검증과
실제 Slack 연결은 별개이며, 정식 출시 준비는 아직 완료되지 않았습니다.

## 로컬 실행

이 README가 있는 체크아웃 루트에서 실행합니다. 의존성과 PostgreSQL을 먼저 준비하고
[설정·실행·종료 안내](scripts/README.md)를 확인하세요. 기존 `.env`는 덮어쓰지 마세요.
현재 표준 작업 위치는 원래 프로젝트 루트 `main`, Python 환경은 `.venv`입니다.
과거 `.worktrees/review-hitl-v2-design` 또는 `.venv-task4-r3-review`를 지정하지 않습니다.

```powershell
.\scripts\test-provider.ps1     # 설정만 확인: 외부 호출 없음
.\scripts\test-services.ps1     # 실제 DB 연결·스키마 확인
.\scripts\start.ps1             # 실제 서비스 모드
.\scripts\status.ps1
.\scripts\stop.ps1              # 이 런처가 시작한 서버만 종료
.\scripts\restart.ps1
```

브라우저: <http://localhost:3000> · 백엔드: <http://127.0.0.1:8000/health>
이미 다른 방법으로 실행한 서버는 해당 터미널에서 먼저 종료하세요.

## 문서

- [scripts 사용법과 기존 .env 업데이트](scripts/README.md)
- [실제 GraphRAG final demo: 준비 상태와 남은 연결 작업](docs/superpowers/runbooks/final-demo.md)
- [현재 로드맵](plan.md)
- [작업 인수인계](docs/superpowers/runbooks/session-handoff.md)
- [변경·검증 이력](docs/portfolio-log.md)

기능 검증 통과는 상용 출시 승인이나 유료 모델 실행 승인을 의미하지 않습니다.
