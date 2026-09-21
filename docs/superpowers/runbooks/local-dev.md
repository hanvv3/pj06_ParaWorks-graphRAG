# 로컬 개발

표준 작업 위치는 원래 저장소 루트 `main`이다. [작업 위치 안내](workspace.md)를 따른다.
설정은 루트 `.env`, Python은 루트 `.venv`, frontend는 루트 `frontend`를 사용한다.

```powershell
uv sync --locked
npm --prefix frontend ci
docker compose up -d postgres
.\scripts\test-provider.ps1
.\scripts\test-services.ps1
.\scripts\start.ps1
```

기존 `.env`와 서명/API 키를 보존한다. 신규 설치에서만 `.env.example`을 복사하거나
`scripts/admin/bootstrap_local_env.py`로 새 비밀값을 준비한다. 기존 DB의 서명 키를
재생성하는 것은 단순 환경 정리가 아니다. frontend용 별도 env를 만들지 않는다.

DB 초기화/마이그레이션은 [pgvector 안내](pgvector-dev.md)를 따른다.
Redis는 eager 작업 모드에서 필요 없고, Neo4j는 GraphRAG에 필요한 별도 인스턴스다.
실제 source의 OAuth/API 설정과 권한은 따로 준비하며 실제 Slack source는 아직 별도 선택 사항이다.

```powershell
.\scripts\status.ps1
.\scripts\stop.ps1
.\scripts\restart.ps1
```

자세한 옵션·설정은 [scripts README](../../../scripts/README.md).
일반 서버 시작이 실제 GraphRAG final demo 완료를 뜻하지 않는다.
[Final demo 준비와 미구현 연결 항목](final-demo.md)을 확인한다.
