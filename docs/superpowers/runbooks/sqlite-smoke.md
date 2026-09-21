# SQLite UI smoke — final demo 아님

Docker나 provider 없이 UI를 점검하는 보조 모드다. 실제 PostgreSQL/Neo4j/모델 호출을
검증하려면 [final demo 안내](final-demo.md)를 따른다.

저장소 루트에서 실행:

```powershell
.\scripts\demo\start-smoke.ps1 -DatabasePath .tmp/paraworks-ui-smoke.db
.\scripts\status.ps1
.\scripts\stop.ps1
```

명시한 SQLite DB를 초기화하고 키 bootstrap 후 demo seed를 넣는다. 기존 orphaned DB를
강제로 채택하지 않으며 키를 바꾸어 오류를 우회하지 않는다. 원본 DB는 지우지 말고 보존한다.
smoke는 child process의 provider 키/유료 기능을 비활성화하고 `.next-smoke`를 사용한다.
포트가 사용 중이면 거절하며 다른 서버를 재사용하거나 종료하지 않는다.

브라우저: `http://localhost:3000`, API health: `http://127.0.0.1:8000/health`.
일반 PostgreSQL 데이터/계정과 이 모드의 seeded 데이터/계정은 다르다.

자동 visual 테스트는 `scripts/demo/run-visual-smoke.ps1`을 사용한다.
이 스크립트는 자신이 시작한 서버를 finally에서 종료하며 오래된 자동 sync POST는 수행하지 않는다.
실제 브라우저 테스트는 별도 설치된 Playwright 런타임이 필요하다.
