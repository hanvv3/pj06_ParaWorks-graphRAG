# 현재 작업 위치

2026-09-21 사용자 승인에 따라 원래 저장소 루트를 단일 개발·실행 위치로 통합했다.

- 폴더: `C:\Users\hanvv\Study\potenup3\pj06_ParaWorks+graphRAG`
- 브랜치: `main` (로컬). 원격 push하지 않음.
- Python: 루트 `.venv`; `uv sync --locked`로 동기화.
- 설정: 루트 `.env`; 이전 검증된 작업트리의 값을 그대로 복사, 키 재발급/회전 없음.
- frontend: 루트 `frontend`; `npm --prefix frontend ci`로 설치.

`codex/rag-orchestrator-agent`의 `e62fc65`까지 루트 main에 fast-forward했다.
기존 main은 `codex/pre-root-unification-20260921`에 보존했다.
그 뒤의 scripts 정리 커밋은 루트 main에 작성한다. 이전 작업트리를 다시 실행하지 않는다.

## 과거 작업트리 정리

등록 worktree를 9개에서 2개로 줄였다. 다음 7개 체크아웃을 제거했고 각 브랜치·커밋은 보존했다.

- langchain-langgraph-dependency-compatibility
- task6-fix-ask-index
- task6-fix-ingestion
- task6-fix-reconciliation-pagination
- task6-fix-recovery
- task6-fix-review-v2-fixtures
- task6-fix-serving-authority

미추적 메모·토큰 파일·테스트 산출물 등 비캐시 로컬 파일은 삭제하지 않고
`.tmp/worktree-preservation/20260921/<이전 폴더명>/`에 보존했다.
가상환경/패키지 캐시/pycache 같은 재생성 가능한 파일은 제거된 체크아웃과 함께 정리했다.
필요한 과거 코드는 보존한 브랜치에서 복구할 수 있다. DB Docker 볼륨은 건드리지 않았다.

남은 `.worktrees/review-hitl-v2-design`은 **개발 위치가 아니라 로컬 데이터 보존용**이다.
여기에는 기존 SQLite DB, 실행 이력 및 표시되는 문서 변경이 남아 있어 강제 삭제하지 않았다.
기존 이 작업트리의 Next/uvicorn 서버는 종료했다. DB/기록을 최종 이관하기 전까지 보관한다.
루트와 이 폴더의 `.env`는 자동 동기화되지 않으므로 앞으로 수정할 파일은 루트 `.env`다.

## 일상 명령

```powershell
git branch --show-current
.\scripts\test-provider.ps1
.\scripts\test-services.ps1
.\scripts\start.ps1
.\scripts\stop.ps1
```

과거 터미널에 `UV_PROJECT_ENVIRONMENT=.venv-task4-r3-review`가 남아 있다면
`Remove-Item Env:UV_PROJECT_ENVIRONMENT -ErrorAction SilentlyContinue`로 해제한다.
별도의 가상환경 이름을 외울 필요가 없다. 실제 final demo 조건은 [이 안내](final-demo.md)를 따른다.
