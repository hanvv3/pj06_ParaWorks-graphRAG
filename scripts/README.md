# 로컬 서비스 실행 및 설정

Windows PowerShell 기준입니다. 모든 명령은 해당 체크아웃 루트에서 실행합니다.
일상 작업·실행 위치는 원래 저장소 루트 `main`으로 통일했습니다. 루트 `.env`와 `.venv`를
사용하며 과거 `.worktrees` 폴더에서 실행하지 않습니다. 실제 키는 출력·커밋하지 마세요.
런처는 `.env` 값을 그대로 읽으며 `${OTHER_VARIABLE}` 치환은 하지 않습니다.
같은 이름의 터미널 환경변수가 있으면 파일보다 우선합니다. 기존
`PARAWORKS_DATABASE_URL` 별칭은 일반 모드에서 `DATABASE_URL`보다 우선합니다.

## 준비

의존성은 루트 `pyproject.toml`/lockfile과 `frontend/package-lock.json`을 기준으로 설치합니다.
런처는 의존성을 자동 설치하지 않습니다. Python은 `-PythonPath`,
`UV_PROJECT_ENVIRONMENT`, 루트 `.venv` 순으로 선택합니다.

신규 환경 설치 명령(기존 환경이 준비되어 있으면 반복할 필요 없음):

```powershell
uv sync --locked
npm --prefix frontend ci
```

과거 터미널에 임시 가상환경 선택값이 남아 있다면 해제합니다:

```powershell
Remove-Item Env:UV_PROJECT_ENVIRONMENT -ErrorAction SilentlyContinue
```

신규 설치만 `.env.example`을 `.env`로 복사합니다. `admin/bootstrap_local_env.py`는 새 환경의
비밀값 생성 도구입니다. 기존 DB의 fingerprint 키를 교체할 수 있으므로 기존 환경
정리 목적으로 실행하지 마세요.

Docker Desktop을 사용한다면 필요한 서비스만 명시적으로 시작합니다.

```powershell
docker compose up -d postgres
```

다른 포트라면 `PARAWORKS_POSTGRES_PORT`와 `DATABASE_URL`을 함께 맞춥니다.
Compose 포트 변수만 바꿔도 애플리케이션 DSN이 바뀌는 것은 아닙니다.
기존 DB 접속정보는 보존하세요. Neo4j는 현재 Compose에 없으며 별도로 필요합니다.
`CELERY_TASK_ALWAYS_EAGER=true`이면 로컬 작업에 Redis/별도 worker가 필요하지 않습니다.

## 기존 .env에 추가할 항목

예제는 가능한 설정을 설명하는 템플릿이며 실제 `.env`는 필요한 설정만 선언할 수 있습니다.
항목 수가 같을 필요는 없습니다. 누락된 항목은 코드 기본값을 사용합니다.
기존 OpenAI 키, DB 접속정보, fingerprint 키·버전, 인증/OAuth 비밀값은 유지하세요.
파일을 통째로 복사하거나 기존 키를 새 키로 교체하지 마세요.

이번 로컬 비교에서 기존 키가 존재했고 중복 변수명은 없었습니다. 다음은 없는 항목만
추가하여 안전한 기본 실행 의도를 명시하는 예입니다. 이미 있으면 중복 추가하지 마세요.

```dotenv
PARAWORKS_SEED_DEMO_DATA=false
LANGGRAPH_RAG_V2_MODE=disabled
LANGGRAPH_RAG_V2_STAGE=none
RAG_RETRIEVAL_BACKEND=keyword
RAG_GRAPH_ENRICHMENT_ENABLED=false
RAG_ANSWER_CACHE_ENABLED=false
ASSISTANT_EMAIL_AGENT_ENABLED=false
GOOGLE_DRIVE_SYNC_ENABLED=false
GMAIL_SYNC_ENABLED=false
```

마지막 세 항목은 별도 이메일 AI와 Google 자동 동기화의 명시적 opt-in을 위한 설정입니다.
이미 사용하는 기능이라면 무조건 false로 바꾸지 말고 유지 여부를 결정하세요.
일반 실행은 `PARAWORKS_DEMO_MODE=false`이며, `AUTO_REVIEW_MODE=disabled`,
`AUTO_REVIEW_ENFORCE_PERCENTAGE=0`을 유지합니다. 기존 rollout 승인을 대신하지 않습니다.

| 사용 목적 | 확인하거나 직접 작성할 값 |
|---|---|
| PostgreSQL 실행 | `DATABASE_URL`, `AGENT_RUNTIME_FINGERPRINT_SECRET`, `AUTH_SESSION_SECRET`: 기존 값 재사용. fingerprint는 최소 32바이트 비밀값이며 기존 DB 키 임의 교체 금지 |
| OpenAI 연결 확인 | 기존 `OPENAI_API_KEY`; 모델 선택 `AGENT_LLM_OPENAI_MODEL=gpt-5.4-mini` |
| source-agent 추론 | `AGENT_LLM_ENABLED=true`, 현재 입력/출력 단가 `AGENT_LLM_INPUT_COST_PER_1M_TOKENS`, `AGENT_LLM_OUTPUT_COST_PER_1M_TOKENS`, 상한 `AGENT_LLM_MAX_ESTIMATED_COST_USD` |
| 유료 임베딩 | `OPENAI_EMBEDDING_MODEL`, `OPENAI_EMBEDDING_DIMENSIONS`, 현재 단가 `OPENAI_EMBEDDING_INPUT_COST_PER_1M_TOKENS`, 상한 `RAG_EMBEDDING_MAX_ESTIMATED_COST_USD` |
| Neo4j 연결 | `RAG_NEO4J_URI`, `RAG_NEO4J_USERNAME`, `RAG_NEO4J_PASSWORD`, `RAG_NEO4J_DATABASE=neo4j` |
| 비동기 worker | `CELERY_TASK_ALWAYS_EAGER=false`, `REDIS_URL`, 별도 Celery worker |
| Google/Slack 연결 | 공급자 자격증명·redirect URI·허용 source. 사용하지 않는 공급자 키는 작성 불필요 |

단가에는 최신 공식 가격을 확인한 숫자를 입력하세요. 주석 처리된 빈 숫자 항목을 그대로
활성화하면 설정 파싱에 실패합니다. 코드의 기존 기본 단가를 현재 가격으로 간주하지 마세요.
상한이 작아 preflight에서 거절되면 예상 비용을 확인하고 조정합니다.
C.5 자동 검토는 별도의 서버 가격 레지스트리·승인 계약을 사용합니다.

GraphRAG/캐시는 연결값만 채워도 켜지지 않습니다. 승인 근거, 인덱스/파생 projection,
허용된 RAG 경로가 필요합니다. flag만 켜서 정식 release 경계를 우회할 수 없습니다.
관리자용 provider-safety/release authority 변수는 일반 실행을 위해 추가할 필요 없습니다.

## 시작·상태·종료

```powershell
.\scripts\test-services.ps1
.\scripts\start.ps1
.\scripts\status.ps1
.\scripts\test-services.ps1 -IncludeApi
.\scripts\stop.ps1
.\scripts\restart.ps1
```

빈 개발 DB를 의도적으로 초기화할 때만 `start.ps1 -InitializeDatabase`를 사용합니다.
기존 DB는 백업 및 마이그레이션 절차를 먼저 확인하세요. 기본 start는 초기화나 데모
삽입을 하지 않으므로 기본 계정도 자동 생성된다고 가정하지 마세요.
SQLite 데모의 admin 로그인과 실제 DB의 사용자 준비는 별개입니다.

선택 인자: `-BackendPort 8000 -FrontendPort 3000 -HostAddress 127.0.0.1`,
`-SkipFrontend`, `-PythonPath <python.exe 경로>`.
`start.ps1 -SkipApp`은 서비스 준비/점검만 하고 앱은 실행하지 않습니다.
변경한 포트와 별도 클라이언트/OAuth redirect 설정도 일치시켜야 합니다.
`restart.ps1`은 기존 관리 상태의 포트·호스트·smoke 여부를 보존하고 설정을 다시 읽습니다.
구성을 바꾸려면 stop 후 원하는 인자로 start하세요.
로그는 `.tmp/local-runtime/*.log`, 프로세스 관리 상태는 `.tmp/local-runtime/state.json`에
저장합니다. 로그는 로컬에 보관하고 외부 공유 전 민감한 내용 포함 여부를 확인하세요.

stop은 저장된 PID뿐 아니라 생성 시각·명령 식별 정보도 확인합니다. 다른 터미널이나
예전 런처로 시작한 서버는 해당 터미널에서 Ctrl+C로 종료하세요. 포트만 보고 강제
종료하지 않습니다. stop은 Docker를 중지하거나 DB 볼륨을 지우지 않습니다.
직접 시작한 Compose DB를 닫으려면 `docker compose stop postgres`를 사용합니다.
`docker compose down -v`는 데이터 삭제이므로 일반 종료 방법이 아닙니다.

## provider 연결 확인과 실제 AI 테스트

```powershell
.\scripts\test-provider.ps1        # 설정만 확인, 네트워크 호출 없음
.\scripts\test-provider.ps1 -Live  # OpenAI 모델 메타데이터 조회 1회
```

`-Live`는 기존 키로 설정된 모델 메타데이터를 조회합니다. 생성/임베딩이나 회사 데이터
전송은 없고, timeout 10초·재시도 0회입니다. 성공은 연결 증거일 뿐, 실제 생성 성공·
모델 품질·전체 GraphRAG 준비 완료의 증거는 아닙니다.

실제 추론은 로그인한 앱에서 권한 있는 문서/메일을 준비한 뒤 기존 source-agent
preflight와 유료 실행 확인 흐름으로 테스트하세요. 결과는 pending Review로 남으며,
승인 없이 공식 지식으로 승격되지 않습니다. `AGENT_LLM_ENABLED=true`는 일반 source-agent
작업에도 영향을 주므로 활성화 후 모든 호출이 무과금이라고 가정하지 마세요.
스크립트는 유료 실행 authority나 RAG V2 cutover 승인을 만들어 주지 않습니다.

## scripts 구성

| 위치 | 용도 |
|---|---|
| 최상위 6개 `.ps1` | start / stop / restart / status / test-services / test-provider |
| `internal/` | 실행기의 내부 구현. 직접 사용할 필요 없음 |
| `admin/` | env 초기화, 체크포인트 관리, 데이터 reset, 명시적 Slack 수집, Celery worker |
| `checks/` | DB 스키마·pgvector 점검, 별도 release 검증 |
| `demo/` | SQLite smoke·visual 테스트. 실제 final demo와 다름 |

`paraworks-docker.ps1`, `paraworks-docker.sh`, `start-pgvector-dev.ps1`,
`run_e2e_demo.py`는 삭제했습니다. 기존 명령은 위 최상위 실행 명령과 명시적 Docker Compose
명령으로 대체하세요. 과거 spec/이력에 나온 경로는 당시 기록이며 현재 실행 안내가 아닙니다.

관리 도구 예: `.venv\Scripts\python.exe scripts/checks/check_db_schema.py --help`.
`admin/reset_connector_data.py`는 삭제 도구이며 일반 시작/종료에는 필요 없습니다.
`admin/sync_slack.py --execute`는 명시적 수집만 수행하며 실제 source/접근권한이 필요합니다.

## GraphRAG final demo

[Final demo 준비 상태와 절차](../docs/superpowers/runbooks/final-demo.md)를 먼저 읽으세요.
일반 서버 시작과 fully functional GraphRAG 시연은 다릅니다. 실제 V2 provider 설정,
V2 인덱싱·그래프 갱신의 운영 진입점이 아직 필요하며 flag만 켜서 완료되지 않습니다.

전문 진단·reset 도구 일부는 원래 예외 메시지를 출력합니다. 오류 출력을 그대로
공유하지 마세요. 일반 서비스 점검은 비밀값이 출력되지 않는 `test-services.ps1`을 사용하세요.

정식 release는 capability P1 때문에 NOT CLEAN입니다. 이 문서 정리는 그 보류 사항을
해제하거나 실제 Slack 연결을 승인하지 않습니다.
