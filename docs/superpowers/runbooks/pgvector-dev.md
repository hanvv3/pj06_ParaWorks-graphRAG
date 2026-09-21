# PostgreSQL + pgvector 로컬 실행

현재 실행 진입점은 저장소 루트의 `scripts/start.ps1`입니다.
[통합 안내](../../../scripts/README.md), [GraphRAG final demo](final-demo.md)를 함께 확인하세요.
이 문서는 PostgreSQL/legacy pgvector 준비를 설명하며 V2 GraphRAG 전체 준비 완료를 뜻하지 않습니다.

## 서비스 및 설정

```powershell
docker compose up -d postgres
.\scripts\test-services.ps1
```

기존 외부 DB를 쓰면 해당 DB를 켜고 루트 `.env`의 `DATABASE_URL`을 맞춥니다.
Compose 기본 DB/user/password는 로컬 개발용 `paraworks`, 기본 포트는 5432입니다.
포트 충돌 시 `PARAWORKS_POSTGRES_PORT`와 `DATABASE_URL`을 함께 변경하세요.
자동 포트 fallback이나 다른 프로그램 종료는 하지 않습니다.
`PARAWORKS_DATABASE_URL`이 있으면 일반 모드에서 우선하므로 오래된 override도 확인하세요.

## 스키마

기존 DB는 백업하고 마이그레이션을 검토합니다. 빈 개발 DB를 명시적으로 준비할 때만:

```powershell
.\scripts\start.ps1 -InitializeDatabase -SkipApp
.\scripts\test-services.ps1
.\scripts\start.ps1
```

초기화는 Alembic과 기존 init 경로를 사용합니다. 일반 start는 스키마를 자동 변경하지 않습니다.
현재 legacy vector column은 1536차원이므로 모델/차원을 변경하기 전에 실제 스키마를 확인하세요.
세부 점검 도구는 `scripts/checks/check_db_schema.py`, `scripts/checks/check_pgvector_dev.py`입니다.
기존 키와 DB fingerprint를 임의 재생성하지 마세요. 새 env 생성기는
`scripts/admin/bootstrap_local_env.py`이며 기존 설정 정리용으로 실행하지 않습니다.

## 인덱싱과 worker

유료 임베딩 전 기존 API 키, 선택 모델의 현재 단가, 비용 상한, 승인된 source를 준비합니다.
관리자 인증이 필요한 기존 `/api/v1/rag/reindex`/`jobs`의 dry-run으로 먼저 범위를 확인합니다.
이 API는 legacy 인덱싱입니다. V2 GraphRAG 인덱싱 완료로 간주하지 마세요.
인증 없는 sync/reindex POST나 키가 없는 fake 결과를 실제 provider 검증으로 사용하지 않습니다.

로컬에서는 `CELERY_TASK_ALWAYS_EAGER=true`로 Redis 없이 실행할 수 있습니다.
별도 worker가 필요할 때만 Redis를 켜고 `CELERY_TASK_ALWAYS_EAGER=false`와 `REDIS_URL`을
설정한 뒤 `scripts/admin/start-celery-worker.ps1`을 별도 터미널에서 실행합니다.

## 종료

```powershell
.\scripts\stop.ps1
docker compose stop postgres
```

stop은 관리 중인 앱만 종료하며 Docker 데이터는 보존합니다. 일반 종료에 `down -v`를 쓰지 않습니다.
