# Keel: two components (backend/, frontend/) + Postgres. See README for first-time setup.
.PHONY: setup sync db migrate seed api worker web backend frontend dev test e2e lint typecheck build openapi langfuse langfuse-services langfuse-down

# `uv run` may re-sync the venv and macOS then re-hides the editable-install .pth files (Python 3.13 skips them).
# `sync` fixes that once, then every run uses --no-sync.
KEEL = cd backend && uv run --no-sync keel

sync:
	@cd backend && uv sync --extra ocr --extra observability -q
	@chflags nohidden backend/.venv/lib/python*/site-packages/*.pth 2>/dev/null || true

setup:            ## install backend + frontend dependencies
	cd backend && uv sync --extra ocr --extra observability
	@# macOS can mark .pth files hidden; Python 3.13 then skips them and `import keel` fails.
	@if command -v chflags >/dev/null; then chflags nohidden backend/.venv/lib/python*/site-packages/*.pth; fi
	cd frontend && pnpm install

db:               ## start Postgres 17 + pgvector in Docker (skip if you run Postgres yourself)
	docker compose up -d postgres

migrate: sync          ## apply migrations (as the owner role)
	$(KEEL) migrate

seed: sync             ## demo tenant: owner@demo.keel / keel-demo-2026
	$(KEEL) seed

api: sync
	$(KEEL) api --reload

worker: sync
	$(KEEL) worker

web:
	cd frontend && pnpm dev

backend: sync          ## api :8000 + worker in one terminal (Ctrl-C stops both)
	@trap 'kill 0' INT; ($(KEEL) api --reload) & ($(KEEL) worker) & wait

frontend:         ## web :3000 in its own terminal
	cd frontend && pnpm dev

dev: sync              ## api + worker + web together (Ctrl-C stops all)
	@trap 'kill 0' INT; ($(KEEL) api --reload) & ($(KEEL) worker) & (cd frontend && pnpm dev) & wait

LANGFUSE = docker compose -f docker-compose.yml -f docker-compose.langfuse.yml

langfuse:         ## whole stack in Docker + self-hosted Langfuse on :3001, Keel pre-wired to trace
	docker compose up -d --wait postgres
	docker compose exec -T postgres psql -q -U postgres -v ON_ERROR_STOP=1 < backend/scripts/langfuse-db.sql
	$(LANGFUSE) up -d --build
	@echo "Keel http://localhost:3000 · Langfuse http://localhost:3001 (admin@keel.local / keel-langfuse-local)"

langfuse-services: ## only Langfuse (+ its Postgres/ClickHouse/Redis/MinIO) for use with `make dev`
	docker compose up -d --wait postgres
	docker compose exec -T postgres psql -q -U postgres -v ON_ERROR_STOP=1 < backend/scripts/langfuse-db.sql
	$(LANGFUSE) up -d langfuse-web langfuse-worker
	@echo "Add to backend/.env: LANGFUSE_BASE_URL=http://localhost:3001 LANGFUSE_PUBLIC_KEY=pk-lf-keel-local LANGFUSE_SECRET_KEY=sk-lf-keel-local"

langfuse-down:    ## stop everything started by `make langfuse` (data volumes are kept)
	$(LANGFUSE) down

openapi:          ## regenerate the typed frontend client from the backend's OpenAPI spec
	cd backend && uv run keel openapi > ../openapi.json
	cd frontend && pnpm exec openapi-ts

lint:
	cd backend && uv run ruff check keel tests migrations ../packages/model-router && uv run ruff format --check keel tests migrations ../packages/model-router
	cd frontend && pnpm exec eslint .

typecheck:
	cd backend && uv run mypy keel ../packages/model-router/src
	cd frontend && pnpm exec tsc --noEmit

test:             ## backend integration tests against Postgres (keel_test database)
	cd backend && uv run pytest -q tests ../packages/model-router/tests

e2e:              ## browser test against a running stack (make dev first)
	cd frontend && pnpm exec playwright test

build:
	cd frontend && pnpm build
