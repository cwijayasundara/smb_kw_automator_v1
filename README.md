# Keel

Keel turns small food producers' paperwork into orders, invoices, batch records and food-safety logs. Owners photograph or upload what they already use: order pads, batch sheets, HACCP logs, scans and emailed orders. Every value links back to where it was written, and nothing is created, sent or posted without a person's approval.

Keel is two components:

- **`backend/`**: FastAPI with LangGraph and deepagents. It handles tenants and users, documents and the AI pipeline, approval-gated workflows, invoicing and the Ask Keel agent.
- **`frontend/`**: Next.js 16. It provides the Desk, the review screen with evidence boxes, orders, invoices, production, food safety, lot trace, catalog, team and the Ask Keel panel.

Postgres holds everything. Row-level security isolates each business's data.

## What works today

| Area | Details |
|---|---|
| Tenancy | Sign up a business, sign in with a password or magic link, invite teammates with roles (owner/admin/member/viewer), switch between businesses. Postgres RLS on every tenant table; the app's database role cannot bypass it. |
| Paper in | Drag and drop, file picker, or a phone camera upload. The same photo uploaded twice is processed only once. |
| Reading | PDF text layer → CPU OCR (PP-OCR via RapidOCR) → Gemini 3.8 Flash extraction (or an offline rules engine) → validation-driven re-read when checks fail. |
| Checks | Line and total maths, grouped pricing, crossed-out lines, unreadable fields (left empty, never guessed), dates, and instructions written on the paper (ignored). |
| Evidence | Each value is boxed on the page image. Hover a field to see where it was read; click to correct it. |
| Approvals | LangGraph workflow pauses at a gate. The approval is bound to a hash of exactly what will be written; edits made after approval need a new approval. |
| Orders → invoices | Prices come from the customer's price list or the catalog, never the paper. Issuing an invoice is a separate approval. Invoices export as PDF and as QuickBooks or Xero CSV. |
| Batch records | Batch sheets become signed-off batches: product, output lot, ingredient lots. A blank lot code blocks sign-off until a person records it as a trace gap. Formula deviations over 2% are flagged. Batch tables are append-only (trigger + revoked UPDATE/DELETE). |
| Food safety | Critical control points with limits. HACCP logs become verified readings; a blank reading is *missing*, never passed, and every missing or out-of-range reading needs a written corrective action. One-click inspection binder PDF. |
| Corrections | Signed batch records are never edited. A correction is a new version that supersedes the old one, behind its own approval, with the reason on record; the old version stays readable and trace follows the latest. |
| Lot trace | Recursive SQL from any lot: back to supplier lots, forward through batches to the orders and customers that received it, with trace gaps called out. Lots are allocated on order lines. |
| Search | Every page Keel reads is searchable: exact words (Postgres full-text), misspellings and shorthand (pg_trgm), and with an OpenAI key similar meaning (pgvector, `text-embedding-3-small` at 512 dims), fused by reciprocal rank fusion. Vectors record their model; `keel reindex` re-embeds after a model change. |
| Onboarding | A Desk checklist computed from real records: read one real paper first, then catalog, customers, control points and the business profile. The profile (year end, books, allergens) changes only through an approved diff. Ask Keel's `keel-onboard` and `keel-router` skills name one next step at a time. |
| Ask Keel | A deepagents agent with skills (`backend/skills/`), shared rules, tenant memory, read-only tools and streamed answers. It works offline without API keys. |
| Tracing | Optional self-hosted Langfuse for extraction and agent calls, tagged by org; off unless keys are set. `make langfuse` runs it locally, pre-wired. |
| Metering | The Desk shows token and page cost for every model call, per tenant. |
| Model routing | Separate `harness-model-router` package picks the cheapest available profile meeting task/capability requirements. Model IDs, prices and provider settings come from `backend/keel/models.yml`, with `.env` overrides. Agent calls reserve tenant budget and meter actual models, including child and summary calls. |

## Run it locally

You need Python 3.13 with [uv](https://docs.astral.sh/uv/), Node 22 with pnpm, and either Docker (for Postgres) or a local Postgres 16+ with pgvector.

```bash
make setup                 # backend + frontend dependencies
make db                    # Postgres 17 + pgvector in Docker (creates roles keel_owner / keel_app)
cp backend/.env.example backend/.env   # optional: add OPENAI_API_KEY / GOOGLE_API_KEY
make migrate
make seed                  # demo business: catalog, CCPs, a signed batch, sample order/batch/HACCP papers
```

Then start the two halves, each in its own terminal so the logs stay separate:

```bash
make backend               # terminal 1: API :8000 (auto-reload) + worker
make frontend              # terminal 2: web :3000
```

`make dev` runs all three in one terminal with interleaved logs. Use `make api`, `make worker` and `make web` to run each process on its own.

Open http://localhost:3000 and sign in as **owner@demo.keel** with password **keel-demo-2026**, or create your own business.

Without API keys, Keel runs fully offline:
- The local rules engine reads typed and OCR'd order sheets.
- Ask Keel answers through an offline tool-calling model.

Add `GOOGLE_API_KEY` for Gemini document parsing. Add `OPENAI_API_KEY` and/or `FIREWORKS_API_KEY`
for routed agent answers; semantic search embeddings need the OpenAI key.

Routine tasks start with the cheapest eligible profile; reasoning requests require a higher configured
suitability tier. Current YAML candidates include Luna, both Fireworks Flash models, Sol and Gemini.
Qwen can be added as a priced profile with an approved endpoint. No model identifier is hard-coded
in the router. Unknown prices and unavailable providers are excluded.

Set `KEEL_ROUTER_MODE=shadow` to record candidate routes while keeping `KEEL_MODEL_DEFAULT`,
or `off` to use the configured default. Changing YAML/.env requires restarting the API and worker.
Run `make migrate` for tenant budget reservations. The monthly budget gate applies to agent calls;
document and embedding spend also contributes to the amount considered by that gate.
Task tiers are configured rules, not measured quality guarantees; refine them using real task evals.
See [the implementation report](docs/MODEL_ROUTER_IMPLEMENTATION.md) for configuration,
validation and the scope of the first version.

**Postgres without Docker:** create the roles and databases in `backend/scripts/init-db.sql` (for example `sudo -u postgres psql -f backend/scripts/init-db.sql`), then continue from `make migrate`.

**Everything in Docker:** `docker compose up --build`, then `docker compose exec api keel seed`.

## Checks

```bash
make test        # backend unit and integration tests against real Postgres (RLS, approvals, append-only, trace, search, injection, OCR, agent)
make lint        # ruff + eslint
make typecheck   # mypy --strict + tsc
make e2e         # Playwright: order → invoice → Ask Keel, and batch sheet → HACCP log → lot trace (needs make dev)
```

## Tracing (Langfuse)

```bash
make langfuse            # whole stack in Docker + Langfuse; open http://localhost:3001
make langfuse-services   # only Langfuse, for use with `make dev` (prints the three backend/.env lines)
make langfuse-down
```

Langfuse shares Keel's Postgres server (its own `langfuse` database and role; it cannot read Keel's tables) and
adds ClickHouse, Redis and MinIO. A project with fixed local API keys and an admin user
(`admin@keel.local` / `keel-langfuse-local`) is created on first boot, so traces arrive with no setup. All of
its secrets are local-dev defaults (`docker-compose.langfuse.yml`); override them for anything shared. Traces
contain document text. For Cloud Run, point `LANGFUSE_BASE_URL` at a Langfuse you run elsewhere; the pilot
deploy doesn't host one.

## Parser bake-off

```bash
cd backend
uv run keel evals parsers --engines local-rules,luna,gemini,sol,reducto,ade --gold evals/gold/synthetic
```

Reports field accuracy, **invented values** (a value where the paper is empty), unreadable recall, $ per 1k pages and
latency, by engine, document type and field. Engines without a key are skipped. Real pages stay outside git; see
`backend/evals/gold/README.md`. Tier-4 engines (Reducto, ADE, GPT-6 Sol) only run in the pipeline when
`KEEL_TIER4_ENGINE` names one, and only on pages still failing checks after the Gemini re-read.

## Deploy

Cloud Run pilot: Neon Postgres, a private Google Cloud Storage bucket (no storage keys: the service account authenticates), a scheduled worker job, deploys gated on green CI. See
`infra/cloudrun/README.md`.

CI (`.github/workflows/ci.yml`) runs the same checks on every push, then both Playwright journeys against a live stack, and fails if the generated API client is out of date.

## Docs

- [`intent.md`](intent.md) and [`plan.md`](plan.md): AI-native SDLC artifacts (intent and build plan).
- [`docs/FINAL_REVIEW.md`](docs/FINAL_REVIEW.md): locked decisions; [`docs/PRODUCT_EVALUATION.md`](docs/PRODUCT_EVALUATION.md): what to build first and why.
- [`docs/PROPOSAL.md`](docs/PROPOSAL.md) and [`docs/TECH_DEEP_DIVE.md`](docs/TECH_DEEP_DIVE.md): vision and technology research.
- [`CLAUDE.md`](CLAUDE.md) and [`REVIEW.md`](REVIEW.md): conventions and review policy.
