# Model router implementation

Implemented on 4 October 2026. The separate package is
[`harness-model-router`](../packages/model-router/README.md), with a provider-neutral
core and optional async LangChain middleware. Its core depends only on Pydantic;
Keel owns configuration, credentials, provider clients, task rules and the budget store.
The [research and design proposal](MODEL_ROUTER_DESIGN.md) describes the LangChain
reference and longer-term evaluation plan.

## Current behavior

Document parsing remains pinned to the configured Gemini 3.8 Flash model. Existing
validation, review, specialist opt-ins and approval gates remain in place.
Conversational routing selects the cheapest eligible available model, comparing
estimated input and output costs in USD with Decimal arithmetic. Eligibility checks
task family, suitability tier, required capabilities, context and output limits.
Unknown prices and missing credentials exclude a candidate.

| Task | Initial configured choice when all provider keys are available |
|---|---|
| Bounded lookups, lists, concise summaries | Luna |
| Reasoning, comparisons, reconciliation, ambiguous requests | Fireworks GLM Flash |
| Repeated tool errors | Raise the suitability floor; select the cheapest remaining eligible profile |
| Document parsing | Gemini Flash, configured independently of conversational routing |

These choices are derived from configuration rather than model-name branches.
DeepSeek Flash, Sol and Gemini are additional eligible candidates. Qwen support
uses an explicit OpenAI-compatible endpoint; no Qwen deployment is enabled by default.

The middleware keeps the selected profile within a user task, rechecks eligibility
when context grows, and resets on a new user turn. Two consecutive tool errors can
raise the tier, with at most two upgrades. Availability failures can fall back twice;
all attempts count toward the eight-call task limit. After streaming begins, errors
end the attempt instead of replaying a second model into the same response.
Child agents receive routing explicitly. Summary and child calls use the same
per-request usage recorder and model-client lifecycle.

## Configuration

Edit [`backend/keel/models.yml`](../backend/keel/models.yml): identifiers, provider,
task tiers, capabilities, limits, prices and provider parameters all live there.
Use `KEEL_MODEL_REGISTRY_PATH` to load another YAML file. Registry aliases can be
bound to `.env` fields, whose nonempty values override YAML defaults. When replacing
a model, configure its price and capability contract too; an unpriced override
cannot be automatically selected. See [`backend/.env.example`](../backend/.env.example).

`KEEL_ROUTER_MODE=enabled` is the default. `shadow` records the proposed candidate
while executing the configured default; `off` executes that default directly.
All modes meter paid agent calls. Restart the API and worker after config changes.
Apply migrations with `make migrate` before starting the updated services.

## Metering and budget

Every paid agent call reserves estimated spend atomically under tenant RLS and a
Postgres advisory lock, then settles actual provider usage exactly once. Usage
records contain the actual model/profile, cache reads, routing reason, latency
and price version. Missing provider usage is marked estimated rather than free.
The precision migration retains small cached-call costs.

The monthly budget gates agent calls. Document and embedding usage contributes
to recorded monthly spend but those workflows do not reserve against this gate.
Stale reservations remain held until reconciled; expiration never silently erases
possible spend from a crashed process. There is no automated crash reconciliation
job in this version. Reservations and usage are isolated per tenant.

## Validation and limits

Unit and Postgres integration checks cover selection, YAML/env overrides, unknown
prices, fallback, stream boundaries, task persistence, call limits, budget races,
tenant isolation, metering, client cleanup and existing workflows. The package
builds and imports independently of Keel. Backend wheels include the YAML registry;
Docker builds from the repository root to include the separate package.

Final checks: 79 tests passed, Ruff lint/format and strict mypy passed, both wheels
built, the router imported in an isolated environment without Keel, and the Docker
image loaded its packaged YAML successfully. Existing FPDF warnings remain.

The opt-in synthetic probe (`cd backend && uv run python -m
scripts.probe_model_router --live`) passed streaming, schema-conforming tool calls
and usage reporting for Luna, GLM Flash, DeepSeek Flash, Sol and Gemini. These are
provider contract checks, not task-quality benchmarks. No tenant data was sent.

This version uses deterministic task rules and configured suitability tiers,
avoiding a paid classifier call. It does not implement the proposal's classifier,
learned retry-cost optimization, latency SLO ranking or gold-data quality promotion.
Evaluate complete task outcomes before adjusting eligibility or claiming savings.
The Bugs, Security and Compliance review preserves tenant-bound read-only tools
and existing write approval gates; provider error logs omit raw exception payloads.
