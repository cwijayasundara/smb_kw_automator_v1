# Harness Model Router

Independent Python package for selecting the cheapest configured model that meets
task, quality-tier, capability, context, output and budget requirements. No Keel,
database or provider SDK dependency in the core.

Install `harness-model-router[langchain]` for the optional async middleware. Hosts
supply profiles, available providers, task classification and a model factory.
`Router.select(RouteRequest(...))` returns an explainable `RouteDecision`; an
unknown price or unmet requirement raises `NoEligibleModel` instead of choosing
an unsuitable/free-looking model. Prices use Decimal and distinguish cache hits.

The middleware preserves a route within a user task, re-evaluates on a new task,
escalates on repeated tool errors, and supports bounded availability fallback
before streaming starts. Tenant identity, metering and atomic budget reservations
belong to the host. It never executes business tools itself.

Profile tiers are operator-supplied suitability constraints, not claimed benchmark
results. Evaluate task outcomes before refining eligibility. Document parsing can
be pinned by the host and need not use conversational routing.

From this repository: `uv build --directory packages/model-router`; run tests with
`cd backend && uv run pytest ../packages/model-router/tests`.
