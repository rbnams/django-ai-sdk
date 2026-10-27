# Changelog

## Unreleased

### Oversight

A new app, `django_ai_sdk.oversight`, audits every agent turn on any channel (chat, webhooks, workflow steps, your own code) and can act on it. See [Oversight](docs/content/oversight.md) and the [manual](docs/content/manual/oversight.md).

- **Lenses**: typed questions about a turn: `Flag`, `Choice`, `Score`, `Entities` and `Text`, about the question, the answer, the exchange, the sources or a tool call.
- **Answerers**, picked per lens: Jev (`TypeSafe`) for most yes/no, label and score lenses; `LLMJudge` on any SDK generator and `AgentJudge` on a registered agent, for text and escalations; local detectors (`PII` by checksum, `Secrets`, `Keywords`, `Patterns`, language, length, refusal, lexical support); `Moderation`, `HFClassifier`, `Presidio`, `Function` and `Human`. A lens asks them in order until one is sure; if none is, the surest answer stands. Secrets are masked before any turn content leaves.
- **Grounding**: groundedness and context relevance read the turn's sources and its tool results, Jev scores the RAG triad in one call, and a turn with no evidence is not scored for grounding.
- **Rules**: a condition over lens answers (`L("topic") == "hr"`) and actions: `Block`, `Redact`, `RestrictTools`, `Queue`, `Escalate`, `Notify`, `Alert`, `Allow`. Each rule is `off`, `shadow` (test mode) or `enforce`.
- **Stages**: `gate` before the model, `watch` while it works (each tool call, the streaming answer), `review` before the user sees it, `reflect` afterwards in the background, and `retro` replays with `oversight_replay`. `catalog.STAGE_INFO` holds each stage's label, what runs there and what a rule can do. Inline stages have deadlines and a per-stage `FAIL_MODE`.
- **Policies**: the `security`, `safety`, `quality` and `classify` packs, and `default` with all four. Policies extend each other, or `pick` single lenses and rules from a pack, and can be tuned from settings or a database overlay.
- **Ledger**: turns, observations, decisions, access and overlay changes on an append-only hash chain, with signed Merkle checkpoints (`oversight_seal`), verification (`oversight_verify`), per-thread encrypted evidence with crypto-shredding, and retention (`oversight_prune`).
- **Service and console API**: `OversightService` and `get_oversight_router()` for Django Ninja: turns, reviews, policies, the policy editor's writes, dry runs, agent assignment, answerer aliases, checkpoints, the access log and NDJSON export.
- **Exporters**: `Log` and a signed `Webhook` (Standard Webhooks); anchoring exporters receive every checkpoint.
- **Extras**: `[oversight]` (`cryptography`, for encryption and signed checkpoints), `[typesafe]` (Jev) and `[presidio]`. `[all]` leaves out `[presidio]`, which needs a spaCy model.
- **Migration**: `django_ai_sdk_oversight.0001_initial`. Run `manage.py migrate` after adding the app to `INSTALLED_APPS`.
