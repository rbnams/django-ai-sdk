---
title: Oversight
type: docs
weight: 127
---

The [Oversight guide](/oversight/) covers the ideas. This page documents the classes, the stages, the models, the service and the commands.

## Lenses

`django_ai_sdk.oversight.lenses`. Subclass one, or construct one with keyword arguments.

| Attribute | Default | Meaning |
| --- | --- | --- |
| `name` | required | Key within the policy. Observations key on it. |
| `question` | required | Asked verbatim of model answerers. |
| `guidance` | `""` | Extra instructions for judges: scale anchors, edge cases. |
| `subject` | `Subject.INPUT` | `input`, `output`, `exchange`, `sources`, `tool_call`. |
| `stages` | `(Stage.REVIEW,)` | Any of `gate`, `watch`, `review`, `reflect`. |
| `answerers` | required | Tried in order; instances, aliases, or optional aliases (`"judge?"`). |
| `confident` | `0.75` | An answer below this asks the next answerer. |
| `sample` | `1.0` | Share of turns the reflect stage judges. Inline stages never sample. |
| `purposes` | security, quality | Which purpose views show its answers. |

| Kind | Value | TypeSafe |
| --- | --- | --- |
| `Flag` | `bool` | Noul (`yes` / `no` criteria) |
| `Choice` | one of `labels` (`default` when a detector finds none) | Choice (`criteria`) |
| `Score` | `0..1` (`low` / `high` describe the ends) | Score |
| `Entities` | sorted list of labels | — |
| `Text` | string, `max_chars` | — |

## Answerers

`django_ai_sdk.oversight.answerers`. `answer(lens, view) -> Answer | None`; `None` abstains.

System one (`system_one = True`: free, local, no budget):

| Class | Answers |
| --- | --- |
| `Patterns(patterns)` | Regexes, by label or as one list. |
| `Keywords({label: [words]})` | Whole words; the label with most hits wins. Classifies every message. |
| `PII(kinds=None, extra=None)` | `email`, `iban` (mod-97), `credit_card` (Luhn), `bsn` (11-check), `phone`, `ip_address`, `postcode_nl`. |
| `Secrets(extra=None)` | OpenAI, GitHub, AWS, Slack keys, JWTs, private keys, password assignments. |
| `Language()` | The language (Choice), or "answered in another language" (Flag). |
| `Length(max_chars)`, `Refusal()`, `EmptyRetrieval()`, `ToolErrors()` | As named. |
| `LexicalSupport()` | Word-overlap groundedness (output) and context relevance (sources). Low confidence by design: a floor under the judges. |
| `Presidio(entities, language)` | Microsoft Presidio, local. `[presidio]` extra. |
| `HFClassifier(model, positive, labels, endpoint)` | A Hugging Face classifier; system one when local. |
| `Function(fn)` | Any function `fn(lens, view)` returning a value, an `Answer` or `None`. |
| `Human(queue, priority)` | Queues the lens for a person; the answer arrives later. |

Models:

| Class | Answers with |
| --- | --- |
| `LLMJudge(model, llm=None)` | Any SDK generator factory (`llm`, a callable or dotted path; OpenAI Responses by default). |
| `AgentJudge(agent)` | A registered agent, by id, name or class name. Oversight is off inside the judge's run. |
| `TypeSafe(model=None)` | Jev, through `django_ai_sdk.typesafe.evaluate`. `[typesafe]` extra and `TYPESAFE_API_KEY`. |
| `Moderation(model)` | OpenAI's moderation endpoint. |

Combinators: `Cascade(*answerers, confident=None)`, `Vote(*answerers, min_agree=1)`, `Sample(answerer, rate)`. A lens with several answerers is a `Cascade`.

## Rules and conditions

```python
Rule(name, when=Condition, then=Action | [Action], mode=Mode.SHADOW, stages=None,
     message="", description="")
```

Conditions are built from `L(lens)`: `.is_true`, `.is_false`, `== value`, `!= value`, `> < >= <=`, `.in_(values)`, `.contains(label)`, `.any`, `.missing`, `.present`, `.confident(x)`, `.unsure(x)`, combined with `&`, `|` and `~`. Each serializes to JSON (`{"all": [...]}`, `{"any": [...]}`, `{"not": ...}`, `{"lens", "op", "value"}`), which is what the database overlay stores. A comparison on a lens without a usable answer is false.

A rule without `stages` is evaluated once every lens it reads has had its chance, so it can combine a gate lens and a reflect lens. A rule fires once per turn.

| Action | Enforced at | Outcome |
| --- | --- | --- |
| `Block(message)` | gate, watch, review | `blocked` |
| `RestrictTools(names)` (`[]` = all) | gate, watch | `restricted` |
| `Redact(lenses=None, template="[{label}]")` | gate, watch, review, reflect | `redacted` |
| `Queue(queue, priority)`, `Escalate(lens, answerer)`, `Notify(exporters, severity)`, `Alert(severity)` | any | `flagged` |
| `Allow()` | any | `allowed` |

## Policies

```python
class Policy:
    name: str
    extends: list[str]          # packs or policies
    lenses: list[Lens | type[Lens]]
    rules: list[Rule]
    capture: Capture | None     # none, metadata, redacted, full
    stream_mode: str | None     # live, hold
    retention: dict[str, int]   # security, quality, metadata (days)
```

Register with `@register` in an app's `oversight.py`. Agents choose with `oversight = "name"` or `get_oversight_policy()`.

Every effective value has a source, in this order: the extended packs (`pack:<name>`), the class (`code`), `AI_SDK_OVERSIGHT["POLICIES"]` (`settings`), then `PolicyOverride` rows (`db`). Overrides may set a lens's `enabled`, `sample`, `answerers`, `confident` and `stages`, a rule's `mode`, `when` (thresholds only, unless the rule itself came from the database), `then`, `message` and `stages`, and a policy's `capture`, `stream_mode` and `retention`. They may also add lenses and rules (`field="definition"`). An override that cannot apply is listed under `rejected` and ignored.

## The session

`TurnSession.start(agent, channel, user, thread_id, messages | input)` returns an inert session unless a policy applies. Then `gate()`, `note_text(chunk)` + `watch()`, `check_tool(call)`, `set_sources(...)`, `add_tool_call(...)`, `add_usage(...)`, `review(output)`, `close()`. Chat (`Agent.as_view` and `Stream`), webhooks (`run_inbound`) and workflow `AgentStep` call it already; channels are `chat`, `webhook:<integration>`, `workflow:<workflow>:<step>`, or what you pass. `unobserved()` suppresses sessions inside a block; `active(session)` makes one visible to `get_tools()` and the tool hook.

## Models

| Model | Holds |
| --- | --- |
| `Turn` | Channel, agent, agent fingerprint, policy, thread and message ids (no foreign keys), pseudonym, sealed actor, model, tokens, cost, latency, `outcome`, `shadow_outcome`; labels and severity (denormalized). |
| `Evidence` | The content at the policy's capture level, encrypted per scope, with a salted digest and an expiry. |
| `Observation` | One answer to one lens: answerer, model, value, confidence, rationale, spans, trail, cost, latency, reviewer. |
| `Decision` | A rule that held: stage, mode, actions, whether it acted, the condition. |
| `ReviewItem` | A turn waiting for a person, per queue. |
| `AccessRecord` | A read of the ledger: who, purpose, action, turn, reason. |
| `PolicyOverride` | One operator change, never edited; a new row withdraws an old one. |
| `LedgerEntry`, `LedgerHead` | The chain and its lock. |
| `Checkpoint`, `DataKey`, `DailySpend` | Signed roots, per-scope keys, spend per day and answerer. |

## Service and HTTP

`OversightService` (domain `oversight`): `list_turns`, `get_turn`, `reveal`, `list_reviews`, `queues`, `resolve_review`, `add_verdict`, `list_policies`, `get_policy`, `set_override`, `list_overrides`, `overview`, `checkpoints`, `verify`, `seal_now`, `access_log`, `export`, `forget_thread`. Unknown ids return `None`; refused operations raise `PermissionDenied`; bad input raises `UserError`.

`get_oversight_router()` serves them under `/oversight/`: `turns/`, `turns/{id}/`, `turns/{id}/reveal/`, `turns/{id}/verdicts/`, `reviews/`, `reviews/{id}/`, `queues/`, `policies/`, `policies/{name}/`, `policies/{name}/overrides/`, `overview/`, `checkpoints/`, `verify/`, `access/`, `export/` (NDJSON), `threads/{id}/forget/`.

## Commands and tasks

| Command | Does |
| --- | --- |
| `oversight_verify [--verifier KEY] [--json]` | Walks the chain and the latest checkpoint; exits non-zero on tampering. |
| `oversight_seal` | Writes a checkpoint over new entries and sends it to anchoring exporters. |
| `oversight_prune [--thread ID]` | Applies retention, or crypto-shreds one thread. |
| `oversight_keys` | Prints new keys, or the verifier key for the configured one. |
| `oversight_replay AGENT --file golden.jsonl \| --reviewed N` | Retro: replays questions and checks the verdicts held. |

Tasks (`django_ai_sdk.oversight.tasks`): `reflect_turn` (queued on commit when the policy has reflect lenses or an escalation), `seal_checkpoint`, `prune_expired`.

## Exporters

`AI_SDK_OVERSIGHT["EXPORTERS"]` lists `Log()`, `Webhook(url, secret)` (HMAC-SHA256 in `X-Oversight-Signature`) and `OpenTelemetry()` (`gen_ai.evaluation.result` events on a GUARDRAIL span), as instances or `{"class": ..., **kwargs}`. `anchor=True` also sends every checkpoint.

## System checks

| Id | When |
| --- | --- |
| `oversight.E001` | A policy was not registered (the reason is in the message). |
| `oversight.E002` | A policy does not resolve (unknown parent, a cycle). |
| `oversight.E003` | A rule reads a lens its policy does not have. |
| `oversight.E004` | A lens names an alias `ANSWERERS` does not define (optional `?` aliases excepted). |
| `oversight.E005` | An alias cannot be built. |
| `oversight.E006` | `DEFAULT_POLICY` names an unknown policy. |
| `oversight.W004` | Evidence is stored in the clear. |
| `oversight.W005` | Checkpoints are unsigned. |
