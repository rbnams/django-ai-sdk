---
title: Oversight
type: docs
weight: 127
---

The [Oversight guide](/oversight/) covers the ideas. This page documents the classes, the stages, the models, the service and the commands.

## Installing

`pip install "django-ai-sdk[oversight]"` adds `cryptography`, for encrypted evidence and signed checkpoints; without it the ledger still chains and verifies and content is stored in the clear. `[typesafe]` adds the Jev client. `[presidio]` adds Microsoft Presidio's analyzer; `[all]` leaves it out on purpose, because it also needs a spaCy language model you download yourself.

## Settings

`AI_SDK_OVERSIGHT`, one dict, every key optional (`django_ai_sdk.oversight.conf.DEFAULTS`). Dict values merge one level deep with the defaults, so `{"FAIL_MODE": {"gate": "closed"}}` keeps the other stages open.

| Key | Default | Meaning |
| --- | --- | --- |
| `ENABLED` | `True` | Kill switch: off leaves every agent unwatched, whatever its policy says. |
| `DEFAULT_POLICY` | `""` | The policy of agents that name none. Empty: they are unwatched. |
| `CAPTURE` | `"full"` | The most content the ledger keeps (`none`, `metadata`, `redacted`, `full`); a policy may keep less, never more. |
| `STREAM_MODE` | `"live"` | `hold` shows an answer only after review passed. |
| `POLICIES` | `{}` | Per policy, the tuning the console allows, kept in settings: `{"support": {"rules": {"block_injection": {"mode": "enforce"}}}}`. |
| `ANSWERERS` | `{}` | Named answerers: `{"alias": {"class": ..., **kwargs}}`, or the name of another alias. |
| `BUDGET_PER_DAY` / `CURRENCY` | `None` / `"EUR"` | Daily cap on metered answerers' spend, across the deployment. `None`: no cap. |
| `GATE_DEADLINE_MS` | `1000` | What the gate may add before the model starts: room for Jev and code, not for a model judge. |
| `MAX_INPUT_CHARS` | `100000` | A longer question is refused at the gate, unread. `None`: no limit. |
| `REVIEW_DEADLINE_MS` | `2000` | What review may add after the answer is complete. |
| `WATCH_EVERY_CHARS` | `400` | Streamed characters between watch checks. |
| `WATCH_DEADLINE_MS` | `300` | How long a watch check may pause the stream; late lenses answer `deadline` and are asked again in reflect. |
| `FAIL_MODE` | all `"open"` | Per inline stage, what a failed or late answerer means. `open` lets the turn through. `closed` treats an enforced rule that blocks, redacts or takes tools away as matched when a lens it reads failed; rules that only alert, queue or notify, and shadow rules, never fire on a failure. |
| `CLIENT_DETAIL` | `"actions"` | What `data-oversight` tells the chat client. `actions`: only rules that changed the turn. `full`: every stage's answers and matches too, for a developer view. `staff`: that view only for users who may view oversight, since it shows what each rule tests. |
| `RETENTION` | `{"content": 90, "metadata": 730}` | Days a turn's content (evidence) and its metadata (the turn, its answers and decisions) are kept. A policy may set either; other keys are rejected. |
| `PSEUDONYM_SECRET` | derived from `SECRET_KEY` | HMAC secret for actor pseudonyms. |
| `MASTER_KEY` | `""` | base64 32 bytes that wrap the per-thread content keys. Empty stores content in the clear (`oversight.W004`). |
| `SIGNING_KEY` / `SIGNING_NAME` | `""` | base64 Ed25519 private key for signed checkpoints, and the key's name (defaults to the log origin). |
| `LOG_ORIGIN` | first `ALLOWED_HOSTS` entry + `/oversight` | The checkpoint's origin line. |
| `EXPORTERS` | `[]` | Where notifying decisions and checkpoints are sent. |
| `MODEL_CHOICES` | `[]` | Models the console offers for judges: `[{"id", "label", "group", "kinds"}]`, or the dotted path of a function returning that list. |
| `POLICY_CACHE_SECONDS` | `10` | How long a resolved policy (code plus database overlay) is reused per process. |

`AI_SDK_MODEL_PRICES` ([Settings](/manual/settings/)) prices the judges; `BUDGET_PER_DAY` never counts a model without a price (`oversight.W010`).

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

| Kind | Value | TypeSafe |
| --- | --- | --- |
| `Flag` | `bool` | Noul (`yes` / `no` criteria) |
| `Choice` | one of `labels` (`default` when a detector finds none) | Choice (`criteria`) |
| `Score` | `0..1` (`low` / `high` describe the ends) | Score |
| `Entities` | sorted list of labels | — |
| `Text` | string, `max_chars` | — |

## Answerers

`django_ai_sdk.oversight.answerers`. `answer(lens, view) -> Answer | None`; `None` abstains.

Local (`metered = False`, `external = ""`: nothing is charged and no turn content leaves):

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
| `HFClassifier(model, positive, labels, endpoint)` | A Hugging Face classifier; local without `endpoint`, metered and sent to Hugging Face with one. |
| `Function(fn)` | Any function `fn(lens, view)` returning a value, an `Answer` or `None`. |
| `Human(queue, priority)` | Queues the lens for a person; the answer arrives later. |

Remote (`external` names who receives the turn's content, secrets masked first):

| Class | Answers with |
| --- | --- |
| `TypeSafe(model=None)` | Jev (`external = "TypeSafe"`, not metered). `[typesafe]` extra and `TYPESAFE_API_KEY`; without them it abstains (`oversight.W008`). A failed request is the error `jev: <exception>`. One stage's questions about the same state go out as one request. |
| `TypeSafeSources(model=None)` | Jev on each source of the turn, for an `Entities` lens such as `taken_from`: one yes/no per source in one request; the value is the titles of the sources the answer is taken from, the rationale each source's probability, the confidence the least certain source's. A source an earlier turn found is named "(found earlier)". With no source at all, Jev is asked whether the reply states facts: one that does, or may, is taken from none, and so is an empty answer. |
| `LLMJudge(model, llm=None)` | Any SDK generator factory (`llm`, a callable or dotted path; OpenAI Responses by default). Metered. |
| `AgentJudge(agent)` | A registered agent, by id, name or class name: its model and instructions; its tools are not run. Oversight is off inside the judge's run. Metered. |
| `Moderation(model)` | OpenAI's moderation endpoint. |

A lens's answerers are asked in order until one is sure (its confidence reaches the lens's `confident`); otherwise the most confident usable answer stands, so a fallback that is less sure cannot overrule the first. Every other answer is kept in its `trail`.

A lens that judges the answer against evidence (subject `sources`, or `needs_evidence = True` such as `groundedness`) reads the turn's retrieved sources *and* its tool results, and is not asked on a turn that has neither: an answer from general knowledge is not "ungrounded". A lens on the answer also reads the sources earlier turns of the thread found, after this turn's: a follow-up may answer from them. A lens on the sources judges this turn's search alone. Jev reads every source whole, labelled `S1`, `S2`, ..., up to `SOURCE_CHARS` (8,000) each and `SOURCES_CHARS` (32,000) together, earlier turns' sources the first to be left out; a search tool's result that only repeats sources already listed is read once. Reflect asks no lens on the answer of a turn the gate blocked: there is none.

## Rules and conditions

```python
Rule(name, when=Condition, then=Action | [Action], mode=Mode.SHADOW, stages=None,
     message="", description="")
```

Conditions are built from `L(lens)`: `.is_true`, `.is_false`, `== value`, `!= value`, `> < >= <=`, `.in_(values)`, `.contains(label)`, `.any`, `.missing`, `.present`, `.confident(x)`, `.unsure(x)`, combined with `&`, `|` and `~`. Each serializes to JSON (`{"all": [...]}`, `{"any": [...]}`, `{"not": ...}`, `{"lens", "op", "value"}`), which is what the database overlay stores. A comparison on a lens without a usable answer is false.

A rule without `stages` is evaluated once every lens it reads has had its chance, so it can combine a gate lens and a reflect lens. A rule fires once per turn.

Watch can only stop an answer. A match there is kept when it blocks or when the rule's `stages` name watch (then it is recorded and changes nothing); every other one is left for review, which judges the whole answer. A redaction found while streaming is applied when the stream ends: review masks the stored answer and tells the client to replace its copy. A tool call rule is judged on every call, so a refused call stays refused when the model retries it; the ledger records it once.

Reflect masks only the stored message, and only when the stored text is the text it judged (under `redacted` capture a masked evidence copy differs, so nothing is written). It appends a `reflect` notice to the message, which a reloaded thread shows as adjusted. Lenses that missed an inline deadline are asked again there, never sampled away; their late answers reach only rules whose `stages` include reflect or that have no `stages`. Tool call lenses are not asked again, and a turn that kept no text (`metadata` capture, shredded) is judged only on the sources it kept.

| Action | Enforced at | Outcome |
| --- | --- | --- |
| `Block(message)` | gate, watch, review | `blocked` |
| `RestrictTools(names)` (`[]` = all) | gate, watch | `restricted` |
| `Redact(lenses=None, template="[{label}]")` | gate, review, reflect | `redacted` |
| `Queue(queue, priority)`, `Escalate(lens, answerer)`, `Notify(exporters, severity)`, `Alert(severity)` | any | `flagged` |
| `Allow()` | any | `allowed` |

A severity is `info`, `warning`, `high` or `critical` (the old `error` reads as `high`). A blocked turn is at least `high`.

## Policies

```python
class Policy:
    name: str
    extends: list[str]          # packs or policies
    lenses: list[Lens | type[Lens]]
    rules: list[Rule]
    capture: Capture | None     # none, metadata, redacted, full
    stream_mode: str | None     # live, hold
    retention: dict[str, int]   # content, metadata (days); either alone
```

Register with `@register` in an app's `oversight.py`. Agents choose with `oversight = "name"` or `get_oversight_policy()`.

`pick(items, *names, mode=None)` takes single lenses or rules out of a pack instead of extending it whole: `pick(SecurityPack.lenses, "prompt_injection")`, `pick(SecurityPack.rules, "block_injection", mode=Mode.ENFORCE)`. They come back in the order named; rules are copies, so `mode` leaves the pack's own alone. An unknown name raises `ImproperlyConfigured`, and `mode` on lenses raises `TypeError`. Picked lenses and rules have the source `code`.

Every effective value has a source, in this order: the extended packs (`pack:<name>`), the class (`code`), `AI_SDK_OVERSIGHT["POLICIES"]` (`settings`), then `PolicyOverride` rows (`db`). A parent's settings and console rows reach the policies that extend it, with the source `db:<parent>` for the latter. Overrides may set a lens's `enabled`, `sample`, `answerers`, `confident` and `stages`, a rule's `mode`, `when` (thresholds only, unless the rule itself came from the database), `then`, `message` and `stages`, and a policy's `capture`, `stream_mode` and `retention`. They may also add lenses and rules (`field="definition"`). An override that cannot apply is listed under `rejected` and ignored.

What the console writes names answerers only by alias or as one of the catalog's kinds (`django_ai_sdk.oversight.catalog`), with that kind's listed parameters: `ALIAS_KINDS` for an alias, `INLINE_KINDS` inside a lens. Any other class or keyword is refused when saved and rejected when loaded, so the console never imports host code. Settings and code are not limited this way; a settings alias with options the console cannot write is shown read-only.

A policy that fails to load (a missing parent, a cycle, a broken row) leaves its agents unwatched rather than failing the chat; the reason is logged once per `POLICY_CACHE_SECONDS` and shown in the policy list.

## The session

`TurnSession.start(agent, channel, user, thread_id, messages | input)` returns an inert session unless a policy applies. Then `gate()`, `note_text(chunk)` + `watch()`, `check_tool(call)`, `set_sources(...)`, `add_tool_call(...)`, `add_usage(...)`, `review(output)`, `close()`. Chat (`Agent.as_view` and `Stream`), webhooks (`run_inbound`) and workflow `AgentStep` call it already; channels are `chat`, `webhook:<integration>`, `workflow:<workflow>:<step>`, or what you pass. `unobserved()` suppresses sessions inside a block; `active(session)` makes one visible to `get_tools()` and the tool hook.

A headless `agent.run(..., tools=True)` inside `active(session)` reports its tool calls, their results and its usage to the session, so review judges what the loop did. A workflow `AgentStep` withholds a result review blocked, and rebuilds a structured result from the masked JSON (a result that no longer validates fails the step). Known gap: an `AgentStep` without tools reports no token usage, so its turns have no cost.

## Models

| Model | Holds |
| --- | --- |
| `Turn` | Channel, agent, agent fingerprint, policy, thread and message ids (no foreign keys), pseudonym, sealed actor, model, tokens, cost, latency, `outcome`, `shadow_outcome`; labels and severity (denormalized); `expires_at`, when the policy's metadata retention runs out. |
| `Evidence` | The content at the policy's capture level, encrypted per scope, with a salted digest and an expiry. |
| `Observation` | One answer to one lens: answerer, model, value, confidence, rationale, spans, trail, cost, latency, reviewer. A `Text` lens's value is kept in the evidence (`texts`) instead, and rationales are kept only at `redacted` or `full` capture and cleared when the content is shredded or expires; neither is hashed, so the chain still verifies. |
| `Decision` | A rule that held: stage, mode, actions, whether it acted, the condition. |
| `ReviewItem` | A turn waiting for a person, per queue. |
| `AccessRecord` | A read of the ledger: who, purpose, action, turn, reason. |
| `PolicyOverride` | One operator change; a new row withdraws an old one. `active` marks the latest row per field, is left out of the hash, and is the one field updated after insert. |
| `LedgerEntry`, `LedgerHead` | The chain and its lock. |
| `Checkpoint`, `DataKey`, `DailySpend` | Signed roots, per-scope keys, spend per day and answerer. |

## Service and HTTP

`OversightService` (domain `oversight`): `list_turns` (narrowed by a `TurnFilter`), `get_turn`, `reveal`, `list_reviews`, `queues`, `resolve_review`, `add_verdict`, `list_policies`, `get_policy`, `overview`, `checkpoints`, `verify`, `seal_now`, `access_log`, `export`, `forget_thread`, and the console's part (`console.ConsoleMixin`): `set_override`, `create_policy`, `delete_policy`, `list_agents`, `set_agent_policy`, `catalog`, `set_answerer`, `try_answerer`, `simulate`, `try_lens`, `changes`. Unknown ids return `None`; refused operations raise `PermissionDenied`; bad input raises `UserError`.

`get_oversight_router()` serves them under `/oversight/`:

| Route | Method | Service |
| --- | --- | --- |
| `turns/`, `turns/{id}/` | GET | `list_turns` (query: `purpose` and the `TurnFilter` fields), `get_turn` |
| `turns/{id}/reveal/`, `turns/{id}/verdicts/` | POST | `reveal`, `add_verdict` |
| `reviews/`, `queues/` | GET | `list_reviews`, `queues` |
| `reviews/{id}/` | POST | `resolve_review` |
| `policies/`, `policies/{name}/` | GET | `list_policies`, `get_policy` |
| `policies/` | POST | `create_policy` |
| `policies/{name}/overrides/`, `policies/{name}/delete/` | POST | `set_override`, `delete_policy` |
| `policies/{name}/simulate/`, `policies/{name}/try-lens/` | POST | `simulate`, `try_lens` (dry runs; nothing is stored) |
| `agents/` | GET | `list_agents` |
| `agents/{id}/` | POST | `set_agent_policy` |
| `catalog/` | GET | `catalog` |
| `answerers/{alias}/`, `answerers-try/` | POST | `set_answerer`, `try_answerer` |
| `changes/`, `overview/`, `access/` | GET | `changes`, `overview`, `access_log` |
| `checkpoints/` | GET, POST | `checkpoints`, `seal_now` |
| `verify/` | POST | `verify` |
| `export/` | GET | `export`, as NDJSON |
| `threads/{id}/forget/` | POST | `forget_thread` |

Turn reads need `VIEW_OVERSIGHT`, or `AUDIT_OVERSIGHT` for `purpose=audit`. Reviews, queues and verdicts need `REVIEW_OVERSIGHT`; `reveal` needs `REVEAL_ACTOR`; checkpoints, `verify` and the access log need `AUDIT_OVERSIGHT`. Console writes, dry runs, `seal_now`, `export` and `forget_thread` need `MANAGE_OVERSIGHT`. See [Permissions](/manual/permissions/).

## Commands and tasks

| Command | Does |
| --- | --- |
| `oversight_verify [--verifier KEY] [--json]` | Walks the chain and the latest checkpoint; exits non-zero on tampering. |
| `oversight_seal` | Writes a checkpoint over new entries and sends it to anchoring exporters. |
| `oversight_prune [--thread ID]` | Applies retention, or crypto-shreds one thread. |
| `oversight_keys` | Prints new keys, or the verifier key for the configured one. |
| `oversight_replay AGENT --file golden.jsonl \| --reviewed N` | Retro: replays questions and checks the verdicts held (`simulate.replay` from code). |

Schedule both `oversight_prune` and `oversight_seal`, hourly or daily, with cron or your scheduler: nothing runs them for you. Without prune, retention never applies; without seal, no checkpoint is written and nothing is anchored.

Tasks (`django_ai_sdk.oversight.tasks`): `reflect_turn`, queued on commit when the policy has reflect lenses or an escalation, and `export_events`, which sends a turn's notifications from the worker so a slow receiver never holds a stream open.

## Exporters

`AI_SDK_OVERSIGHT["EXPORTERS"]` lists `Log()` and `Webhook(url, secret)`, or your own `Exporter` subclass, as instances or `{"class": ..., **kwargs}`. `anchor=True` also sends every checkpoint.

A `Webhook` with a `secret` signs like [Standard Webhooks](https://www.standardwebhooks.com/): headers `webhook-id`, `webhook-timestamp` and `webhook-signature` (`v1,` + base64 HMAC-SHA256 of `{id}.{timestamp}.{body}`).
To verify, recompute the HMAC over the raw body with the same secret (base64-decoded when it starts with `whsec_`) and compare in constant time.
Refuse timestamps more than a few minutes old and ids you have seen, so a captured request cannot be replayed:

```python
import base64, hashlib, hmac, time

def verified(headers, body: bytes, secret: bytes) -> bool:
    msg_id, stamp = headers["webhook-id"], headers["webhook-timestamp"]
    if abs(time.time() - int(stamp)) > 300:
        return False
    signed = f"{msg_id}.{stamp}.".encode() + body
    expected = "v1," + base64.b64encode(hmac.new(secret, signed, hashlib.sha256).digest()).decode()
    return any(hmac.compare_digest(expected, s) for s in headers["webhook-signature"].split())
```

An exporter's `export(event)` returns a reference to what the receiver stored, or `None`. It may raise: the error is logged and the turn goes on.

## System checks

| Id | When |
| --- | --- |
| `oversight.E001` | A policy was not registered (the reason is in the message). |
| `oversight.E002` | A policy does not resolve (unknown parent, a cycle). |
| `oversight.E003` | A rule reads a lens its policy does not have. |
| `oversight.E004` | A lens names an alias `ANSWERERS` does not define (optional `?` aliases excepted). |
| `oversight.E005` | An alias cannot be built. |
| `oversight.E006` | `DEFAULT_POLICY` names an unknown policy. |
| `oversight.E007` | An enforced rule reads a lens no answerer here can answer. |
| `oversight.W011` | An enforced rule reads lenses whose answerers are all optional (`"jev?"`) and none is defined: it never acts until one is. |
| `oversight.W004` | Evidence is stored in the clear. |
| `oversight.W005` | Checkpoints are unsigned. |
| `oversight.W006` | A lens has no answerer here (its optional aliases are not defined, or Jev has no key). Console aliases are checked live. |
| `oversight.W007` | A lens lists answerers it never asks: a detector before them is sure when it finds nothing. |
| `oversight.W008` | An alias is Jev, but `TYPESAFE_API_KEY` is unset or `typesafe-sdk` is missing. |
| `oversight.W009` | A lens reads the question at watch or review, which only judge the answer. |
| `oversight.W010` | `BUDGET_PER_DAY` is set, but a judge alias's model has no price in `AI_SDK_MODEL_PRICES`. |
