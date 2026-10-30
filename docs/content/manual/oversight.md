---
title: Oversight
type: docs
weight: 127
---

The [Oversight guide](/oversight/) covers the ideas. This page documents the classes, settings, stages, models, service, routes, commands and checks.

## Installing

Add `"django_ai_sdk.oversight"` to `INSTALLED_APPS` and run `migrate`. The app needs no extra to run: the ledger chains and verifies without one.

- `pip install "django-ai-sdk[oversight]"` adds `cryptography`. Encrypted evidence needs it and a `MASTER_KEY`; signed checkpoints need it and a `SIGNING_KEY`; `oversight_keys` needs it to make keys. Without a `MASTER_KEY`, content is stored in the clear (`ai_sdk.oversight.W001`, a deploy check). A key that is set but unusable (wrong length, or `cryptography` missing) is an error (`ai_sdk.oversight.E008`, `ai_sdk.oversight.E009`): turns are then recorded without their content and without who asked, never in the clear; content sealed earlier reads as `unreadable`; and no checkpoint can be signed.
- `pip install "django-ai-sdk[typesafe]"` adds the client for the optional `TypeSafe` answerer (`django_ai_sdk.oversight.answerers.TypeSafe`), a hosted commercial service.

## Settings

`AI_SDK_OVERSIGHT`, one dict, every key optional (`django_ai_sdk.oversight.conf.DEFAULTS`). A dict value merges one level deep with its default, so `{"FAIL_MODE": {"gate": "closed"}}` keeps the other stages open. A key nothing reads is reported (`ai_sdk.oversight.W010`).

| Key | Default | Meaning |
| --- | --- | --- |
| `ENABLED` | `True` | Kill switch: off leaves every agent unwatched, whatever its policy says. |
| `DEFAULT_POLICY` | `""` | The policy of agents whose `oversight` is `None`, and of runtime agents whose base class names none. Empty: they are unwatched (`ai_sdk.oversight.W012` warns about runtime agents). |
| `CAPTURE` | `"masked"` | The most content the ledger keeps (`none`, `metadata`, `masked`, `full`); a policy may keep less, never more. |
| `STREAM_MODE` | `"live"` | `live` streams the answer as it is written; `hold` shows it once finish passed it. A policy may set its own. |
| `POLICIES` | `{}` | Per policy, the tuning the console allows, kept in settings: `{"support": {"rules": {"block_injection": {"mode": "enforce"}}, "capture": "masked"}}`. |
| `ANSWERERS` | `{}` | Named answerers: `{"alias": {"class": ..., **kwargs}}`, or the name of another alias. A class without a dot is one of `django_ai_sdk.oversight.answerers`. |
| `BUDGET_PER_DAY` | `None` | Cap on metered answerers' spend per UTC day, across the deployment. `None`: no cap. |
| `CURRENCY` | `"USD"` | The currency `AI_SDK_MODEL_PRICES` and `BUDGET_PER_DAY` are in: a label, never converted. |
| `GATE_DEADLINE_MS` | `1000` | What the gate may add before the model starts. Room for code and a fast classifier; a model judge (`LLMJudge`) usually needs longer, and a lens still running at the deadline fails, so under the default open fail mode its enforced rule does not act. Also the deadline of each tool call check and of masking a tool result. |
| `FINISH_DEADLINE_MS` | `2000` | What finish may add once the answer is complete. |
| `MAX_INPUT_CHARS` | `100000` | A longer question is refused at the gate, unread, when the policy has gate lenses. `None`: no limit. |
| `CLIENT_DETAIL` | `"actions"` | What `data-oversight` tells the chat client. `actions`: the stage, outcome and message when an enforced rule acted, no rule names. `full`: rule names and every stage's answers too, for a developer view, each value as the record keeps it (see [Models](#models)). `staff`: that view only for users with `VIEW_OVERSIGHT`. |
| `WATCH_EVERY_CHARS` | `400` | Streamed characters between watch checks. |
| `WATCH_DEADLINE_MS` | `300` | How long a watch check may pause the stream. Late lenses answer `deadline` and are asked again in reflect. |
| `FAIL_MODE` | `{"gate": "open", "watch": "open", "finish": "open"}` | Per inline stage, what a failed or late answerer means. `open` lets the turn through. `closed` treats an enforced rule that blocks, masks or takes tools away as matched when a lens it reads failed, and withholds a tool result it cannot check. Rules that only alert, queue or notify, and rules in report mode, never fire on a failure. A policy that cannot be chosen or loaded follows the gate's: closed refuses the turn, open runs it unwatched and logs. Open is the default so an answerer outage never takes chat down; the price is that a deadline a slow answerer misses (a long question can make it slower) lets the turn through: set `closed` where a rule must hold. |
| `RETENTION` | `{"content": 90, "metadata": 730}` | Days a turn's content (evidence, with the reasons, `Text` answers and notes kept in it) and its metadata (the turn, its answers and decisions) are kept. A policy may set either. |
| `PSEUDONYM_SECRET` | `""` (derived from `SECRET_KEY`) | HMAC secret for actor pseudonyms. |
| `MASTER_KEY` | `""` | base64 of 32 bytes; wraps the per-thread content keys. Empty stores content in the clear (`ai_sdk.oversight.W001`). |
| `SIGNING_KEY` | `""` | base64 Ed25519 private key (32 bytes) for signed checkpoints. Empty: checkpoints are unsigned (`ai_sdk.oversight.W002`). |
| `SIGNING_NAME` | `""` (the log origin) | The key's name in the signature; a verifier key carries it too. |
| `LOG_ORIGIN` | `""` (first `ALLOWED_HOSTS` entry + `/oversight`) | The checkpoint's origin line. |
| `EXPORTERS` | `[]` | Where notifying decisions and checkpoints are sent: instances, dotted paths, or `{"class": "<dotted path>", **kwargs}` (unlike `ANSWERERS`, always the full path: `"django_ai_sdk.oversight.exporters.Webhook"`). One that cannot be built is logged and left out. |
| `MODEL_CHOICES` | `[]` | Models the console offers for judges: `[{"id", "label", "group", "kinds"}]`, or the dotted path of a function returning that list. |
| `POLICY_CACHE_SECONDS` | `10` | How long a resolved policy (code plus database overlay) is reused per process. |

Outside the dict:

| Setting | Default | Meaning |
| --- | --- | --- |
| `AI_SDK_MODEL_PRICES` | `{}` | Prices the judges ([Settings](/manual/settings/)). `BUDGET_PER_DAY` never counts a model without a price (`ai_sdk.oversight.W007`). |
| `TYPESAFE_API_KEY` | unset | The `TypeSafe` answerer's key; without it that answerer abstains (`ai_sdk.oversight.W005`). |
| `TYPESAFE_MODEL` | unset | A TypeSafe model to pin; unset leaves the choice to `typesafe-sdk`. `TypeSafe(model=...)` overrides it. |

## Agents and runs

| On `Agent` | Meaning |
| --- | --- |
| `oversight` | A `Policy` subclass (registered when the agent class is defined), a registered policy's name, `None` (the default: `DEFAULT_POLICY`) or `False` (no oversight). Anything else (an instance, `True`, `""`) raises `ImproperlyConfigured` when the class is defined. |
| `async get_oversight(*, channel, user, **kwargs)` | The policy for this turn; returns `oversight` unless overridden. It may read the database; keep `**kwargs`, as more about the turn may be passed later. Channels: `chat`, `run`, `api`, `webhook:<integration>`, `workflow:<workflow>:<step>`, `workflow:adhoc:<step>` (a submitted definition), or what a caller passes. A policy only this returns is registered when a turn first uses it: `@register` it, so the console and the checks see it. |
| `run(messages, ..., *, channel="run", oversight=<own>)` | One turn under the policy. Left out, `oversight` is the agent's own; a class or a name replaces it for this call; `None` is `DEFAULT_POLICY` (like `oversight = None` on the class); `False` runs unwatched (trusted code only). Raises `Blocked`. |
| `as_view(...)` | Chat: one turn on channel `chat`. A refusal streams the policy's message. |

A turn's policy is, first match wins: `run(oversight=...)`; the console's assignment for the agent (a name, or "off"); `get_oversight()`; `DEFAULT_POLICY` when that returned `None`. `False` anywhere in that order means no oversight. A runtime agent (an `AgentSettings` row) without an assignment of its own takes its base class's `oversight` (`AI_SDK_RUNTIME_AGENT_BASES`), then `DEFAULT_POLICY`. A hook that raises, or a policy that cannot load, follows `FAIL_MODE["gate"]`; the console lists such an agent with the reason `error`.

`django_ai_sdk.errors.Blocked(message, *, stage="", turn_id="")`, error code `blocked` (HTTP 400, like `content_filtered`, through Ninja's `register_error_handlers` and DRF's `django_ai_sdk.contrib.drf.exception_handler`, with `message` as the body's message). Raised by `Agent.run()` when the gate refuses the history or the question, or finish withholds the answer, and when a masked structured result no longer validates. `message` is the policy's text for the person asking. The run endpoints answer `{"result", "thread_id", "blocked"}` instead (`thread_id` is `""` for the stateless one).

Each agent runs under its own policy. A run inside an active turn joins it when it resolves the same policy, or none: no gate or finish of its own, its tool calls watched and recorded in the outer turn. Under another policy (its own, or an explicit `oversight=`) it opens a child turn, with its own gate, watch, finish and record, and `parent_turn_id` naming the outer turn. A subagent's delegation does the same with its task as the question; a refusal is the tool result the coordinator reads. A delegation run synchronously (Haystack's `run()`) cannot open a turn: when it would need one, it answers `UNWATCHABLE` instead. `run(oversight=False)` runs unwatched and does not join. A closed turn is never joined, even by a task that copied it.

`run()` judges what the caller sent on copies: only the last user message of the list is masked in place, so what the caller stores is the masked copy. History the server stored itself is judged on its own turn, not again: with a `thread_id` whose thread is stored, under a policy whose gate masks, the stored messages are the history (the thread run endpoints, as chat); a history that is the stored thread as stored stands (a webhook's). Any other history is judged by `TurnSession.check_history()` before the question.

A subclass's own `run()` is wrapped when the class is defined: the turn opens around the outermost `run()`, its arguments read as `Agent.run` takes them, and the `super().run()` inside joins. An override that answers without calling `super()` is judged too. `run_turn` in `django_ai_sdk.oversight.headless` is what `run()` wraps around the work; `open_turn` decides between joining and a child turn; `thread_history` is the stored-thread rule chat shares.

## Lenses

`django_ai_sdk.oversight.lenses`. Subclass one, or construct one with keyword arguments (an unknown keyword raises `ImproperlyConfigured`).

| Attribute | Default | Meaning |
| --- | --- | --- |
| `name` | required | Key within the policy. Observations key on it, so renaming starts a new series. |
| `label` | `""` | Shown in the console; defaults to the name. |
| `question` | `""` | Asked verbatim of model answerers. |
| `guidance` | `""` | Extra instructions for judges: scale anchors, edge cases. |
| `subject` | `Subject.INPUT` | `input`, `output`, `exchange`, `sources`, `tool_call`. |
| `stages` | `(Stage.FINISH,)` | Any of `gate`, `watch`, `finish`, `reflect`. Gate and reflect read the question; watch and finish only the answer and tool calls. |
| `answerers` | required | Tried in order: instances, aliases, or optional aliases (`"judge?"`). |
| `min_confidence` | `0.75` | An answer below this asks the next answerer. |
| `sample` | `1.0` | Share of turns the reflect stage judges. Inline stages never sample. |
| `enabled` | `True` | Off: the lens never runs. |
| `needs_evidence` | `False` | Only asked when the turn has sources or tool results to judge against. |

| Kind | Value | Own attributes |
| --- | --- | --- |
| `Flag` | `bool` | `yes`, `no`: what each means |
| `Choice` | one of `labels` | `labels`, `criteria` (label to meaning), `default` (a label; what a detector that finds none means) |
| `Score` | `0..1` | `low`, `middle`, `high`: what the ends and the middle mean |
| `Entities` | sorted list of labels, with spans | `labels` (when set, other values are dropped); `per_source` (an answerer that judges source by source answers with the sources that hold, `items` each one's weight) |
| `Text` | string | `max_chars` (500) |

A `Choice` whose `default` is not one of its `labels` is refused. Only an `Entities` lens says where in the text it found something, so only it can drive a `Mask`.

## Answerers

`django_ai_sdk.oversight.answerers`. `answer(lens, view) -> Answer | None`; `None` abstains. An `Answer` has `value`, `confidence`, `rationale`, `spans`, `items`, `trail`, `error`, `queued`, `cost`, `tokens`, `latency_ms`, `answerer` and `model`. Every answerer also takes `name=`, the name its answers are recorded under. The package exports the answerers; helpers such as `base.resolve` and `pii.mask` stay in their modules.

Local (`external = ""`: nothing leaves the process; not metered):

| Class | Answers |
| --- | --- |
| `PII(kinds=None, *, extra=None)` | Personal data. Default kinds (`DEFAULT_KINDS`): `email`, `iban` (mod-97), `credit_card` (Luhn), `phone` (international format), `ip_address`. Country kinds (`COUNTRY_KINDS`), only when named: `bsn` (11-check), `phone_nl`, `postcode_nl`. `extra`: label to regex. |
| `Secrets(*, extra=None)` | `api_key`, `stripe_key`, `google_api_key`, `github_token`, `aws_key`, `slack_token`, `jwt`, `private_key`, `password_assignment`. |
| `Patterns(patterns, *, flags=re.IGNORECASE, miss_confidence=0.9)` | Regexes, by label or as one list. |
| `Keywords({label: [words]}, *, miss_confidence=0.5)` | Whole words; the label with most hits wins. |
| `Language()` | The language (a `Choice`), or "answered in another language" (a `Flag`). |
| `Length(max_chars)`, `Refusal()`, `EmptyRetrieval()`, `ToolErrors()` | As named. |
| `LexicalSupport()` | Word-overlap groundedness (output) and context relevance (sources). Low confidence by design: a floor under the judges. |
| `Function(fn, *, name="", metered=False)` | Any function `fn(lens, view)`, sync or async, returning a value, an `Answer` or `None`; or its dotted path. |
| `Human(queue="default", *, priority=0)` | Queues the lens for a person; the answer arrives later. Also the built-in alias `human`. |

A detector's hit has confidence 1; its "found nothing" has its `miss_confidence` (`PII` 0.85, `Secrets` 0.9). When that reaches the lens's `min_confidence`, answerers after the detector are never asked (`ai_sdk.oversight.W004`).

Remote (`external` names who receives the turn's content; secrets are masked first):

| Class | Answers with |
| --- | --- |
| `LLMJudge(model=None, *, llm=None, instructions=..., generation_kwargs=None, timeout=None)` | Any chat model an SDK generator factory builds (`llm`: the factory or its dotted path; OpenAI Responses by default). `external = "model gateway"`, metered, 45 s timeout. Structured output where the generator supports it, parsed JSON where not. |
| `TypeSafe(model=None)` | TypeSafe, a hosted commercial classification service. `external = "TypeSafe"`, not metered, 20 s timeout. Needs the `typesafe` extra and `TYPESAFE_API_KEY`, else abstains (`ai_sdk.oversight.W005`). Import it from `django_ai_sdk.oversight.answerers`. Answers `Flag`, `Choice` and `Score` lenses, and an `Entities` lens with `per_source = True` by asking one yes/no per source: the value is the titles of the sources that answer yes, `items` each source's probability. One stage's questions about the same turn go out as one request. A failed request is an `unavailable` error, so `FAIL_MODE` applies and reflect asks again. |

What each remote answerer sends, per lens subject and with every limit, is in the guide's [Where turn content goes](/oversight/#where-turn-content-goes) and in the two modules' docstrings (`answerers/llm.py`, `answerers/typesafe.py`).

A lens's answerers are asked in order until one is sure (its confidence reaches the lens's `min_confidence`). When none is, the last one that answered decides: confidences from different answerers are on different scales and are never compared. Every other answer is kept in its `trail`. A chain that ends on an error or the deadline after only unsure answers counts as failed for `FAIL_MODE`, and keeps the unsure answer for the record.

A lens that judges the answer against evidence (subject `sources`, or `needs_evidence = True` such as `groundedness`) reads the turn's retrieved sources *and* its tool results, and is not asked on a turn that has neither. A lens on the answer also reads the sources earlier turns of the thread found, after this turn's. Reflect asks no lens on the answer of a turn the gate blocked. Reflect reads the turn from its evidence; under `masked` capture it reads the question and answer from the thread's stored messages instead, where there are any, so its masks reach the stored answer.

## Rules and conditions

```python
Rule(name, *, when=Condition, then=Action | [Action], mode=Mode.REPORT, stages=None,
     message="", description="")
```

Conditions are built from `L(lens)`: `.is_true`, `.is_false`, `== value`, `!= value`, `> < >= <=`, `.in_(values)`, `.contains(label)`, `.any`, `.missing`, `.present`, `.confident(x)`, `.unsure(x)`, combined with `&`, `|` and `~`. Each serializes to JSON (`{"all": [...]}`, `{"any": [...]}`, `{"not": ...}`, `{"lens", "op", "value"}`), which is what the database overlay stores. A comparison on a lens without a usable answer is false.

A rule without `stages` is evaluated once every lens it reads has had its chance, so it can combine a gate lens and a reflect lens. A rule fires once per turn. `message` replaces the action's message when the rule blocks.

Watch can only stop an answer. A match there is kept when it blocks or when the rule's `stages` name watch (then it is recorded and changes nothing); any other is left for finish, which judges the whole answer. A tool call rule is judged on every call, so a refused call stays refused when the model retries it; the ledger records it once. `RestrictTools` at watch refuses the calls it names (all, when it names none), and those tools stay away for the rest of the turn.

| Action | Enforced at | Outcome |
| --- | --- | --- |
| `Block(message="")` | gate, watch, finish | `blocked` |
| `RestrictTools(names=None)` (none = all) | gate, watch | `restricted` |
| `Mask(lenses=None, template="[{label}]")` | gate, finish, reflect | `masked` |
| `Queue(queue="default", priority=0)`, `Notify(exporters=None, severity="warning")`, `Alert(severity="warning")` | any | `flagged` |

A severity is `info`, `warning`, `high` or `critical`; a blocked turn is at least `high`. `Block.DEFAULT_MESSAGE` is "This request can't be answered here."

`Mask` masks the spans of `lenses` (default: every lens the rule reads), each replaced by `template` with the label in capitals (`[EMAIL]`). Every lens it masks must be an `Entities` lens on text (not a `tool_call` lens: arguments are judged, never rewritten; block the call or take the tool away instead), or the policy is refused when it is built. Tool results are masked by the gate's mask rules. An enforced `Mask` that cannot point at what it should mask (a lens it reads failed while the stage fails closed, or answered without spans) blocks instead. Reflect masks only the stored message, and only when the stored text is the text it judged; it appends a `reflect` notice to the message.

## Policies

```python
class Policy:
    name: str
    label: str
    description: str
    extends: list[str]          # packs or policies
    lenses: list[Lens | type[Lens]]
    rules: list[Rule]
    capture: Capture | None     # none, metadata, masked, full; None inherits
    stream_mode: str | None     # live, hold; None inherits
    retention: dict[str, int]   # content, metadata (days); either alone

    def get_lenses(self) -> list[Lens]: ...   # override to build them at runtime
    def get_rules(self) -> list[Rule]: ...
```

A policy is registered by `@register` (in any module; an installed app's `oversight.py` is imported on startup), or by naming the class on an agent (`oversight = SupportPolicy`) or in `run(oversight=...)`. A policy that fails validation is not registered (`ai_sdk.oversight.E001`); its agents' turns follow `FAIL_MODE["gate"]`. Two classes that declare one name are an error (`ai_sdk.oversight.E010`): the first one registered keeps the name.

`pick(items, *names, mode=None)` takes single lenses or rules out of a pack instead of extending it whole: `pick(SecurityPack.lenses, "prompt_injection")`, `pick(SecurityPack.rules, "block_injection", mode=Mode.ENFORCE)`. They come back in the order named; rules are copies, so `mode` leaves the pack's own alone. An unknown name raises `ImproperlyConfigured`, and `mode` on lenses raises `TypeError`.

Every effective value has a source, in this order: the extended packs (`pack:<name>`), the class (`code`), `AI_SDK_OVERSIGHT["POLICIES"]` (`settings`), then `PolicyOverride` rows (`db`). A parent's settings and console rows reach the policies that extend it, with the source `db:<parent>` for the latter. Overrides may set a lens's `enabled`, `sample`, `answerers`, `min_confidence` and `stages`, a rule's `mode`, `when` (thresholds only, unless the rule itself came from the database), `then`, `message` and `stages`, and a policy's `capture`, `stream_mode` and `retention`. They may also add lenses and rules (`field="definition"`). An override that cannot apply is listed under `rejected` and ignored.

What the console writes names answerers only by alias or as one of the catalog's kinds (`django_ai_sdk.oversight.catalog`), with that kind's listed parameters: `ALIAS_KINDS` for an alias, `INLINE_KINDS` inside a lens. Any other class or keyword is refused when saved and rejected when loaded, so the console never imports host code. Settings and code are not limited this way.

A policy that fails to load (a missing parent, a cycle, a broken row) follows `FAIL_MODE["gate"]`: open (the default) leaves its agents unwatched rather than failing the chat, closed refuses their turns. The reason is logged once per `POLICY_CACHE_SECONDS` and shown in the policy list.

## Packs

`django_ai_sdk.oversight.packs`. Every pack rule is in report mode. A lens answered only by an optional alias is skipped where that alias is undefined.

| Pack | Lens | Kind, subject | Stage | Answerers |
| --- | --- | --- | --- | --- |
| `security` | `prompt_injection` | Flag, input | gate | `classifier?` |
| | `pii_input` | Entities, input | gate | `PII()` |
| | `secrets_input` | Entities, input | gate | `Secrets()` |
| | `tool_secrets` | Entities, tool call | watch | `Secrets()` |
| | `tool_pii` | Entities, tool call | watch | `PII()` |
| | `pii_output` | Entities, output | finish | `PII()` |
| | `exfiltration` | Flag, output | watch, finish | `Patterns` (Markdown links and images that carry data in their URL) |
| `safety` | `self_harm` | Flag, input | gate | `classifier?` |
| | `harm_input` | Score, input | gate | `classifier?` |
| | `harm_output` | Score, output | finish | `classifier?`, `judge?` |
| `quality` | `context_relevance` | Score, sources | reflect | `classifier?` |
| | `groundedness` | Score, output, needs evidence | reflect | `classifier?` |
| | `taken_from` | Entities, output, `per_source` | reflect | `classifier?` |
| | `answer_relevance` | Score, exchange | reflect | `classifier?` |
| | `helpfulness` | Score, exchange | reflect | `classifier?`, `judge?` |
| | `refusal` | Flag, output | finish | `classifier?` |
| | `language_mismatch` | Flag, exchange | finish | `classifier?` |
| | `empty_retrieval` | Flag, sources | finish | `EmptyRetrieval()` |
| | `tool_errors` | Score, exchange | finish | `ToolErrors()` |

Rules: `security` has `block_injection`, `mask_secrets_in_question`, `mask_pii_in_question`, `refuse_secrets_in_tools`, `mask_pii_in_answer` and `block_exfiltration`; `safety` has `person_at_risk` (never blocks: queues at priority 100 and notifies), `harmful_request` and `harmful_answer`; `quality` has `review_ungrounded`, `review_unsourced`, `review_irrelevant_context`, `note_empty_retrieval` and `note_tool_failures`. `default` extends all three.

## The session

`TurnSession` (`django_ai_sdk.oversight.session`) is the seam every turn passes through. `Agent.run()`, chat and the SDK's entry points open it for you; open one yourself only for a turn the SDK does not run.

`await TurnSession.start(agent, *, channel, user=None, thread_id=None, messages=None, input=None, oversight=None)` returns an inert session unless a policy applies (every call then returns at once and allows everything); `oversight` replaces the agent's own as in `run()`, and `False` opens none. Then `check_history(messages)` (history the server did not store; returns a `GateResult`), `gate()`, `note_text(chunk)` with `watch()`, `check_tool(call)`, `mask_result(call_id, text)`, `set_sources(...)`, `add_tool_call(...)`, `add_usage(...)`, `finish(output)` and `close(error="")`, which writes the turn and never raises.

`unobserved()` suppresses sessions inside a block (the SDK's judges and dry runs run under it); `active(session)` makes one visible to `get_tools()`, the tool hooks and nested runs; `current_session()` returns it, and `None` once it is closed. `OversightToolHook` (before each tool call) and `OversightToolResultHook` (after) are in `django_ai_sdk.agents.tool_agent.default_hooks(agent)`, so every tool agent, subagents included, reports to the active session. Tool results that are not text (images, files) pass as they are.

A headless `agent.run(..., tools=True)` reports its tool calls, their results and its usage to the session, so finish judges what the loop did. A run without tools reports no token usage, so its turn has no cost.

## Client notices

`data-oversight` stream parts (`DataEvent(data_type="oversight")`):

| Key | When |
| --- | --- |
| `turn_id`, `stage`, `outcome` | Always. |
| `message` | The turn was blocked. |
| `replace` | Finish or a watch stop replaced the answer: the text to show. |
| `input` | The gate masked the question: the masked text. |
| `rules`, `detail` | `CLIENT_DETAIL` `full`, or `staff` for users with `VIEW_OVERSIGHT`. Each answer's `value` is the record's: `null` for a `Text` lens, and a name an `Entities` answerer gave in its own words as `"[found]"`, so the client never reads what a mask took out. |

A refused question or withheld answer ends with finish reason `content-filter` (`django_ai_sdk.oversight.stream.CONTENT_FILTER`); `OpenAIProtocolHandler` sends `content_filter` and no `data-*` parts. The notices are stored on the message under `metadata["oversight"]`, with `replace` reduced to `"replaced": true`.

## Models

| Model | Holds |
| --- | --- |
| `Turn` | Channel, agent, agent fingerprint, policy, thread and message ids (no foreign keys), pseudonym, sealed actor (cleared, unhashed, when the thread is forgotten), model, tokens, cost, latency, `outcome`, `report_outcome` (what report-mode rules would have made it); labels (the observations' values, below) and severity (denormalized); `expires_at`, when the policy's metadata retention runs out. |
| `Evidence` | The content at the policy's capture level, sealed under its scope's key (or in the clear without `MASTER_KEY`), with a salted digest and an expiry. Beside it, outside what the digest commits to: `texts`, by observation id, what each answer said in words (its reason, a `Text` answer, the names an `Entities` answerer gave, its per-item scores by key, the same of the answers on its trail); and `notes`, by review item id, what reviewers wrote. Both are added as answers and verdicts arrive, and go when the evidence goes. |
| `Observation` | One answer to one lens: answerer, model, value, confidence, spans (offsets and labels), trail, error, cost, latency, reviewer. Nothing that can quote the turn: a `Text` lens's value is `null`, and an `Entities` value keeps a label the lens declares, or a kind its detector reports (`PII`, `Secrets`, `Patterns`), while any other name (a judge's, `TypeSafe`'s source titles, a person's) is `"[found]"`, one per name, so `.any` and counts still hold. The words are in the evidence's `texts`. |
| `Decision` | A rule that held: stage, mode, actions, whether it acted, the condition. |
| `ReviewItem` | A turn waiting for a person, per queue; its resolution holds who closed it (a pseudonym), the value as the record keeps it, and whether they agreed. Their note is in the evidence's `notes`. |
| `AccessRecord` | A recorded read: who, purpose, action, turn, query, reason. |
| `PolicyOverride` | One operator change; a new row withdraws an old one. `active` marks the latest row per field, is left out of the hash, and is the one field updated after insert. |
| `LedgerEntry`, `LedgerHead` | The chain and its lock. |
| `Checkpoint`, `DataKey`, `DailySpend` | Signed roots, per-scope keys (`thread:<id>`, or `turn:<id>` without a thread), spend per day and answerer. |

Ledger entry kinds: `turn`, `observation`, `decision`, `access`, `override`, `shred` (a forgotten scope) and `prune` (a run of entries whose rows retention deleted; it has no row).

The Django admin registers `Turn`, `Observation`, `Decision`, `ReviewItem` and `Checkpoint` read-only, without the sealed actor. Reasons, `Text` answers and notes are not in those rows, so the admin never shows them. Admin reads are not recorded.

## Service and HTTP

`OversightService` (`django_ai_sdk.oversight.services`, permission domain `oversight`): `list_turns` (narrowed by a `TurnFilter`), `get_turn`, `reveal`, `list_reviews`, `queues`, `resolve_review`, `add_verdict`, `list_policies`, `get_policy`, `overview`, `checkpoints`, `verify`, `seal_now`, `access_log`, `export`, `forget_thread`, and the console's part (`console.ConsoleMixin`): `set_override`, `create_policy`, `delete_policy`, `list_agents`, `set_agent_policy`, `catalog`, `set_answerer`, `try_answerer`, `simulate`, `try_lens`, `changes`. Unknown ids return `None`; refused operations raise `PermissionDenied`; bad input raises `UserError`.

`django_ai_sdk.contrib.ninja.get_oversight_router()` serves them; there is no DRF viewset yet, so a DRF project calls `OversightService` from its own views. The router brings no auth of its own: it runs under the API's (e.g. `NinjaAPI(auth=SessionAuth())`), and each service call checks the permission in the table for `request.user`. The routes carry the `oversight/` prefix, so mount it with `api.add_router("/", get_oversight_router())`.

| Route | Method | Service | Permission | Recorded |
| --- | --- | --- | --- | --- |
| `oversight/turns/` | GET | `list_turns` (query: `purpose`, `limit`, `offset` and the `TurnFilter` fields) | `VIEW_OVERSIGHT`, or `AUDIT_OVERSIGHT` for `purpose=audit` | yes |
| `oversight/turns/{turn_id}/` | GET | `get_turn` (`purpose`) | as above | yes |
| `oversight/turns/{turn_id}/reveal/` | POST | `reveal` (`{"reason"}`, at least 8 characters) | `REVEAL_OVERSIGHT_ACTOR` | yes |
| `oversight/turns/{turn_id}/verdicts/` | POST | `add_verdict` (201) | `REVIEW_OVERSIGHT` | chained as an observation |
| `oversight/reviews/` | GET | `list_reviews` (`queue`, `status`, `purpose`, `limit`, `offset`) | `REVIEW_OVERSIGHT`, and the purpose's | yes |
| `oversight/reviews/{item_id}/` | POST | `resolve_review` (a verdict, `dismiss`); 409 when already closed | `REVIEW_OVERSIGHT` | yes |
| `oversight/queues/` | GET | `queues`: open items per queue | `REVIEW_OVERSIGHT` | no |
| `oversight/policies/` | GET | `list_policies` | `VIEW_OVERSIGHT` | no |
| `oversight/policies/` | POST | `create_policy` (201) | `MANAGE_OVERSIGHT` | chained as an override |
| `oversight/policies/{name}/` | GET | `get_policy` | `VIEW_OVERSIGHT` | no |
| `oversight/policies/{name}/` | DELETE | `delete_policy` (`?note=`; 204) | `MANAGE_OVERSIGHT` | chained as an override |
| `oversight/policies/{name}/overrides/` | POST | `set_override` | `MANAGE_OVERSIGHT` | chained as an override |
| `oversight/policies/{name}/simulate/` | POST | `simulate` (dry run; nothing stored) | `MANAGE_OVERSIGHT` | no |
| `oversight/policies/{name}/try-lens/` | POST | `try_lens` (dry run; nothing stored) | `MANAGE_OVERSIGHT` | no |
| `oversight/agents/` | GET | `list_agents` | `VIEW_OVERSIGHT` | no |
| `oversight/agents/{agent_id}/` | POST | `set_agent_policy` (`policy`: a name, `""` for off; or `reset`) | `MANAGE_OVERSIGHT` | chained as an override |
| `oversight/catalog/` | GET | `catalog` | `VIEW_OVERSIGHT` | no |
| `oversight/answerers/{alias}/` | POST | `set_answerer` | `MANAGE_OVERSIGHT` | chained as an override |
| `oversight/answerers-try/` | POST | `try_answerer` (dry run) | `MANAGE_OVERSIGHT` | no |
| `oversight/changes/` | GET | `changes` (`policy`, `limit`) | `VIEW_OVERSIGHT` | no |
| `oversight/overview/` | GET | `overview` (`days`, `agent_id`, `policy`) | `VIEW_OVERSIGHT` | no |
| `oversight/checkpoints/` | GET | `checkpoints` (`limit`) | `AUDIT_OVERSIGHT` | no |
| `oversight/checkpoints/` | POST | `seal_now` (201, or 204 when nothing was added) | `MANAGE_OVERSIGHT` | no |
| `oversight/verify/` | POST | `verify` | `AUDIT_OVERSIGHT` | yes |
| `oversight/access/` | GET | `access_log` (`turn_id`, `limit`, `offset`) | `AUDIT_OVERSIGHT` | no |
| `oversight/export/` | GET | `export` as NDJSON (`purpose`, default `audit`; `days`, default 30) | `MANAGE_OVERSIGHT`, and the purpose's | yes |
| `oversight/threads/{thread_id}/forget/` | POST | `forget_thread` (`{"reason"}`, at least 8 characters; 204) | `MANAGE_OVERSIGHT` | yes, as `shred` |

By default (`OversightDefaultPermission`) staff have `VIEW_OVERSIGHT`, `REVIEW_OVERSIGHT` and `AUDIT_OVERSIGHT`, and only superusers `REVEAL_OVERSIGHT_ACTOR` and `MANAGE_OVERSIGHT`: any staff user may read every user's turns, content included as far as `CAPTURE` keeps it. Override the `oversight` domain in `AI_SDK_PERMISSIONS` to narrow that. See [Permissions](/manual/permissions/). A recorded read is an `AccessRecord` chained as an `access` entry; the same reader repeating the same list or view within 60 seconds is recorded once. Dry runs with a remote answerer send their sample text to it.

The `audit` purpose shows decisions, verdicts as the record keeps them (value, confidence, answerer, trail) and the chain: no evidence, reasons, spans, items, `Text` answers, names given in an answerer's own words, reviewers' notes, or thread and message ids, and who asked only as the pseudonym, even to a reader who revealed it. The `review` purpose shows the evidence (question, answer, what was shown, sources, tool calls, history) and every verdict with what the evidence keeps of it: its reason, a `Text` answer, the names, per-item scores (`items`, which the response carries under the same field names), and reviewers' notes (`resolution["note"]`). A turn whose evidence is gone or never kept text shows none of these.

`reveal` returns `forgotten: true`, and no user, for a turn whose thread was forgotten. Evidence that `MASTER_KEY` cannot open (another key, none, or one that is set but unusable) reads as `evidence_state: "unreadable"`.

## Commands and tasks

| Command | Does |
| --- | --- |
| `oversight_verify [--verifier KEY] [--since-checkpoint] [--json]` | Walks the chain, every record and the newest checkpoint; exits non-zero on a problem. For every turn whose evidence it can open it recomputes the evidence hash (`evidence altered`; `evidence missing` for evidence neither shredded nor holding content). It checks that the chain still reaches `LedgerHead` as it stood when the walk began (`chain ends early`, `head mismatch`). It reports how many entries retention pruned (`pruned`). `--verifier` checks every signed checkpoint against a verifier key (as `oversight_keys` prints it). `--since-checkpoint` starts after the newest checkpoint signed by that key (or the configured `SIGNING_KEY`) and checks the entries before it against its root; a checkpoint the ledger no longer gives fails. |
| `oversight_seal` | Writes a checkpoint over the entries added since the last, and sends it to anchoring exporters. |
| `oversight_prune [--thread ID]` | Applies retention: shreds expired evidence (with the reasons, `Text` answers and notes kept in it), deletes expired turns with their observations and decisions, and chains a `prune` entry over their entries. `--thread` forgets one thread now: its key, its content and who asked in it. |
| `oversight_keys` | Prints a new `MASTER_KEY` and `SIGNING_KEY` with the verifier key, or, with `SIGNING_KEY` set, the verifier key for it. |

Schedule `oversight_prune` and `oversight_seal`, hourly or daily: nothing runs them for you. Without prune, retention never applies; without seal, no checkpoint is written and nothing is anchored. Publishing checkpoints where the deployment cannot change them is the operator's job; the SDK does not compare against published copies.

What verify cannot tell from the database alone:

- **A `prune` entry is a claim.** Anyone who can write the ledger tables can append one over rows they deleted, and verify then counts those rows as pruned. A prune entry only covers entries before it, never an access, override or shred entry. Treat prune entries as evidence of deletion: check `pruned`, and the turns' dates and policies, against retention.
- **The newest entries.** Deleting them is caught (`chain ends early`) unless `LedgerHead` is rolled back with them. Only a checkpoint published where the database's writers cannot change it protects the end of the chain.

Tasks (`django_ai_sdk.oversight.tasks`, `django-tasks` on the `default` queue): `reflect_turn`, queued on commit when the policy has reflect lenses or a lens is to be asked again (it claims the turn first, so a doubled task does nothing), and `export_events`, which sends a turn's notifications from the worker so a slow receiver never holds a stream open.

## Exporters

`AI_SDK_OVERSIGHT["EXPORTERS"]` lists `Log()` and `Webhook(url, *, secret="", timeout=5.0)` (both in `django_ai_sdk.oversight.exporters`), or your own `Exporter` subclass, as instances, dotted paths, or `{"class": "<dotted path>", **kwargs}`. Every exporter takes `name` and `anchor`; `anchor=True` also sends every checkpoint. A `Notify` action sends `{"type": "oversight.decision", "turn_id", "rule", "stage", "severity", "outcome", "agent", "channel"}`: no turn content.

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
| `ai_sdk.oversight.E001` | A policy was not registered; the message says why (a lens without a name or answerers, a duplicate lens or rule, a rule without `then`, a `Mask` over a lens that is not `Entities` or that judges a tool call, a `Choice` whose `default` is not a label). |
| `ai_sdk.oversight.E002` | A policy does not resolve (an unknown parent, a cycle, an inherited lens a rule cannot use). |
| `ai_sdk.oversight.E003` | A rule reads a lens its policy does not have. |
| `ai_sdk.oversight.E004` | A lens names an alias `ANSWERERS` does not define (optional `?` aliases excepted). |
| `ai_sdk.oversight.E005` | An alias cannot be built. |
| `ai_sdk.oversight.E006` | `DEFAULT_POLICY` names an unknown policy. |
| `ai_sdk.oversight.E007` | An enforced rule reads a lens no answerer here can answer. |
| `ai_sdk.oversight.E008` | `MASTER_KEY` is set but unusable (not base64 of 32 bytes, or `cryptography` is not installed). |
| `ai_sdk.oversight.E009` | `SIGNING_KEY` is set but unusable. |
| `ai_sdk.oversight.E010` | Two policy classes declare the same name; the first one registered keeps it. |
| `ai_sdk.oversight.E011` | An agent class's `oversight` names no policy (set after the class was defined). |
| `ai_sdk.oversight.W001` | Deploy check: evidence is stored in the clear, no `MASTER_KEY`. |
| `ai_sdk.oversight.W002` | Deploy check: checkpoints are unsigned, no `SIGNING_KEY`. |
| `ai_sdk.oversight.W003` | Lenses of the policies in use (`DEFAULT_POLICY`, the ones agent classes name, and what they extend) have no answerer here and are skipped on every turn, listed in one warning by policy. Console aliases and console assignments are checked live. |
| `ai_sdk.oversight.W004` | A lens lists answerers it never asks: a detector before them is sure when it finds nothing. |
| `ai_sdk.oversight.W005` | An alias's answerer cannot run here, such as `TypeSafe` without `TYPESAFE_API_KEY` or `typesafe-sdk`. |
| `ai_sdk.oversight.W006` | A lens reads the question at watch or finish, which only judge the answer: it never runs there. |
| `ai_sdk.oversight.W007` | `BUDGET_PER_DAY` is set, but a judge alias's model has no price in `AI_SDK_MODEL_PRICES`. |
| `ai_sdk.oversight.W008` | An enforced rule reads lenses whose answerers are all optional aliases and none is defined: it never acts. |
| `ai_sdk.oversight.W009` | An agent class names a policy no app registers: its turns follow `FAIL_MODE["gate"]` unless the console defines it. |
| `ai_sdk.oversight.W010` | `AI_SDK_OVERSIGHT` has a key nothing reads (a typo). |
| `ai_sdk.oversight.W011` | A policy in use masks or blocks the answer at finish but streams it live: the client has the text before finish judges it. Use `stream_mode = "hold"` or a client that applies the `replace` notice. |
| `ai_sdk.oversight.W012` | Oversight is set up, `DEFAULT_POLICY` is empty and a runtime base class names no policy: runtime agents run unwatched until the console assigns them one. |
| `ai_sdk.oversight.W013` | An agent with a policy streams chat through a `get_pipeline_adapter()` annotated to return something other than the SDK's `Stream`: the answer is never watched or finished. |
| `ai_sdk.oversight.W014` | `POLICIES` tunes a policy nothing registers: its entries tune nothing. |
| `ai_sdk.oversight.W015` | A `POLICIES` entry cannot apply (an unknown lens or rule, a value out of range): it is ignored. |

`E008`, `E009`, `W001`, `W002` and `W012` are checked only while oversight is enabled; `W001` and `W002` only by `manage.py check --deploy`.
