---
title: Oversight
type: docs
weight: 10
---

**Oversight** checks and records every turn an agent runs: the question, what retrieval found, the tools it called and the answer. A turn is checked the same way whichever way the agent runs: chat, `Agent.run()`, the HTTP run endpoints, webhooks and workflow steps. Each turn lands on an append-only, hash-chained ledger, with the answers to the questions your policy asks about it and what its rules did.

It is separate from [tracing](/manual/tracing/). Traces are debug data: mutable, deleted with their thread, all-or-nothing about content. The ledger is evidence: chained, kept apart from threads, kept as long as its policy's retention says, and read through a service that records who read what (see [Who reads what](#who-reads-what) for the reads it does not record).

## The idea in four words

A lens asks one question about every turn. A rule decides what to do with the answer. Rules start in report mode.

- **Lenses** ask one typed question about a turn: a yes/no (`Flag`), one of a set of labels (`Choice`), a 0–1 `Score`, which `Entities` occur (with where), or a short `Text`.
- **Answerers** answer lenses: code (a regex, a checksum), a hosted classifier, any chat model the SDK can build, or a person. You list them per lens, quickest first; a later one is asked only while the earlier ones are unsure.
- **Rules** read the answers and act: `Block`, `Mask`, `RestrictTools`, `Queue` for a person, `Notify` an exporter, `Alert`. Every rule is `off`, `report` (recorded as what it would have done, nothing changes) or `enforce`.
- **Policies** bundle lenses and rules. The SDK ships packs (`security`, `safety`, `quality`, and `default`, which is all three), and an app declares its own.

The question a lens asks is fixed; who answers it is not. You can change the answerer of a lens, compare answerers against people's verdicts, and keep one comparable series.

## Turning it on

```bash
pip install "django-ai-sdk[oversight]"    # cryptography: encrypted evidence, signed checkpoints
python manage.py oversight_keys           # prints a MASTER_KEY and a SIGNING_KEY
```

```python
INSTALLED_APPS = [..., "django_ai_sdk.oversight"]

AI_SDK_OVERSIGHT = {
    "DEFAULT_POLICY": "default",   # every agent that names no policy of its own
    "MASTER_KEY": env("OVERSIGHT_MASTER_KEY"),
    "SIGNING_KEY": env("OVERSIGHT_SIGNING_KEY"),
}
```

```bash
python manage.py migrate
```

That is enough to start: every turn is recorded and every pack rule is in report mode, so nothing is blocked. With no answerer aliases defined, only the lenses code answers run: personal data by checksum, secrets by pattern, exfiltration links, and facts about the turn such as failed tools. The `ai_sdk.oversight.W003` check lists every lens of the policies in use that is skipped, in one warning. Define the aliases ([below](#picking-the-answerer)) and the other pack lenses run too. `manage.py check --deploy` also warns when `MASTER_KEY` or `SIGNING_KEY` is missing (`W001`, `W002`).

## Attaching a policy to an agent

A policy is a class. Name it on the agent:

```python
from django_ai_sdk import Agent
from django_ai_sdk.oversight import PII, Entities, L, Mask, Mode, Policy, Queue, Rule, Stage, Subject, pick
from django_ai_sdk.oversight.packs import SecurityPack


class SupportPolicy(Policy):
    name = "support"
    stream_mode = "hold"  # the answer is shown once finish has masked it
    lenses = [
        *pick(SecurityPack.lenses, "prompt_injection"),
        Entities(
            name="pii",
            question="Which kinds of personal data does the answer contain?",
            subject=Subject.OUTPUT,
            stages=(Stage.FINISH,),
            answerers=[PII()],
        ),
    ]
    rules = [
        *pick(SecurityPack.rules, "block_injection", mode=Mode.ENFORCE),
        Rule("mask_pii", when=L("pii").any, then=Mask(), mode=Mode.ENFORCE),
        Rule("iban_to_people", when=L("pii").contains("iban"), then=Queue("privacy"), mode=Mode.REPORT),
    ]


class SupportAgent(Agent):
    oversight = SupportPolicy
```

What `oversight` can be:

| Value | Means |
| --- | --- |
| A `Policy` subclass | That policy. Naming it on an agent class registers it, so the console sees it; a typo fails at import. |
| A policy's name (`"support"`) | A policy registered elsewhere: a pack, or one decorated with `@register` in an app's `oversight.py`. A policy no app registers (and the console does not define) cannot load: its turns follow `FAIL_MODE["gate"]`, refused when closed, unwatched when open (`ai_sdk.oversight.W009`). |
| `None` (the default) | `AI_SDK_OVERSIGHT["DEFAULT_POLICY"]`, or no oversight when that is empty. |
| `False` | No oversight, not even the default policy. |

Anything else (an instance, `True`, `""`) raises `ImproperlyConfigured` when the class is defined.

Use `@register` for a policy no agent class names: one that runtime agents pick by name, or that the console assigns.

To choose per instance, per channel or per user, override `get_oversight()`. It is async, so it may read the database, and returns the same kinds of values. Keep `**kwargs`: more about the turn may be passed later.

```python
from django_ai_sdk.oversight import register


@register  # so the console and the checks see it before a turn uses it
class PublicSupportPolicy(SupportPolicy):
    name = "public-support"


class SupportAgent(Agent):
    oversight = SupportPolicy

    async def get_oversight(self, *, channel, user, **kwargs):
        if channel.startswith("webhook:"):
            return PublicSupportPolicy     # strict where anyone can write
        return self.oversight
```

A hook that raises, like a policy that cannot load, follows `FAIL_MODE["gate"]`: closed refuses the turn, open runs it unwatched and logs it.

Which policy a turn runs under, first match wins:

1. `run(oversight=...)` for that one call (trusted code; see below).
2. The console's assignment for the agent, including "off". An operator's choice wins over what the code says.
3. `get_oversight(channel=..., user=...)`, which returns `oversight` unless you override it.
4. `DEFAULT_POLICY`, when the steps above gave `None`.

A runtime agent (an `AgentSettings` row) has no class of its own: without a console assignment it runs under its base class's `oversight` (`AI_SDK_RUNTIME_AGENT_BASES`), then `DEFAULT_POLICY`. Any user who may create an agent can make one, so set `DEFAULT_POLICY` or give the base class a policy; `ai_sdk.oversight.W012` warns when neither is set.

## Every way an agent runs is a turn

A policy is a promise about an agent, not about one endpoint. If chat checked the question but `POST /agents/{id}/run/` answered the same question unchecked, the policy would hold only for whoever came in through the checked door. So every entry point opens a turn under the agent's policy, and an agent without a policy (and no `DEFAULT_POLICY`) does no oversight work at all.

| Entry point | Channel | When the policy refuses |
| --- | --- | --- |
| Chat, `Agent.as_view()` | `chat` | The policy's message streams as the reply, ending with finish reason `content-filter`. A refused question is not stored. |
| `Agent.run()` | `run`, or what you pass as `channel` | Raises `Blocked`. |
| HTTP run endpoints (`POST /agents/{id}/run/`, `POST /threads/{id}/run/`, Ninja and DRF) | `api` | `{"result": <the policy's message>, "thread_id": ..., "blocked": true}`. Raised elsewhere, `Blocked` is an HTTP 400. |
| Webhooks (`run_inbound`) | `webhook:<integration>` | A refused question gets the policy's message as the reply and is not stored. A withheld answer is replaced by the message, which is posted and stored. |
| Workflow `AgentStep` | `workflow:<workflow>:<step>`; `workflow:adhoc:<step>` for a definition a client submitted | The step fails: `refused by oversight: ...` or `withheld by oversight: ...`. |

### `Agent.run()` and `Blocked`

```python
from django_ai_sdk.errors import Blocked

try:
    answer = await agent.run(messages, user=request.user, channel="api")
except Blocked as refused:
    answer = refused.message        # the policy's text for the person asking
    log.info("stopped at %s, turn %s", refused.stage, refused.turn_id)
```

`Blocked` lives in `django_ai_sdk.errors`, so you can catch it without the oversight app installed. It carries `message` (show it as the reply), `stage` (`"gate"` or `"finish"`) and `turn_id`. A refusal is an exception, not a returned string: with a `response_format` there is no way to return it as the model, and a refusal stored silently as "the answer" would be worse.

Under a policy, `run()`:

1. **Gates the history and the question.** History the server did not store is judged first (see [History](#history)). When a rule masks the question, the last user message in the list you passed is masked in place, so what you store afterwards is the masked copy. The rest of your list is left as it was: history is judged and masked on copies.
2. **Watches the tools** (with `tools=True`): every call is checked before it runs, and every result is masked before the model reads it.
3. **Finishes the answer.** A masked answer comes back masked. A structured result is judged as its `model_dump_json()`, so a field excluded from that (`exclude=True`) is never judged; it is rebuilt from its masked JSON, and one that no longer validates raises `Blocked` rather than pass as the original.

`run()` takes two keyword arguments for this:

- `channel`: what the turn is recorded as on, such as `"api"` or `"webhook:slack"`. `get_oversight()` sees it too.
- `oversight`: left out, the agent's own policy. A policy for this call instead (a class or a name); `None` for `DEFAULT_POLICY`, as `oversight = None` on the class means; or `False` to run unwatched. Use `False` only in code you trust with the agent unwatched, such as the SDK's own title generation. The SDK's HTTP endpoints never pass it.

### Nested runs and subagents

Each agent runs under its own policy. A run that starts inside another turn joins that turn when it resolves the same policy, or none: one request is one record. There is no second gate (the outer gate already judged the question); the nested run's tool calls are checked and recorded in the outer turn; and its output reaches the outer agent as a tool result, masked like any other, and the outer turn's finish judges the answer built from it.

A run under another policy (the nested agent's own, or an explicit `run(oversight=...)`) opens a child turn instead: its own gate, watch, finish and record, linked to the outer turn. Subagents follow the same rule. A delegation to a subagent whose policy is the parent's, or none, joins the parent's turn; one whose policy differs runs as a child turn under it, with the delegated task as its question. When that policy refuses the task or withholds the subagent's answer, the coordinator reads the policy's message as the tool result. A delegation run synchronously (Haystack's `run()` rather than `run_async()`) cannot open a turn, so when it would need one it is refused.

A turn that has closed is never joined: a task started inside it that outlives it opens turns of its own.

### Overriding `run()`

The policy wraps the outermost `run()`. A subclass's own `run()` is wrapped when the class is defined: the turn opens around it (the gate before it starts, the finish on what it returns), and the `super().run()` inside it joins that turn. So an override cannot skip the policy, and one that answers by itself (a command, a canned reply) is judged too. Its arguments are read as `Agent.run` takes them (`messages`, `thread_id`, `user`, `channel`, `oversight`). A workflow step that overrides `AgentStep.run()` and calls the agent itself passes `channel=self.channel(ctx)`, so the turn is recorded as the step's.

Two doors are yours to keep closed:

- `stream_response()` called from your own view opens no turn: only `as_view()` does.
- A `get_pipeline_adapter()` that returns an adapter other than the SDK's `Stream` gets the gate only: its answer is never watched or finished. `ai_sdk.oversight.W013` warns when its return annotation says so.

A run without `tools=True` records no token usage, so its turn has no cost.

## Your own policy

Declare lenses and rules on a `Policy`, starting from packs if you like:

```python
from django_ai_sdk.oversight import Block, Choice, L, Mode, Policy, Queue, Rule, Stage


class Topic(Choice):
    name = "topic"
    question = "What is the message about?"
    labels = ["billing", "hr", "other"]
    criteria = {"billing": "Invoices, refunds, payments.", "hr": "Salary, leave, contracts."}
    # The question (the default subject), in the background: routing needs no hurry.
    stages = (Stage.REFLECT,)
    answerers = ["classifier?", "judge?"]


class SupportPolicy(Policy):
    name = "support"
    extends = ["security", "quality"]
    lenses = [Topic]
    rules = [
        Rule("hr_goes_to_people", when=L("topic") == "hr", then=Queue("hr")),
        Rule("block_injection", when=L("prompt_injection").is_true, then=Block(), mode=Mode.ENFORCE),
    ]
```

A lens or rule declared on the policy replaces an inherited one of the same name, so the `block_injection` above replaces the security pack's. A lens's `subject` (what it reads) defaults to the question and its `stages` to finish; gate and reflect read the question, watch and finish only the answer (`ai_sdk.oversight.W006` names a lens that would never run).

### Start small

`extends` brings in every lens and rule of a pack. To take only what you need, use `pick`:

```python
from django_ai_sdk.oversight import Mode, Policy, pick
from django_ai_sdk.oversight.packs import QualityPack, SecurityPack


class SupportPolicy(Policy):
    name = "support"
    lenses = [
        *pick(SecurityPack.lenses, "prompt_injection", "pii_input", "secrets_input", "pii_output"),
        *pick(QualityPack.lenses, "groundedness"),
        Topic,
    ]
    rules = [
        *pick(SecurityPack.rules, "block_injection", "mask_pii_in_answer"),
        *pick(SecurityPack.rules, "mask_pii_in_question", "mask_secrets_in_question", mode=Mode.ENFORCE),
    ]
```

A picked rule is a copy, so `mode` changes it only here. A rule may only read lenses its policy has (`ai_sdk.oversight.E003`). To change a pack lens, subclass it and list the subclass:

```python
from django_ai_sdk.oversight.packs import security

class PromptInjection(security.PromptInjection):
    answerers = ["classifier?", "judge?"]
```

## Picking the answerer

Pick the answerer per lens:

- **Code** (a local detector) when you need exact spans: personal data by checksum and secrets by pattern give the characters to mask. A failed tool or an exfiltration link is a fact about the turn, so code answers it too. Code sends nothing anywhere.
- **A classifier** for most yes/no, label and score lenses.
- **A model judge** for written text, or as a second opinion where the classifier is unsure (`["classifier?", "judge?"]`).

The packs and the console's examples name answerers by alias, and a deployment maps each alias it has to an answerer:

| Alias | Answers |
| --- | --- |
| `classifier` | Yes/no, label and score lenses. |
| `judge` | A chat model: text lenses, and a second opinion. |
| `strong` | A stronger model, for long reasoning. |
| `human` | Always defined: the lens goes to a review queue. |

```python
AI_SDK_OVERSIGHT = {
    "ANSWERERS": {
        # Any model your OpenAI-compatible endpoint (OPENAI_API_URL) serves.
        "classifier": {"class": "LLMJudge", "model": "your-small-model"},
        "judge": {"class": "LLMJudge", "model": "your-small-model"},
        "strong": {"class": "LLMJudge", "model": "your-large-model"},
    },
    "BUDGET_PER_DAY": 5.0,
    "GATE_DEADLINE_MS": 3000,      # a model judge at the gate needs more than the default 1000
}
# The budget counts only what it can price (ai_sdk.oversight.W007 names the rest).
AI_SDK_MODEL_PRICES = {
    "your-small-model": {"input": 0.4, "output": 1.6},
    "your-large-model": {"input": 2.0, "output": 8.0},
}
```

A gate lens still running at `GATE_DEADLINE_MS` fails, and under the default open fail mode its enforced rule then does not act: with a model judge on `prompt_injection`, raise the deadline or set the gate's `FAIL_MODE` to `closed`.

A `"class"` without a dot is one of the SDK's answerers; a dotted path names your own. An alias may also name another alias (`"fast": "judge"`). `LLMJudge` takes `llm=` (an SDK generator factory or its dotted path) to use another provider, such as `"django_ai_sdk.generators.anthropic_chat"` or `"django_ai_sdk.generators.ollama_chat"` for a model on your own hardware.

`"classifier?"` names an alias *optionally*: it is asked when `ANSWERERS` defines `classifier` and skipped otherwise. So the packs work with whatever you define: a lens whose answerers are all undefined optional aliases is skipped on every turn, `ai_sdk.oversight.W003` lists every such lens of the policies in use in one warning, and `ai_sdk.oversight.W008` names enforced rules that therefore never act. A lens that names an alias without `?` that nothing defines is an error (`ai_sdk.oversight.E004`).

### TypeSafe

`TypeSafe` is an answerer for TypeSafe, a hosted, commercial classification service. It is optional: it needs the `typesafe` extra (`pip install "django-ai-sdk[typesafe]"`) and the `TYPESAFE_API_KEY` Django setting, and without either it abstains (`ai_sdk.oversight.W005`). It is exported from `django_ai_sdk.oversight.answerers`, not from `django_ai_sdk.oversight`. Nothing in the SDK uses it unless a deployment maps an alias to it:

```python
"ANSWERERS": {"classifier": {"class": "TypeSafe"}}
```

It answers `Flag`, `Choice` and `Score` lenses, and judges an `Entities` lens with `per_source = True` (the quality pack's `taken_from`: which sources the answer was taken from) source by source. One stage's questions about the same turn go out as one request. `TYPESAFE_MODEL` pins a model.

### How a lens decides

A lens asks its answerers in order. The first answer whose confidence reaches the lens's `min_confidence` (0.75 by default) wins. When none is sure enough, the last one that answered decides: each answerer's confidence is on a scale of its own, so they are never compared, and a lens lists its answerers from quick to thorough. Every answer given on the way is kept in the result's `trail`.

Metered answerers (the model judges) are charged to a daily budget (`BUDGET_PER_DAY`, priced by `AI_SDK_MODEL_PRICES`). Once it is spent they answer with the error `budget`, and the lens keeps the answers it has. Code and `TypeSafe` are not counted against the budget.

## Personal data and secrets

`PII()` finds personal data by kind, with checksums where a kind has one: a number shaped like an IBAN that fails mod-97 is not reported.

- **Default kinds**, the same in every country: `email`, `iban` (mod-97), `credit_card` (Luhn), `phone` (international format, `+31 6 ...` or `0031 ...`), `ip_address` (IPv4).
- **Country kinds**, only when you name them: `bsn` (Dutch citizen service number, 11-check), `phone_nl` (`06-12345678`), `postcode_nl` (`1234 AB`).

```python
from django_ai_sdk.oversight import PII
from django_ai_sdk.oversight.answerers.pii import DEFAULT_KINDS

PII(kinds=[*DEFAULT_KINDS, "bsn", "postcode_nl"], extra={"customer_id": r"\bCUST-\d{6}\b"})
```

`Secrets()` finds `api_key`, `stripe_key`, `google_api_key`, `github_token`, `aws_key`, `slack_token`, `jwt`, `private_key` and `password_assignment`. The security pack's lenses use `PII()` with the default kinds; give your policy its own lens to add country kinds.

A `Mask` replaces each span with its label in capitals: `Mail me at [EMAIL]`.

## When lenses run

| Stage | Label | Runs | A rule can |
| --- | --- | --- | --- |
| `gate` | Before the model | On the question, before the model sees it | Block it, mask parts of it or take tools away |
| `watch` | While it works | On each tool call before it runs, and on the answer as it streams | Refuse the tool call or stop the answer |
| `finish` | Before the user sees it | On the finished answer; the stream is fixed when it ends | Mask or replace the answer, or block it |
| `reflect` | Afterwards | In the background, seconds after the reply | Send the turn to review, alert, or mask the stored copy |

Gate, watch and finish have deadlines (`GATE_DEADLINE_MS`, `WATCH_DEADLINE_MS`, `FINISH_DEADLINE_MS`). An answerer still running at the deadline is abandoned, the lens keeps the answer it has, and reflect asks again later. What a failed or late lens means is the stage's `FAIL_MODE`: `open` (the default) lets the turn through; `closed` treats an enforced rule that blocks, masks or takes tools away as matched when a lens it reads failed. Rules that only alert, queue or notify, and rules in report mode, never fire on a failure.

`FAIL_MODE` is open by default so an answerer outage or a slow judge never takes chat down. The price is the deadline: a lens that misses it (and a long question can make a model judge slower) lets the turn through. Where a rule must hold whatever happens, set that stage to `closed`.

Reflect and notifications run as `django-tasks` tasks on the `default` queue: configure a worker backend so they run off the request.

## Masking

- **The question** is masked at the gate, in place, before the model or the thread sees it. The client is told what the masked question reads (`input` in the `data-oversight` part), so it can replace its copy. Under `full` capture the ledger also keeps what was said.
- **The history**: see [History](#history).
- **Tool results** are masked once, when they arrive, before the model, the screen and the store see them, all three the same masked text. They are masked by the gate's `Entities` lenses that an enforced `Mask` rule reads. A result that cannot be checked (a masking lens failed or ran out of time) follows the watch fail mode: `closed` replaces the whole result with `[withheld: could not be checked]`; `open` passes it as it is and records the failure.

Masking history or a tool result asks those lenses' answerers about that text as if it were the question. A remote answerer on such a lens (a judge, `TypeSafe`) is sent each earlier message and each tool result it masks as the question, with the rest of what an `input` lens sends ([the tables below](#where-turn-content-goes)). Keep masking lenses on code answerers (`PII()`, `Secrets()`) where that must not happen.
- **Sources** that retrieval found are masked like tool results, before the citations card shows them, the stored citations keep them, or `/threads/{id}/sources/` serves them.
- **Attachments**: the gate judges what the model reads, the message and each attachment's context (its file name, a document's summary, an image's caption); attachments are resolved before the turn opens. An image's pixels are judged only through its caption.
- **Tool results that are not text** (images, files) pass as they are: no lens reads them.
- **The answer** is masked at finish, or afterwards by reflect (the stored copy only).

A mask that cannot point at what it should mask blocks instead: when an enforced `Mask` rule matches but a lens it reads failed under a `closed` fail mode, or found something without saying where, the stage blocks with the rule's message. A turn is never recorded as masked when nothing was. Only an `Entities` lens says where; a `Mask` over a `Flag`, `Choice`, `Score` or `Text` lens is refused when the policy is built.

## History

History the server did not store itself is judged at the gate on every turn: every earlier user and assistant message is checked by the gate's `input` lenses (`TurnSession.check_history`), masked message by message, and the turn is refused if any is refused. That is the history a stateless run endpoint (`POST /agents/{id}/run/`), a chat without a thread, a `run()` from your code or a workflow's `history` inputs bring: a client can put anything in it. So a stateless caller pays for judging its history on every request.

A thread has its history judged once, each message on its own turn:

- Under a policy whose gate masks the question, the model gets the thread's own stored history (already masked) and only the new question from the request, in chat (`as_view`) and on `POST /threads/{id}/run/` alike.
- Under any other policy, a client copy that is the thread as stored stands as it is; a copy that differs (edited, made up) is judged like a stateless one. A policy in report mode never replaces the client's copy.

Webhooks send the thread's stored messages back, so their history is not judged again.

## What the person sees: live and hold

- **Live** (the default). The answer reaches the client as the model writes it. Watch checks it every `WATCH_EVERY_CHARS` characters and can only stop it; it never rewrites text already shown. Finish fixes it when the stream ends: when finish blocks or masks, the last part tells the client to replace the answer on screen, and the stored message is already the fixed one.
- **Hold** (`stream_mode = "hold"` on the policy, or `STREAM_MODE`). The answer, and what its tools were given and found, is shown only once finish passed it; meanwhile the chat shows which tools run. The cost is the live stream. Reasoning is neither shown nor stored: no lens judges it. Tool cards show what the model saw, its tool results masked at watch: to keep something out of the tool cards, mask it at watch. Rows an artifact tool wrote with an answer that is withheld are deleted with it.
- **Reflect** fixes only the stored copy. It runs after the stream has closed and pushes nothing to the client. The stored message is masked and carries a notice, so a reloaded thread shows it as adjusted.

{{< callout type="warning" >}}
In live mode the client has the text before finish runs, and between watch checks: an enforced finish `Mask` or `Block` then only replaces what the client already showed, and only in a client that applies the `replace` notice (`ai_sdk.oversight.W011` warns). Stock clients that ignore `data-*` parts, and the OpenAI protocol, never do. A Markdown image in the answer, rendered by the client, makes it fetch the image's URL, with whatever data the URL carries, before the exfiltration lens can stop it. For an agent that reads sensitive data, use `stream_mode = "hold"`, or a client that does not load remote images.
{{< /callout >}}

## What the client is told

What the gate, watch and finish did reaches the chat client as a `data-oversight` stream part:

```json
{"turn_id": "...", "stage": "gate", "outcome": "masked", "input": "Mail me at [EMAIL]"}
```

| Key | When |
| --- | --- |
| `turn_id`, `stage`, `outcome` | Always. `outcome` is `blocked`, `restricted`, `masked`, `flagged` or `allowed`. |
| `message` | The turn was blocked: the policy's text. |
| `replace` | Finish replaced the answer: the text to show instead. |
| `input` | The gate masked the question: the masked question. |
| `rules`, `detail` | Only with `CLIENT_DETAIL` `"full"`, or `"staff"` for users who may view oversight: rule names and every stage's answers say what the policy tests. |

With the default `CLIENT_DETAIL = "actions"`, a part is sent only when an enforced rule acted, and it names no rules. A refused tool call reaches the client as that tool's result, `{"refused": "<the rule's message>"}`. A refused question or a withheld answer ends with finish reason `content-filter` (`content_filter` for `OpenAIProtocolHandler`, which does not send `data-*` parts). The notices are stored on the message, so a reloaded thread shows them again.

## Where turn content goes

Code answerers never send content anywhere. Remote answerers send the parts of the turn their lens's subject covers and nothing else: no user id, thread id, sources' URLs or metadata. Every one masks secrets (the `Secrets` patterns) before anything is cut or sent. Personal data is not masked on the way out, except where a gate rule already masked the question; lenses at the gate read the question as asked. A longer text keeps its start and its end. A masking lens with a remote answerer also receives the history and tool results it masks, each as the question (see [Masking](#masking)).

What `LLMJudge` sends to your model endpoint, in characters:

| Lens subject | History | Question | Sources | Answer | Tool call |
| --- | --- | --- | --- | --- | --- |
| `input` | last 6, 500 each | 6,000 | | | |
| `exchange` | last 6, 500 each | 6,000 | 8 sources, 1,500 each (1) | 8,000 | |
| `output` | | 6,000 | 8 sources, 1,500 each (1) | 8,000 | |
| `sources` | | 6,000 | 8 sources, 1,500 each (2) | | |
| `tool_call` | | 6,000 | | | name (200), arguments as JSON (4,000) |

(1) This turn's sources and tool results, then the sources earlier turns found. (2) This turn's sources and tool results only. A source is its title (200) and its text; a tool result counts as a source titled "<tool> result". With them go the judge's instructions, the lens's question and guidance and, for a model without structured output, the verdict's JSON schema.

What `TypeSafe` sends to TypeSafe, in characters:

| Lens subject | History | Question | Answer | Sources (3) | Tool call |
| --- | --- | --- | --- | --- | --- |
| `input` | last 6, 500 each | 30,000 | | | |
| `exchange` | last 6, 500 each | 30,000 | 30,000 | | |
| `output` | | 30,000 | 30,000 | yes, earlier turns' too | |
| `sources` | | 30,000 | | yes, this turn's only | |
| `tool_call` | | 30,000 | | | name (200), arguments (4,000 of JSON) |
| `per_source` | | 30,000 | 30,000 | one per question | |

(3) Each source and tool result: its title (200) and text, up to 8,000 together; all of them up to 32,000, past which the rest are left out, earlier turns' sources first. With them go each lens's question (its question and guidance, and what yes, no, each label or each score level means) and the model id. One request carries at most 100,000 characters of the turn.

The console shows each answerer's recipient, so an operator sees where a lens sends turns before switching it on.

## What is recorded

A policy's `capture` says how much of a turn's content the ledger keeps; the deployment's `CAPTURE` is a ceiling a policy may stay under, never exceed.

Everything that can quote a turn lives in its **evidence**: the question, the answer, the history, the sources and tool calls, and beside them what was said about the turn in words: judges' reasons, `Text` answers, the names an `Entities` answerer gave in its own words, per-item scores keyed by source, and reviewers' notes. Capture decides what of that is kept, `MASTER_KEY` seals it per thread, and forgetting the thread or expiring the content destroys all of it at once.

The **rows** the chain hashes (the turn, its observations and decisions) and the review items keep, at every level, only what cannot quote the turn: the pseudonym, outcome, labels and severity; each answer's value, confidence, answerer and error, and where in the text it pointed (offsets and labels); the decisions; who closed a review and whether they agreed. An answer's value there is the record's: a flag, a label, a score, and for an `Entities` lens the labels it declares or the kinds its detector reports (`["email", "iban"]`). A name it gave in its own words (a judge's, `TypeSafe`'s source titles, a person's) is `"[found]"`, one per name, so `L(...).any` and counts still hold. A `Text` answer's value is `null`.

| Capture | The evidence keeps |
| --- | --- |
| `none` | Nothing: there is no evidence, and no reasons, `Text` answers, names, scores or notes anywhere. |
| `metadata` | The lengths of the question and answer, the sources' kinds and the tool names. No text at all: no title, URL, source key, reason, `Text` answer, name, score or note. |
| `masked` | Text, every string of it masked: the question, the answer, the history, every field of the sources (content, title, URL, key) and of the tool calls (arguments and results), reasons, `Text` answers, names, per-item scores (their keys masked) and notes. Masked by the built-in detectors (every `PII()` kind, the country kinds too, and `Secrets()`); then every stretch of the question, the answer or the sources that any lens of the turn pointed at (three characters or more) is replaced, wherever it occurs, by the label the lens gave it: `[PERSON]`. |
| `full` | All of that, unmasked, and what was shown where it differs from what was said. |

A stretch a lens found in a tool call is masked only where a detector finds it too: the call's text is not kept to point into. Reflect and reviewers write later, into the same evidence: under `masked` a reviewer's words are masked by the detectors, and reflect's also by what its lenses pointed at.

Under `masked`, reflect does not judge the masked evidence where a chat thread stores the turn: it reads the question and answer as the thread stores them, so a mask it asks for reaches the stored answer. A turn with no stored messages (a webhook, a run endpoint) is judged as the evidence keeps it, masked.

Retention is per policy: `{"content": 90, "metadata": 730}` days by default. `oversight_prune` applies it (schedule it).

## Who reads what

`OversightService` is the seam an HTTP layer sits on, and `django_ai_sdk.contrib.ninja.get_oversight_router()` is a ready-made one. The router is Django Ninja's; there is no DRF viewset yet, so a DRF project calls `OversightService` from its own views. It brings no auth of its own: it runs under your API's auth, and every service call checks the `oversight` permission domain (by default staff view, review and audit, so any staff user may read every user's turns; superusers reveal who asked and change policy; override the domain in `AI_SDK_PERMISSIONS` to narrow it). Turns are read through one of two purposes:

- `review`: the question, the answer, the sources, tool calls, history, and every verdict with what the evidence keeps of it (its reason, a `Text` answer, the names, per-item scores, reviewers' notes); the person as a pseudonym.
- `audit`: decisions and verdicts as the rows keep them (value, confidence, answerer) and the chain. No content, no reasons, no `Text` answers, no names given in an answerer's words, no scores per source, no reviewers' notes, no thread or message ids, and the person only as a pseudonym, even to a reader who revealed who it is. It needs `AUDIT_OVERSIGHT`.

A verdict's value is metadata: it is hashed into the chain and shown in audit and, with `CLIENT_DETAIL` `full` or `staff`, to the chat client. So it never holds a name: see [What is recorded](#what-is-recorded).

Recorded on the ledger, as an access entry with the reader and purpose: listing and opening turns, listing review items, resolving a review, revealing who asked (needs `REVEAL_OVERSIGHT_ACTOR` and a reason), verifying the chain, exporting, and forgetting a thread. The same reader repeating the same list or view within a minute (a page polling) is recorded once. A person's verdict is chained as an observation.

Not recorded: the overview, the queue counts, the checkpoint list and the access log itself; policy, agent, catalog and change listings; the Django admin (read-only, metadata only: no evidence, reasons or notes); management commands; and anything that reads the database directly.

## Encryption and forgetting

Evidence, with the reasons, answers, names, scores and notes kept in it, is encrypted only when `MASTER_KEY` is set (with the `oversight` extra installed): one key per thread, or per turn for a turn without a thread, wrapped by the master key. Who asked is sealed under a key of its own per pseudonym. Without `MASTER_KEY` all of it is stored in the clear, and `ai_sdk.oversight.W001` says so on every `check --deploy`. A `MASTER_KEY` that is set but unusable fails `ai_sdk.oversight.E008`: turns are then recorded without their content and without who asked rather than in the clear, and evidence sealed earlier reads as `unreadable`, as it does under another key.

Forgetting a thread (`OversightService.forget_thread`, `POST /oversight/threads/{id}/forget/`, or `manage.py oversight_prune --thread ID`) destroys its key, clears any content stored in the clear, and with them every reason, `Text` answer, name, score and note kept in its evidence. It clears who asked from the thread's turns, which keep only the pseudonym (a reveal says "forgotten"), and chains a `shred` entry. The chain still verifies: none of that is hashed. A new turn in a forgotten thread stores no content and not who asked, whatever its capture. Deleting a thread (`ThreadService.delete_thread`, the admin, a cascade) forgets it the same way.

What forgetting does not reach:

- **Backups.** A backup taken before still holds the wrapped key, readable with `MASTER_KEY` until the backup is gone. Without `MASTER_KEY` it holds the clear text.
- **The thread itself.** Its chat messages and traces are the conversation's own storage; delete the thread for those.
- **Copies already made**: exported files, and what was sent to remote answerers.
- **The turn's metadata**: its pseudonym, labels, verdict values and decisions, which hold no text of the turn (see [What is recorded](#what-is-recorded)), kept until the policy's metadata retention runs out.
- **Turns without a thread** (a stateless run endpoint, a workflow step): each has its own key, and forgetting works by thread, so only retention removes them.

## Proving it

Each ledger entry is chained: `entry_hash = sha256(seq, kind, ref, payload_hash, prev_hash)`. Edit a recorded row and its payload no longer matches; edit or drop an entry and every later link breaks. A turn commits to its evidence by a salted hash of the content (not of the reasons and notes beside it, which are added later). `manage.py oversight_verify` recomputes every link and every record, recomputes the evidence hash of every turn whose evidence it can open, checks that the chain still reaches its head, and checks the newest checkpoint. With `--since-checkpoint` it starts from the newest checkpoint your key signed and checks the entries before it against that checkpoint's root instead of reading them again.

Retention deletes old turns and leaves a `prune` entry on the chain naming the run of entries whose rows went; verify accepts a missing row only under one, and reports how many it accepted (`pruned`). A row deleted any other way is reported as missing. A prune entry is a claim, though: whoever can write the ledger tables can append one over rows they deleted. Treat prune entries as evidence of a deletion, and check the pruned count against your retention.

Deleting the newest entries is caught too (`chain ends early`), unless the head row is rolled back with them. Only a published checkpoint protects the end of the chain against whoever controls the database.

`manage.py oversight_seal` writes a checkpoint: an RFC 6962 Merkle root, signed as a C2SP signed note when `SIGNING_KEY` is set, and sent to every exporter marked `anchor`. A checkpoint stored next to the ledger proves little against whoever controls the database. Publishing checkpoints somewhere you do not control is your job: then history before one cannot be rewritten without disagreeing with the published root, for anyone who compares it. The SDK does not fetch or compare published checkpoints.

Schedule `oversight_prune` and `oversight_seal`: nothing runs them for you.

See the [Oversight manual](/manual/oversight/) for every class, setting, check and endpoint.
