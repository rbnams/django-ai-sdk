---
title: Oversight
type: docs
weight: 10
---

**Oversight** audits every turn an agent runs (the question, what retrieval found, the tools it called, the answer) on any channel: chat, webhooks, workflow steps and your own code. Each turn lands on an append-only, hash-chained ledger, judged by the questions your policy asks about it.

It is separate from [tracing](/manual/tracing/). Traces are debug data: mutable, deleted with their thread, all-or-nothing about content. The ledger is evidence: chained, kept apart from threads, retained per purpose, readable only through a purpose and only on the record.

## The idea in four words

- **Lenses** ask one typed question about a turn: a yes/no (`Flag`), one of a set of labels (`Choice`), a 0–1 `Score`, which `Entities` occur, or a short `Text`.
- **Answerers** answer lenses. Anything can: a regex, a checksum, Presidio, a Hugging Face classifier, TypeSafe's Jev, any chat model the SDK can build, a registered agent, or a person. A lens lists them cheapest first, and later ones are only asked while earlier ones are unsure.
- **Rules** read the answers and act: `Allow`, `Alert`, `Redact`, `RestrictTools`, `Block`, `Queue` for a person, `Escalate` to a stronger answerer, `Notify` an exporter. Every rule is `off`, `shadow` (records what it *would* have done) or `enforce`.
- **Policies** bundle lenses and rules. The SDK ships packs (`security`, `safety`, `quality`, `classify` and `default`, which is all four), and every app declares its own.

Because the question is fixed and the answerer is not, you can start with patterns, add a model judge later, compare both against human verdicts, and keep one comparable series throughout.

## Turning it on

```python
INSTALLED_APPS = [..., "django_ai_sdk.oversight"]

AI_SDK_OVERSIGHT = {
    "DEFAULT_POLICY": "default",   # every agent without its own
}
```

```bash
pip install "django-ai-sdk[oversight]"    # encryption and signed checkpoints
python manage.py migrate
python manage.py oversight_keys           # prints MASTER_KEY and SIGNING_KEY
```

That is enough: every turn is recorded with system-one answers (patterns, checksums, keywords, word overlap) and every rule in shadow mode. No model is called and nothing is blocked.

## Your own policy

Declare it in any installed app's `oversight.py`:

```python
from django_ai_sdk.oversight import (
    Choice, Flag, Keywords, L, Mode, Patterns, Policy, Queue, Redact, Rule, register,
)

class Topic(Choice):
    name = "topic"
    question = "What is the message about?"
    labels = ["billing", "hr", "other"]
    answerers = [Keywords({"billing": ["invoice", "refund"], "hr": ["salary", "leave"]}), "fast?"]

@register
class Support(Policy):
    name = "support"
    extends = ["security", "quality"]
    lenses = [Topic]
    rules = [
        Rule("hr_goes_to_people", when=L("topic") == "hr", then=Queue("hr")),
        Rule("block_injection", when=L("prompt_injection").is_true, then=Block(), mode=Mode.ENFORCE),
    ]
```

and point agents at it:

```python
class SupportAgent(Agent):
    oversight = "support"     # False opts out even of DEFAULT_POLICY
```

`"fast?"` names an answerer alias *optionally*: it is asked when `AI_SDK_OVERSIGHT["ANSWERERS"]` defines `fast`, and skipped otherwise. That is how the packs work on a deployment with no model and get better on one with.

## Models: system one or any other

```python
AI_SDK_OVERSIGHT = {
    "ANSWERERS": {
        "fast": {"class": "django_ai_sdk.oversight.answerers.LLMJudge", "model": "gpt-4.1-mini"},
        "judge": "fast",
        "strong": {"class": "django_ai_sdk.oversight.answerers.LLMJudge",
                   "llm": "django_ai_sdk.generators.anthropic_chat", "model": "claude-sonnet-4-5"},
        "local": {"class": "django_ai_sdk.oversight.answerers.LLMJudge",
                  "llm": "django_ai_sdk.generators.ollama_chat", "model": "llama3.2"},
        "guard": {"class": "django_ai_sdk.oversight.answerers.HFClassifier",
                  "model": "meta-llama/Llama-Prompt-Guard-2-86M", "positive": ["MALICIOUS"]},
        "moderation": {"class": "django_ai_sdk.oversight.answerers.Moderation"},
        "typesafe": {"class": "django_ai_sdk.oversight.answerers.TypeSafe"},
    },
    "BUDGET_PER_DAY": 5.0,
}
```

A lens says `"judge"`, settings say which model that is, and the console can switch a lens to another alias without a deploy. Judges get the turn fenced as evidence and are told it may try to address them; they answer in the lens's typed schema, structured output where the provider supports it and parsed JSON where it does not. `Vote("fast", "local", min_agree=2)` asks several models at once and records their disagreement. Model answerers are charged to a daily budget; once it is spent they abstain and the cascade keeps what it has.

## When lenses run

| Stage | When | Can still |
| --- | --- | --- |
| `gate` | Before the question is stored or reaches the model | block, redact the question, take tools away |
| `watch` | While the answer streams, and before every tool call | stop the answer, refuse a tool call |
| `review` | When the answer is complete, before it is stored | withhold or redact the answer |
| `reflect` | In the background, on the task worker | queue, notify, redact the stored message after the fact |
| `retro` | `manage.py oversight_replay`, on a golden set or reviewed turns | tell you whether a change kept the verdicts |

Gate and review have deadlines (`GATE_DEADLINE_MS`, `REVIEW_DEADLINE_MS`); an answerer still running is abandoned and, by default, the turn goes on (`FAIL_MODE`). Chat streams live by default: on an enforced redaction the client is sent the replacement. A policy with `stream_mode = "hold"` shows the answer only once review passed.

The client learns of every intervention as a `data-oversight` stream part: `{turn_id, stage, outcome, rules, message?, replace?, input?}`. A refused answer ends with finish reason `content-filter`.

## Reading the ledger

`OversightService` is the seam an HTTP layer sits on, and `django_ai_sdk.contrib.ninja.get_oversight_router()` is a ready-made one. Every read goes through a **purpose**:

- `security`: questions, answers and tool calls, the person as a pseudonym.
- `quality`: everything including sources, the person hidden.
- `audit`: decisions and the chain, verdicts without their reasons, no content at all.

Every read is itself written to the ledger. Revealing who asked needs `REVEAL_ACTOR` and a reason, and is recorded next to the reveal.

## Proving it

Each row is chained: `entry_hash = sha256(seq, kind, ref, payload_hash, prev_hash)`. `manage.py oversight_verify` recomputes every link and every record. `manage.py oversight_seal` writes a checkpoint, an RFC 6962 Merkle root signed as a C2SP signed note, and sends it to every exporter marked `anchor`. Publish it somewhere you do not control, and history before it can no longer be rewritten quietly.

Content lives apart from the chain, encrypted with one key per thread. Forgetting a thread (`OversightService.forget_thread`, `manage.py oversight_prune --thread`) deletes its key: every copy becomes unreadable at once, and the chain still verifies.

See the [Oversight manual](/manual/oversight/) for every class, setting and endpoint.
