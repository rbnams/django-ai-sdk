---
title: Oversight
type: docs
weight: 10
---

**Oversight** audits every turn an agent runs (the question, what retrieval found, the tools it called, the answer) on any channel: chat, webhooks, workflow steps and your own code. Each turn lands on an append-only, hash-chained ledger, judged by the questions your policy asks about it.

It is separate from [tracing](/manual/tracing/). Traces are debug data: mutable, deleted with their thread, all-or-nothing about content. The ledger is evidence: chained, kept apart from threads, kept as long as its policy's retention says, readable only through a view and only on the record.

## The idea in four words

A lens asks one question about every turn. A rule decides what to do with the answer. Rules start in test mode.

- **Lenses** ask one typed question about a turn: a yes/no (`Flag`), one of a set of labels (`Choice`), a 0–1 `Score`, which `Entities` occur, or a short `Text`.
- **Answerers** answer lenses. Anything can: code (a regex, a checksum, Presidio, a Hugging Face classifier), TypeSafe's Jev, any chat model the SDK can build, a registered agent, or a person. You pick them per lens, cheapest first; later ones are only asked while earlier ones are unsure.
- **Rules** read the answers and act: `Allow`, `Alert`, `Redact`, `RestrictTools`, `Block`, `Queue` for a person, `Escalate` to a stronger answerer, `Notify` an exporter. Every rule is `off`, `shadow` (test mode: records what it *would* have done) or `enforce`.
- **Policies** bundle lenses and rules. The SDK ships packs (`security`, `safety`, `quality`, `classify` and `default`, which is all four), and every app declares its own.

Because the question is fixed and the answerer is not, you can change who answers a lens, compare answerers against human verdicts, and keep one comparable series throughout.

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

That is enough: every turn is recorded and every rule is in test mode, so nothing is blocked. With no answerer aliases defined, only the lenses answered by code run: personal data by checksum, secrets by pattern, and facts about the turn such as failed tools. Define the aliases (below) and the other pack lenses are answered too.

## Your own policy

Declare it in any installed app's `oversight.py`:

```python
from django_ai_sdk.oversight import Block, Choice, L, Mode, Policy, Queue, Rule, register

class Topic(Choice):
    name = "topic"
    question = "What is the message about?"
    labels = ["billing", "hr", "other"]
    criteria = {"billing": "Invoices, refunds, payments.", "hr": "Salary, leave, contracts."}
    answerers = ["jev?", "judge?"]

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

### Start small

`default` and `extends` bring in every lens of a pack. To start small, list only the lenses and rules you need with `pick`:

```python
from django_ai_sdk.oversight import Mode, Policy, pick, register
from django_ai_sdk.oversight.packs.quality import QualityPack
from django_ai_sdk.oversight.packs.security import SecurityPack

@register
class Support(Policy):
    name = "support"
    lenses = [
        *pick(SecurityPack.lenses, "prompt_injection", "pii_input", "pii_output"),
        *pick(QualityPack.lenses, "groundedness"),
        Topic,
    ]
    rules = [
        *pick(SecurityPack.rules, "block_injection", "redact_pii_in_question", "redact_pii_in_answer"),
        *pick(SecurityPack.rules, "redact_secrets_in_question", mode=Mode.ENFORCE),
    ]
```

A picked rule is a copy, so `mode` changes it only here. A rule must only read lenses the policy has (`oversight.E003`). To change a pack lens, subclass it and list the subclass: `class PromptInjection(security.PromptInjection): answerers = ["jev?", "judge?"]`, with `from django_ai_sdk.oversight.packs import security`.

Then point agents at the policy:

```python
class SupportAgent(Agent):
    oversight = "support"     # False opts out even of DEFAULT_POLICY
```

`"jev?"` names an answerer alias *optionally*: it is asked when `AI_SDK_OVERSIGHT["ANSWERERS"]` defines `jev`, and skipped otherwise. The `oversight.W006` check names every lens that is left with no answerer at all.

## Picking the answerer

Pick the answerer per lens:

- **Jev** answers most yes/no, label and score lenses, in about 0.3 s, so it fits before the model.
- **Code** (a local detector) when you need exact spans: personal data by checksum and secrets by pattern give the characters to mask. A failed tool or an exfiltration link is a fact about the turn, so code answers it too.
- **A model judge** when the answer is written text, or as a second opinion where Jev is unsure (`["jev?", "judge?"]`).

The packs and the console's examples name these aliases, and a deployment defines the ones it has:

| Alias | Answers |
| --- | --- |
| `jev` | TypeSafe's Jev: yes/no, label and score lenses. |
| `judge` | A cheap chat model: text lenses, and a second opinion where there is time. |
| `strong` | A stronger model for escalations and long reasoning. |

`moderation` (OpenAI's endpoint, only on OpenAI itself) is an optional extra for the safety pack, and `human` always exists.

```python
AI_SDK_OVERSIGHT = {
    "ANSWERERS": {
        "jev": {"class": "django_ai_sdk.oversight.answerers.TypeSafe"},
        # Any model your OpenAI-compatible endpoint (OPENAI_API_URL) serves.
        "judge": {"class": "django_ai_sdk.oversight.answerers.LLMJudge",
                  "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"},
        "strong": {"class": "django_ai_sdk.oversight.answerers.LLMJudge",
                   "model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"},
    },
    "BUDGET_PER_DAY": 5.0,
}
```

An alias may also name another alias (`"fast": "judge"`), so an old name keeps resolving after a rename.

A lens says `"judge"`, settings say which model that is, and the console can switch a lens to another alias without a deploy. To ask more of a pack lens, give your own policy a lens of the same name with, say, `answerers = ["jev?", "strong?"]`. Judges get the turn fenced as evidence and are told it may try to address them; they answer in the lens's typed schema, structured output where the provider supports it and parsed JSON where it does not. Metered answerers (the model judges) are charged to a daily budget; once it is spent they abstain and the lens keeps the answers it has. Jev and code are never metered.

## When lenses run

| Stage | Label | Runs | A rule can |
| --- | --- | --- | --- |
| `gate` | Before the model | On the question, before the model sees it | Block it, mask parts of it or take tools away |
| `watch` | While it works | On each tool call before it runs, and on the answer as it streams | Refuse the tool call or stop the answer |
| `review` | Before the user sees it | On the finished answer; the stream is fixed when it ends | Mask or replace the answer, or block it |
| `reflect` | Afterwards | In the background, seconds after the reply | Send the turn to review, alert, or mask the stored copy |
| `retro` | Later, on demand | `manage.py oversight_replay`, on a golden set or reviewed turns | Tell you whether a change kept the verdicts |

The console reads the first four lines from `catalog.STAGE_INFO`.

Gate, watch and review have deadlines (`GATE_DEADLINE_MS`, `WATCH_DEADLINE_MS`, `REVIEW_DEADLINE_MS`); an answerer still running is abandoned, the lens keeps the best answer it already had, reflect asks again later, and by default the turn goes on (`FAIL_MODE`).

What the reader sees follows from when each stage runs:

- **The stream stays live.** The answer reaches the client as the model writes it. Watch can only stop it; it never rewrites text already shown.
- **Review fixes it at stream end.** When review blocks or redacts, the last stream part tells the client to replace the answer on screen, and the stored message is already the fixed one.
- **Reflect fixes only the stored copy.** It runs after the stream has closed, and nothing is pushed to the client then. The stored message is masked and carries a notice, so a reloaded thread shows it as adjusted.
- **Hold is opt-in.** A policy with `stream_mode = "hold"` shows the answer only once review passed, at the cost of a live stream.

The client learns of every intervention as a `data-oversight` stream part: `{turn_id, stage, outcome, rules, message?, replace?, input?}`. A refused answer ends with finish reason `content-filter`.

## Where turn content goes

Code answerers never send content anywhere. Remote answerers send only the parts of the turn their lens's subject covers, and every one masks secrets (keys, tokens, private keys, password assignments) first. Personal data is masked only where a gate rule redacts it.

| Answerer | Sends | To |
| --- | --- | --- |
| `TypeSafe` (Jev) | The question, the answer, up to 8 sources (1,500 characters each), the tool call's name and arguments, and the last 6 messages (500 characters each), as the subject needs | TypeSafe |
| `LLMJudge` | The same parts (the question up to 6,000 characters, the answer up to 8,000), fenced as evidence, with the lens's question and guidance | Your model endpoint (the gateway, OpenAI, or the `llm` you configure) |
| `AgentJudge` | The same, with the agent's own instructions | The agent's model endpoint |
| `Moderation` | The subject's text, up to 20,000 characters | OpenAI |
| `HFClassifier(endpoint=...)` | The subject's text | Hugging Face (local without `endpoint`) |

The console shows each answerer's recipient, so an operator sees where a lens sends turns before switching it on.

## Reading the ledger

`OversightService` is the seam an HTTP layer sits on, and `django_ai_sdk.contrib.ninja.get_oversight_router()` is a ready-made one. Every read goes through one of two views (the `purpose`):

- `review`: the question, the answer, the sources it drew on, tool calls, history, and every verdict with its reason; the person as a pseudonym.
- `audit`: decisions and the chain, verdicts without their reasons, no content at all. It needs `AUDIT_OVERSIGHT`.

Every read is itself written to the ledger. Revealing who asked needs `REVEAL_ACTOR` and a reason, and is recorded next to the reveal.

## Proving it

Each row is chained: `entry_hash = sha256(seq, kind, ref, payload_hash, prev_hash)`. `manage.py oversight_verify` recomputes every link and every record. `manage.py oversight_seal` writes a checkpoint, an RFC 6962 Merkle root signed as a C2SP signed note, and sends it to every exporter marked `anchor`. Publish it somewhere you do not control, and history before it can no longer be rewritten quietly.

Content lives apart from the chain, encrypted with one key per thread. Forgetting a thread (`OversightService.forget_thread`, `manage.py oversight_prune --thread`) deletes its key: every copy becomes unreadable at once, and the chain still verifies.

See the [Oversight manual](/manual/oversight/) for every class, setting and endpoint.
