---
title: TypeSafe
type: docs
weight: 9
---

**TypeSafe** (Jev) answers typed questions — Choice, Noul, Score — about a state. It is not a chat model. Code owns the workflow; Jev supplies calibrated judgments you can threshold, gate tools with, or show as score cards.

Install the optional extra:

```bash
pip install "django-ai-sdk[typesafe]"
```

Set `TYPESAFE_API_KEY` (and optionally `TYPESAFE_MODEL`, default `jev-latest`). When the key or package is missing, screening fails open and the turn continues.

## Screen a chat turn

Implement `TypeSafeScreener` (`screen_input` / `screen_output`) and attach it to an agent. Questions stay in your app — the SDK only defines the protocol and the Stream/webhook hooks.

```python
from django_ai_sdk import Agent
from django_ai_sdk.typesafe import evaluate

class MyScreener:
    async def screen_input(self, text: str) -> dict | None:
        summary = await evaluate(
            state={"message": text},
            questions={
                "should_use_tool": {
                    "type": "noul",
                    "instructions": (
                        "Would a capable assistant need an external tool "
                        "for this message?"
                    ),
                    "criteria": {
                        "true": "Needs live data, tools, or files.",
                        "false": "Greeting or answerable from the message alone.",
                    },
                },
            },
        )
        if not summary:
            return None
        payload = {"side": "input", **summary}
        for j in summary.get("judgments") or []:
            if j.get("id") == "should_use_tool" and float(j.get("value") or 0) < 0.35:
                payload["control"] = {"strip_tools": True}
                payload["acted"] = ["stripped_tools"]
                break
        return payload

    async def screen_output(self, user_text: str, reply: str) -> dict | None:
        summary = await evaluate(
            state={"user_message": user_text, "assistant_reply": reply},
            questions={
                "quality": {
                    "type": "score",
                    "instructions": "How helpful is the assistant_reply?",
                    "criteria": [
                        "Unhelpful or off-topic.",
                        "Partially useful.",
                        "Clear and on-topic.",
                    ],
                },
            },
        )
        return {"side": "output", **summary} if summary else None


class MyAgent(Agent):
    typesafe_screener = MyScreener
```

On Stream, inbound runs after `MessageStart` (and may strip tools); outbound runs before `MessageEnd`. Both emit a `DataEvent` with `data_type="typesafe"` (Vercel AI SDK part `data-typesafe`).

## Tool selection

A screener may attach a `control` dict. The SDK applies it generically:

| Hint | Effect |
| --- | --- |
| `strip_tools` | Empty the tool list for this turn |
| `skip_rag` | Drop tools whose name contains `rag` |
| `remove_tool_names` | Drop exact tool names |

On web chat, Stream mutates the Haystack agent’s tools after inbound screening. On webhooks, set `agent.typesafe_control` from the same payload before `agent.run(tools=True)` — `get_tools()` filters with `apply_tool_control`. Clear `typesafe_control` after the run.

## Webhooks

If the agent resolved by a webhook integration has `typesafe_screener`, [`run_inbound`](/manual/webhooks/) screens the inbound text, applies control, then posts plain-text TypeSafe cards as separate channel replies (inbound → answer → outbound). Cards are channel chrome only — they are not stored on the thread.

## Call `evaluate` anywhere

Workflows, management commands, and host steps can call `evaluate(state, questions)` directly. Prefer that over pretending Jev is an Agent `model`.

See the [TypeSafe reference](/manual/typesafe/) for payload and control details.
