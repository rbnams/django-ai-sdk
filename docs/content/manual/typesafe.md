---
title: TypeSafe
type: docs
weight: 126
---

Internals for the optional TypeSafe / Jev integration (`django_ai_sdk.typesafe`). User walkthrough: [TypeSafe](/typesafe/).

## Package

| Module | Role |
| --- | --- |
| `client.evaluate` | Fail-open System One call → `{model, judgments}` |
| `screen.TypeSafeScreener` | Protocol: `screen_input` / `screen_output` |
| `control` | `apply_tool_control`, `apply_control_to_agent_component`, `enrich_payload_acted` |
| `format` | Plain-text cards for Slack / Telegram |

Install: `django-ai-sdk[typesafe]`. Settings: `TYPESAFE_API_KEY`, `TYPESAFE_MODEL` (see [Settings](/manual/settings/)).

## Agent

| Attribute | Default | Purpose |
| --- | --- | --- |
| `typesafe_screener` | `None` | Screener class; instantiated per turn |
| `typesafe_control` | `None` | Per-request `control` dict for `get_tools()` |

## Stream lifecycle

1. `MessageStartEvent`
2. Inbound `DataEvent(data_type="typesafe")` — may apply `control` to Haystack tools
3. Pipeline (LLM + tools)
4. Outbound `DataEvent(data_type="typesafe")`
5. `MessageEndEvent`
6. Suggestions…

## `data-typesafe` payload

```json
{
  "side": "input",
  "model": "jev-latest",
  "judgments": [
    {"id": "should_use_tool", "type": "noul", "label": "Should use tool", "value": 0.12}
  ],
  "control": {"strip_tools": true},
  "acted": ["stripped_tools"]
}
```

Judgment shapes:

| `type` | Fields |
| --- | --- |
| `noul` | `value` ∈ [0, 1] (`P(yes)`) |
| `choice` | `value` (winner), `probabilities`, `confidence` |
| `score` | `value` (level position), `legend`, `probabilities`, `confidence` |

## Control hints

Applied by `apply_tool_control(tools, control)`:

- `strip_tools: true` → `[]`
- `skip_rag: true` → drop names containing `rag`
- `remove_tool_names: [...]` → drop exact names

Stream applies inbound `control` via `apply_control_to_agent_component`. Webhooks assign `agent.typesafe_control` before `run()` and clear it in `finally`.

## Webhooks

`integrations.webhooks.tasks.run_inbound` screens when the resolved agent has a screener, then posts `format_typesafe_card` results as separate replies. See [Webhooks](/manual/webhooks/).
