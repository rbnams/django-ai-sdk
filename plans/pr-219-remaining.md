# Remaining fixes for PR 219

Reviewed tip: `2fb7b4d695af9cbe92b72a757c33902724302af1` (`feat/oversight-cleanup`), on `lean/5-haystack-plugin` (`a77769f`).

The disconnect bug is fixed: `run_agent` awaits the Haystack run directly, so cancelling the stream cancels the agent. These five are still real. Nothing else from the last review is worth a change.

## 1. Vouched history still skips attachment text

`django_ai_sdk/oversight/plugin/gate.py`, the `judged = [...]` list in `_gate`.

Chat rebuilds `attachment.context` onto the stored messages, then the stream folds that context into the Haystack text (`get_user_text`). `_gate` copies only role and `.text` into a `ChatMessage` with no attachments, then calls `gate_history(..., history=not vouched)`. On a vouched thread that skips earlier bodies. The attachment loop in `guard/gate.py` never runs, because those copies have no attachments. The model still receives the summaries and captions. The comment above `_gate` says attachment context is always judged. It is not. The new question is fine: `gate()` reads its combined string.

Fix:

- Keep the pre-Haystack `ChatMessage` list (the one that still has attachments) on the `TurnRequest`. Chat and `Agent.run` already have it.
- Run `gate_history` on that list, not on the text-only copies. `history=False` when vouched, so earlier bodies stay skipped and every `attachment.context` is still judged.
- Set the question `gate()` reads from the question's content, not from the concatenated Haystack text, so the new question's context is judged once.
- After both checks, rebuild each Haystack user message with `get_user_text` so the model sees the masked context.

Test: a masking policy, a stored thread, a second turn whose earlier message has an attachment whose summary contains an IBAN or an injection. The model must see the masked or withheld context. A non-vouched history must still be judged as sent.

## 2. A mask that waited drops spans the gate already found

`django_ai_sdk/oversight/reflect.py`, `answers_from` and `_reflect_masks`.

A mask rule waits until every lens it reads has answered. At reflect, `collect_masks` only sees answers from that reflect run. `answers_from` rebuilds earlier answers with value and confidence and no `spans`. A gate lens that already located an email is not in the list, so the mask does not cover it. Finish did not mask either, because the rule was still waiting. The stored answer keeps the text.

Fix:

- When rebuilding an answer, copy `row.spans` into `Answer.spans`.
- Pass those earlier answers into `collect_masks` together with the answers this reflect run just produced. A lens that reflect re-ran uses the new answer. A lens that already answered keeps its spans.
- Leave the "no spans at all" path as it is: that still masks the whole answer.

Test: an enforced mask whose condition is an email lens at the gate and a reflect-only lens. The email is in the stored answer before reflect and gone after, replaced by the mask. A reflect-only entities lens with no spans still masks the whole answer.

## 3. A review note can land after forget

`django_ai_sdk/oversight/services/reviews.py` (`note` in the resolve update) and `forget_turns` in `django_ai_sdk/oversight/content.py`.

`resolve_review` writes `note` whenever the item is still open. It does not look at the turn. `forget_turns` blanks notes that are already non-empty, and sets `content_forgotten` only on turns that are still open. A review item sits on a closed turn, whose note is empty until resolve. Forget commits, then resolve writes the note. A note that quotes the question puts that text back on a forgotten turn.

Fix:

- Set `content_forgotten` on every turn `forget_turns` locks, not only the open ones. Keep blanking notes in that same transaction, after the flag is set.
- In `resolve_review`, lock the turn and, when `content_forgotten` is set, store an empty note. Take the turn lock before the item update, so the two transactions cannot pass each other.

Test: forget a closed turn, then resolve its open review item with a note. The note is empty and `content_forgotten` stays set. Resolve first, then forget: the note is blanked.

## 4. A lookup can cache a policy a save just replaced

`django_ai_sdk/oversight/resolve.py`, the `_cache[name] = ...` store in `aget_policy`. `register_policy` already calls `invalidate`.

`aget_policy` builds, then stores, with no check that the cache was still empty. A save's `loader._load()` runs on a worker thread, registers the new class, and clears the cache without the asyncio lock. A lookup that already passed `aload_policies()` can write the policy it built from the previous class. The loader's stamp then matches the database, so the next load does nothing. That old definition stays until some later save invalidates the name. The cache has no TTL.

Fix:

- Bump a process-wide generation in `invalidate`.
- Read the generation before `_build`. Store the result only when the generation is unchanged. If it changed, return the result for this call and do not cache it. The next lookup builds again.

Test: start `aget_policy` so it is inside `_build`, save a tighter definition of that policy, then let the lookup finish. The next `aget_policy` returns the saved definition, not the one from before the save.

## 5. A refused tool call streams its arguments when the answer is not held

`django_ai_sdk/oversight/stream.py`, `pass_event`.

Stored arguments are cleared in `_drop_refused_inputs`, and a held stream does not release them. `pass_event` forwards the event immediately when `holds_answer()` is false. The adapter emits `ToolInputCompleteEvent` as soon as the arguments parse, which is before the tool hook can refuse. A policy whose only enforced block is on tool calls does not hold. The client receives the credential. The stored copy is then wiped.

Fix:

- Hold tool-input events until the tool hook has decided, whether or not the answer itself is held.
- On a refusal (`on_tool_results` with the refused marker), release those events with an empty `tool_input`.
- On a real tool result, release the original input. The call ran.
- Leave answer text on the existing hold rule.

Test: an enforced tool-stage secret block, no finish mask or block, `stream_mode` auto. The client never receives the secret arguments. The stored message has empty arguments. A call the policy allows still shows its arguments.
