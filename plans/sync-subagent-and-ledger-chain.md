# Sync subagent regression, and binding judge text into the ledger

Notes from review of the oversight stack. Not a user-facing manual page.

Reviewed code:

- Sync-subagent bug: rewritten tip `b3f656a6c4d2dc422b0e8b18fb51b17a04e28813` (`oversight/8-integrations`).
- Ledger: `5a2e3ee5ea13065bcbaa0470960dcf302b626ac8`. The rewrite to `b3f656a` did not change `ledger/`, `vault/`, or conversation storage, so the ledger section applies to both.

## 1. Bug: oversight refuses every sync subagent delegation

### What happens

With the oversight app enabled, `SubagentStreamFilter.run` (the sync Haystack path) always returns the unguardable stub and does not run the subagent. That includes a subagent with `oversight = False`, a subagent that would join the current turn, and a deployment with no switches overlay.

The async path (`run_async`) is unaffected. It still opens a child turn or joins the outer one.

### Why

`policy_needs_await` treats any callable `get_oversight` as a per-turn hook that sync code cannot read:

```python
def policy_needs_await(agent: Any) -> bool:
    return callable(getattr(agent, "get_oversight", None)) or (
        type(overlay_source()) is not OverlaySource
    )
```

`Agent.get_oversight` is defined on the base class and only returns `self.oversight`. `getattr(agent, "get_oversight")` is therefore callable for every agent. `_needs_own_turn` returns true as soon as `policy_needs_await` is true, before `code_policy` and `joins_turn`.

`run` then does:

```python
if self._needs_own_turn():
    return _answered(str(UNGUARDABLE))
```

### Where

On `b3f656a`:

- `django_ai_sdk/oversight/agents.py` `policy_needs_await` (about lines 53–61)
- `django_ai_sdk/agents/subagent.py` `_needs_own_turn` (the `policy_needs_await` return, about line 375)
- `django_ai_sdk/agents/base.py` `Agent.get_oversight` (about lines 521–538)

### The test that misses it

`tests/oversight/test_nested_runs.py` covers an overlay subclass and an instance attribute assigned on top of the base method. Both make `policy_needs_await` true for an extra reason. Neither asserts that a default agent, or one with `oversight = False` and no overlay, still runs synchronously.

### Fix

Treat “needs an async read” as an override, not as the presence of the base method.

1. Class override: `getattr(type(agent), "get_oversight", None) is not Agent.get_oversight`.
2. Instance assignment, which the existing test uses: `"get_oversight" in getattr(agent, "__dict__", {})`.
3. Keep the overlay check. A switches overlay can assign a policy that sync code does not see. Refusing those delegations is the safe direction. Do not let that check stand in for (1). With no overlay and no override, fall through to `code_policy` and `joins_turn`, as the sync path did before this change.

Add a regression that fails on the current tip:

- Oversight enabled, base `OverlaySource`, subagent `oversight = False`: `run` executes the subagent.
- Same setup, subagent policy equal to the current turn’s policy: `run` executes and joins.
- Class `oversight` set to a different policy: `run` still returns `UNGUARDABLE`.
- Instance `get_oversight` assigned, and a non-base overlay installed: `run` still returns `UNGUARDABLE`.

Optional follow-up, not required to close the regression: give the overlay a sync read and consult it from `_needs_own_turn`, so installing switches does not refuse a delegation whose effective policy is still “off” or the same as the outer turn. Until that exists, refusing every sync delegation while a non-base overlay is installed is acceptable. Refusing them when the only reason is the base method is not.

## 2. Plan: put judge text and reviewer notes on the chain

### What is already bound

`LensAnswer` rows are chained. The payload includes lens, stage, answerer, model, value, confidence, spans, trail, error, and reviewer. `RuleMatch` rows are chained, including `acted`.

Those columns are the non-quoting record. `split_answer` (`django_ai_sdk/oversight/persist.py`) moves anything that can quote the turn into the content blob:

- rationale
- entity names a judge wrote (`stored_value` keeps `"[found]"` on the row; the name goes to `texts`)
- per-item scores
- trail values that were reduced the same way

Reviewer notes go to `content["notes"]`, keyed by review item id. `TurnContent`’s docstring names both maps `SIDE_SECTIONS`.

### What is not bound

`content_digest` hashes the content after dropping `texts` and `notes`:

```python
hashed = {key: value for key, value in content.items() if key not in SIDE_SECTIONS}
```

The content chain entry stores that digest, not the ciphertext (`chain._digest`). `add_to_content` re-encrypts the whole blob when reflect or a review appends quotes, and it does not append a chain entry. Verify recomputes the digest from the decrypted content minus those two keys.

So a holder of the thread’s data key can change a rationale, an entity name, or a reviewer note, or put back an older envelope whose question and answer are unchanged, and `ledger_verify` still returns ok. The chain stays blind to the words, which is right. It is also blind to whether those words were swapped.

This was deliberate so later judge and reviewer writes would not move a digest taken at close. The cost is that the quoting copy is not part of the audit trail.

Chat `Message.result` rows stay outside the vault and outside the chain. This plan does not encrypt them. `actor_id` stays out of the turn hash so forget can clear a pseudonym. That is a separate decision and is not part of this change.

### Recommendation

Do this before the ledger PRs merge. Nothing deployed has a chain to migrate, and the rule in `chain.py` is that hashed fields of an existing kind do not change after release. After merge, the same behavior needs a new entry kind instead of a new meaning for the content digest.

Bind the side sections with a second keyed digest, and chain a new content entry whenever that digest changes. Do not copy the plaintext into `LedgerEntry`.

### Design

1. **Digest the quotes separately.** Add `aside_digest(scope, texts, notes)` next to `content_digest`. It is `vault.digest` over the canonical JSON of `{"texts": ..., "notes": ...}` only, including empty maps, so “no quotes” is a value. Keyed, same as the content digest, so the chain still does not reveal a short rationale. Keep `content_digest` excluding `SIDE_SECTIONS`, so the commitment taken at close does not change meaning when quotes arrive later.

2. **Store it beside the core digest.** `LedgerCustody.store` returns `{"vault": envelope, "digest": core, "aside": aside_digest}`. The envelope still holds the full content, quotes included, so shredding the scope key still destroys the words. Plain custody ignores the extra field.

3. **New chain kind, do not edit `CONTENT`.** Add kind `aside` (name it in `KINDS` only if there is a row; otherwise treat it like `prune`: a span-style entry with no model). Preferred shape, because the quotes are not their own table:

   - `kind = "aside"`
   - `ref = <TurnContent pk>`
   - `payload_hash = sha256(canonical {"digest": aside_digest})`
   - `rechained`: only the newest `aside` entry for that content id must match the row. Older entries stay in the hash chain as history.

   Do not add `aside` to the `CONTENT` payload. Existing content entries keep hashing `id`, `scope`, `capture`, `expires_at`, and the core digest.

4. **Append on every write that can add quotes.**

   - Close path: when `texts` is non-empty, `LedgerCustody.chained` already runs in that transaction. Append the `aside` entry in that same `append_many`, after the content entry.
   - Later path: `add_to_content` updates `stored` and returns. Extend custody with something like `content_extended(row)` and call it inside `add_to_content`’s transaction, after the `stored` update, only when the aside digest changed. `PlainCustody` no-ops. `LedgerCustody` re-reads the row and `append_many([(ASIDE, row)])`.
   - Forgotten scope: `store` raises `Forgotten`, `add_to_content` returns false, no new entry.
   - Empty update: if the merged maps do not change the digest, do not append.

5. **Lock order.** `add_to_content` already holds the content row (`select_for_update`) and then would take `LedgerHead`. Take the head lock only after the content update, at the end of that transaction. Do not take the head lock first. Reflect and review run after close has committed, so this does not nest inside the close transaction’s head lock.

6. **Verify.**

   - Decrypt as today. Recompute the core digest. A mismatch is still `content altered`.
   - Recompute the aside digest from `texts` and `notes` (missing keys count as `{}`).
   - Newest `aside` entry for that content id must equal it. Mismatch: `quotes altered`.
   - Non-empty `texts` or `notes` and no `aside` entry: `quotes not chained`.
   - Empty quotes and no `aside` entry: ok. That is every turn whose capture keeps no quoting text, and every turn closed before any judge text was stored.
   - Older `aside` entries are not compared to the row. Their `entry_hash` / `prev_hash` links still have to hold.
   - `erase` already covers `CONTENT`. Extend it to cover `aside` entries for those content ids, or accept a missing content row as also excusing the aside entries that name it. A leftover aside entry whose content row was erased should not be `record missing`.

7. **Docs.** In `docs/content/oversight-ledger.md`, replace the sentence that side sections change nothing in the chain. Say the core digest still omits them, and the newest `aside` entry commits the keyed digest of judge text and reviewer notes. The chain still holds no plaintext. Shredding the scope key still makes both digests unverifiable as opened text; verify already accepts shredded content when the `forget` entry is present.

### Tests

Add to `tests/oversight/ledger/`:

- Close a turn whose lens answer has a rationale. Verify ok. The chain has an `aside` entry and the ledger row does not contain the rationale string.
- Reflect or a review appends another rationale or a note. A second `aside` entry appears. Verify ok. The first entry’s link still checks.
- Re-encrypt the same core content with a different rationale (or swap in an older envelope that matches the core digest only). Verify reports `quotes altered`.
- Delete `texts` inside the envelope and re-encrypt. Verify reports `quotes altered` or `quotes not chained`, not ok.
- Shred the scope. Verify still ok, and the words do not decrypt.
- Capture that stores no quoting text. No `aside` entry. Verify ok.
- Plain custody, ledger not installed: `add_to_content` still updates the JSON and does not import the ledger.

### Out of scope

- Encrypting `django_ai_sdk_messages`. The chat thread stays a separate store.
- Hashing `Turn.actor_id`. Leaving it out is what lets forget clear a pseudonym without breaking the turn entry.
- Inclusion proofs. Verify keeps walking the whole chain.
