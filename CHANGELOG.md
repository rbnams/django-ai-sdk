# Changelog

## Unreleased

### Oversight

A new app, `django_ai_sdk.oversight`, audits every agent turn on any channel (chat, webhooks, workflow steps, your own code) and can act on it. See [Oversight](docs/content/oversight.md) and the [manual](docs/content/manual/oversight.md).

- **Lenses**: typed questions about a turn: `Flag`, `Choice`, `Score`, `Entities` and `Text`, about the question, the answer, the exchange, the sources or a tool call.
- **Answerers**, picked per lens: Jev (`TypeSafe`) for most yes/no, label and score lenses; `LLMJudge` on any SDK generator and `AgentJudge` on a registered agent, for text and escalations; local detectors (`PII` by checksum, `Secrets`, `Keywords`, `Patterns`, language, length, refusal, lexical support); `Moderation`, `HFClassifier`, `Presidio`, `Function` and `Human`. A lens asks them in order until one is sure; if none is, the surest answer stands. Secrets are masked before any turn content leaves.
- **Grounding**: groundedness and context relevance read the turn's sources and its tool results, Jev scores the RAG triad in one call, and a turn with no evidence is not scored for grounding. Jev reads every source whole and labelled; `taken_from` (quality pack, answered by `TypeSafeSources`) names the sources the answer is taken from, one yes/no per source in one request, and `review_unsourced` queues an answer taken from none, also when Jev can't settle a source. Every answer is asked: one that states facts with no source at all, and an empty answer, are taken from none. Sources carried over from earlier turns no longer count as found in this one, but an answer is judged against them too, named "(found earlier)". Reflect no longer judges the empty answer of a turn the gate blocked.
- **Rules**: a condition over lens answers (`L("topic") == "hr"`) and actions: `Block`, `Redact`, `RestrictTools`, `Queue`, `Escalate`, `Notify`, `Alert`, `Allow`. Each rule is `off`, `shadow` (test mode) or `enforce`.
- **Stages**: `gate` before the model, `watch` while it works (each tool call, the streaming answer), `review` before the user sees it, `reflect` afterwards in the background, and `retro` replays with `oversight_replay`. `catalog.STAGE_INFO` holds each stage's label, what runs there and what a rule can do. Inline stages have deadlines and a per-stage `FAIL_MODE`.
- **Policies**: the `security`, `safety`, `quality` and `classify` packs, and `default` with all four. Policies extend each other, or `pick` single lenses and rules from a pack, and can be tuned from settings or a database overlay.
- **Ledger**: turns, observations, decisions, access and overlay changes on an append-only hash chain, with signed Merkle checkpoints (`oversight_seal`), verification (`oversight_verify`), per-thread encrypted evidence with crypto-shredding, and retention (`oversight_prune`).
- **Service and console API**: `OversightService` and `get_oversight_router()` for Django Ninja: turns, reviews, policies, the policy editor's writes, dry runs, agent assignment, answerer aliases, checkpoints, the access log and NDJSON export.
- **Exporters**: `Log` and a signed `Webhook` (Standard Webhooks); anchoring exporters receive every checkpoint.
- **Extras**: `[oversight]` (`cryptography`, for encryption and signed checkpoints), `[typesafe]` (Jev) and `[presidio]`. `[all]` leaves out `[presidio]`, which needs a spaCy model.
- **Migration**: `django_ai_sdk_oversight.0001_initial`. Run `manage.py migrate` after adding the app to `INSTALLED_APPS`.
- **What the model reads is checked too**: with a stored thread, the model gets the thread's own (masked) history and only the new question from the request; without one, the gate's masks run over the history the client sends. Tool results are masked the same way before the model reads them (`OversightToolResultHook`, in `default_hooks`). What a result's masking found is recorded at watch, the lens's answer without its spans and the rule once, so the turn says it was masked while the agent worked.
- **Long input**: detectors scan the whole text in windows; Jev, model judges and classifiers read the start and end of a long text; a question over `MAX_INPUT_CHARS` (100,000) is refused at the gate, unread.
- **Hold means hold**: in `stream_mode = "hold"` tool cards show that each tool ran while the answer is held, and their input and output follow review (or a "withheld" note); a withheld answer stores its tool calls by name only; a masked or withheld answer stores no reasoning; a turn that ends before review (a disconnect, an error) stores no answer; review first watches the end of the answer a watch lens has not seen.
- **Fail modes cover the gate and tool checks**: an error there follows `FAIL_MODE` instead of failing the request, and a condition that can't be evaluated (`confident` without a number, `in` without a list) is refused when the rule is built.
- **Checks**: `oversight.W011` replaces `E007` when the lenses an enforced rule reads have only optional answerers (`"jev?"`) and none is defined, so a deployment without Jev's key still starts.

### Agents

- **Runtime agents search their knowledge bases**: `Agent.get_linked_rag_tools()` links an agent's `memories` to the thread and builds their search tools with the turn's citation registry, so what they find reaches citations and oversight. `RuntimeAgent` calls it, so a database-defined agent with a `rag_provider` and memories gets one `search_<slug>` tool per memory.

### Knowledge bases

- **An edited document replaces its old chunks**: `QdrantBM25HybridRAG.add_documents` removes a document's earlier chunks first. A chunk's id is a hash of its text, so before, an edit left the old chunks searchable beside the new ones.
- **One write per shared index**: an `Entry` save or delete is embedded once for a server index, not once per agent class with a RAG provider (`RAGProvider.add_documents(..., written=)`).
- **A mixed PDF is read**: a text PDF with a few scanned pages (a cover, a signature page) is read without those pages, which are logged, instead of being refused. A mostly scanned PDF still goes to OCR.
- **Chunks say which document they are from**: an `Entry`'s name is now its `title`, which `meta_fields_to_embed` embeds with every chunk (it named a field no entry had). Two documents with the same kind of article, such as two CAOs, no longer read alike. Rebuild existing indexes with `reindex_memories --force`.
