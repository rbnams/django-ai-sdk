# Changelog

## Unreleased

### Oversight

A new app, `django_ai_sdk.oversight`, checks and records every turn an agent runs, whichever way it runs. Install `django-ai-sdk[oversight]`, add the app and run `migrate`. See [Oversight](docs/content/oversight.md) and the [manual](docs/content/manual/oversight.md).

- **A policy per agent**: `oversight = SupportPolicy` on an `Agent` (a `Policy` subclass, registered automatically) or a registered policy's name; `False` opts out and `None` takes `DEFAULT_POLICY`. Anything else raises `ImproperlyConfigured` when the class is defined. `async def get_oversight(self, *, channel, user, **kwargs)` chooses per turn and may read the database, and an operator's assignment in the console wins over code. A runtime agent without an assignment runs under its base class's policy, then `DEFAULT_POLICY`.
- **Every way an agent runs is a turn**: chat, `Agent.run()` (with `channel=` and, for trusted code, `oversight=`), the HTTP run endpoints (which answer `"blocked": true`), webhooks and workflow `AgentStep`s (a submitted definition runs on `workflow:adhoc:<step>`). A refusal raises `django_ai_sdk.errors.Blocked` with the policy's message. An override of `run()` is the turn: the policy wraps the outermost `run()`.
- **Each agent runs under its own policy**: a nested run or a subagent's delegation joins the turn it runs in when it resolves the same policy, or none; under a policy of its own it opens a child turn (its own gate, watch, finish and record), linked to the outer one. A refusal inside a delegation is the tool result the coordinator reads.
- **Lenses, answerers and rules**: a lens asks one typed question about a turn (`Flag`, `Choice`, `Score`, `Entities`, `Text`); an answerer answers it: local detectors (`PII` by checksum, `Secrets`, `Patterns`, `Keywords`, ...), a model judge (`LLMJudge` on any SDK generator, or on an OpenAI-compatible server of your own named in `ENDPOINTS`: Ollama, vLLM, LM Studio), the optional hosted `TypeSafe` classifier (`django_ai_sdk.oversight.answerers.TypeSafe`), a function or a person. `Classifier` is the base for any model that answers typed questions, `Detector` for span finders; `catalog.endpoint_models` lists every server's chat models for the console. A rule's actions are `Block`, `Mask`, `RestrictTools`, `Queue`, `Notify` and `Alert`, and every rule is `off`, `report` or `enforce`.
- **Four stages**: `gate` before the model, `watch` on each tool call and while the answer streams, `finish` before the answer is shown or stored, `reflect` in the background, with deadlines and a per-stage fail mode. The question, history the server did not store, attachment context and tool results are judged before the model reads them; sources are masked like tool results before a citations card, the stored citations or `/threads/{id}/sources/` show them. `stream_mode = "hold"` shows an answer only once finish passed it, and keeps no reasoning; the default, `auto`, holds it exactly while an enforced rule may mask or withhold it at finish.
- **Packs that start in report mode**: `security`, `safety`, `quality`, and `default` with all three. Their lenses ask optional aliases (`"classifier?"`, then `"judge?"`) that a deployment maps to answerers; with none defined, the lenses code answers still run and a system check lists the rest for the policies in use. `pick` takes single lenses and rules from a pack.
- **What the client sees**: `data-oversight` stream parts say what the gate, watch or finish changed; a refused question or withheld answer ends with finish reason `content-filter` (`content_filter` for OpenAI clients). `Blocked` maps to HTTP 400, like `CONTENT_FILTERED`.
- **An audit ledger**: turns, verdicts, decisions, recorded reads and policy changes on an append-only hash chain, with signed Merkle checkpoints (`oversight_seal`), verification (`oversight_verify`), evidence encrypted per thread when a `MASTER_KEY` is set, forgetting a thread (deleting a thread forgets it too), and retention (`oversight_prune`). `CAPTURE` defaults to `masked`.
- **Service and console API**: `OversightService` and `get_oversight_router()` for Django Ninja (DRF projects call `OversightService`): turns through a `review` or `audit` purpose, review queues, the policy editor, dry runs, agent assignment, answerer aliases, checkpoints, the access log, NDJSON export, and exporters (`Log`, a signed `Webhook`). New operations in the `oversight` permission domain: `VIEW_OVERSIGHT`, `REVIEW_OVERSIGHT`, `AUDIT_OVERSIGHT`, `REVEAL_OVERSIGHT_ACTOR`, `MANAGE_OVERSIGHT`.
- **System checks** `ai_sdk.oversight.E001`–`E011` and `W001`–`W015`; `W001`/`W002` (no `MASTER_KEY`, no `SIGNING_KEY`) are deploy checks.

### Breaking changes

- **`Agent.run()` raises `Blocked`** when the agent has a policy (its own or `DEFAULT_POLICY`) and the policy refuses the question or withholds the answer. Under a policy it masks the last user message of the list you pass in place when the gate masks the question; the rest of the list is left as it was.

### Agents

- **Runtime agents search their knowledge bases**: `Agent.get_linked_rag_tools()` links an agent's `memories` to the thread and builds their search tools with the turn's citation registry, so what they find reaches citations and oversight. `RuntimeAgent` calls it, so a database-defined agent with a `rag_provider` and memories gets one `search_<slug>` tool per memory.

### Knowledge bases

- **An edited document replaces its old chunks**: `QdrantBM25HybridRAG.add_documents` removes a document's earlier chunks first. A chunk's id is a hash of its text, so before, an edit left the old chunks searchable beside the new ones.
- **One write per shared index**: an `Entry` save or delete is embedded once for a server index, not once per agent class with a RAG provider (`RAGProvider.add_documents(..., written=)`).
- **A mixed PDF is read**: a text PDF with a few scanned pages (a cover, a signature page) is read without those pages, which are logged, instead of being refused. A mostly scanned PDF still goes to OCR.
- **Chunks say which document they are from**: an `Entry`'s name is now its `title`, which `meta_fields_to_embed` embeds with every chunk (it named a field no entry had). Two documents with the same kind of article, such as two CAOs, no longer read alike. Rebuild existing indexes with `reindex_memories --force`.
