# DSPy Integration Plan for Nanobot

## Goal

Integrate DSPy (v3.1.3) into nanobot for observability, tiered model
selection, conversation-driven optimization, and inference-time
self-improvement. All phases are independent and can be implemented in any
order, except Phase 1 which is a prerequisite for everything else.

### Key Architectural Decisions

**Tiered LLMs from day one.** Three named `dspy.LM` instances are configured
in Phase 1 and used consistently across all phases:

| Tier | Use case | Example model |
|------|----------|---------------|
| `quick` | Simple subtasks, formatting, classification, RLM sub-queries, memory consolidation | `gemini-2.5-flash-lite` |
| `normal` | Default agent conversation and tool use | `gemini-3-pro-preview` |
| `deep` | Self-evaluation, optimization, complex reasoning, escalation when stuck | reasoning model (o3, gpt-5, etc.) |

The agent runs on `normal` by default. It can explicitly delegate to `quick`
or escalate to `deep` via a `think` tool. Some operations are pre-wired to
a specific tier. The LiteLLM fallback chain handles rate limits transparently.

**Conversational architecture.** DSPy is task-oriented (inputs in, outputs
out) -- it has no built-in conversational loop. However, it has `dspy.History`,
a first-class type for passing multi-turn conversation context into any
module. DSPy's adapters format `History` as proper multi-turn messages, and
optimizers understand it.

Nanobot's `AgentLoop` is the conversational orchestrator. The integration
pattern is:

- Nanobot manages sessions, memory, channels, and the outer message loop.
- Each turn, nanobot feeds `dspy.History` (built from session history) into
  DSPy modules (`Predict`, `ReAct`) that handle the actual LLM interaction.
- DSPy provides observability, structured tool use, and optimization hooks
  around those per-turn LLM calls.

Phase 4 refactors `_run_agent_loop` itself to use `dspy.ReAct` with
`dspy.History`, making the core agent loop optimizable while nanobot retains
full control of the conversational state.

**Agent-controlled optimization.** The optimization pipeline (Phases 5-6) is
exposed as tools the agent invokes. The agent reviews proposed changes and
decides what to apply using its existing filesystem tools. Nothing is
auto-applied.

---

## Phase 1: LM Layer + Tiered Models (prerequisite)

**Effort:** small (half a day)
**Files:** `nanobot/providers/`, `nanobot/config/`, `pyproject.toml`

Set up three `dspy.LM` instances (quick, normal, deep) routed through the
LiteLLM proxy. This is the foundation everything else builds on.

### Tasks

1. Add `dspy>=3.1.3` to `pyproject.toml` dependencies.

2. Create `nanobot/providers/dspy_provider.py` implementing `LLMProvider` by
   delegating to `dspy.LM`. Map `LLMResponse` fields from DSPy's response
   format.

3. Configure three tiers in `config.json`:
   ```json
   {
     "providers": {
       "quick": {
         "type": "dspy",
         "model": "gemini/gemini-2.5-flash-lite",
         "api_base": "http://localhost:4000",
         "temperature": 0.3
       },
       "normal": {
         "type": "dspy",
         "model": "gemini/gemini-3-pro-preview",
         "api_base": "http://localhost:4000",
         "temperature": 0.7
       },
       "deep": {
         "type": "dspy",
         "model": "openai/o3",
         "api_base": "http://localhost:4000",
         "temperature": 1.0
       }
     }
   }
   ```

4. Initialize all three at startup in `AgentLoop.__init__`:
   ```python
   self.lm_quick  = dspy.LM(cfg.quick.model,  api_base=..., ...)
   self.lm_normal = dspy.LM(cfg.normal.model, api_base=..., ...)
   self.lm_deep   = dspy.LM(cfg.deep.model,   api_base=..., ...)
   dspy.configure(lm=self.lm_normal)  # default
   ```

5. Keep `LiteLLMProvider` as a fallback; the two are interchangeable.

6. Verify: run a CLI conversation and confirm responses are identical to
   the pre-DSPy behavior.

### Notes

- `dspy.LM` uses `litellm` under the hood, same as `LiteLLMProvider`. The
  swap is transparent.
- `dspy.LM` adds caching, retry with backoff, and rollout_id support for free.
- All three tiers route through the same LiteLLM proxy, so the existing
  fallback chain and rate limit handling apply.
- The `deep` tier can use reasoning models; DSPy's `dspy.Reasoning` type
  handles reasoning model output automatically.

---

## Phase 2: Observability (callbacks + usage tracking)

**Effort:** small (half a day)
**Depends on:** Phase 1
**Files:** new `nanobot/providers/dspy_callbacks.py`

### Tasks

1. Implement a `NanobotCallback(dspy.utils.callback.BaseCallback)` with:
   - `on_lm_start` / `on_lm_end`: log **which tier** (quick/normal/deep),
     model name, latency, and token counts via loguru.
   - `on_module_start` / `on_module_end`: log which DSPy module ran (once we
     add Predict/ReAct modules in later phases).
   - `on_tool_start` / `on_tool_end`: log tool name, duration, success/failure
     (available once tools are wrapped as `dspy.Tool`).
2. Register the callback globally:
   `dspy.configure(lm=lm_normal, callbacks=[NanobotCallback()])`.
3. Wrap each `_process_message` call in `dspy.track_usage()` to get
   per-conversation token totals broken down by tier. Emit in the session
   log (`sessions/*.jsonl`).
4. Optionally expose a `/stats` slash command showing cumulative usage
   per tier.

### Value

- Immediate visibility into token spend per tier, per model, per conversation.
- Answers questions like "how often does the agent escalate to deep?" and
  "is quick being used for tasks that need normal?"
- Foundation for the judge pipeline (Phase 5) -- tier usage is part of the
  signal.

---

## Phase 3: RLM as a Tool

**Effort:** small-medium (1 day)
**Depends on:** Phase 1
**Files:** new `nanobot/agent/tools/rlm_tool.py`

Add `dspy.RLM` as a nanobot tool that the agent can invoke for tasks requiring
programmatic exploration of large data (logs, long documents, code analysis).

### Tasks

1. Create `RLMTool` implementing `BaseTool` in nanobot's tool registry.
   - Input: `task` (str), `context` (str -- the large data to explore).
   - Internally creates a `dspy.RLM("context, task -> result")` and calls
     `forward(context=..., task=...)`.
   - Returns the RLM's `result` field.
2. Register it in `AgentLoop._register_default_tools()`.
3. **Pre-wired tier**: RLM's `sub_lm` uses `self.lm_quick` for sub-queries.
   The outer RLM reasoning uses `self.lm_normal`.
4. Set `max_iterations=15`, `max_llm_calls=30` as sensible defaults.

### Value

- Nanobot can handle "analyze this 50k-line log" type tasks that would
  otherwise exceed the context window.
- The REPL sandbox (Deno/Pyodide/WASM) is secure -- no shell escape risk.
- RLM supports custom tools, so nanobot's Jira/Confluence MCP tools could be
  passed through for complex multi-source queries.
- Sub-queries run on `quick` -- fast and cheap even with 30+ calls.

---

## Phase 4: Refactor Agent Loop to dspy.ReAct + dspy.History + think Tool

**Effort:** medium (2-3 days)
**Depends on:** Phase 1
**Files:** `nanobot/agent/loop.py`, new `nanobot/agent/dspy_agent.py`,
new `nanobot/agent/tools/think_tool.py`

Refactor `AgentLoop._run_agent_loop` to use `dspy.ReAct` with `dspy.History`
instead of the hand-rolled tool-calling loop. Add a `think` tool for explicit
tier selection by the agent.

### Current Architecture

```
_process_message()
  -> context.build_messages(session history, current message)
  -> _run_agent_loop(messages)
       while iteration < max_iterations:
         provider.chat(messages, tools)
         if tool_calls: execute tools, append results
         else: return final_content
  -> session.add_message(user, assistant)
```

### Target Architecture

```
_process_message()
  -> build dspy.History from session.get_history()
  -> react_agent(message=current_message, history=history)
       dspy.ReAct internally handles:
         thought -> tool_name -> tool_args -> observation loop
         automatic context window truncation
         structured trajectory
       agent can call think(prompt, mode) within the loop
  -> session.add_message(user, assistant)
```

### Tasks

1. Define the conversational signature:
   ```python
   class AgentTurn(dspy.Signature):
       """You are a helpful AI assistant with access to tools."""
       message: str = dspy.InputField(desc="Current user message")
       history: dspy.History = dspy.InputField(desc="Conversation history")
       response: str = dspy.OutputField(desc="Response to the user")
   ```

2. Wrap nanobot's existing tools (filesystem, shell, web, message, spawn,
   cron, MCP) as `dspy.Tool` instances. Each tool's `func` delegates to the
   existing `BaseTool.execute()` method.

3. **Create the `think` tool** (`think_tool.py`):
   ```python
   class ThinkTool(BaseTool):
       """Run a prompt through a different thinking mode."""
       name = "think"
       parameters = {
           "prompt": {"type": "string", "description": "What to think about"},
           "mode": {
               "type": "string",
               "enum": ["quick", "deep"],
               "description": "quick = fast/cheap for simple subtasks. "
                              "deep = slow/powerful for hard problems."
           }
       }

       async def execute(self, prompt: str, mode: str) -> str:
           lm = self.lm_quick if mode == "quick" else self.lm_deep
           with dspy.context(lm=lm):
               result = dspy.Predict("prompt -> result")(prompt=prompt)
           return result.result
   ```

   The agent uses this within ReAct's tool loop:
   - `think("Is this user asking about the same topic?", mode="quick")` --
     fast classification.
   - `think("Why did my last 3 responses about Jira fail?", mode="deep")` --
     careful self-reflection.

4. Create `NanobotReAct(dspy.Module)` in `dspy_agent.py`:
   - Wraps `dspy.ReAct(AgentTurn, tools=[...], max_iters=20)`.
   - `forward(message, history)` calls ReAct and returns the response.
   - Loads system prompt / persona from workspace files (`SOUL.md`, etc.)
     and injects it into the signature instructions.
   - Runs on `lm_normal` by default.

5. Refactor `_process_message` to:
   - Build `dspy.History` from `session.get_history(max_messages=memory_window)`.
   - Call `react_agent(message=msg.content, history=history)`.
   - Extract `response` from the prediction.
   - Session management, memory consolidation, channel routing remain unchanged.

6. Add `dspy.Refine` wrapper with a lightweight reward function for
   automatic retry with blame-based feedback on failure:
   ```python
   agent = dspy.Refine(
       module=NanobotReAct(tools),
       N=3,
       reward_fn=lambda args, pred: 1.0 if len(pred.response.strip()) > 10 else 0.0,
       threshold=1.0,
   )
   ```

7. **Pre-wired tiers**:
   - Main ReAct loop: `lm_normal`
   - Memory consolidation (`_consolidate_memory`): `lm_quick`
   - `think(mode="quick")`: `lm_quick`
   - `think(mode="deep")`: `lm_deep`
   - Cron jobs: per-job `tier` field (default `quick`), passed through
     `process_direct(tier=...)` and selected via `dspy.context(lm=...)`
     at execution time.

   **Note**: Cron tier plumbing is already in place (`CronPayload.tier`,
   tool parameter, serialization, `on_cron_job` callback). The actual LM
   switching activates when `process_direct` uses `dspy.context(lm=...)`.

8. Add tier description to `AGENTS.md`:
   ```markdown
   ## Thinking Modes

   You run in **normal** mode by default. You also have access to two
   other modes via the `think` tool:

   - **quick** -- fast and cheap. Use for simple subtasks: reformatting
     text, extracting a single field, yes/no classification, summarizing
     short content. Subsecond response. Low cost.
   - **deep** -- slow and powerful. Use when you're struggling with a
     task, when accuracy is critical, or for self-reflection and
     evaluation. Has extended reasoning capabilities. Use sparingly.

   You don't need to call `think` for normal work -- just respond as
   usual. Only escalate to deep when normal isn't working, and delegate
   to quick when the subtask is trivial.

   When scheduling cron jobs, you can set a **tier** to control which
   model runs the job. Use `quick` for simple recurring tasks (weather,
   reminders), `normal` for moderate tasks, and `deep` for complex
   analysis or report generation. If omitted, cron jobs default to
   `quick`.
   ```

### What Stays the Same

- `AgentLoop.run()` -- the outer message bus consumer loop.
- `SessionManager` -- session creation, persistence, history retrieval.
- `_consolidate_memory()` -- memory consolidation (now uses `lm_quick`).
- Channel routing, `/new`, `/help` commands.
- MCP server connections.
- Tool implementations -- only the registration layer changes.

### What Changes

- `_run_agent_loop` is replaced by a `dspy.ReAct` call.
- Tool definitions go through `dspy.Tool` instead of nanobot's `ToolRegistry`
  OpenAI-format JSON schemas.
- Context building uses `dspy.History` instead of manual message list
  construction.
- The agent gains a `think` tool for explicit tier selection.

### Value

- The core agent loop becomes a `dspy.Module` -- fully optimizable by
  SIMBA/GEPA (Phase 6).
- Built-in context window truncation (ReAct trims oldest tool calls
  automatically when the window is exceeded).
- Structured trajectories (thought/action/observation) for debugging.
- `Refine` gives inference-time self-correction without training data.
- `dspy.History` is understood by optimizers, so few-shot demos can include
  multi-turn context naturally.
- Tier selection is explicit and observable -- the callback (Phase 2) tracks
  which tier was used for each call, enabling analysis of tier usage patterns.

---

## Phase 5: Conversation Judge Pipeline

**Effort:** medium (2-3 days)
**Depends on:** Phase 1 (Phase 2 helps but is not required)
**Files:** new `nanobot/optimization/judge.py`, `nanobot/optimization/extractor.py`,
new `nanobot/agent/tools/optimize_tool.py`

Build the LLM-as-judge pipeline that mines implicit feedback from conversation
logs to produce labeled training data for optimization. Expose it as a tool
the agent can invoke.

### Tasks

1. **Conversation segmenter** (`extractor.py`):
   - Parse `sessions/*.jsonl` into conversation segments.
   - Detect topic boundaries (long pause, `/new` command, semantic shift).
   - For each (user_query, bot_response) pair, capture the user's next action
     as implicit feedback signal:
     - User rephrases the same question -> score 0.0 (bad response)
     - User says "thanks" / moves to new topic -> score 1.0 (good response)
     - User gives up / leaves -> score 0.0 (bad response)
     - User asks a clarifying follow-up -> score 0.5 (partial)

2. **Judge module** (`judge.py`):
   - `dspy.Signature`: `conversation_context, bot_response, user_followup -> score: float, feedback: str`
   - `dspy.ChainOfThought` predictor for the judge so it reasons about why
     a response was good/bad.
   - **Pre-wired tier**: judge runs on `lm_normal`.
   - The judge itself can be bootstrapped with a small hand-labeled seed set
     (20-30 examples) using `dspy.BootstrapFewShot`.

3. **`review_conversations` tool** (in `optimize_tool.py`):
   - The agent calls this tool to run the judge over recent session logs.
   - Returns a summary: number of turns scored, average score, worst turns
     with the judge's feedback, and the path to the full labeled dataset.
   - The agent can read the dataset, inspect specific bad turns, and decide
     whether to proceed with optimization.

4. **Validation set**:
   - Hold out 20% of labeled data for evaluation.
   - Use `dspy.Evaluate` to measure judge consistency (compare LLM judge
     scores against the implicit signal baseline).

### Value

- Automatic, continuous collection of training signal from real usage.
- No manual labeling required beyond the initial seed set.
- The labeled dataset feeds directly into Phase 6.
- The agent decides when to run the judge, not a cron job.

---

## Phase 6: Prompt Optimization (SIMBA + GEPA)

**Effort:** heavy (3-5 days)
**Depends on:** Phase 4 (for optimizable module), Phase 5 (for training data)
**Files:** new `nanobot/optimization/optimize.py`, extends `optimize_tool.py`

Use DSPy optimizers to improve nanobot's prompts, instructions, and few-shot
examples based on the labeled conversation data from Phase 5. The agent
triggers optimization and decides what to do with the results.

### Optimization Targets

With Phase 4 complete, there are two `dspy.Module`s to optimize:

1. **`NanobotReAct`** -- the main agent turn. This is the primary target.
   SIMBA/GEPA optimize the agent's instructions, tool-selection reasoning,
   and few-shot demos. Because the module uses `dspy.History`, optimized
   demos naturally include multi-turn conversational context.

2. **Memory consolidation** -- wrap `_consolidate_memory`'s prompt as a
   `dspy.Predict` module (secondary target, simpler). Optimize what gets
   extracted into MEMORY.md and HISTORY.md.

### Agent-Driven Optimization Tools

The optimization pipeline is exposed as tools the agent can invoke, inspect,
and act on. The agent is in control -- nothing is auto-applied.

#### Tool: `review_conversations` (Phase 5)

Run the judge over recent session logs. Returns a report.

```
Agent: "Let me review how my recent conversations went."
-> review_conversations()
-> Returns:
   Reviewed 142 turns across 23 sessions.
   Average score: 0.72
   Turns scoring below 0.3: 18
   Top issues:
   - 6 turns: user had to rephrase Jira-related questions
   - 4 turns: incomplete answers about deployment procedures
   - 3 turns: missed context from earlier in conversation
   Full dataset: workspace/optimization/labeled_2026-02-13.json
```

The agent reads the report and decides whether there's enough signal to
optimize.

#### Tool: `optimize_prompts`

Run SIMBA and/or GEPA on the labeled data. Returns proposed changes as a
structured result -- does NOT auto-apply. **Pre-wired tier**: runs on
`lm_deep` (this is a high-stakes, infrequent operation that benefits from
maximum reasoning capability).

```
Agent: "There are enough bad Jira turns. Let me run optimization."
-> optimize_prompts(optimizer="simba", dataset="labeled_2026-02-13.json")
-> Returns:
   Optimization complete. Validation score: 0.72 -> 0.81

   Proposed RULES (3 new):
   1. "When summarizing Jira issues, always include the issue key (e.g. PROJ-123)"
   2. "If the user asks about issue priority, check the priority field explicitly"
   3. "For deployment questions, read MEMORY.md first for project-specific context"

   Proposed DEMOS (2 new):
   1. [user: "what are the P1 bugs?" -> good response with keys and summaries]
   2. [user: "summarize OSPRH-19963" -> good response with full context]

   Proposed SOUL.md changes:
   - Added to Communication Style: "Include identifiers (issue keys, PR numbers)
     when referencing external resources"

   Nothing has been written yet. Review and apply with your filesystem tools.
```

The agent then decides:

```
Agent thinks: "Rules 1 and 2 look good. Rule 3 is too generic, I'll skip it.
              Both demos are useful. The SOUL.md change makes sense."

-> write_file("RULES.md", ...) -- writes rules 1 and 2
-> write_file("DEMOS.md", ...) -- writes both demos
-> edit_file("SOUL.md", ...) -- applies the communication style addition
```

Or the agent might reject everything:

```
Agent thinks: "The validation improvement is marginal (0.72 -> 0.74),
              and these rules seem too specific. I'll wait for more data."
```

Or the agent might edit a rule before applying:

```
Agent thinks: "Rule 1 is good but I want to generalize it."
-> Writes to RULES.md: "When referencing external resources (Jira issues,
   PRs, Confluence pages), always include the identifier."
```

#### Tool: `evaluate_self`

Run the current agent configuration against the validation set to measure
baseline performance. Useful before and after applying changes.

```
Agent: "Let me check my current score before changing anything."
-> evaluate_self(dataset="labeled_2026-02-13.json")
-> Returns: Current validation score: 0.72 (142 turns)

[agent applies some changes]

Agent: "Did that help?"
-> evaluate_self(dataset="labeled_2026-02-13.json")
-> Returns: Current validation score: 0.79 (142 turns)
```

### Implementation

1. **`optimize_tool.py`** exposes three tools:
   - `review_conversations(sessions_path?, since?)` -- judge pipeline,
     runs on `lm_normal`.
   - `optimize_prompts(optimizer, dataset, target?)` -- run SIMBA/GEPA,
     runs on `lm_deep`.
   - `evaluate_self(dataset)` -- score current config, runs on `lm_normal`.

2. All three return text reports -- no files are written automatically.
   The agent uses its existing `write_file` / `edit_file` tools to apply
   changes it agrees with.

3. Proposed changes are also saved to `workspace/optimization/proposals/`
   as timestamped JSON so the agent (or user) can review them later.

4. Add `"RULES.md"` and `"DEMOS.md"` to `ContextBuilder.BOOTSTRAP_FILES`
   so they're automatically included in the system prompt when present.

### Optimizer Details

**SIMBA** (recommended first):
```python
simba = dspy.SIMBA(
    metric=judge_metric,
    bsize=32,
    num_candidates=6,
    max_steps=8,
    max_demos=4,
)
optimized = simba.compile(
    student=nanobot_react,
    trainset=labeled_data,
    valset=validation_data,
)
```
- Finds hard examples (the rephrase pairs), introspects, generates rules.
- Output: proposed rules + selected few-shot demos.

**GEPA** (alternative/complement):
```python
gepa = dspy.GEPA(
    metric=gepa_feedback_metric,
    auto_run="medium",
)
result = gepa.compile(
    student=nanobot_react,
    trainset=labeled_data,
    valset=validation_data,
)
```
- Genetically evolves instruction text with per-predictor feedback.
- Output: proposed SOUL.md rewrite.

**Stacking**: The agent can run GEPA first, apply instruction changes, then
run SIMBA to add rules and demos on top. Or vice versa. The agent decides.

### Value

- The agent controls its own self-improvement loop.
- Nothing is auto-applied -- the agent reviews, edits, and selectively
  applies proposed changes.
- Everything is in plain Markdown -- no opaque state.
- The agent can also manually add rules or demos without running the
  optimizer (e.g., "I notice I keep getting timezone questions wrong" ->
  adds a rule to RULES.md directly).
- The user can review and override at any time by editing the same files.
- Optimization becomes a conversation: "review my recent performance" ->
  "run optimization" -> "apply these two rules" -> "check if it helped."
- Optimization runs on `lm_deep`, so the agent's best reasoning is applied
  to its own improvement.

---

## Phase 7: GRPO Finetuning (optional, future)

**Effort:** heavy (5+ days)
**Depends on:** Phase 4, Phase 5, finetuning infrastructure for local GLM
**Files:** new `nanobot/optimization/finetune.py`

Use `dspy.GRPO` for RL-based finetuning of the local GLM-4.7-Flash model
using the labeled conversation data. This is the most impactful but also
most infrastructure-heavy change.

### Prerequisites

- The local `llama-server` must expose a finetuning API (or use an external
  finetuning service).
- Sufficient conversation data (~1000+ labeled examples).
- GPU resources for training.

### Tasks

1. Configure `dspy.LM` with finetuning support for the local model.
2. Set up GRPO with the judge metric and training data.
3. Train and evaluate against the validation set.
4. Deploy the finetuned model back to `llama-server`.
5. Potentially replace `lm_quick` with the finetuned model -- a local model
   optimized for nanobot's specific tasks would be an ideal quick tier.

### Value

- The local fallback model gets meaningfully better at nanobot's specific
  tasks, reducing dependence on cloud API rate limits.
- Trained on real conversation patterns, not generic benchmarks.
- A finetuned local model as `lm_quick` would make the quick tier both
  fast and specifically good at nanobot's common subtasks.

---

## Implementation Order (recommended)

```
Week 1:  Phase 1 (LM layer + 3 tiers) + Phase 2 (observability)
         Immediate value: token tracking per tier, latency logs.

Week 2:  Phase 3 (RLM tool using quick for sub-queries)
         + Phase 5 (judge pipeline + review_conversations tool)
         Large-context capability + start collecting training data.

Week 3:  Phase 4 (ReAct + History + think tool refactor)
         Core agent loop becomes a dspy.Module with tier selection.

Week 4:  Phase 6 (SIMBA/GEPA + optimize_prompts/evaluate_self tools)
         Agent-driven optimization using deep tier.

Later:   Phase 7 (GRPO) when enough data + infra is ready.
```

Each phase is independently valuable and can be shipped/tested before
starting the next. Phase 4 is the keystone: it makes Phase 6 dramatically
more valuable by making the entire agent turn optimizable, not just
individual sub-steps.

---

## Architecture Diagram

```
  user message
       |
       v
  AgentLoop.run()                   <-- nanobot (unchanged)
       |
  SessionManager                    <-- nanobot (unchanged)
       |
  build dspy.History                <-- Phase 4: bridge layer
       |
       v
  NanobotReAct(dspy.Module)         <-- Phase 4: core refactor
    |         |
    |    dspy.ReAct loop             <-- thought -> tool -> observation
    |    runs on lm_normal
    |         |
    |    dspy.Tool wrappers          <-- Phase 4: thin wrappers
    |      |   |   |   |   |
    |      |   |   |   |   |
    |      v   v   v   v   v
    |    fs  shell web MCP  think    <-- nanobot tools (unchanged)
    |                        |            + new think tool
    |                   lm_quick or
    |                   lm_deep
    |         |
    |    RLM tool                    <-- Phase 3
    |    outer: lm_normal
    |    sub_lm: lm_quick
    |
    v
  dspy.LM tiers -> LiteLLM proxy    <-- Phase 1: LM layer
    lm_quick   (fast/cheap)
    lm_normal  (default)
    lm_deep    (reasoning)
       |
  NanobotCallback                    <-- Phase 2: observability
  dspy.track_usage() per tier
       |
  dspy.Refine                        <-- Phase 4: inference-time retry
       |
       v
  Session logs (*.jsonl)             <-- nanobot (unchanged)
       |
  review_conversations tool          <-- Phase 5: agent-invoked
  (judge on lm_normal)
       |
  optimize_prompts tool              <-- Phase 6: agent-invoked
  (SIMBA/GEPA on lm_deep)
       |
  evaluate_self tool                 <-- Phase 6: agent-invoked
  (scoring on lm_normal)
       |
       v
  Agent decides:                     <-- agent is in control
  write_file / edit_file
  -> RULES.md, DEMOS.md, SOUL.md
       |
  GRPO finetuning                    <-- Phase 7: future
  (finetuned model -> lm_quick)
```

---

## Tier Usage Summary

| Operation | Tier | Rationale |
|-----------|------|-----------|
| Main agent turn (ReAct) | `normal` | Default conversation quality |
| `think(mode="quick")` | `quick` | Agent delegates simple subtask |
| `think(mode="deep")` | `deep` | Agent escalates hard problem |
| RLM sub-queries | `quick` | Many calls, need speed |
| RLM outer reasoning | `normal` | Needs decent planning |
| Memory consolidation | `quick` | Routine summarization |
| Cron jobs / heartbeat tasks | configurable | Per-job tier, default `quick` |
| `review_conversations` judge | `normal` | Needs decent judgment |
| `optimize_prompts` SIMBA/GEPA | `deep` | High-stakes, rare, needs best reasoning |
| `evaluate_self` scoring | `normal` | Just running the metric |
| `Refine` retry attempts | `normal` | Same tier as main loop |

The agent learns optimal tier usage through SIMBA optimization: if labeled
data shows wasted deep calls on easy tasks, SIMBA generates a rule. If it
shows the agent struggled on tasks it should have escalated, another rule.

---

## Dependencies

- `dspy>=3.1.3` (pulls in `litellm`, `pydantic`, `anyio`)
- `gepa[dspy]>=0.0.26` (only needed for Phase 6 GEPA, optional)
- Deno runtime (only needed for Phase 3 RLM, the PythonInterpreter sandbox)
