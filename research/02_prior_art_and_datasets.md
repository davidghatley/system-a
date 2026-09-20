# Prior Art and Public Agent-Trajectory Datasets

Research date: 2026-09-20. This survey prioritizes public artifacts and primary sources, with emphasis on releases from 2025-2026. The target is data that can train a model to choose the **next consequential agent action from the current task, history, and environment observation**, not merely to name a tool, route a request, score an answer, or imitate a final response.

## Executive finding

There is abundant public tool-use and agent data, but much less clean data for **trajectory control-policy distillation**. The best currently inspectable sources are:

1. **Nebius SWE-agent trajectories** for large-scale, outcome-labelled coding trajectories, including failures.
2. **Kwai-Klear SWE-smith mini-swe-agent-plus 66K** for newer, large-scale harness-specific coding SFT and reported downstream scaling.
3. **Hermes Agent Reasoning Traces** for broad real-tool execution across terminal, browser, repository, file, and delegation tasks.
4. **Exgentic Multi-Benchmark LLM Agent Traces** for cross-domain and cross-harness comparisons with full per-call inputs/outputs and tool definitions.
5. **AgentTrek** as the strongest web-specialist complement, subject to resolving its missing dataset license and weaker outcome metadata.

No inspected public corpus is a turnkey general policy dataset. A usable training mixture requires reconstructing turn-level `(state, available actions, action, observation, outcome)` examples, normalizing harness actions, preserving failed trajectories, and preventing benchmark leakage.

## What counts as control-policy data

The following categories are often conflated:

| Category | Supervised target | Typical evidence | Is it trajectory control-policy distillation? |
|---|---|---|---|
| Routing | Which model, agent, skill, or workflow receives a request | Request plus route label | No. It chooses a component, not the environment action sequence. |
| Tool choice | Which API/function to call, often without longitudinal state | Prompt plus tool schema and one call | Usually no. It becomes policy data only when embedded in repeated observation/action transitions. |
| Next-agent-action | Next message, tool call, command, click, or delegate action conditioned on history | Full prefix plus next action | Yes at the behavioral-cloning level, if observations and action semantics are preserved. |
| Judging/verifying | Score, preference, critique, pass/fail, or patch ranking | Candidate output plus label | No, but useful for reward models, filtering, or outcome-conditioned training. |
| Trajectory control-policy distillation | Repeated choices that actually change an environment, with resulting observations and preferably terminal outcomes | Task, state/history, available actions, executed action, observation, terminal reward | Yes. This is the target class. |

Chain-of-thought alone is not a trajectory. A list of tool names is not an action policy. OpenTelemetry timing spans are not sufficient unless the input messages, output action, tool definitions, and resulting observations can be ordered and linked. Successful-only demonstrations support imitation but cannot teach recovery or discriminate good from bad behavior as directly as mixed-outcome rollouts.

## Shortlist

| Dataset | Why shortlist it | Principal limitation | Best role |
|---|---|---|---|
| [nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) | 80,036 full SWE-agent trajectories; explicit solved label, exit status, patch, and evaluation logs; both successes and failures | Coding-only; 2024 generation; mixed model-output license obligations and source-repository licenses | Backbone for outcome-aware coding policy and verifier experiments |
| [Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k](https://huggingface.co/datasets/Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k) | 65,994 newer end-to-end coding conversations; real harness observations; MIT; published downstream data-scaling result | Card exposes only `instance_id` and messages; no per-row reward, teacher, or failure/success field documented | High-volume coding behavior cloning in a compact harness |
| [lambda/hermes-agent-reasoning-traces](https://huggingface.co/datasets/lambda/hermes-agent-reasoning-traces) | 14,701 real-execution trajectories, 174,550 tool calls, explicit tool definitions, broad task taxonomy, Apache-2.0 | No terminal reward/evaluator field; generated tasks and retained reasoning need quality and policy review | Generalist next-action SFT and action-taxonomy transfer |
| [Exgentic/agent-llm-traces](https://huggingface.co/datasets/Exgentic/agent-llm-traces) | 1,781 rich traces over six benchmarks, five harnesses, and multiple frontier/open teachers; exact LLM inputs/outputs/tool schemas in spans | Small, very token-heavy, reward fields are not documented in the public schema; reconstruction is nontrivial | Cross-harness normalization, ablations, and mixed-domain policy seed |
| [xlangai/AgentTrek](https://huggingface.co/datasets/xlangai/AgentTrek) | 52,594 web-agent training rows/turns from tutorial-guided replay; browsing/shopping/retrieval action sequences | No license declared on Hub; one-column flattened conversations; no explicit outcome/reward in schema | Web-policy specialist after provenance/license review |

If legal simplicity and direct outcome supervision dominate, use the first four and replace AgentTrek with a separately collected licensed web corpus. If broad generality dominates, Hermes plus Exgentic are essential even though their outcome supervision is weaker.

## Detailed dataset assessments

### 1. Nebius SWE-agent trajectories

**Primary sources:** [dataset card and schema](https://huggingface.co/datasets/nebius/SWE-agent-trajectories), [collection/training technical blog](https://nebius.com/blog/posts/training-and-search-for-software-engineering-agents), [SWE-agent harness](https://github.com/SWE-agent/SWE-agent).

- **Teacher(s):** multiple action-generator models. The public card does not provide a complete teacher list in prose; each row has `model_name`. The card separately warns that use of model outputs must comply with the Llama 3.1 license.
- **Harness:** a software-engineering agent based on SWE-agent.
- **Size:** 80,036 trajectories, about 5.63 GB uncompressed. The card reports 13,389 resolved and 66,647 unresolved trajectories.
- **Decision points:** model reasoning/actions under `role: ai`, interleaved with environment observations under `role: user`; average 31.3 steps for resolved and 58.4 for unresolved attempts.
- **Domains/actions:** GitHub issue resolution from SWE-bench dev and `nebius/SWE-bench-extra`; shell/navigation/search/edit/test/submit actions exposed by the SWE-agent environment.
- **Observations:** tool/environment text is retained in the trajectory; system prompt is retained. This makes turn-prefix reconstruction plausible.
- **Outcomes/rewards:** explicit Boolean `target`, `exit_status`, final patch, and test `eval_logs`. Reward is terminal and binary rather than dense.
- **Candidate reconstruction:** parse each trajectory into assistant action and following environment response; train next action on all prefixes. Attach terminal success to every transition only for outcome conditioning or return-weighting, not as a claimed dense reward. Retain failed runs for recovery and contrastive sampling.
- **License:** CC BY 4.0 at dataset level. The card requires respecting each source repository's license and flags Llama 3.1 output-license obligations.
- **Quality/reasoning:** strongest public outcome metadata in this survey. Failures are longer and likely contain loops/context exhaustion, which is useful but requires deduplication and loop filtering. The `mask` field should be inspected before deciding which turns receive loss. Raw hidden reasoning may include model-specific artifacts.
- **Contamination:** instances derive from public GitHub issues and SWE-bench-family data. Do not mix overlapping repositories/issues into train and SWE-bench evaluation. Split by issue/repository and audit exact `instance_id` overlap.
- **Suitability:** excellent domain policy corpus for coding; weak as a sole general policy corpus.

### 2. Kwai-Klear SWE-smith mini-swe-agent-plus trajectories 66K

**Primary sources:** [dataset card/schema](https://huggingface.co/datasets/Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k), [mini-swe-agent-plus repository](https://github.com/Kwai-Klear/mini-swe-agent-plus), [SWE-smith source dataset](https://huggingface.co/datasets/SWE-bench/SWE-smith).

- **Teacher(s):** not stated on the dataset card. The trained student is described as Qwen3-8B/Klear-AgentForge-8B-SFT; that is not evidence that Qwen3-8B generated the trajectories.
- **Harness:** `mini-swe-agent-plus` on issues derived from SWE-smith.
- **Size:** 65,994 end-to-end trajectories, about 4.64 GB uncompressed and 1.58 GB Parquet.
- **Decision points:** interleaved `messages[{role, content}]`; the card does not publish aggregate turn count.
- **Domains/actions:** repository issue resolution through the harness's terminal/file-edit workflow.
- **Observations:** embedded as conversation messages. Tool schema and structured call IDs are not separate dataset columns.
- **Outcomes/rewards:** no reward, success flag, patch, or evaluator column is documented. The card reports the downstream student reaches 39.0% on SWE-bench Verified and that performance rises approximately linearly with log data scale from 1K to 66K, but that is corpus-level evidence, not row-level reward.
- **Candidate reconstruction:** role-aware message parsing can produce next-action examples. Recovering action/observation boundaries requires validating the harness serialization against samples. Outcome-aware filtering is not possible from documented columns alone.
- **License:** MIT according to Hub metadata. Source issue/repository licensing should still be audited.
- **Quality/reasoning:** unusually valuable evidence that trajectory scale helps an 8B coding student. Absence of documented teacher, selection procedure, and row-level outcomes makes quality weighting difficult.
- **Contamination:** SWE-smith is generated from real repositories; enforce repository/time/issue separation from SWE-bench Verified and any downstream coding benchmark.
- **Suitability:** very strong coding behavior-cloning corpus; unsuitable by itself for outcome-model or generalist policy training.

### 3. Hermes Agent Reasoning Traces

**Primary sources:** [dataset card/schema](https://huggingface.co/datasets/lambda/hermes-agent-reasoning-traces), [Hermes Agent and generator](https://github.com/nousresearch/hermes-agent).

- **Teachers:** Moonshot Kimi-K2.5 (7,646 samples) and ZhipuAI GLM-5.1-FP8 (7,055).
- **Harness:** Hermes Agent and `hermes-agent-generator`, with actual terminal, file, browser, repository, memory, delegation, planning, and scheduling tools.
- **Size:** 14,701 trajectories; 320,716 conversation turns; 174,550 tool calls. Kimi averages 24.3 turns/13.9 tool calls, GLM 19.1/9.7.
- **Decision points:** each `gpt` message can contain `<think>` and `<tool_call>` blocks; each execution result appears in a tool message. Tool definitions are supplied per example.
- **Domains/actions:** nine categories: Terminal & Coding, Agent Tools, Repository Tasks, Browser Automation, Multi-Tool, File Operations, Scheduling, Planning & Organization, and Conversational.
- **Observations:** actual `<tool_response>` outputs, not language-model-fabricated tool results according to the card.
- **Outcomes/rewards:** no explicit success, grader score, or terminal reward field in the six-column schema (`id`, conversations, tools, category, subcategory, task).
- **Candidate reconstruction:** deterministic extraction of every tool-call prefix and matching tool response is feasible. Strip or separately gate `<think>` tokens; retain tool schemas and category. Add post-hoc executable checks only where the original environment can be recreated.
- **License:** Apache-2.0.
- **Quality/reasoning:** high action volume and genuine observations are major strengths. The Kimi and GLM reasoning distributions differ sharply (reported average thinking depth 414 versus 70 words), creating teacher-style confounding. The community filtered fork reduces Kimi rows from 7,646 to 3,679, evidence that structural filtering is worthwhile, but that fork's filter is not a substitute for outcome evaluation.
- **Contamination:** tasks include GitHub repositories and web interaction; exact source/task generation and benchmark overlap need hash and URL audits. Exposed chain-of-thought also raises provider-policy and privacy review requirements even when the dataset license is permissive.
- **Suitability:** best broad action-imitation candidate; weaker for reward-conditioned control because outcomes are absent.

### 4. Exgentic Multi-Benchmark LLM Agent Traces

**Primary source:** [dataset card, schema, and trace format](https://huggingface.co/datasets/Exgentic/agent-llm-traces).

- **Teachers:** DeepSeek-V3.2, Kimi-K2.5, Claude Opus 4.5, Gemini 3 Pro Preview, and GPT-5.2 appear in the published model table. Coverage differs by benchmark/harness.
- **Harnesses:** Claude Code, `openai_solo`, generic tool calling, tool calling with shortlisting, and `smolagents_code`.
- **Size:** 1,781 session traces, 39 Parquet shards, about 984 MB. Sessions span AppWorld (406), BrowseCompPlus (133), SWE-bench (391), tau2-airline (196), tau2-retail (469), and tau2-telecom (186).
- **Decision points:** OpenTelemetry spans retain each LLM call's complete input messages, output messages, tool definitions, model, token use, finish state, timestamps, and errors. Median turns range from 7 to 35 by benchmark; maximum is 158.
- **Domains/actions:** personal assistant/API use, deep web research, software engineering, airline/retail service, and telecom support. Actions vary by harness and include direct function calls, code-agent commands, and shortlisted tool calls.
- **Observations:** because later inputs include conversation history and tool outputs, transitions can be recovered by ordering spans, but the public schema has no explicit parent span in the Hub preview even though the card's example documents one. Reconstruction must be tested against raw rows.
- **Outcomes/rewards:** the card reports failed LLM-call percentages, which are operational errors, not task rewards. No benchmark reward/result field is documented in the eight-column dataset schema. Do not treat span success as task success.
- **Candidate reconstruction:** order spans within `session_id`; extract output tool calls and locate their results in subsequent inputs/messages. Canonicalize equivalent tools across five harnesses while preserving the original tool schema. Join benchmark rewards only from authoritative run artifacts if task IDs are recoverable.
- **License:** CDLA-Permissive-2.0.
- **Quality/reasoning:** uniquely useful for harness/model comparison and action normalization. It is extremely context-heavy: the card reports roughly 106K-1.67M average successful-call tokens per trace depending on benchmark, so naive prefix duplication is wasteful. Some provider reasoning may be omitted or summarized by APIs.
- **Contamination:** all six named benchmarks are evaluation suites. Training on these traces invalidates evaluation on overlapping tasks. Preserve held-out task IDs and treat benchmark-derived traces as contaminated for headline scores.
- **Suitability:** strongest general cross-harness seed, but too small and weakly rewarded to be the sole policy corpus.

### 5. AgentTrek

**Primary sources:** [dataset card/schema](https://huggingface.co/datasets/xlangai/AgentTrek), [ICLR 2025 paper](https://openreview.net/forum?id=EEgYUccwsV), [official repository](https://github.com/xlang-ai/AgentTrek).

- **Teacher:** the method synthesizes trajectories by guiding replay with web tutorials. The Hub card does not name the generating model; verify the exact paper/configuration before assigning teacher provenance.
- **Harness:** AgentTrek web-agent collection pipeline; tutorial instructions guide replay in a browser environment.
- **Size:** 52,594 Hub rows labelled by the card as dialogue turns; one `messages[{role, content}]` column; about 431 MB Parquet.
- **Decision points:** browser action/reaction dialogue serialized into messages. The one-column schema omits explicit task IDs, tool schemas, and reward fields.
- **Domains/actions:** web browsing, shopping, product comparison, information retrieval, and site navigation.
- **Observations:** browser feedback is represented in the dialogue, but fidelity and action grounding must be inspected from samples before conversion.
- **Outcomes/rewards:** no explicit outcome/reward column documented on Hub.
- **Candidate reconstruction:** segment message lists into observation-action pairs; infer neither success nor executable action type unless explicitly encoded. Tutorial and URL metadata may need recovery from the original release files/repository.
- **License:** **unknown/not declared on the Hub card**. Do not train or redistribute until the project authors' applicable data license is confirmed.
- **Quality/reasoning:** guiding replay is more grounded than free-form trajectory synthesis and provides web-domain diversity. Tutorial imitation can bias paths toward scripted behavior and does not guarantee task completion.
- **Contamination:** web tutorials and public benchmark sites may overlap web-agent tests; preserve URLs/domains and split by site/task template.
- **Suitability:** promising specialist web policy data, not a legally or evaluatively clean default until gaps are resolved.

### 6. AgentBank

**Primary sources:** [public mirror/schema](https://huggingface.co/datasets/Solaris99/AgentBank), [paper](https://aclanthology.org/2024.findings-emnlp.116/).

- **Teacher/harness:** heterogeneous; AgentBank aggregates and synthesizes interaction trajectories across many existing environments rather than one teacher/harness. Provenance must be resolved per configuration from the paper and upstream source.
- **Size:** 53,205 rows across 19 configurations in the inspected Hub mirror. Examples include ALFRED 623, ALFWorld 3,321, APPS 4,408, HotpotQA 4,273, InterCode SQL 4,522, Mind2Web 7,770, WebArena 658, and WebShop 4,958.
- **Decision points/domains:** embodied/text worlds, coding, math/reasoning, QA, shell/SQL, and web actions. Some subsets are real multi-turn interactions; others are conventional solution dialogues.
- **Observations/outcomes:** retained in `conversations`; no common reward field exists in the mirror schema. `meta_information` appears only for some configs.
- **Candidate reconstruction:** process per source, not through one universal parser. Classify rows into genuine environment trajectories versus static QA/reasoning before training.
- **License:** Apache-2.0 on the mirror, but upstream component licenses and benchmark terms still apply.
- **Quality/contamination:** exceptional breadth but old, heterogeneous, and heavily benchmark-derived. It risks teaching answer generation under the label “agent trajectory.”
- **Suitability:** useful auxiliary generalist mixture; not a clean modern control-policy backbone.

### 7. SWE-Gym OpenHands SFT Trajectories

**Primary sources:** [dataset card/schema](https://huggingface.co/datasets/SWE-Gym/OpenHands-SFT-Trajectories), [SWE-Gym paper](https://openreview.net/forum?id=Cq1BNvHx74), [OpenHands](https://github.com/All-Hands-AI/OpenHands).

- **Teacher/harness:** OpenHands trajectories; the minimal card does not identify the teacher model.
- **Size:** 491 rows in `train.success.oss`, one messages column, 31.8 MB uncompressed.
- **Decision points:** OpenHands conversation turns containing actions and observations.
- **Domains/actions:** software issue resolution with shell/file/editor interactions.
- **Observations/outcomes:** split name indicates successful open-source trajectories; there is no explicit reward field in the published schema.
- **Candidate reconstruction:** straightforward message-prefix SFT after validating action serialization. All rows appear success-filtered, so combine with failures elsewhere.
- **License:** MIT.
- **Quality/contamination:** small and success-only; valuable as a clean seed or harness adapter, not a standalone corpus. SWE-Gym/SWE-bench overlap must be controlled.
- **Suitability:** domain-specific supplementary data.

### 8. AgentSuite tau-bench trajectories

**Primary sources:** [dataset card](https://huggingface.co/datasets/AgentSuite/tau-bench-trajectories), [tau-bench repository](https://github.com/sierra-research/tau-bench), [tau-bench paper](https://openreview.net/forum?id=roNSXZpUDN).

- **Teachers:** 30 named models, including GPT-4/5 families, Claude 4/4.5, Gemini 2.5, DeepSeek V3/R1, Kimi K2, Qwen3, and GPT-OSS.
- **Harness:** tau-bench's tool-calling conversational agent and user simulator.
- **Size:** nominally 30 models x 165 tasks = 4,950 runs, one JSONL file per model. Hub Dataset Viewer generation currently fails, so this count comes from the card rather than a parsed table.
- **Decision points:** `messages` across assistant, simulated user, and business tools; fields include sampling settings and `eval_result`.
- **Domains/actions:** airline and retail customer support; database-backed policy-constrained tool calls plus user communication.
- **Observations/outcomes:** tool results and user replies in messages; `eval_result` provides outcome data, but its exact structure must be inspected from raw files.
- **Candidate reconstruction:** extract next assistant tool call/message from each history; preserve dual-control user turns; derive reward only from documented `eval_result` semantics.
- **License:** no dataset license declared on the inspected card. tau-bench code license does not automatically license generated model outputs.
- **Quality/contamination:** broad model comparison on identical tasks is ideal for preference/contrastive studies, but benchmark-task training precludes clean tau-bench evaluation. User-simulator variance is another confound.
- **Suitability:** promising customer-service policy and judging data once license and raw schema are verified.

### 9. Gelato OSWorld agent trajectories

**Primary sources:** [dataset repository/schema](https://huggingface.co/datasets/mlfoundations/gelato-osworld-agent-trajectories), [OSWorld](https://github.com/xlang-ai/OSWorld).

- **Teacher/harness:** not documented in the dataset card; “Gelato” naming is insufficient evidence of a specific model or control algorithm.
- **Size:** 13,500 image rows, about 5.0 GB Parquet, grouped into roughly 369 UUID class labels in the inspected schema.
- **Decision points/domains:** desktop GUI interaction across OSWorld applications; screenshots are preserved.
- **Observations/outcomes:** the Hub-visible schema has only `image` and UUID `label`. No actions, ordering, task text, reward, or outcome are documented in the card.
- **Candidate reconstruction:** not safely possible from the Hub table alone. The underlying image-folder paths/manifests would need to establish frame order, action association, task, and terminal score.
- **License:** MIT in Hub metadata.
- **Quality/contamination:** screenshots are valuable observations, but calling this a policy dataset without action metadata would be an error. OSWorld benchmark overlap invalidates evaluation on included tasks.
- **Suitability:** potential GUI observation source; **not shortlisted** until manifests/actions are verified.

### 10. HAL traces and Agent-Eval-Refine trajectories

**Primary sources:** [HAL traces repository](https://huggingface.co/datasets/agent-evals/hal_traces), [HAL](https://hal.cs.princeton.edu/), [Agent-Eval-Refine trajectory repository](https://huggingface.co/datasets/Agent-Eval-Refine/Agent-Trajectories).

- **Coverage:** HAL publishes benchmark-evaluation traces and Agent-Eval-Refine bundles WebArena GPT-4 CoT, Android Auto-UI/CogAgent/human demonstrations, and other evaluated trajectories.
- **Verification status:** HAL has no README and its Hub Dataset Viewer currently fails; Agent-Eval-Refine is a ZIP-oriented artifact with sparse Hub metadata and no declared license visible in the search/card metadata.
- **Use:** useful evaluation/debugging prior art and possibly reconstruction material after downloading and auditing raw archives.
- **Why not shortlisted:** insufficient public schema, license, teacher/run metadata, and normalized outcomes for immediate policy-distillation use.

## Important near-misses and category errors

### Tool-use instruction corpora

ToolBench, ToolAlpaca, API-Bank, Gorilla/OpenFunctions, xLAM function-calling data, ToolACE, and similar corpora are important for API grounding and function-call syntax. Most examples primarily supervise tool selection, arguments, or short ReAct exchanges. Unless a release preserves repeated real execution observations and terminal outcomes, it should be treated as **tool choice**, not full trajectory-control data. These can initialize action syntax but should not dominate a control-policy mixture.

### Routing and multi-agent delegation data

RouterBench-like model routing, skill classifiers, MoA orchestration labels, and supervisor-to-worker assignments train **routing**. Even a “next agent” label is only a control action if delegation is one action within a longer observed trajectory and the delegated result returns to the policy state. Synthetic enterprise workflow/event logs such as `juliensimon/open-agent-traces` are useful for process mining and conformance, but not evidence that an agent chose actions in a live environment.

### Judges, verifiers, and preference datasets

SWE-bench patches with test outcomes, WebArena evaluators, tau-bench rewards, process-reward data, and trajectory critiques can train judges or reward models. They become policy data only after joining labels to complete action prefixes. A scalar attached only to a final answer does not reveal which intermediate decision caused success.

### Observability traces

OpenTelemetry and production traces can be excellent source material, but latency/token/status fields alone optimize operations, not behavior. Exgentic qualifies as reconstructable because it also includes messages, output tool calls, and tool definitions. Many “agent traces” datasets do not.

### GUI foundation-model datasets

Mind2Web, Android-in-the-Wild, GUIAct, SeeClick, Aguvis, UI-TARS, and OpenCUA provide important screenshots, grounding annotations, action sequences, or training mixtures. Public releases vary in whether they expose full trajectories, teachers, outcomes, and licenses. They are often excellent **domain action-grounding** data, but cannot be assumed to be outcome-labelled control trajectories. The inspected Gelato/OSWorld Hub artifact illustrates this gap: screenshots are visible, actions are not.

## Hugging Face Agent Traces support

Hugging Face added a native trace viewer on 2026-04-07. The [official Agent Traces documentation](https://huggingface.co/docs/hub/en/agent-traces) says raw sessions from Claude Code, Codex, and Pi can be uploaded without conversion; the launch announcement also names Hermes Agent and Factory Droid. The Hub auto-detects supported formats, applies `format:agent-traces`, and renders sessions, turns, tool calls, results, reasoning, and model responses in Data Studio or Storage Buckets ([announcement](https://huggingface.co/changelog/agent-trace-viewer)).

For custom harnesses, the [Session Trace Simple Format](https://huggingface.co/docs/hub/en/session-traces-format) is JSONL with:

- one session header containing `type: session`, `harness`, and unique `id`;
- one line per message with `role` and `content`;
- optional `reasoningContent`, `toolCalls`, `toolCallId`, timestamp, and model;
- tool-call/result linkage through call IDs.

This is a **storage and visualization interoperability layer**, not a dataset-quality or policy-learning standard. It has no required task definition, environment version, available-action snapshot, reward, terminal status, provenance, license, privacy audit, or train/test split. For this mission, extend it with a sidecar manifest containing at least task/source IDs, environment and harness versions, tool schemas, terminal outcome, evaluator version, teacher/model parameters, and licensing/provenance.

The official docs explicitly warn that traces may contain prompts, command output, local paths, screenshots, secrets, private code, and personal data. Public trace ingestion therefore requires secret scanning, PII review, repository authorization checks, and deduplication before any training use. Viewer support should not be mistaken for permission to train on uploaded traces.

## Recommended normalized representation

Use an append-only episode record and materialize training examples from it:

```text
episode_id, task_id, source, source_revision, license
teacher_model, harness, harness_version, environment, environment_version
initial_task, system_prompt, tools/action_space
steps[]:
  observation_before
  visible_history_or_state_reference
  reasoning_optional
  action_type, action_name, action_arguments
  observation_after
  timestamp, execution_status
terminal_status, reward, evaluator, evaluator_version, artifacts
```

Do not fill absent values by guessing. Preserve original raw records and conversion code. Produce three explicit targets:

1. `next_action`: behavioral cloning over valid action turns.
2. `outcome`: terminal success/failure or benchmark score, only where authoritative.
3. `value/advantage`: derived data with the derivation and discounting recorded, never presented as source reward.

For cross-harness training, maintain both canonical and raw actions. For example, canonical `shell(command)` may map to SWE-agent command text, Claude Code Bash, Hermes terminal, or an OpenTelemetry function call. The raw action is required for replay and audit.

## Quality, reasoning, and contamination policy

- **Keep failures.** Deduplicate loops and infrastructure failures, but preserve genuine bad decisions and recovery attempts.
- **Separate execution failure from task failure.** API errors, context exhaustion, timeout, invalid syntax, evaluator failure, and incorrect solution are different labels.
- **Do not train blindly on hidden reasoning.** Teacher reasoning can be unavailable, policy-restricted, inconsistent across providers, or causally disconnected from actions. An action-only target with short observable rationale is safer for a first pass.
- **Split by source entity.** For coding, use repository plus issue/commit time; for web, site/domain plus task template; for GUI, app plus task; for support, policy/task template and database seed.
- **Audit benchmark overlap.** SWE-bench, SWE-smith, SWE-Gym, WebArena, OSWorld, AppWorld, BrowseCompPlus, and tau-bench-derived traces contaminate evaluation on matching tasks even when prompts differ.
- **Record teacher and harness jointly.** A trajectory is generated by a model-scaffold pair. “Distilling model X” is inaccurate when the harness supplies planning, context management, retries, or tool shortlisting.
- **Replay a sample.** Before large-scale training, replay or evaluator-check a stratified sample from every source. Conversation syntax alone does not prove that observations follow actions or that terminal labels are correct.

## Suggested first experiment

Build a legally conservative mixture from Nebius, Kwai-Klear, Hermes, and Exgentic; keep domains and harnesses as explicit conditioning metadata. Train action-only SFT first, with Nebius success/failure used for stratification rather than filtering. Evaluate on temporally and entity-disjoint coding tasks plus newly authored terminal/browser tasks, not on source benchmarks. Add AgentTrek only after license confirmation and raw-schema inspection; add tau-bench only after its output license and `eval_result` semantics are verified.

## Remaining unknowns

- The exact original mission candidate list was not present in this workspace; this survey therefore verified the prominent named families discoverable in current Hub/web search and searched beyond them.
- Complete teacher lists and sampling ratios for Nebius and Kwai-Klear are not documented in their cards.
- Kwai-Klear has no documented row-level success/outcome field.
- Hermes has no documented evaluator or terminal reward.
- Exgentic does not document benchmark reward/task IDs in the Hub-visible schema.
- AgentTrek has no declared Hub license and no explicit reward field.
- AgentSuite tau-bench has no declared dataset license, and the Hub viewer currently fails.
- Gelato OSWorld lacks a substantive card and Hub-visible action/order/outcome metadata.
- HAL and Agent-Eval-Refine need raw archive inspection and license clarification before use.
- Provider terms governing redistribution/training on outputs from closed teachers must be checked independently of dataset-level licenses.

## Primary-source index

- Hugging Face, [Agent Traces](https://huggingface.co/docs/hub/en/agent-traces), [Session Trace Simple Format](https://huggingface.co/docs/hub/en/session-traces-format), and [2026 launch announcement](https://huggingface.co/changelog/agent-trace-viewer).
- Nebius, [SWE-agent trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) and [training/search report](https://nebius.com/blog/posts/training-and-search-for-software-engineering-agents).
- Kwai-Klear, [66K SWE-smith trajectories](https://huggingface.co/datasets/Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k) and [mini-swe-agent-plus](https://github.com/Kwai-Klear/mini-swe-agent-plus).
- Lambda/Nous, [Hermes Agent Reasoning Traces](https://huggingface.co/datasets/lambda/hermes-agent-reasoning-traces) and [Hermes Agent](https://github.com/nousresearch/hermes-agent).
- Exgentic, [Multi-Benchmark LLM Agent Traces](https://huggingface.co/datasets/Exgentic/agent-llm-traces).
- XLang, [AgentTrek dataset](https://huggingface.co/datasets/xlangai/AgentTrek), [paper](https://openreview.net/forum?id=EEgYUccwsV), and [code](https://github.com/xlang-ai/AgentTrek).
- AgentBank, [Hub mirror](https://huggingface.co/datasets/Solaris99/AgentBank) and [EMNLP Findings paper](https://aclanthology.org/2024.findings-emnlp.116/).
- SWE-Gym, [OpenHands SFT trajectories](https://huggingface.co/datasets/SWE-Gym/OpenHands-SFT-Trajectories) and [paper](https://openreview.net/forum?id=Cq1BNvHx74).
- AgentSuite/Sierra, [tau-bench trajectories](https://huggingface.co/datasets/AgentSuite/tau-bench-trajectories), [benchmark code](https://github.com/sierra-research/tau-bench), and [paper](https://openreview.net/forum?id=roNSXZpUDN).
- ML Foundations/XLang, [Gelato OSWorld trajectories](https://huggingface.co/datasets/mlfoundations/gelato-osworld-agent-trajectories) and [OSWorld](https://github.com/xlang-ai/OSWorld).
