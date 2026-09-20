# 2025-2026 Prior-Art Update: Typed Small-Model Next-Control Distillation

Research date: 2026-09-20

## Bottom Line

**VERIFIED:** No inspected public 2025-2026 work is essentially identical to System-A. The closest actual policy-distillation work trains small autoregressive agents on longitudinal reason/action/real-observation trajectories. Separate work provides typed probabilistic decisions or confidence-based escalation, but no verified source combines all four properties:

1. a genuinely small student;
2. next control conditioned on authentic prior actions and environment observations;
3. a finite, typed, non-generative decision head rather than token generation; and
4. calibrated selective execution that escalates the predicted control decision to a stronger policy.

**INFERRED:** System-A's defensible novelty is the conjunction, not any component alone. Agent trajectory imitation, compact tool agents, structured outputs, confidence routing, and selective classification all predate it.

**UNKNOWN:** TypeSafe has not publicly disclosed enough of Jev's architecture, training data, RLCD objective, parameter count, calibration protocol, or agent-trajectory use to determine whether an internal Jev system already implements this conjunction.

## Category Boundary

| Category | Decision learned | Why it is or is not System-A prior art |
|---|---|---|
| Router | Which model/agent handles a request or step | Adjacent when confidence triggers escalation, but it does not learn the environment action. |
| Isolated tool classifier | Tool name/arguments from one prompt | Typed action selection, but lacks longitudinal observed transitions and recovery. |
| Next-action behavioral cloning | Next reason/action from a trajectory prefix | Direct policy-distillation precedent, usually autoregressive and uncalibrated. |
| Verifier/reward model | Correctness, preference, critique, or reward | Can supervise/filter a policy but is not itself the policy. |
| Hierarchical controller | Subgoal, skill, model, or worker selection | Policy-like only if its output is an executable control in the continuing observed process. |
| Actual policy distillation | Student policy trained from teacher/environment trajectories | Closest class; System-A adds typed heads and selective escalation. |

## Ranked Closest Works

### 1. Agent Distillation: Distilling LLM Agent into Small Models with Retrieval and Code Tools (Kang et al., 2025; arXiv:2505.17612v2)

**VERIFIED:** This is the closest direct precedent. A Qwen2.5-32B teacher generates repeated thought/action/real-observation trajectories; observations come from retrieval or code execution and are excluded from the loss; 0.5B, 1.5B, 3B, and 7B students are LoRA-tuned to generate the teacher's reasoning and actions. About 2,000 trajectories per domain are used. The paper explicitly describes this as cloning full task-solving behavior and reports out-of-domain factual and mathematical evaluations. Repository HEAD inspected: `8884b80ea3d22e53a2e1e4b9fd324600d13e0430`.

**VERIFIED:** Its self-consistent action generation samples executable candidate code actions, retains a failed action only when all candidates fail, and votes over resulting observations. This is robustness through execution and sampling, not calibrated abstention or escalation.

**Difference:** It is an autoregressive reason-and-code policy, not a typed decision model. It neither estimates calibrated next-action correctness nor hands uncertain states to a stronger policy. Its action space is narrow: retrieval and Python/code tools for QA/math rather than general coding-agent controls.

**Consequence:** System-A cannot claim to introduce small-model agent-policy distillation, trajectory-prefix next-action learning, or observation-conditioned tool behavior. It may claim a typed/selective formulation if demonstrated.

### 2. SCoRe: Student-Centered Distillation Narrows the Agentic Gap Between Small and Large LLMs (2025; arXiv:2509.14257v1)

**VERIFIED:** SCoRe explicitly treats teacher trajectory imitation as behavior cloning, cold-starts on teacher trajectories, then lets the student act and has the teacher correct the first critical error. Corrected trajectories feed another SFT round, and RL begins from the correct prefix with key-step rewards. It evaluates a Qwen2.5-7B student against a 72B teacher on 12 reasoning/search benchmarks.

**Difference:** This is stronger policy learning than static offline imitation and directly addresses compounding error, but it remains a generative ReAct policy. Teacher intervention is a training procedure, not calibrated inference-time escalation. Seven billion parameters is also much larger than System-A's 421M encoder.

**Consequence:** Any claim that System-A solves distribution shift or long-horizon compounding error is unsupported by Experiment 001. Its novelty must remain offline typed imitation and selective prediction unless closed-loop corrective training is added later.

### 3. MENTOR: RL via Teacher-Optimized Rewards for Tool-Use Distillation (Park et al., 2025; arXiv:2510.18383v1)

**VERIFIED:** MENTOR trains a small tool-using policy with RL and a dense composite reward derived from a teacher reference trajectory. Its stated purpose is to improve cross-domain generalization beyond static teacher-trajectory SFT and sparse terminal reward.

**Difference:** MENTOR is actual teacher-guided policy optimization, not a router or verifier. It does not use typed non-generative outputs or calibrated reject/escalate behavior, and the teacher trajectory primarily shapes reward rather than serving as System-A-style hard typed next-control labels.

**Consequence:** System-A should describe Experiment 001 as behavioral cloning, not policy optimization or full knowledge distillation. A later outcome-aware experiment should compare against dense teacher-guided or verifier-guided RL, not only SFT.

### 4. LiteSearch-VL (2026; arXiv:2608.29357v1)

**VERIFIED:** LiteSearch-VL LoRA-tunes Qwen3-VL-2B/4B on released, tool-interleaved OpenSearch-VL trajectories and adds GPT-5-generated step-level preference pairs for five local control failures: premature answer, wrong tool, weak query, repeated query, and ignored image. Full-trajectory SFT changes the 2B model from almost never producing a usable answer to 28.4% macro Pass@1; extra search steps mostly turn abstentions into wrong-entity answers.

**Difference:** This is multimodal generative trajectory distillation with step-level preference refinement. Its `no_answer` outcome is observed behavior, not a trained calibrated reject option, and it does not escalate to a stronger policy.

**Consequence:** Merely showing that a small model learns an "agent contract" or tool syntax is not novel. Experiment 001 needs structural baselines and action-semantic tests showing task/history-conditioned control rather than harness-format acquisition.

### 5. Adaptive VLM Routing for Computer Use Agents (2026; arXiv:2603.12823v1)

**VERIFIED:** Adaptive VLM Routing operates at each computer-use tool-call step. A 120M difficulty classifier and a small VLM's normalized log-probability confidence decide whether to accept the small model or route to a larger VLM; risky actions can be sent directly to the strongest model. It reports projected cost reductions on grounding/routing data.

**Difference:** AVR supplies the closest step-level confidence/escalation mechanism but routes among action-generating models instead of learning a distilled typed action policy.

**Consequence:** System-A cannot claim confidence escalation as an invention. Its remaining novelty is calibrating a distilled typed next-control predictor itself for selective execution.

## Jev Architecture Comparator

**VERIFIED:** TypeSafe's 15 September 2026 announcement describes unstructured program state in and predeclared typed probabilistic decisions out, parallel sampling, confidence on every answer, and Reinforcement Learning for Calibrated Decisions. Its workflow evaluation uses the average of Astra and Fable as reference probabilities. These are vendor-reported decision-imitation/evaluation claims, not a published agent-policy-distillation experiment. The inspected page ETag is `23c98b3268253caab6a80b329bd77178`.

**Difference:** Jev supplies the closest output/calibration substrate but no verified trajectory distillation. It is therefore an architecture/objective comparator, not one of the five closest demonstrated policy-learning works.

**Consequence:** System-A cannot claim typed probabilistic automation as an invention. Its remaining novelty is applying a Jev/Laya-like typed predictor to authentic agent next-control prefixes and calibrating that predictor for selective execution.

## Trace-Claim Corrections

**VERIFIED:** `lambda/hermes-agent-reasoning-traces` at revision `b92885e4f0161d4b2536512710e004d4892cac6e` contains 7,646 Kimi-K2.5 and 7,055 GLM-5.1-FP8 trajectories. Its card states that `<tool_response>` values are real executions and publishes tool definitions. This is suitable source material for next-control reconstruction, but no terminal evaluator/reward is documented.

**VERIFIED:** The pinned Kimi K3 release used by Experiment 001 is not suitable for observation-conditioned policy distillation. Although its card claims arguments, results, and corrections are retained, the local full-shard audit at commit `33a874c3affbdb97e142752a9144e6624ef5bd07` found 18,291 prior tool calls and zero tool-result messages. Codex is only an acceptance verifier in that release, not a demonstration teacher. The card claim and serialized artifact therefore conflict; measured data controls.

**VERIFIED:** Exgentic revision `70036b93a04e61b0ea2706a68b962f4f26774587` provides reconstructable traces from Claude Code and other harnesses, with Claude Opus 4.5, Kimi-K2.5, DeepSeek-V3.2, Gemini, and GPT-5.2 teachers. This verifies Claude-harness/model traces, not Claude/Codex/Pi/OpenCode policy distillation.

**VERIFIED:** Hugging Face documentation says raw Claude Code, Codex, and Pi session JSONL is viewer-compatible. That is a serialization/viewer capability, not evidence of a public training corpus, outcomes, licenses, or distillation.

**VERIFIED:** The Fable-named aggregate at revision `15ba38ac01b610cc986b728a2d55fcb8db9c2096` is not a clean Fable teacher corpus. Its card claims 9,057,143 logical records, while the Hub structure exposes about 2.0M rows; it mixes static math/science/code reasoning, NVIDIA sets, personal Codex captures, 586 Claude Code records, 200 OpenCode records, and 188 Pi records. The card says personal Codex traces are not redistributable despite repository-level CC-BY-4.0 metadata. `opencode-reasoning-2` denotes NVIDIA OpenCodeReasoning data and must not be conflated with the 200 OpenCode-client traces.

**UNKNOWN:** No inspected primary artifact establishes that the named Fable or Astra models generated longitudinal agent traces suitable for System-A. In TypeSafe materials they are reference-probability models. Likewise, viewer support alone does not establish usable Codex, Claude, Pi, or OpenCode datasets.

## Remaining Novelty, Precisely

**INFERRED:** If Experiment 001 succeeds after the data blocker is repaired, the narrow contribution is:

> A roughly 421M bidirectional encoder learns a coherent finite next-control bundle from authentic observation-bearing frontier-agent trajectory prefixes, and a separately calibrated correctness score permits statistically controlled abstention/escalation on held-out trajectories within the same harness.

This excludes claims of inventing policy distillation, typed decisions, tool routing, uncertainty estimation, or escalation. It also excludes executable-policy quality, cross-harness transfer, recovery, and teacher-policy probability recovery.

**UNKNOWN:** A hard one-hot action from one successful rollout does not identify the teacher's action distribution, action optimality, or acceptable alternatives. Independent typed marginals do not identify a joint bundle probability.

## Consequences for Experiment 001

1. **RECOMMENDED:** Keep the existing stop. Do not train on the Kimi K3 artifact without authentic tool results. Its zero-observation prefixes would test action-sequence/text imitation, not observed-state control.
2. **RECOMMENDED:** Re-source the pilot from a pinned observation-bearing corpus, with Hermes Kimi/GLM as the simplest currently verified candidate and Exgentic as a smaller cross-harness candidate. Audit outcome fields, licenses, duplicates, and benchmark leakage before freezing.
3. **RECOMMENDED:** Add a parameter-budget-matched autoregressive next-action BC baseline. Agent Distillation and LiteSearch-VL make typed heads, not trajectory SFT itself, the experimental distinction.
4. **RECOMMENDED:** Preserve the repaired coherent bundle target and calibrate only the event `predicted bundle exactly matches the recorded bundle`; do not multiply independent marginal probabilities. Escalation should be selected with trajectory-disjoint calibration and simultaneous risk control.
5. **RECOMMENDED:** Treat Experiment 001 as offline recorded-action imitation. To claim useful control, run a later closed-loop comparison against generative BC and confidence routing, with task outcomes, failures, recovery states, and escalation cost.

## Evidence Limits

**VERIFIED:** Raw query results, downloaded primary HTML, derived text, response headers, repository HEADs, and Hub revision metadata are retained under `artifacts/prior_art/`. URLs, revisions, and access dates are indexed in `artifacts/prior_art/source_index.json`.

**UNKNOWN:** Several papers are arXiv preprints and results were not independently reproduced. TypeSafe claims are vendor claims. This review establishes documented method overlap, not empirical correctness.
