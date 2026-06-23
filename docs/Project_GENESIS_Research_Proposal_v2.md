# PROJECT GENESIS

## Emergent Search Algorithm Discovery via Hierarchical Reinforcement Learning Under Resource Constraints

*Where Agents Do Not Merely Select Search Strategies — They Compose New Ones*

**Target Venues:** NeurIPS | ICLR | ICRA
**Status:** Ongoing Research | May 2026

---

## 1. Abstract

Every classical search algorithm — A\*, RRT, potential fields — was designed by humans reasoning about a specific computational insight. None were designed for the problem of searching under a finite computational budget, with bounded working memory (K=16 slots), in a non-stationary adversarially generated state space. That constraint profile has no existing algorithm optimized for it.

Project GENESIS proposes a Hierarchical Neuro-Symbolic Controller operating over 13 atomic search primitives — the decomposed operations of classical search algorithms — under a finite energy budget and adversarial curriculum pressure generated via the PAIRED framework (Dennis et al., 2020). The controller does not execute search. It orchestrates it, deciding at every timestep which search operation to invoke, when to invest in information gathering versus action, and how to manage finite cognitive resources.

The central hypothesis: under sufficient pressure, the controller will compose these primitives into stable, reusable, transferable search procedures optimized for its unique constraint profile — procedures that encode insights about resource-constrained search that no existing algorithm contains. These procedures are extracted as human-readable pseudocode, validated as standalone algorithms, and benchmarked against classical baselines.

Hardware validation confirms feasibility: the target GPU (RTX 5070 Ti, 12 GB) achieves 444,000 environment steps per second with 1,024 parallel JAX-vectorized environments — 8.9x the design target.

---

## 2. The Problem

### 2.1 The Constraint Profile No Algorithm Solves

Every existing search algorithm assumes at least one of: unlimited computation, unlimited memory, a static state space, or a known state space. Real autonomous systems violate all four simultaneously. A drone with finite battery, limited onboard memory, navigating a dynamic building with no prior map cannot run A\* (too expensive), RRT to completion (too slow), potential fields (trapped by local minima), or wall-following (fails with dynamic obstacles).

The optimal search procedure for this constraint profile — finite budget, bounded memory, dynamic and adversarial state spaces — is an open problem. GENESIS is designed to discover it.

### 2.2 Why a Controller, Not a Player

A standard RL approach trains a model that directly outputs movement commands: go left, go right, move forward. This model IS the player. Its action space is directions. No matter how long you train it, it cannot discover A\*, because A\* is not a direction — it is a procedure involving graph expansion, priority queues, and heuristic evaluation. You cannot express algorithmic reasoning in an action space of movement commands.

GENESIS separates the layers. The player (execution layer) is fixed — it knows how to move, follow walls, expand graphs. The controller (reasoning layer) is learned. Its action space contains cognitive operations: "expand a graph node," "fire a sensor burst," "write a waypoint to memory," "scout ahead." These operations enable algorithmic reasoning to emerge from learning.

> [!important] The Key Distinction
> When a player solves a problem, you can ask "what path did you take?" The answer is a route — it works for one instance.
>
> When a controller solves a problem, you can ask "what procedure did you follow?" The answer is an algorithm — it works for any instance.
>
> A player discovers routes. A controller discovers algorithms. GENESIS builds the controller.

### 2.3 How This Differs From AlphaDev

AlphaDev (Mankowitz et al., 2023) discovered faster sorting algorithms by searching over assembly instructions with a binary correctness oracle. The environment is static. The search space is constrained. GENESIS has no correctness oracle, a co-evolving adversarial fitness landscape, high-level cognitive primitives (not instruction-level operations), and discovers reusable procedures across a distribution of problems the agent has never seen.

---

## 3. Core Thesis

> [!abstract] The Central Claim
> - The model is not the player. The model is the controller.
> - The search algorithms are tools. The controller learns when to use them, how to sequence them, and how to compose new ones.
> - Under adversarial pressure and computational constraints, selection evolves into composition.
> - Whatever emerges is a search algorithm: it takes a start state and finds a goal state.
> - The proof is transferability: procedures that work on unseen environments encode principles, not memorized permutations.

### 3.1 The Primitives Are Search Operations

The 13 primitives are the decomposed atomic operations of classical search algorithms instantiated in a 2D spatial domain. Graph node expansion (P3) is A\*'s inner loop. Frontier sampling (P4) is RRT's core. Memory write/read (P6/P7) is maintaining a visited set. Gradient step (P5) is greedy descent. The 2D grid world is a state space graph where search is expensive enough that resource management matters.

### 3.2 The Permutation Problem

13 primitives in sequences produce a finite combinatorial space. The test that distinguishes memorized permutations from genuine algorithms is generalization. A memorized sequence works in the context where it was learned and fails elsewhere. An algorithm captures an abstract principle that applies across contexts. Zero-shot transfer (Experiment C) is the proof.

---

## 4. Research Contributions

Four contributions. The first three are guaranteed methodological contributions. The fourth is the empirical bet.

### 4.1 Sub-Algorithmic Search Primitives as an Action Space

| ID | Primitive | Search Operation | Cost | Enables |
|----|-----------|-----------------|------|---------|
| P0 | Idle / Observe | Passive sensing | 0 | Energy management |
| P1 | Motor Primitive | Single-step state transition | 1 | Direct traversal |
| P2 | Wall Surface Follow | Traversal heuristic (one step) | 2 | Structured exploration |
| P3 | Graph Node Expand | A\* inner loop: expand best node | 3 | Optimal path planning |
| P4 | Frontier Sample | RRT inner loop: random extension | 3 | Probabilistic coverage |
| P5 | Gradient Step | Greedy descent toward goal | 2 | Fast exploitation |
| P6 | Memory Write | Store to visited set (FIFO, K=16) | 1 | Temporal reasoning |
| P7 | Memory Read | Retrieve from visited set | 1 | Backtrack planning |
| P8 | Sensor Burst | Full local state observation | 5 | Information gathering |
| P9 | Backtrack | Reverse to stored state | 2 | Error recovery |
| P10 | Scout Fork | Virtual forward sim (N steps) | 8 | Lookahead / futures |
| P11 | Commit Path | Execute path uninterrupted | 1/step | Decisive execution |
| P12 | Subgoal Set | Decompose into sub-problems | 1 | Hierarchical search |

**Primitive temporal granularity:** P0–P9 and P12 are one-tick atomic operations — the primitive executes one step and returns control to the controller immediately. P10 (Scout Fork) is multi-step by necessity: meaningful lookahead requires simulating N steps forward. P11 (Commit Path) is intentionally multi-step: it executes a stored path uninterrupted. This design maximises controller decision frequency and enables interleaved compositions.

**Critical primitives for emergence:** P6/P7 (temporal reasoning), P10 (future simulation), P12 (hierarchical decomposition). Without these, the controller can only react. With them, it can plan, simulate, and decompose — the prerequisites for algorithmic reasoning.

### 4.2 Computational Energy Budget with Dynamic Traversal Costs

Every primitive consumes energy from a finite budget B₀. Primitives exceeding remaining budget are masked. The reward includes λ·(Bₜ/B₀) for budget preservation. Success reward is set at 10–50x maximum episode cost to prevent deliberate-failure reward hacking.

**Dynamic traversal cost:** Each cell tracks a visit count. Traversal cost multiplies by (1 + visit_count). This penalty eliminates the P2 (wall-follow) spam failure mode where the agent discovers that cheap looping solves most environments. Revisit penalties force the agent to use memory (P6/P7) and lookahead (P10) to optimize routes, making cognitive primitives load-bearing rather than optional.

### 4.3 PAIRED Adversarial Curriculum

The adversary uses the PAIRED framework (Dennis et al., 2020) instead of standard POET-style failure reward. PAIRED introduces a three-player game: the Adversary generates state spaces, the Antagonist (a baseline solver) attempts to solve them, and the Protagonist (the main controller) attempts to solve them. The Adversary is rewarded proportional to regret: the gap between Antagonist performance and Protagonist performance.

This guarantees mathematically that the adversary only generates solvable, productive curriculum steps. Impossible environments earn zero reward (Antagonist fails too). Trivial environments earn zero reward (Protagonist succeeds). Only environments at the frontier of Protagonist capability — solvable but currently unsolved — earn reward. This prevents the catastrophic forgetting cycles that plague standard adversarial curriculum.

The Antagonist is a smaller, simpler network (~500K parameters) serving as a difficulty gauge, not a subject of study. Training begins against randomly generated environments (no adversary) to establish baseline competence, then PAIRED activates.

### 4.4 Three-Criteria Evaluation Framework

> [!note] What Constitutes a Discovered Search Algorithm
> A primitive sequence qualifies if and only if:
> 1. **Stability** — Deployed consistently (>80%) in a recognizable context class.
> 2. **Transferability** — Works on environments absent from training.
> 3. **Non-triviality** — Cannot be replicated by a decision tree with fewer than K nodes.
>
> Pre-registered criteria. This framework is itself a contribution — no prior work formalizes "RL discovering an algorithm."

---

## 5. The Search Environment

The environment is a procedurally generated 2D grid world. Not a maze — a configurable state space with five dimensions of complexity, each designed to defeat a different class of simple strategies. A 3D environment and Isaac Lab port are planned for Phase 4 (post-paper) to support ICRA submission. The core algorithm discovery hypothesis is fully testable in 2D — what drives emergence is graph complexity, adversarial pressure, and resource constraints, not spatial dimensionality.

### 5.1 Five Dimensions of Complexity

#### Dimension 1: Topology (Static Graph)
Walls, corridors, rooms, density, dead-end frequency, loop frequency, topology class. The base graph structure.

#### Dimension 2: Dynamics (Non-Stationary Graph)
Moving walls on timers, doors that open/close cyclically, cells that collapse after traversal. Edges that existed moments ago may disappear. Breaks static-graph algorithms. Forces temporal reasoning.

#### Dimension 3: Information (Partial Observability)
Fog of war with configurable radius, sensor noise by zone. Sensor Burst (P8) reveals more but costs 5 energy. Forces the agent to solve the value-of-information problem: when is certainty worth its computational price?

#### Dimension 4: Resources (Non-Uniform Cost Landscape)
Energy recharge stations and penalty zones. Dynamic traversal costs (revisit multiplier). Route planning must account for resource logistics and avoid revisiting explored space.

#### Dimension 5: Objectives (Multi-Goal Search)
Single target, multi-checkpoint, time-limited, minimum-budget-at-arrival. Forces hierarchical decomposition and visiting-order optimization.

> [!important] Why All Five Dimensions
> - Topology alone → classical algorithms suffice.
> - Add dynamics → static-graph algorithms break.
> - Add fog → complete-information algorithms break.
> - Add resources + revisit costs → shortest-path and wall-spam algorithms break.
> - Add multi-goal → single-objective algorithms break.
> - All five together demand genuinely novel search procedures.

### 5.2 Adversary Control Space (15 Parameters)

| Parameter | Range | Dimension | Effect |
|-----------|-------|-----------|--------|
| Wall Density | 0.1–0.9 | Topology | Graph connectivity |
| Corridor Width | 1–5 | Topology | Passage constraints |
| Dead End Freq. | 0–0.5 | Topology | Backtrack requirement |
| Loop Frequency | 0–0.4 | Topology | Cycle confusion |
| Topology Seed | Categorical | Topology | Base structure class |
| Moving Walls | 0–0.3 | Dynamics | Edge instability |
| Door Frequency | 0–0.4 | Dynamics | Wait-or-detour |
| Collapse Zones | 0–0.2 | Dynamics | One-use edges |
| Fog Radius | 3–20 | Information | Observation horizon |
| Sensor Noise | 0–0.3 | Information | Observation reliability |
| Recharge Stations | 0–5 | Resources | Budget refill |
| Penalty Zones | 0–0.2 | Resources | High-cost regions |
| Checkpoints | 0–4 | Objectives | Multi-target |
| Time Limit | 0.5–1.0 | Objectives | Search deadline |
| Goal Distance | 0.3–1.0 | Objectives | Search depth |

*Note: Floor Count removed from adversary control space (2D phase). Restored in Phase 4 3D extension.*

---

## 6. System Architecture

### 6.1 Environment Engine

Custom 2D procedural grid world in JAX. JIT-compiled to XLA, fully on-GPU. Raycast observation (N-ray). Vectorized across 1,024 parallel environments. Phase 4: Isaac Lab port for ICRA submission.

### 6.2 The Hierarchical Controller

#### Observation Space

| Component | Dimensions | Description |
|-----------|-----------|-------------|
| LiDAR Scan | N-ray | Raycast distances, 360° |
| Local Occupancy | 32×32 | Accumulated grid map |
| Goal Vector | 2-dim (per goal) | Direction + distance |
| Agent State | 4-dim | Position + velocity |
| Budget State | 2-dim | Energy: absolute + ratio |
| Primitive History | 13×16 | One-hot of last 16 primitives |
| Memory Buffer | K=16 slots | Waypoints + local map hashes |
| Visit Counts | Local grid | Revisit tracking for dynamic costs |
| Dynamic State | 8-dim | Timer states for doors/walls |
| Resource Map | 8×8 | Known recharge locations |

#### Policy Architecture

**Phase 1–4 (current):** MLP controller (2 layers, 128 hidden). Confirmed working before Transformer upgrade.

**Phase 5+ (upgrade):** Transformer encoder — 4 layers, 4 heads, 128-dim. Attention over primitive history provides compositional memory. LiDAR/occupancy via CNN, concatenated with scalars. Output: categorical distribution over 13 primitives, masked by budget. PPO + GAE. ~2M parameters.

### 6.3 PAIRED Three-Player Training Loop

Three networks trained concurrently:

- **Protagonist** (2M params) — The main controller. Trained with PPO to maximize navigation reward.
- **Antagonist** (~500K params) — Baseline solver. Same action space, simpler architecture. Trained with PPO on the same environments.
- **Adversary** (~300K params) — State space generator. Controls 15 parameters. Reward = max(0, antagonist_score − protagonist_score) (regret). Diversity bonus via entropy over parameter distributions.

Training warmup: random environments (no adversary) for baseline competence. Then PAIRED activates. Dimensions added incrementally: topology first, then dynamics, then fog, then resources, then multi-goal.

---

## 7. The Emergence Hypothesis

### 7.1 Developmental Stages

**Stage 1 — Random Search.** Primitives sampled uniformly. Poor performance.

**Stage 2 — Strategy Selection.** Context-dependent preferences. Gradient steps in open space, wall-following in corridors. A decision tree could replicate this.

**Stage 3 — Strategy Composition.** PAIRED defeats simple selection by generating environments where no single strategy works. The controller chains primitives into procedures. Stable, transferable search strategies emerge.

### 7.2 Concrete Example: Parallel Evaluation

Two paths from a junction: path A leads to goal, path B is a dead end. Budget: 40. Selection-based controller picks one (possibly wrong), wastes 15–20 energy. Composition-based controller:

- P6 → P10 → P7 → P10 → P7 → P12 → P5×2 → P11 (cost: 25, saved: 20+)

Bookmark position, scout path A virtually, return, scout path B virtually, return, compare, commit to winner. A parallel evaluation procedure — no single primitive or classical algorithm does this. It manages computational investment as part of the search process.

### 7.3 Anti-Spam Mechanism

Without countermeasures, the agent discovers that P2 (wall-follow, cost 2) solves most simple environments. Dynamic traversal costs prevent this: revisiting cells multiplies cost by (1 + visit_count). Wall-following loops become prohibitively expensive, forcing the agent to use memory (P6/P7) to track explored space and lookahead (P10) to evaluate before committing. This makes cognitive primitives essential rather than optional.

### 7.4 Falsifiability

If behavior is replicable by a decision tree with <K nodes: composition claim rejected. If procedures don't transfer: they are permutations, not algorithms. Negative results reported transparently.

---

## 8. Experimental Design

### Experiment A: Search Efficiency Scaling

**Hypothesis:** GENESIS cost scales sub-linearly with state space complexity.

**Method:** 1,000 state spaces × 10 complexity levels × 5 baselines (A\*, RRT, end-to-end PPO, algorithm-selection baseline, GENESIS).

### Experiment B: Search Procedure Mining

**Hypothesis:** Stable primitive sequences emerge meeting all three criteria.

**Method:** Log every primitive call. N-gram and motif mining across millions of episodes. Test stability, transferability, non-triviality.

### Experiment C: Zero-Shot Transfer (Critical Test)

**Hypothesis:** Procedures generalize to unseen state space types.

**Method:** Train on one topology class with heavy domain randomization. Test on unseen topology classes, dynamic configurations, and multi-checkpoint objectives. Measure both success rate AND whether same procedures deploy in new contexts.

*PAIRED's regret-based curriculum provides theoretical grounding for transfer. Domain randomization across all 15 adversary parameters during training ensures the controller cannot memorize spatial layouts.*

### Experiment D: Algorithm Extraction

**Hypothesis:** Emergent procedures distillable into standalone pseudocode algorithms.

**Method:** Formalize, implement standalone, benchmark against classical algorithms AND full controller.

**If an extracted algorithm outperforms classical baselines on resource-constrained environments, that is a concrete, verifiable, novel search algorithm discovered by a machine.**

### Experiment E: Ablation Suite

- No budget constraint — does composition emerge without resource pressure?
- No adversary (random curriculum only) — is PAIRED necessary?
- Full algorithms as actions (A\*, RRT as atomic) — does sub-algorithmic granularity matter?
- No memory primitives (P6/P7 removed) — is temporal reasoning essential?
- Static state spaces only (no dynamics) — are non-stationary graphs necessary?
- No dynamic traversal costs (flat revisit penalty) — does anti-spam mechanism drive composition?

---

## 9. Hardware Feasibility

| Benchmark | Measured | Target | Headroom |
|-----------|----------|--------|----------|
| Matrix Multiply | 167 ops/sec | — | Fast NN confirmed |
| Env Steps (1,024 parallel) | 444,000/sec | 50,000/sec | 8.9x |

| Component | VRAM | Notes |
|-----------|------|-------|
| JAX Envs (1,024) | ~2.0 GB | Grid + dynamics + resources + visit counts |
| Observation Buffers | ~0.3 GB | Raycast × 1,024 envs |
| Protagonist (2M) | ~0.5 GB | Including Adam |
| Antagonist (500K) | ~0.2 GB | Baseline solver |
| Adversary (300K) | ~0.1 GB | Generator |
| PPO Buffer | ~1.5 GB | Rollout data |
| XLA + Margin | ~7.4 GB | Cache + peaks + OS |
| **Total** | **~12.0 GB** | **Fits RTX 5070 Ti** |

Entire system on a single consumer laptop (Lenovo Legion 5 Pro, RTX 5070 Ti 12 GB, Intel Core Ultra 9). No cloud infrastructure. Accessibility is part of the contribution.

---

## 10. Execution Roadmap

### Phase 1: Environment + Primitives
- 2D grid world (all 5 complexity dimensions) + 13 primitives + energy budget + dynamic traversal costs.
- MLP controller. PPO training loop confirmed working.
- **Deliverable:** Running system, primitive usage logs, first training signal.

### Phase 2: PAIRED + Scale
- Antagonist + Adversary. PAIRED three-player training loop. 1,024 parallel envs.
- Ablation infrastructure. Domain randomization across all 15 adversary parameters.
- **Deliverable:** Trained system, preliminary compositions, ablation results.

### Phase 3: Validation + Paper
- Experiments A–E. Algorithm extraction. LaTeX. ArXiv. GENESIS-Bench release.
- Upgrade controller to Transformer (4 layers, 4 heads, 128-dim).
- **Deliverable:** ArXiv preprint, polished repo, benchmark.

### Phase 4: 3D Extension + ICRA
- Isaac Lab port. 3D voxel environment. Multi-floor topology. Sim-to-real.
- Multi-agent extension: one controller, N bodies, distributed search discovery.
- **Deliverable:** ICRA submission, extended benchmark.

---

## 11. Related Work

### Hierarchical RL
Options framework (Sutton et al., 1999) and successors define options as learned motor sub-policies. GENESIS uses classical search procedures as options. Controller learns invocation/sequencing, not execution.

### Algorithm Selection
Portfolio solvers (SATzilla, AutoFolio) choose among complete algorithms offline. GENESIS operates at sub-algorithmic granularity online, enabling compositions no portfolio expresses.

### Algorithm Discovery
AlphaDev (sorting) and AlphaTensor (matrix multiplication) target narrow, verifiable, static domains. GENESIS targets open-ended, non-stationary domains with co-evolving fitness via PAIRED.

### Unsupervised Environment Design
POET (Wang et al., 2019) uses failure-based reward for environment co-evolution. PAIRED (Dennis et al., 2020) uses regret-based reward with a baseline antagonist, providing mathematical guarantees on curriculum coherence and zero-shot transfer bounds. GENESIS adopts PAIRED.

### Tool Use in AI
Toolformer and Gorilla demonstrate LLM tool invocation. GENESIS extends this to embodied search agents with physical tool cost and algorithmic tools.

---

## 12. Risk Analysis

| Risk | Severity | Mitigation |
|------|----------|------------|
| No compositions emerge | High | 6-ablation suite identifies causes. Methodological contributions stand independently. |
| Compositions don't transfer | High | Experiment C + PAIRED transfer bounds detect this. Domain randomization maximizes probability. |
| Agent spams cheap primitives | Medium | Dynamic traversal costs + revisit penalties eliminate by design. |
| PAIRED instability | Medium | Staged warmup: random curriculum first, then activation. Antagonist provides stability. |
| Reward hacking (deliberate failure) | Medium | Success reward at 10–50x max episode cost. Ablation verifies. |
| VRAM insufficient | Low | 8.9x headroom. Fallback: 512 envs. |
| Below top-venue bar | Medium | Target AAAI, AAMAS, NeurIPS workshops. |

---

## 13. Open-Source Strategy

All code, models, and data released as GENESIS-Bench under MIT license: procedural grid world generator (5 dimensions, 15 parameters), 13-primitive action space with budget + dynamic traversal costs, PAIRED adversary framework with pre-trained checkpoints, and full evaluation scripts.

---

## 14. Future Extension: Distributed Search

If single-agent results are strong, a natural extension: one controller managing N parallel search agents. Not multi-agent RL (separate brains). A single brain with multiple bodies — an executive allocating search resources across parallel processes.

Two new primitives: Signal (broadcast discovered information, cost 2) and Listen (integrate others' information, cost 1). Emergent procedures become distributed search protocols: "send agent 1 as scout, deploy agent 2 with heavy sensing, keep agent 3 as spatial anchor." Such protocols are exceptionally difficult to design by hand and valuable in practice (drone swarms, warehouse robots, disaster response).

---

## 15. Conclusion

Project GENESIS reframes algorithm discovery. Instead of narrow, static, verifiable domains, we ask whether RL agents can compose novel search procedures from atomic operations under real-world constraints: finite computation, bounded memory, dynamic adversarial environments, and exponential revisit penalties.

The architecture is honest. Three guaranteed methodological contributions (sub-algorithmic action space, PAIRED-based curriculum, three-criteria evaluation framework) stand independent of the empirical bet. The anti-spam mechanisms (dynamic traversal costs, reward scaling) prevent degenerate solutions. The PAIRED framework provides mathematical curriculum guarantees. Six ablations isolate causal mechanisms.

Hardware is validated: 444,000 steps/sec, 8.9x headroom, single consumer GPU. The constraint profile is novel. The experiments are rigorous. The framework is open-source.

> ***Can a machine, given the atomic operations of search and enough pressure, discover a search algorithm that no human designed?***
