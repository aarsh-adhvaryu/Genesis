# PROJECT GENESIS — Reference & Build Prompt

**Status:** Build Phase Active  
**Environment:** 2D first. 3D is Phase 4 (post-paper, Isaac Lab port).  
**Hardware:** Lenovo Legion 5 Pro · RTX 5070 Ti 12GB · Intel Core Ultra 9 · JAX verified on GPU

---

## What This Document Is

This is not a timeline. It is a living reference you keep open while building. It tells Claude Code exactly what GENESIS is, what decisions are locked, and what to build next. Update it as decisions get made.

---

## 1. What GENESIS Is (Read This Before Everything Else)

GENESIS is not a maze-solving project. It is a **search algorithm discovery** project.

The distinction matters for every implementation decision:

- A standard RL agent learns to navigate. Its action space is directions. It discovers routes.
- The GENESIS controller learns to orchestrate search operations. Its action space is cognitive primitives. It discovers **procedures** — reusable strategies that transfer to unseen environments.

The central hypothesis: under adversarial pressure and resource constraints, the controller will compose the 13 primitives into stable, transferable search procedures that encode insights no existing algorithm contains. The proof is zero-shot transfer — procedures that work on environment configurations never seen during training.

**The model is not the player. The model is the controller.**

---

## 2. Locked Architecture Decisions

These are not up for debate during implementation. If something seems wrong, flag it — but don't silently change it.

### The Three Networks

| Network | Role | Parameters | Architecture |
|---|---|---|---|
| Protagonist | Main controller — learns to orchestrate primitives | ~2M | Transformer (4 layers, 4 heads, 128-dim) + CNN |
| Antagonist | Baseline solver — difficulty gauge for PAIRED | ~500K | Smaller Transformer + CNN |
| Adversary | Environment generator — trained on regret signal | ~300K | MLP |

All three trained with **PPO**.

### Training Algorithm: PPO

- Clipped surrogate objective, clip ratio ε = 0.2
- GAE advantage estimation, λ = 0.95
- **Entropy bonus is architecturally critical** — prevents controller collapsing to a subset of primitives. Never remove it.
- Both protagonist and antagonist trained with identical reward structure

### Adversarial Curriculum: PAIRED

Three-player regret-based curriculum (Dennis et al., 2020).

- Adversary reward = `max(0, antagonist_score − protagonist_score)`
- This guarantees: impossible environments earn zero reward (antagonist also fails). Trivial environments earn zero reward (protagonist succeeds). Only environments at the frontier of protagonist capability earn reward.
- **Do not replace this with a two-player setup.** The antagonist baseline is load-bearing.
- Training warmup: random environments first (no adversary) until protagonist has baseline competence. Then PAIRED activates.

### Primitives: Formal Grounding

All 13 primitives are **options** in the Options Framework (Sutton, Precup, Singh, 1999). Each is formally defined as ⟨I, π, β⟩ — initiation set, internal policy, termination condition — forming a Semi-MDP.

**Primitive temporal granularity — Design A (locked):**
- P0–P9, P12: one-tick atomic operations. The primitive executes one step and returns control to the controller immediately.
- P10 (Scout Fork): multi-step by necessity — simulates N steps forward to yield meaningful lookahead.
- P11 (Commit Path): intentionally multi-step — executes a path uninterrupted.

This maximises controller decision frequency and enables interleaved compositions like `P2 → P6 → P2 → P8 → P2` (follow wall, bookmark, follow, scan, follow). You cannot get this composition if primitives hog control for multiple steps.

---

## 3. The 13 Primitives — Full Specification

| ID | Name | Operation | Cost | State Modified | One-Tick? |
|---|---|---|---|---|---|
| P0 | Idle / Observe | Passive sensing, no movement | 0 | None | ✅ |
| P1 | Motor Primitive | Move one step in specified direction | 1 | agent_pos, visit_counts | ✅ |
| P2 | Wall Surface Follow | Move one step hugging the wall (direction auto-computed) | 2 | agent_pos, visit_counts | ✅ |
| P3 | Graph Node Expand | Expand one node from frontier (A* inner loop, one step) | 3 | graph_frontier, visited_set | ✅ |
| P4 | Frontier Sample | Random extension from frontier (RRT inner loop, one step) | 3 | graph_frontier | ✅ |
| P5 | Gradient Step | Move one step toward goal (greedy descent) | 2 | agent_pos, visit_counts | ✅ |
| P6 | Memory Write | Store current position + local map hash to buffer (FIFO, K=16) | 1 | memory_buffer | ✅ |
| P7 | Memory Read | Retrieve most recent entry from memory buffer | 1 | None (read only) | ✅ |
| P8 | Sensor Burst | Full local observation (256-ray LiDAR equivalent), expensive | 5 | observation_cache | ✅ |
| P9 | Backtrack | Move to last stored memory position | 2 | agent_pos, visit_counts | ✅ |
| P10 | Scout Fork | Simulate N steps forward virtually, return projected state | 8 | None (virtual only) | ❌ multi-step |
| P11 | Commit Path | Execute stored path uninterrupted to completion | 1/step | agent_pos, visit_counts | ❌ multi-step |
| P12 | Subgoal Set | Decompose current goal into a sub-goal | 1 | goal_stack | ✅ |

**Critical primitives for emergence:** P6/P7 (temporal reasoning), P10 (lookahead), P12 (hierarchical decomposition). Without these, the controller can only react. With them, it can plan, simulate, and decompose.

**Budget masking:** primitives whose cost exceeds remaining energy are masked from the action distribution. The controller cannot select them.

---

## 4. Resource Systems

### Energy Budget

- Initial budget: B₀ (configurable, default start simple)
- Every primitive deducts its cost from remaining budget
- Reward includes `+λ·(B_t / B₀)` — reward for budget preservation
- Success reward = 10–50× maximum episode cost — prevents deliberate-failure reward hacking
- Primitives with cost > remaining budget are action-masked

### Memory Buffer

- K = 16 slots, FIFO
- Stores waypoints (position) and local map hashes
- P6 writes, P7 reads, P9 uses to backtrack
- This is working memory — the controller's scratch pad

### Dynamic Traversal Costs

- Every grid cell tracks a `visit_count`
- Traversal cost = `base_cost × (1 + visit_count)`
- This is the anti-spam mechanism: wall-following loops become exponentially expensive
- Forces the controller to use P6/P7 to track explored space and P10 to evaluate before committing
- **Makes cognitive primitives load-bearing, not optional**

---

## 5. The Environment (2D Phase)

The environment is a procedurally generated 2D grid world. Not a maze — a configurable state space with five dimensions of complexity.

### Five Complexity Dimensions

| Dimension | What It Does | Simple Algorithms It Breaks |
|---|---|---|
| Topology | Walls, corridors, rooms, dead ends, loops | — (baseline) |
| Dynamics | Moving walls on timers, collapsing edges | Static-graph algorithms (A*) |
| Information | Fog of war, sensor noise | Complete-information algorithms |
| Resources | Energy recharges, penalty zones, revisit costs | Shortest-path, wall-spam |
| Objectives | Multi-checkpoint, time limits, visit ordering | Single-objective algorithms |

All five together demand genuinely novel search procedures. Any subset can be handled by existing algorithms.

### Environment State (SearchState)

```python
@dataclass
class SearchState:
    grid: jnp.ndarray          # (H, W) — wall/empty/goal
    agent_pos: jnp.ndarray     # (2,) — x, y
    goal_pos: jnp.ndarray      # (2,) per goal
    budget: float              # scalar — remaining energy
    memory_buffer: jnp.ndarray # (K=16, ...) — stored waypoints
    visit_counts: jnp.ndarray  # (H, W) — revisit tracking
    dynamic_state: jnp.ndarray # timer states for moving walls
    step_count: int            # episode step counter
```

### Observation Space

| Component | Shape | Description |
|---|---|---|
| LiDAR scan | (N_rays,) | Ray distances to walls, 360° |
| Local occupancy | (32, 32) | Accumulated map around agent |
| Goal vector | (2,) per goal | Direction + distance |
| Agent state | (4,) | Position + velocity |
| Budget state | (2,) | Absolute + ratio (B_t / B₀) |
| Primitive history | (13, 16) | One-hot of last 16 primitives |
| Memory buffer | (K=16, ...) | Stored waypoints |
| Visit counts | local grid | For dynamic cost awareness |

### Adversary Control Space

The adversary outputs 16 parameters controlling environment generation. Domain randomization across all 16 during training prevents the controller from memorising spatial layouts and forces learning of abstract search principles.

---

## 6. Reward Structure

```
R = +R_goal                          # large positive on reaching goal
  + λ · (B_t / B₀)                  # budget preservation bonus
  - time_penalty                     # small per-step cost
  - revisit_costs                    # via dynamic traversal costs
```

- R_goal must be 10–50× the maximum possible episode cost
- This prevents the agent learning "fail deliberately to avoid negative reward"
- λ controls the tradeoff between efficiency and goal-reaching

---

## 7. What Is NOT in Scope (Phase 1–3)

These are real parts of the project but do not belong in the initial build:

- **3D environment** — moved to Phase 4. Algorithm discovery is fully testable in 2D. 3D adds robotics credibility for ICRA but does not change the core hypothesis.
- **Isaac Lab port** — Phase 4 only.
- **Transformer controller** — start with MLP (2 layers, 128 hidden). Upgrade to Transformer once the training loop is confirmed working.
- **Full PAIRED** — start with protagonist only against random environments. Add antagonist + adversary once basic PPO is confirmed learning.
- **Experiment suite** — experiments A–D come after the system is running.

---

## 8. Paper List (Status)

### Tier A — Must Read Before Code

- [x] PPO — Schulman et al., 2017
- [x] Options Framework — Sutton, Precup, Singh, 1999
- [x] PAIRED — Dennis et al., 2020
- [ ] PureJaxRL — Chris Lu et al., 2024 ← **read this before implementing the training loop**
- [ ] GAE — Schulman et al., 2016 ← Section 3 only

### Tier B — Before Writing the Paper

- [ ] AlphaDev — Mankowitz et al., 2023
- [ ] AlphaTensor — Fawzi et al., 2022
- [ ] Option-Critic — Bacon et al., 2017
- [ ] POET — Wang et al., 2019
- [ ] Toolformer — Schick et al., 2023
- [ ] HIRO — Nachum et al., 2018

### Tier C — During Implementation

- [ ] Brax — Freeman et al., 2021 (JAX env reference architecture)
- [ ] Attention Is All You Need — Vaswani et al., 2017 (when upgrading to Transformer)

---

## 9. Tech Stack (Locked)

```
Language:     Python 3.11
Compute:      JAX + jaxlib (GPU) — everything JIT-compiled, no CPU-GPU transfer
Networks:     Equinox
Optimizers:   Optax (Adam)
Distributions: Distrax (categorical policy)
Logging:      WandB
Viz:          Matplotlib
Template:     PureJaxRL (training loop)
```

**Never use PyTorch in the core training loop.** PyTorch is only permitted if Isaac Lab requires it in Phase 4.

---

## 10. Build Sequence

Work through these in order. Do not skip ahead.

### Stage 1 — Environment Core
- [ ] SearchState dataclass in JAX
- [ ] `env.reset()` — generates grid, places agent + goal
- [ ] `env.step(action)` — moves agent, checks goal, returns observation
- [ ] `jax.vmap` over N parallel environments
- [ ] Confirm throughput on GPU

### Stage 2 — Resource Systems
- [ ] Energy budget with per-primitive cost deduction
- [ ] Action masking when budget insufficient
- [ ] Visit count tracking per cell
- [ ] Dynamic traversal cost: `base_cost × (1 + visit_count)`
- [ ] Verify: looping becomes expensive after ~5 revisits

### Stage 3 — All 13 Primitives
- [ ] P0 Idle
- [ ] P1 Motor Primitive
- [ ] P2 Wall Follow (one-tick, direction auto-computed)
- [ ] P3 Graph Node Expand
- [ ] P4 Frontier Sample
- [ ] P5 Gradient Step
- [ ] P6 Memory Write
- [ ] P7 Memory Read
- [ ] P8 Sensor Burst
- [ ] P9 Backtrack
- [ ] P10 Scout Fork (multi-step simulation)
- [ ] P11 Commit Path (multi-step execution)
- [ ] P12 Subgoal Set
- [ ] Unit test each primitive independently before moving on

### Stage 4 — PPO Training Loop
- [ ] MLP controller (2 layers, 128 hidden) — Transformer comes later
- [ ] PPO with clipped objective
- [ ] GAE advantage estimation
- [ ] Entropy bonus — do not remove
- [ ] Rollout buffer
- [ ] Confirm reward increases over time on simple environments

### Stage 5 — Primitive Logging
- [ ] Log every primitive call per episode
- [ ] Histogram of primitive usage
- [ ] First signal: does the agent use any cognitive primitives or spam P1/P2?

### Stage 6 — PAIRED Integration
- [ ] Antagonist network (500K param, simpler than protagonist)
- [ ] Adversary network (300K param, MLP)
- [ ] Regret reward: `max(0, antagonist_score − protagonist_score)`
- [ ] Three-player training loop
- [ ] Verify: adversary not generating trivially easy or impossible environments

### Stage 7 — Scale + Upgrade
- [ ] Scale to 512–1024 parallel environments
- [ ] Upgrade controller to Transformer (4 layers, 4 heads, 128-dim)
- [ ] Full domain randomization across all 5 environment dimensions
- [ ] WandB logging for all metrics
- [ ] Git tag `v0.1`

---

## 11. Falsifiability (Keep This In Mind Throughout)

The project makes a specific claim. These are the conditions under which that claim fails:

- If emergent sequences can be replicated by a decision tree with fewer than K nodes → composition claim rejected
- If sequences don't transfer to unseen environment types → they are permutations, not algorithms
- Both negative results get reported. The three methodological contributions (sub-algorithmic action space, PAIRED curriculum, three-criteria evaluation framework) stand independently of the empirical bet.

---

## 12. Key Design Principles (Do Not Violate)

1. **Controller decides every tick** — never lock the controller out for multiple steps (except P10, P11)
2. **Cognitive primitives must be load-bearing** — dynamic traversal costs enforce this. Without revisit penalties, the agent ignores P6/P7/P10/P12.
3. **Success reward >> episode cost** — prevents deliberate failure as a strategy
4. **Entropy bonus always on** — prevents primitive collapse
5. **Domain randomization throughout training** — not just at test time
6. **Zero-shot transfer is the proof** — if procedures only work on training topologies, they're not algorithms

---

*This document is the authoritative reference for Claude Code sessions. Update it when design decisions change. Do not implement anything that contradicts it without explicit discussion.*
