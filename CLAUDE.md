# CLAUDE.md — Project GENESIS working brief

> Read this first every session. Full design specs: `GENESIS_Preparation_Roadmap_v2.md`
> (authoritative reference) and `Project_GENESIS_Research_Proposal_v2.md` (the paper).
> Detailed approved plan for the current stage:
> `C:\Users\dnaad\.claude\plans\genesis-preparation-roadmap-v2-md-proje-delegated-dijkstra.md`.

## What GENESIS is (one paragraph)
A search-**algorithm-discovery** project. A PPO **controller** (not a player) orchestrates 13 atomic
search primitives under a finite energy budget and a PAIRED adversarial curriculum. The bet: under
resource pressure + adversarial environments it composes stable, transferable search **procedures**.
Proof = zero-shot transfer to unseen environment configs. The model orchestrates search; it does not
navigate. Do NOT introduce 3D, PyTorch in the core loop, or a two-player simplification of PAIRED.

## Where things live (IMPORTANT)
- **Code lives in WSL2 at `~/Genesis`** (Ubuntu-24.04). NOT on `/mnt/d`. Drive it via `wsl.exe -- bash -lc '...'`
  or open the folder in VSCode WSL remote. Activate venv: `source ~/Genesis/.venv/bin/activate`.
- This Windows folder `d:\genesis` holds the design docs + `setup_wsl.sh` + this file. Docs get
  imported into `~/Genesis/docs/` during scaffolding.
- GitHub remote: `https://github.com/aarsh-adhvaryu/Genesis.git`.

## Verified environment facts (2026-05-27)
- GPU visible in WSL: **RTX 5070 Ti, compute_cap 12.0 (sm_120, Blackwell), 12 GB, driver 595.79**.
- Ubuntu 24.04.4, **native Python 3.12.3**, python3-venv + pip present.
- Host CUDA toolkit is 13.1 but **irrelevant**: `jax[cuda12]` bundles its own CUDA-12 libs; only the
  driver matters. GPU smoke test is the arbiter for sm_120.

## Locked decisions (do not silently change)
| Topic | Decision |
|---|---|
| Runtime | WSL2 Ubuntu-24.04, **native Python 3.12** venv (fall back to 3.11 only if a dep breaks) |
| Stack | JAX (`jax[cuda12]`), Equinox, Optax, Distrax, chex. **No PyTorch/TF in core loop** (only if genuine necessity + GPU). PureJaxRL patterns; clone is an external reference at `~/refs/purejaxrl`, never vendored. |
| `SearchState` | `chex.dataclass`, H/W static via `EnvConfig` |
| `reset()` solvability | Fixed-iter **BFS flood-fill** from goal → reachable mask → sample agent from it (`lax.fori_loop`, H·W iters). Same kernel reused later as P3's inner loop. |
| Grid generation | **Parameterized** `reset(key, params)` — domain-randomized now, adversary-driven in Stage 6. Grids must be solvable, never just random. |
| Stage 1 `step()` | **Minimal motor** (4-dir + collision + goal check + minimal obs). Action→effect isolated in `apply_action(state, action)` so Stage 3 extends it to 13 primitives without touching the loop. |
| Grid size | default **32×32**, configurable |

## FIRST PRIORITY — get to "RL is learning and we watch it" ASAP
The north star is a **thin end-to-end vertical slice that trains and is observable**: a runnable
loop where a (minimal) PPO controller acts in the (minimal) env and we can WATCH a learning signal
(reward/success-rate rising on simple maps). Reaching that watchable loop beats polishing any single
unit. Everything below is "develop *toward* that slice." Build the slice narrow but real, confirm it
learns, THEN thicken (more primitives, full obs, energy budget, PAIRED).
- Slice path: minimal env (reset/step/obs) → minimal action interface → minimal PPO (PureJaxRL
  pattern, MLP) → train on easy random maps → **watch reward climb**. Keep each piece as small as
  possible; correctness over completeness; no premature features.
- Still honor the locked decisions, math-first habit, and JAX-native rules below — just aim them at
  the slice. When a "nice to have" would delay the watchable loop, defer it and note it.

## Current status / where to start
- [x] Plan approved. [x] GPU + toolchain verified. [x] **Foundations run** (venv, `GPU OK`, 2026-06-21).
- [x] **Stage 1 env core complete & tested:** `config`, `state`, `generate` (BFS flood-fill,
  0/2000 unsolvable vs independent BFS), `env` (reset/step/obs), 13 pytest tests, `bench` (1024
  envs -> 4.8M steps/sec). GPU calib `XLA_PYTHON_CLIENT_PREALLOCATE=false` (WSL/WDDM).
- [x] **WATCHABLE SLICE ACHIEVED (2026-06-23):** `network` (MLP actor-critic), `rollout`+GAE,
  `ppo` (clipped loss), `train` driver, `render`. On 11x11 easy maps greedy success climbs
  **4.5% -> 73%** in ~32s/300 updates; curve + trajectory saved to `runs/` (gitignored).
- [x] **Thicken #1 (scale):** PPO scales to 32x32 (success 0.8%->43%/325 upd; slower = sparse reward,
  motivates curriculum). [x] **#3 (budget+cost):** `energy` in state, dynamic cost base*(1+visit),
  exhaustion ends episode, `lambda_budget` reward (=0 default), `action_costs`. Slice still learns.
- [x] **#2 foundation (FACTORED action, Option 1):** action = (primitive 13-way, direction 4-way);
  two-head `ActorCritic`; budget `action_mask` (P0 always legal); obs gains budget ratio.
- [x] **#2 memory primitives P6/P7/P9:** `SearchState` gains FIFO ring `memory_buffer` (K=16) +
  `mem_head/mem_count/mem_cursor`; P6 write, P7 read-cursor (advances which waypoint is surfaced),
  P9 backtrack=teleport to cursor-selected waypoint; obs surfaces selected waypoint (rel+flag).
  Implemented now: P0,P1,P5,P6,P7,P9. Slice learns 2.3%->95.5%. 26 tests pass.
  (P7 semantics = read-cursor is OUR interpretation of "retrieve"; flag if you want plain newest-read.)
- **Resume here — remaining #2 primitives** (one unit each, mask flips on as built): P2 wall-follow
  (no new state), P3/P4 frontier (needs frontier+visited set), P8 fog/LiDAR (needs partial obs),
  P10 scout (multi-step sim), P11 commit (stored path), P12 subgoal (goal_stack). Then **#4 PAIRED**
  (Antagonist+Adversary, regret). Scale to 32x32.

## Stage 1 build order (one unit at a time; math first, then code; confirm each before next)
1. **Scaffold** `~/Genesis`: `pyproject.toml` (py3.12, ruff/pytest), `.gitignore` (.venv, __pycache__,
   wandb/, *.npz), `README.md`, `docs/` (copy the two .md from `/mnt/d/genesis`), `genesis/`, `tests/`.
2. **`config.py`** — `EnvConfig`: H, W, base_cost, B0, max_steps, R_goal, time_penalty, gen-param ranges.
3. **`state.py`** — `SearchState` (chex.dataclass): grid(H,W)int8 {0 empty,1 wall,2 goal}, agent_pos(2),
   goal_pos(2), visit_counts(H,W) (tracked now, COST applied Stage 2), step_count(), done(), key.
4. **`generate.py`** — parameterized grid + BFS flood-fill. Math:
   `R_0={G}; R_{t+1}=R_t ∪ (N(R_t) ∩ Free)`, N=4-neighborhood dilation; converges in ≤ H·W iters
   (`lax.fori_loop`). Sample agent only from reachable mask R*. Start empty-room topology, add walls after.
5. **`env.py`** — `reset(key, params)`, `step(state, action)`, `obs(state)`. step math:
   `p'=clip(p+Δ); p_next = p' if grid[p']!=wall else p; visit_counts[p_next]++; reached=(p_next==goal);
   done=reached|(step+1>=max_steps); reward=R_goal*reached - time_penalty`. Branchless (`jnp.where`),
   jit + vmap clean, no Python loops in hot path. Minimal obs = concat[pos/scale,(goal-pos)/scale,KxK patch].
6. **`tests/test_env.py`** (pytest): agent always in R* over many keys (0 unsolvable, verify w/ independent
   BFS); borders are walls; wall_density=0 ⇒ open interior; wall-step is no-op; goal-step sets done&reached;
   visit_counts increments; vmap independence + seeded determinism.
7. **`bench.py`** — `jit(vmap(reset))`, `jit(vmap(step))`, steps/sec = N·T/wall (warmup excluded) for
   N∈{256,512,1024}. Reference target 50k/sec (proposal claims 444k). Records device + steps/sec.
8. **git** — fresh `git init`, `remote add origin <repo>`, one clean commit (scaffold+docs+Stage 1),
   then **`git push --force origin main`** (overwrites the 2 old commits). **Confirm with user immediately
   before the force-push** — it's irreversible on the remote.

## Deferred until AFTER the watchable slice (do NOT build early — thicken later)
Energy budget/masking, dynamic traversal **cost** application, the full 13 primitives, full
observation (LiDAR/occupancy/memory), PAIRED (Antagonist+Adversary), Transformer controller.
`SearchState` + `apply_action` are shaped so each slots in later.
NOTE: a *minimal* PPO loop is NOT deferred — it is the target of the slice. Start with a tiny action
interface (motor / a few primitives), MLP policy, single-env-config training; expand once it learns.

## How to work here (from the user's protocol)
- **Math first**, then code. When code encodes a formula, comment the formula.
- **Decision protocol:** at any fork, STOP — present ≥2 labeled options with trade-offs + a recommendation;
  wait for a decision. Never decide unilaterally. Present a novel 3rd option if you see one.
- **One focused question** when uncertain — not five. Don't fill gaps with assumptions.
- **One unit at a time** — implement, confirm correct, then continue.
- All JAX-native: jit-compiled, vmappable, no Python loops in the hot path.
- Confirm before destructive/outward-facing actions (rm, force-push).
