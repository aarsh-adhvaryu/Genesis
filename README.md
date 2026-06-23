# GENESIS

A search-**algorithm-discovery** project. A PPO **controller** (not a player) orchestrates
13 atomic search primitives under a finite energy budget and a PAIRED adversarial curriculum.
The hypothesis: under resource pressure and adversarial environments, the controller composes
stable, transferable search **procedures**. Proof = zero-shot transfer to unseen environment
configurations.

See [`docs/Project_GENESIS_Research_Proposal_v2.md`](docs/Project_GENESIS_Research_Proposal_v2.md)
for the paper and
[`docs/GENESIS_Preparation_Roadmap_v2.md`](docs/GENESIS_Preparation_Roadmap_v2.md) for the
authoritative build reference.

## Stack
JAX (`jax[cuda12]`), Equinox, Optax, Distrax, chex. No PyTorch/TF in the core loop.
Native Python 3.12 in a WSL2 (Ubuntu-24.04) venv.

## Setup
```bash
source .venv/bin/activate        # deps pinned in requirements.lock.txt
                                 # (sets XLA_PYTHON_CLIENT_PREALLOCATE=false for the WSL/WDDM GPU)
python -m pytest                 # run the test suite (env + primitives)
python -m genesis.bench          # env throughput benchmark
python -m genesis.train          # train the PPO slice and WATCH greedy success climb -> runs/
```

## Status

**Environment core — complete & tested.** JAX-native 2D grid world (jit + vmap, ~4.8M steps/sec at
1024 envs): solvable-by-construction grids (BFS flood-fill, 0 unsolvable vs. an independent BFS),
`reset`/`step`/`obs`, energy budget + dynamic traversal cost `base·(1+visit_count)`.

**Watchable PPO slice — working.** Equinox MLP actor-critic, rollout + GAE, clipped PPO. On easy
11×11 maps greedy success climbs to ~95% in ~30s; learning curve + trajectory render to `runs/`
(gitignored). Confirmed to scale to 32×32 (slower — sparse reward, which the curriculum addresses).

**Factored primitive action interface (in progress).** Action = (primitive 0–12, direction 0–3)
with a two-head policy and budget masking. Implemented so far: **P0 Idle, P1 Motor, P5 Gradient
Step**; the remaining primitives (P2 wall-follow, P3/P4 frontier, P6/P7/P9 memory, P8 sensor, P10
scout, P11 commit, P12 subgoal) are masked off until their subsystems are built.

**Not yet built:** the remaining primitives + their subsystems (memory K=16, frontier, fog,
forward-sim), richer observation, and the PAIRED adversarial curriculum. See the roadmap and
`CLAUDE.md` for the current build status and next units.
