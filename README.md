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
python -m pytest                 # run tests
python -m genesis.bench          # throughput benchmark
```

## Status
**Stage 1 — Environment Core** (in progress): JAX-native 2D grid-world with solvable-by-
construction grids (BFS flood-fill), minimal 4-direction motor, jit + vmap, tests, benchmark.

Later stages (not yet built): resource systems, the 13 primitives, full observation, PPO,
PAIRED. See the roadmap's build sequence.
