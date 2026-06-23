"""generate.py — parameterized, SOLVABLE-BY-CONSTRUCTION grid generation.

The guarantee we need: the agent can always reach the goal. We get it not by checking maps after
the fact, but by *constructing* them so it holds. The key kernel is a BFS flood-fill from the goal
that computes R* = the set of all cells that can reach the goal through open space; we then place
the agent only inside R*. Agent in R*  ==>  a path exists, by definition.

Flood-fill as set-growth (4-connectivity):
    R_0     = {goal}
    R_{t+1} = R_t  ∪  ( N(R_t) ∩ Free )          N = up/down/left/right dilation
Converges to the goal's connected free component in <= H*W iterations (worst case: a snake corridor
through every cell). A fixed H*W-iteration lax.fori_loop is therefore both sufficient and jittable.

This same flood-fill kernel is reused later as P3's inner loop (graph node expansion).

Cell encoding (int8): 0 empty, 1 wall, 2 goal.  Positions are (row, col) int32.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from genesis.config import EnvConfig
from genesis.state import SearchState

WALL = jnp.int8(1)
GOAL = jnp.int8(2)


def _neighbors(R: jax.Array) -> jax.Array:
    """N(R): 4-neighborhood dilation via NON-wrapping shifts (no jnp.roll — that wraps edges).

    Each shifted plane is zero-padded at the boundary it shifts away from, so the goal's
    component never leaks across the grid border.
    """
    up = jnp.zeros_like(R).at[:-1, :].set(R[1:, :])    # pull neighbor below  -> current cell
    down = jnp.zeros_like(R).at[1:, :].set(R[:-1, :])  # pull neighbor above
    left = jnp.zeros_like(R).at[:, :-1].set(R[:, 1:])  # pull neighbor to the right
    right = jnp.zeros_like(R).at[:, 1:].set(R[:, :-1]) # pull neighbor to the left
    return up | down | left | right


def flood_fill_reachable(free: jax.Array, goal_pos: jax.Array, cfg: EnvConfig) -> jax.Array:
    """R* — boolean (H,W) mask of cells that can reach `goal_pos` through `free` cells.

    Implements R_{t+1} = (R_t ∪ N(R_t)) ∩ Free for H*W iterations (the convergence bound).
    """
    H, W = cfg.height, cfg.width
    R0 = jnp.zeros((H, W), dtype=bool).at[goal_pos[0], goal_pos[1]].set(True)

    def body(_, R):
        # grow by one ring, then clip to free space (walls block the flood)
        return (R | _neighbors(R)) & free

    return lax.fori_loop(0, H * W, body, R0)


def _sample_true_cell(key: jax.Array, mask: jax.Array, cfg: EnvConfig) -> jax.Array:
    """Uniformly sample one True cell of `mask` as (row, col) — Gumbel-max over a flat masked field.

    argmax(gumbel) over equal logits = a uniform draw; False cells get -inf so they're never picked.
    Fully static-shape (jit/vmap safe): no boolean indexing, no dynamic sizes.
    """
    H, W = cfg.height, cfg.width
    flat = mask.reshape(-1)
    g = jax.random.gumbel(key, (H * W,))
    logits = jnp.where(flat, g, -jnp.inf)
    idx = jnp.argmax(logits)
    return jnp.stack([idx // W, idx % W]).astype(jnp.int32)


def _build(key: jax.Array, cfg: EnvConfig, wall_density: jax.Array) -> SearchState:
    """Construct one solvable SearchState at the given wall density. Core used by generate() + tests."""
    H, W = cfg.height, cfg.width
    k_wall, k_goal, k_agent = jax.random.split(key, 3)

    # --- walls: random interior cells + an always-solid border ---
    rows = jnp.arange(H)[:, None]
    cols = jnp.arange(W)[None, :]
    border = (rows == 0) | (rows == H - 1) | (cols == 0) | (cols == W - 1)
    interior_walls = jax.random.uniform(k_wall, (H, W)) < wall_density
    walls = border | interior_walls          # (H,W) bool
    free = ~walls

    # --- goal: a uniformly random free cell (free already excludes the border) ---
    goal_pos = _sample_true_cell(k_goal, free, cfg)

    # --- R*: everything that can reach the goal through free space ---
    reachable = flood_fill_reachable(free, goal_pos, cfg)

    # --- agent: uniform over R* \ {goal}; if the goal is isolated, fall back to the goal cell ---
    goal_onehot = jnp.zeros((H, W), dtype=bool).at[goal_pos[0], goal_pos[1]].set(True)
    agent_mask = reachable & (~goal_onehot)
    has_other = agent_mask.any()
    sampled = _sample_true_cell(k_agent, agent_mask, cfg)
    agent_pos = jnp.where(has_other, sampled, goal_pos)   # always inside R*

    # --- encode grid: 0 empty, 1 wall, 2 goal ---
    grid = jnp.where(walls, WALL, jnp.int8(0))
    grid = grid.at[goal_pos[0], goal_pos[1]].set(GOAL)

    return SearchState(
        grid=grid,
        agent_pos=agent_pos,
        goal_pos=goal_pos,
        visit_counts=jnp.zeros((H, W), dtype=jnp.int32),
        energy=jnp.float32(cfg.b0),          # start each episode with the full budget B0
        memory_buffer=-jnp.ones((cfg.memory_k, 2), dtype=jnp.int32),  # empty (FIFO ring)
        mem_head=jnp.int32(0),
        mem_count=jnp.int32(0),
        mem_cursor=jnp.int32(0),
        step_count=jnp.int32(0),
        done=jnp.bool_(False),
        key=key,
    )


def generate(key: jax.Array, cfg: EnvConfig) -> SearchState:
    """Production entry point: sample a wall density from the config range, build a solvable map.

    Domain randomization now; in Stage 6 the adversary supplies these generation params instead.
    """
    k_density, k_build = jax.random.split(key, 2)
    wall_density = jax.random.uniform(
        k_density, (), minval=cfg.wall_density_min, maxval=cfg.wall_density_max
    )
    return _build(k_build, cfg, wall_density)
