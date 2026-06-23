"""env.py — the minimal 2D motor: reset / step / obs.

Everything here is branchless (jnp.where, not Python if) so it is clean under jit + vmap with no
Python loops in the hot path. cfg (EnvConfig) is STATIC: pass it as a static arg to jit.

Stage-1 scope: a 4-direction motor + collision + goal/timeout + a minimal observation. The
action->effect step is isolated in `apply_action` so Stage 3 can replace the 4 moves with the 13
primitives WITHOUT changing step()/the training loop. Energy budget, dynamic traversal COST, and
the full observation arrive in later stages.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from genesis.config import EnvConfig
from genesis.generate import WALL, generate
from genesis.state import SearchState

# Action -> (drow, dcol). 0=up, 1=down, 2=left, 3=right. (row,col) convention.
DELTAS = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=jnp.int32)
N_ACTIONS = 4


def _dest_cell(p: jax.Array, action: jax.Array, grid: jax.Array, cfg: EnvConfig) -> jax.Array:
    """Destination of `action` from p with wall collision (wall -> stay put). Returns (2,) int32."""
    H, W = cfg.height, cfg.width
    p_prime = jnp.clip(p + DELTAS[action], jnp.array([0, 0]), jnp.array([H - 1, W - 1]))
    is_wall = grid[p_prime[0], p_prime[1]] == WALL
    return jnp.where(is_wall, p, p_prime)


def action_costs(state: SearchState, cfg: EnvConfig) -> jax.Array:
    """(N_ACTIONS,) energy cost of each action = base_cost * (1 + visit_count[destination]).

    Exposed for the budget-masking that the expensive primitives need (added with the 13-primitive
    action interface). The motor's costs are near-uniform, so masking rarely binds yet.
    """
    p = state.agent_pos
    dests = jax.vmap(lambda a: _dest_cell(p, a, state.grid, cfg))(jnp.arange(N_ACTIONS))  # (A,2)
    vc = state.visit_counts[dests[:, 0], dests[:, 1]].astype(jnp.float32)                 # (A,)
    return cfg.base_cost * (1.0 + vc)


def apply_action(state: SearchState, action: jax.Array, cfg: EnvConfig) -> SearchState:
    """The motor (Stage-1 = 4-dir move). Updates agent_pos (collision), visit_counts, and energy.

    Math (branchless):
        p_next = clip(p+Δ) if not wall else p            # walls block -> no-op
        cost   = base_cost * (1 + visit_count[p_next])   # dynamic traversal cost (BEFORE increment)
        visit_counts[p_next] += 1 ;  energy -= cost
    """
    p = state.agent_pos
    p_next = _dest_cell(p, action, state.grid, cfg)
    # dynamic cost uses the visit count BEFORE this entry: 1st visit -> base, 2nd -> 2*base, ...
    cost = cfg.base_cost * (1.0 + state.visit_counts[p_next[0], p_next[1]].astype(jnp.float32))
    visit_counts = state.visit_counts.at[p_next[0], p_next[1]].add(1)
    return state.replace(agent_pos=p_next, visit_counts=visit_counts, energy=state.energy - cost)


def step(state: SearchState, action: jax.Array, cfg: EnvConfig):
    """One environment step. Returns (next_state, reward, done).

    Math:
        reached   = (p_next == goal)
        exhausted = (energy <= 0)                          # budget spent -> failure
        done      = reached | (step+1 >= max_steps) | exhausted
        reward    = R_goal*reached - time_penalty + lambda_budget*(max(energy,0)/B0)
    """
    moved = apply_action(state, action, cfg)
    step_count = moved.step_count + jnp.int32(1)
    reached = jnp.all(moved.agent_pos == moved.goal_pos)
    timeout = step_count >= cfg.max_steps
    exhausted = moved.energy <= 0.0
    done = reached | timeout | exhausted
    budget_term = cfg.lambda_budget * (jnp.maximum(moved.energy, 0.0) / cfg.b0)
    reward = cfg.r_goal * reached.astype(jnp.float32) - cfg.time_penalty + budget_term
    next_state = moved.replace(step_count=step_count, done=done)
    return next_state, reward, done


def reset(key: jax.Array, cfg: EnvConfig) -> SearchState:
    """Build a fresh solvable episode. If the (isolated-goal) map placed the agent ON the goal,
    mark it already done so the rollout never sees a degenerate start-on-goal trajectory."""
    state = generate(key, cfg)
    already_solved = jnp.all(state.agent_pos == state.goal_pos)
    return state.replace(done=already_solved)


def obs_size(cfg: EnvConfig) -> int:
    """Length of the minimal observation vector: pos(2) + rel_goal(2) + KxK patch."""
    return 4 + cfg.obs_patch * cfg.obs_patch


def obs(state: SearchState, cfg: EnvConfig) -> jax.Array:
    """Minimal observation = concat[ pos/scale , (goal-pos)/scale , KxK local patch (flattened) ].

    The KxK patch is centered on the agent; the grid is wall-padded by r=K//2 so out-of-bounds
    reads return 'wall' instead of going off-array.
    """
    H, W, K = cfg.height, cfg.width, cfg.obs_patch
    r = K // 2
    scale = jnp.array([H, W], dtype=jnp.float32)
    pos_norm = state.agent_pos.astype(jnp.float32) / scale
    rel_goal = (state.goal_pos - state.agent_pos).astype(jnp.float32) / scale
    # pad with WALL so the window is always valid; cell (i,j) -> (i+r, j+r), so a K-window
    # centered at (i,j) starts at (i, j) in padded coordinates.
    padded = jnp.pad(state.grid, r, constant_values=WALL)
    patch = lax.dynamic_slice(padded, (state.agent_pos[0], state.agent_pos[1]), (K, K))
    return jnp.concatenate([pos_norm, rel_goal, patch.reshape(-1).astype(jnp.float32)])
