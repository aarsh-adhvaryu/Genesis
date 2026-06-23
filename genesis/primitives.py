"""primitives.py — the factored primitive action interface (proposal §4.1, roadmap §3).

Action = (primitive_id in 0..12, direction in 0..3). The direction head is consumed ONLY by P1
(Motor); every other primitive auto-computes its direction or ignores it. This keeps the action
vocabulary the clean "13 primitives" (for procedure mining) while giving P1 directional freedom.

Energy cost model (roadmap §3 table + §4 dynamic costs):
  - each primitive has a cognitive base cost PRIMITIVE_COST[pid];
  - MOVING primitives additionally scale by the anti-spam multiplier (1 + visit_count[dest]) — using
    the primitive cost as the base, so re-entering a cell is linearly more expensive;
  - non-moving primitives pay the flat base cost.
Budget masking forbids unaffordable or not-yet-implemented primitives; P0 (Idle, cost 0) is always
legal so a valid action always exists.

Implemented now: P0 Idle, P1 Motor, P5 Gradient Step (no new subsystems needed). The rest are
declared for a stable 13-wide head but masked off until their machinery (memory K=16, frontier,
fog, scout) is built — added one unit at a time.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from genesis.config import EnvConfig
from genesis.env import DELTAS, WALL, _dest_cell
from genesis.state import SearchState

N_PRIMITIVES = 13
N_DIRECTIONS = 4

# P0..P12 cognitive base costs (roadmap §3 table)
PRIMITIVE_COST = jnp.array(
    [0.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0, 5.0, 2.0, 8.0, 1.0, 1.0], dtype=jnp.float32
)
# which primitives move the agent (pay the dynamic traversal multiplier)
_IS_MOVE = jnp.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0], dtype=bool)
# which are implemented right now (others are masked out of the action distribution)
IMPLEMENTED = jnp.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=bool)  # P0, P1, P5

P0_IDLE, P1_MOTOR, P5_GRADIENT = 0, 1, 5


def _greedy_direction(state: SearchState, cfg: EnvConfig) -> jax.Array:
    """P5's auto-direction: the non-wall neighbor minimizing Manhattan distance to the goal."""
    p = state.agent_pos
    dests = jnp.clip(p + DELTAS, jnp.array([0, 0]), jnp.array([cfg.height - 1, cfg.width - 1]))
    is_wall = state.grid[dests[:, 0], dests[:, 1]] == WALL
    dist = jnp.abs(dests - state.goal_pos).sum(axis=1).astype(jnp.float32)
    dist = jnp.where(is_wall, jnp.inf, dist)            # never step into a wall
    return jnp.argmin(dist).astype(jnp.int32)


def apply_primitive(state: SearchState, pid: jax.Array, direction: jax.Array, cfg: EnvConfig):
    """Apply one primitive. Updates agent_pos / visit_counts / energy. Returns the new SearchState.

    P1 uses `direction`; P5 uses the greedy-toward-goal direction; P0 does not move.
    """
    move_dir = jnp.where(pid == P1_MOTOR, direction,
                         jnp.where(pid == P5_GRADIENT, _greedy_direction(state, cfg), jnp.int32(0)))
    is_move = _IS_MOVE[pid]

    p = state.agent_pos
    p_next = jnp.where(is_move, _dest_cell(p, move_dir, state.grid, cfg), p)
    vc_before = state.visit_counts[p_next[0], p_next[1]].astype(jnp.float32)

    base = PRIMITIVE_COST[pid] * cfg.base_cost
    cost = jnp.where(is_move, base * (1.0 + vc_before), base)   # movers scale with revisits
    visit_counts = state.visit_counts.at[p_next[0], p_next[1]].add(is_move.astype(jnp.int32))
    return state.replace(agent_pos=p_next, visit_counts=visit_counts, energy=state.energy - cost)


def action_mask(state: SearchState, cfg: EnvConfig) -> jax.Array:
    """(N_PRIMITIVES,) bool: primitive is legal iff implemented AND affordable. P0 is always legal."""
    mask = IMPLEMENTED & (PRIMITIVE_COST <= state.energy)
    return mask.at[P0_IDLE].set(True)


def step_primitive(state: SearchState, pid: jax.Array, direction: jax.Array, cfg: EnvConfig):
    """One environment step under the primitive interface. Returns (next_state, reward, done).

    done = reached | timeout | budget exhausted ; reward = R_goal*reached - tp + lambda*(B_t/B0).
    """
    moved = apply_primitive(state, pid, direction, cfg)
    step_count = moved.step_count + jnp.int32(1)
    reached = jnp.all(moved.agent_pos == moved.goal_pos)
    timeout = step_count >= cfg.max_steps
    exhausted = moved.energy <= 0.0
    done = reached | timeout | exhausted
    budget_term = cfg.lambda_budget * (jnp.maximum(moved.energy, 0.0) / cfg.b0)
    reward = cfg.r_goal * reached.astype(jnp.float32) - cfg.time_penalty + budget_term
    return moved.replace(step_count=step_count, done=done), reward, done
