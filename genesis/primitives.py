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
from genesis.state import SearchState, selected_waypoint

N_PRIMITIVES = 13
N_DIRECTIONS = 4

# P0..P12 cognitive base costs (roadmap §3 table)
PRIMITIVE_COST = jnp.array(
    [0.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0, 5.0, 2.0, 8.0, 1.0, 1.0], dtype=jnp.float32
)
# which primitives move the agent into a grid cell (pay the dynamic traversal multiplier)
_IS_MOVE = jnp.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0], dtype=bool)
# which are implemented right now (others are masked out of the action distribution)
IMPLEMENTED = jnp.array([1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0], dtype=bool)  # P0,P1,P2,P5,P6,P7,P9

P0_IDLE, P1_MOTOR, P2_WALLFOLLOW, P5_GRADIENT = 0, 1, 2, 5
P6_WRITE, P7_READ, P9_BACKTRACK = 6, 7, 9

# Right-hand-rule turn tables, indexed by heading (0=up,1=down,2=left,3=right).
# A right turn (clockwise on screen) cycles up->right->down->left->up.
_RIGHT = jnp.array([3, 2, 0, 1], dtype=jnp.int32)   # turn right
_LEFT = jnp.array([2, 3, 1, 0], dtype=jnp.int32)    # turn left
_BACK = jnp.array([1, 0, 3, 2], dtype=jnp.int32)    # reverse


def _greedy_direction(state: SearchState, cfg: EnvConfig) -> jax.Array:
    """P5's auto-direction: the non-wall neighbor minimizing Manhattan distance to the goal."""
    p = state.agent_pos
    dests = jnp.clip(p + DELTAS, jnp.array([0, 0]), jnp.array([cfg.height - 1, cfg.width - 1]))
    is_wall = state.grid[dests[:, 0], dests[:, 1]] == WALL
    dist = jnp.abs(dests - state.goal_pos).sum(axis=1).astype(jnp.float32)
    dist = jnp.where(is_wall, jnp.inf, dist)            # never step into a wall
    return jnp.argmin(dist).astype(jnp.int32)


def _wall_follow(state: SearchState, cfg: EnvConfig):
    """P2's right-hand rule. Returns (chosen_abs_direction, can_move).

    Try directions in priority order [right-turn, straight, left-turn, back]; take the first open
    one. can_move is False only if fully enclosed (no open neighbor).
    """
    h = state.heading
    cand = jnp.stack([_RIGHT[h], h, _LEFT[h], _BACK[h]])             # (4,) abs dirs, priority order
    dests = jnp.clip(state.agent_pos + DELTAS[cand],
                     jnp.array([0, 0]), jnp.array([cfg.height - 1, cfg.width - 1]))
    is_free = state.grid[dests[:, 0], dests[:, 1]] != WALL          # (4,)
    # pick the FIRST free candidate: weight by descending priority, argmax
    chosen = jnp.argmax(is_free.astype(jnp.int32) * jnp.array([4, 3, 2, 1]))
    return cand[chosen], is_free.any()


def apply_primitive(state: SearchState, pid: jax.Array, direction: jax.Array, cfg: EnvConfig):
    """Apply one primitive (branchless). Updates agent_pos / visit_counts / energy / memory.

    Movement: P1 uses `direction`; P5 the greedy-toward-goal direction; P9 teleports to the selected
    waypoint. Memory: P6 writes the current position (FIFO ring), P7 advances the read-cursor.
    """
    K = cfg.memory_k
    p = state.agent_pos

    # --- destination by primitive type ---
    wf_dir, wf_can = _wall_follow(state, cfg)                    # P2 wall-follow direction
    move_dir = jnp.where(pid == P1_MOTOR, direction,
               jnp.where(pid == P5_GRADIENT, _greedy_direction(state, cfg),
               jnp.where(pid == P2_WALLFOLLOW, wf_dir, jnp.int32(0))))
    p_grid = _dest_cell(p, move_dir, state.grid, cfg)           # P1/P2/P5 step destination
    sel_pos, has_mem = selected_waypoint(state, K)              # P9 teleport target
    is_grid_move = (pid == P1_MOTOR) | (pid == P5_GRADIENT) | ((pid == P2_WALLFOLLOW) & wf_can)
    is_backtrack = pid == P9_BACKTRACK
    p_next = jnp.where(is_grid_move, p_grid, jnp.where(is_backtrack & has_mem, sel_pos, p))
    is_move = is_grid_move | (is_backtrack & has_mem)           # backtrack moves only if memory exists
    # P2 updates the heading to the direction it actually stepped
    new_heading = jnp.where((pid == P2_WALLFOLLOW) & wf_can, wf_dir, state.heading)

    # --- energy cost: movers scale by (1+visit_count[dest]); others flat ---
    vc_before = state.visit_counts[p_next[0], p_next[1]].astype(jnp.float32)
    base = PRIMITIVE_COST[pid] * cfg.base_cost
    cost = jnp.where(is_move, base * (1.0 + vc_before), base)
    visit_counts = state.visit_counts.at[p_next[0], p_next[1]].add(is_move.astype(jnp.int32))

    # --- memory updates ---
    is_read = pid == P7_READ
    cursor_after_read = jnp.where(
        is_read & (state.mem_count > 0),
        (state.mem_cursor + 1) % jnp.maximum(state.mem_count, 1),   # cycle to an older waypoint
        state.mem_cursor,
    )
    is_write = pid == P6_WRITE
    head = state.mem_head
    new_buffer = state.memory_buffer.at[head].set(
        jnp.where(is_write, p, state.memory_buffer[head])          # write current pos (else no-op)
    )
    new_head = jnp.where(is_write, (head + 1) % K, head)
    new_count = jnp.where(is_write, jnp.minimum(state.mem_count + 1, K), state.mem_count)
    new_cursor = jnp.where(is_write, jnp.int32(0), cursor_after_read)  # write resets cursor to newest

    return state.replace(
        agent_pos=p_next, visit_counts=visit_counts, energy=state.energy - cost,
        memory_buffer=new_buffer, mem_head=new_head, mem_count=new_count, mem_cursor=new_cursor,
        heading=new_heading,
    )


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
