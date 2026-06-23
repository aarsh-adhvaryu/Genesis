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
from genesis.state import SearchState, current_target, selected_waypoint

N_PRIMITIVES = 13
N_DIRECTIONS = 4

# P0..P12 cognitive base costs (roadmap §3 table)
PRIMITIVE_COST = jnp.array(
    [0.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0, 5.0, 2.0, 8.0, 1.0, 1.0], dtype=jnp.float32
)
# which primitives move the agent into a grid cell (pay the dynamic traversal multiplier)
_IS_MOVE = jnp.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0], dtype=bool)
# which are implemented right now (others are masked out of the action distribution)
IMPLEMENTED = jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1], dtype=bool)  # +P3,P4

P0_IDLE, P1_MOTOR, P2_WALLFOLLOW = 0, 1, 2
P3_EXPAND, P4_FRONTIER, P5_GRADIENT = 3, 4, 5
P6_WRITE, P7_READ, P9_BACKTRACK, P12_SUBGOAL = 6, 7, 9, 12

# Right-hand-rule turn tables, indexed by heading (0=up,1=down,2=left,3=right).
# A right turn (clockwise on screen) cycles up->right->down->left->up.
_RIGHT = jnp.array([3, 2, 0, 1], dtype=jnp.int32)   # turn right
_LEFT = jnp.array([2, 3, 1, 0], dtype=jnp.int32)    # turn left
_BACK = jnp.array([1, 0, 3, 2], dtype=jnp.int32)    # reverse


def _greedy_direction(state: SearchState, cfg: EnvConfig) -> jax.Array:
    """P5's auto-direction: the non-wall neighbor minimizing Manhattan distance to the CURRENT target
    (the top P12 subgoal if any, else the final goal)."""
    p = state.agent_pos
    target = current_target(state)
    dests = jnp.clip(p + DELTAS, jnp.array([0, 0]), jnp.array([cfg.height - 1, cfg.width - 1]))
    is_wall = state.grid[dests[:, 0], dests[:, 1]] == WALL
    dist = jnp.abs(dests - target).sum(axis=1).astype(jnp.float32)
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


def _manhattan_field(target: jax.Array, cfg: EnvConfig) -> jax.Array:
    """(H,W) Manhattan distance from every cell to `target` — the A* heuristic h."""
    rows = jnp.arange(cfg.height)[:, None]
    cols = jnp.arange(cfg.width)[None, :]
    return (jnp.abs(rows - target[0]) + jnp.abs(cols - target[1])).astype(jnp.float32)


def _expand_node(state: SearchState, cell: jax.Array, cfg: EnvConfig):
    """Expand one frontier `cell`: mark it visited, relax its free unvisited neighbors into the
    frontier (g[n] = min(g[n], g[cell]+1)). Returns (frontier, visited, g_cost). Agent does NOT move."""
    nbrs = jnp.clip(cell + DELTAS, jnp.array([0, 0]), jnp.array([cfg.height - 1, cfg.width - 1]))
    free = state.grid[nbrs[:, 0], nbrs[:, 1]] != WALL
    not_visited = ~state.visited[nbrs[:, 0], nbrs[:, 1]]
    addable = free & not_visited
    tentative = state.g_cost[cell[0], cell[1]] + 1.0

    new_g = state.g_cost.at[nbrs[:, 0], nbrs[:, 1]].min(jnp.where(addable, tentative, jnp.inf))
    cur = state.frontier[nbrs[:, 0], nbrs[:, 1]]
    new_frontier = state.frontier.at[nbrs[:, 0], nbrs[:, 1]].set(jnp.where(addable, True, cur))
    new_visited = state.visited.at[cell[0], cell[1]].set(True)
    new_frontier = new_frontier.at[cell[0], cell[1]].set(False)   # expanded -> leaves the frontier
    return new_frontier, new_visited, new_g


def _best_frontier_cell(state: SearchState, cfg: EnvConfig) -> jax.Array:
    """P3 (A*): the frontier cell minimizing f = g + h(goal)."""
    f = jnp.where(state.frontier, state.g_cost + _manhattan_field(state.goal_pos, cfg), jnp.inf)
    idx = jnp.argmin(f.reshape(-1))
    return jnp.stack([idx // cfg.width, idx % cfg.width]).astype(jnp.int32)


def _random_frontier_cell(key: jax.Array, state: SearchState, cfg: EnvConfig) -> jax.Array:
    """P4 (RRT): a uniformly random frontier cell (Gumbel-max over the frontier mask)."""
    flat = state.frontier.reshape(-1)
    logits = jnp.where(flat, jax.random.gumbel(key, flat.shape), -jnp.inf)
    idx = jnp.argmax(logits)
    return jnp.stack([idx // cfg.width, idx % cfg.width]).astype(jnp.int32)


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

    # --- P12 subgoal push: subgoal = clip(agent + stride * direction) ---
    is_subgoal = pid == P12_SUBGOAL
    sg = jnp.clip(p + cfg.subgoal_stride * DELTAS[direction],
                  jnp.array([0, 0]), jnp.array([cfg.height - 1, cfg.width - 1]))
    push_idx = jnp.minimum(state.goal_depth, cfg.goal_stack_size - 1)  # overwrite top when full
    new_stack = state.goal_stack.at[push_idx].set(
        jnp.where(is_subgoal, sg, state.goal_stack[push_idx])
    )
    new_depth = jnp.where(is_subgoal, jnp.minimum(state.goal_depth + 1, cfg.goal_stack_size),
                          state.goal_depth)

    # --- P3 (A* expand) / P4 (RRT sample): grow the abstract search; agent does not move ---
    is_search = (pid == P3_EXPAND) | (pid == P4_FRONTIER)
    k_use, k_next = jax.random.split(state.key)
    cell = jnp.where(pid == P3_EXPAND, _best_frontier_cell(state, cfg),
                     _random_frontier_cell(k_use, state, cfg))
    exp_frontier, exp_visited, exp_g = _expand_node(state, cell, cfg)
    do_expand = is_search & state.frontier.any()                  # nothing to expand -> no-op
    new_frontier = jnp.where(do_expand, exp_frontier, state.frontier)
    new_visited = jnp.where(do_expand, exp_visited, state.visited)
    new_g = jnp.where(do_expand, exp_g, state.g_cost)
    new_key = jnp.where(pid == P4_FRONTIER, k_next, state.key)    # only P4 consumes randomness

    return state.replace(
        agent_pos=p_next, visit_counts=visit_counts, energy=state.energy - cost,
        memory_buffer=new_buffer, mem_head=new_head, mem_count=new_count, mem_cursor=new_cursor,
        heading=new_heading, goal_stack=new_stack, goal_depth=new_depth,
        frontier=new_frontier, visited=new_visited, g_cost=new_g, key=new_key,
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

    # reaching the current SUBGOAL pops it (hierarchical decomposition); the final goal ends the episode
    at_subgoal = (moved.goal_depth > 0) & jnp.all(moved.agent_pos == current_target(moved))
    moved = moved.replace(goal_depth=jnp.where(at_subgoal, moved.goal_depth - 1, moved.goal_depth))

    step_count = moved.step_count + jnp.int32(1)
    reached = jnp.all(moved.agent_pos == moved.goal_pos)
    timeout = step_count >= cfg.max_steps
    exhausted = moved.energy <= 0.0
    done = reached | timeout | exhausted
    budget_term = cfg.lambda_budget * (jnp.maximum(moved.energy, 0.0) / cfg.b0)
    reward = cfg.r_goal * reached.astype(jnp.float32) - cfg.time_penalty + budget_term
    return moved.replace(step_count=step_count, done=done), reward, done
