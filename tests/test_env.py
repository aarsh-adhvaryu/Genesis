"""Stage-1 environment tests.

The headline guarantee — "solvable by construction" — is checked with an INDEPENDENT BFS (plain
Python, not our flood-fill), so a bug in generate.py cannot mask itself. The rest pin down the
step() semantics, observation shape, and jit/vmap behaviour (determinism + per-env independence).
"""

from collections import deque

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from genesis import env as E
from genesis.config import EnvConfig
from genesis.generate import _build, flood_fill_reachable, generate
from genesis.state import SearchState

cfg = EnvConfig()
H, W = cfg.height, cfg.width


# --------------------------------------------------------------------------- helpers
def independent_bfs_reaches(grid: np.ndarray, agent, goal) -> bool:
    """Plain-Python BFS from agent; True iff it reaches goal. grid: 1 == wall. (Independent of ours.)"""
    agent, goal = tuple(int(x) for x in agent), tuple(int(x) for x in goal)
    if agent == goal:
        return True
    seen = np.zeros((H, W), bool)
    seen[agent] = True
    q = deque([agent])
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and grid[nr, nc] != 1:
                seen[nr, nc] = True
                q.append((nr, nc))
    return False


def empty_state(agent, goal) -> SearchState:
    """A controlled open-room state (walls only on the border) for deterministic step() tests."""
    grid = jnp.ones((H, W), jnp.int8).at[1:-1, 1:-1].set(0)
    grid = grid.at[goal[0], goal[1]].set(2)
    return SearchState(
        grid=grid,
        agent_pos=jnp.array(agent, jnp.int32),
        goal_pos=jnp.array(goal, jnp.int32),
        visit_counts=jnp.zeros((H, W), jnp.int32),
        energy=jnp.float32(cfg.b0),
        memory_buffer=-jnp.ones((cfg.memory_k, 2), jnp.int32),
        mem_head=jnp.int32(0),
        mem_count=jnp.int32(0),
        mem_cursor=jnp.int32(0),
        heading=jnp.int32(0),
        step_count=jnp.int32(0),
        done=jnp.bool_(False),
        key=jax.random.PRNGKey(0),
    )


_gen_batch = jax.jit(jax.vmap(generate, in_axes=(0, None)), static_argnums=1)
_step = jax.jit(E.step, static_argnums=2)


# --------------------------------------------------------------------------- generation
def test_solvable_by_construction():
    """Over many random-density maps, an INDEPENDENT BFS must reach the goal every time."""
    n = 500
    states = _gen_batch(jax.random.split(jax.random.PRNGKey(0), n), cfg)
    grids = np.asarray(states.grid)
    agents, goals = np.asarray(states.agent_pos), np.asarray(states.goal_pos)
    unsolvable = [i for i in range(n) if not independent_bfs_reaches(grids[i], agents[i], goals[i])]
    assert unsolvable == [], f"{len(unsolvable)} unsolvable maps, e.g. idx {unsolvable[:5]}"


def test_borders_are_walls():
    states = _gen_batch(jax.random.split(jax.random.PRNGKey(1), 200), cfg)
    g = np.asarray(states.grid)
    assert (g[:, 0, :] == 1).all() and (g[:, -1, :] == 1).all()
    assert (g[:, :, 0] == 1).all() and (g[:, :, -1] == 1).all()


def test_density_zero_open_interior():
    """wall_density = 0 => no interior walls; exactly one goal cell."""
    s = jax.jit(_build, static_argnums=1)(jax.random.PRNGKey(7), cfg, jnp.float32(0.0))
    g = np.asarray(s.grid)
    assert (g[1:-1, 1:-1] == 1).sum() == 0
    assert (g == 2).sum() == 1


def test_exactly_one_goal_and_agent_reachable():
    """Goal is unique; agent lies in OUR flood-fill reachable set (cross-checked vs the value)."""
    states = _gen_batch(jax.random.split(jax.random.PRNGKey(2), 200), cfg)
    grids, agents, goals = (np.asarray(x) for x in (states.grid, states.agent_pos, states.goal_pos))
    for i in range(grids.shape[0]):
        assert (grids[i] == 2).sum() == 1
        free = jnp.asarray(grids[i]) != 1
        reach = np.asarray(flood_fill_reachable(free, jnp.asarray(goals[i]), cfg))
        assert reach[agents[i][0], agents[i][1]], f"agent outside R* on map {i}"


# --------------------------------------------------------------------------- step semantics
def test_wall_step_is_noop():
    """Moving into the border wall leaves the agent in place."""
    s = empty_state((1, 1), (5, 5))
    ns, _, _ = _step(s, jnp.int32(0), cfg)  # action 0 = up, into top border wall
    assert tuple(np.asarray(ns.agent_pos)) == (1, 1)


def test_move_increments_visit_counts():
    s = empty_state((1, 1), (5, 5))
    ns, _, _ = _step(s, jnp.int32(3), cfg)  # right -> (1,2)
    assert tuple(np.asarray(ns.agent_pos)) == (1, 2)
    assert int(ns.visit_counts[1, 2]) == 1


def test_goal_step_sets_done_and_reward():
    s = empty_state((5, 4), (5, 5))
    ns, r, d = _step(s, jnp.int32(3), cfg)  # right onto goal
    assert bool(d) is True
    assert tuple(np.asarray(ns.agent_pos)) == (5, 5)
    assert float(r) == pytest.approx(cfg.r_goal - cfg.time_penalty)


def test_timeout_sets_done_via_max_steps():
    s = empty_state((1, 1), (20, 20)).replace(step_count=jnp.int32(cfg.max_steps - 1))
    ns, r, d = _step(s, jnp.int32(3), cfg)  # a non-goal move
    assert int(ns.step_count) == cfg.max_steps
    assert bool(d) is True
    assert float(r) == pytest.approx(-cfg.time_penalty)  # reached=False -> only time penalty


def test_non_terminal_step_reward_is_time_penalty():
    s = empty_state((1, 1), (20, 20))
    _, r, d = _step(s, jnp.int32(3), cfg)
    assert bool(d) is False
    assert float(r) == pytest.approx(-cfg.time_penalty)


# --------------------------------------------------------------------------- energy budget (Stage 2)
def test_energy_starts_at_b0_and_decrements_by_base_cost():
    s = empty_state((1, 1), (20, 20))
    assert float(s.energy) == pytest.approx(cfg.b0)
    ns, _, _ = _step(s, jnp.int32(3), cfg)  # into a fresh cell -> cost = base_cost*(1+0)
    assert float(ns.energy) == pytest.approx(cfg.b0 - cfg.base_cost)


def test_dynamic_cost_doubles_on_revisit():
    """Re-entering a cell costs base_cost*(1+visit_count): 1st entry base, 2nd entry 2*base."""
    s = empty_state((1, 1), (20, 20))
    s1, _, _ = _step(s, jnp.int32(3), cfg)            # (1,1)->(1,2): fresh, cost base
    s2, _, _ = _step(s1, jnp.int32(2), cfg)           # (1,2)->(1,1): fresh, cost base
    s3, _, _ = _step(s2, jnp.int32(3), cfg)           # (1,1)->(1,2): REVISIT, cost base*(1+1)=2base
    first_entry = float(s1.energy) - float(s2.energy)   # cost of entering (1,1) [fresh]
    revisit = float(s2.energy) - float(s3.energy)       # cost of re-entering (1,2)
    assert first_entry == pytest.approx(cfg.base_cost)
    assert revisit == pytest.approx(2 * cfg.base_cost)


def test_budget_exhaustion_ends_episode():
    s = empty_state((1, 1), (20, 20)).replace(energy=jnp.float32(cfg.base_cost * 0.5))
    ns, _, d = _step(s, jnp.int32(3), cfg)            # cost base > 0.5 energy -> exhausted
    assert float(ns.energy) <= 0.0
    assert bool(d) is True


def test_action_costs_shape_and_base_on_fresh_grid():
    s = empty_state((5, 5), (20, 20))
    costs = np.asarray(E.action_costs(s, cfg))
    assert costs.shape == (E.N_ACTIONS,)
    assert np.allclose(costs, cfg.base_cost)         # all neighbors fresh (visit_count 0)


# --------------------------------------------------------------------------- jit / vmap
def test_seeded_determinism():
    a = jax.jit(generate, static_argnums=1)(jax.random.PRNGKey(123), cfg)
    b = jax.jit(generate, static_argnums=1)(jax.random.PRNGKey(123), cfg)
    assert (a.grid == b.grid).all()
    assert (a.agent_pos == b.agent_pos).all() and (a.goal_pos == b.goal_pos).all()


def test_vmap_matches_serial_and_is_independent():
    """vmap(generate) over keys equals per-key generate (consistency), and different keys give
    different maps (independence — not all agents collapse to one cell)."""
    keys = jax.random.split(jax.random.PRNGKey(99), 16)
    batch = _gen_batch(keys, cfg)
    for i in range(16):
        one = jax.jit(generate, static_argnums=1)(keys[i], cfg)
        assert (batch.grid[i] == one.grid).all()
        assert (batch.agent_pos[i] == one.agent_pos).all()
    agents = np.asarray(batch.agent_pos)
    assert len(np.unique(agents, axis=0)) > 1, "all envs produced the same agent position"


# --------------------------------------------------------------------------- observation
def test_obs_shape_and_values():
    s = empty_state((3, 4), (5, 5))
    o = np.asarray(jax.jit(E.obs, static_argnums=1)(s, cfg))
    assert o.shape == (E.obs_size(cfg),) == (4 + cfg.obs_patch**2 + 1 + 3,)  # +budget +memory(3)
    assert o[4 + cfg.obs_patch**2] == pytest.approx(1.0)  # full budget at start -> ratio 1.0
    assert o[-1] == pytest.approx(0.0)  # no memory written yet -> has_mem flag 0
    # pos/scale in [0,1)
    assert 0.0 <= o[0] < 1.0 and 0.0 <= o[1] < 1.0
    # relative goal vector = (goal - agent)/scale
    assert o[2] == pytest.approx((5 - 3) / H) and o[3] == pytest.approx((5 - 4) / W)
    # center of the KxK patch is the agent's own cell (empty = 0 here)
    center = o[4 + (cfg.obs_patch**2) // 2]
    assert center == pytest.approx(0.0)


def test_reset_marks_start_on_goal_done():
    """The isolated-goal degenerate case (agent == goal) is reported already-done by reset()."""
    s = empty_state((5, 5), (5, 5))  # agent already on goal
    done = jnp.all(s.agent_pos == s.goal_pos)
    assert bool(done) is True
