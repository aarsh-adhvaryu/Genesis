"""Tests for the factored primitive action interface (P0/P1/P5 + budget masking).

Per the roadmap: unit-test each primitive independently. Covers the implemented subset; the masked-
off primitives (P2-P4, P6-P12) get tests as they are built.
"""

import jax
import jax.numpy as jnp
import numpy as np

from genesis import primitives as P
from genesis.config import EnvConfig
from genesis.state import SearchState

cfg = EnvConfig(height=11, width=11)
H, W = cfg.height, cfg.width


def _state(agent, goal, energy=None, grid=None):
    g = jnp.ones((H, W), jnp.int8).at[1:-1, 1:-1].set(0) if grid is None else grid
    g = g.at[goal[0], goal[1]].set(2)
    return SearchState(
        grid=g, agent_pos=jnp.array(agent, jnp.int32), goal_pos=jnp.array(goal, jnp.int32),
        visit_counts=jnp.zeros((H, W), jnp.int32),
        energy=jnp.float32(cfg.b0 if energy is None else energy),
        step_count=jnp.int32(0), done=jnp.bool_(False), key=jax.random.PRNGKey(0),
    )


def test_p0_idle_does_not_move_and_is_free():
    s = _state((5, 5), (8, 8))
    ns = P.apply_primitive(s, jnp.int32(P.P0_IDLE), jnp.int32(0), cfg)
    assert tuple(np.asarray(ns.agent_pos)) == (5, 5)
    assert float(ns.energy) == float(s.energy)             # cost 0
    assert int(ns.visit_counts.sum()) == 0                 # no cell entered


def test_p1_motor_moves_in_given_direction_and_costs_base():
    s = _state((5, 5), (8, 8))
    ns = P.apply_primitive(s, jnp.int32(P.P1_MOTOR), jnp.int32(3), cfg)  # 3 = right
    assert tuple(np.asarray(ns.agent_pos)) == (5, 6)
    assert float(s.energy - ns.energy) == float(P.PRIMITIVE_COST[P.P1_MOTOR])  # base, fresh cell


def test_p5_gradient_moves_toward_goal():
    s = _state((5, 5), (5, 9))  # goal to the right -> greedy should step right
    ns = P.apply_primitive(s, jnp.int32(P.P5_GRADIENT), jnp.int32(0), cfg)
    assert tuple(np.asarray(ns.agent_pos)) == (5, 6)
    # P5 cost = its base (fresh cell)
    assert float(s.energy - ns.energy) == float(P.PRIMITIVE_COST[P.P5_GRADIENT])


def test_mask_only_implemented_and_p0_always_legal():
    s = _state((5, 5), (8, 8))
    m = np.asarray(P.action_mask(s, cfg))
    assert m.shape == (P.N_PRIMITIVES,)
    assert m[P.P0_IDLE] and m[P.P1_MOTOR] and m[P.P5_GRADIENT]
    # unimplemented primitives masked off
    assert not m[P.P0_IDLE + 3]  # P3 not implemented
    # P0 stays legal even with ~zero budget
    s_broke = _state((5, 5), (8, 8), energy=0.0001)
    assert bool(np.asarray(P.action_mask(s_broke, cfg))[P.P0_IDLE])


def test_mask_forbids_unaffordable_primitive():
    # energy below P5's cost (2) but above P1's (1): P5 masked, P1 still legal
    s = _state((5, 5), (8, 8), energy=1.5)
    m = np.asarray(P.action_mask(s, cfg))
    assert m[P.P1_MOTOR] and not m[P.P5_GRADIENT]
