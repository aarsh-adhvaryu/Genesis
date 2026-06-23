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


def _state(agent, goal, energy=None, grid=None, heading=0):
    g = jnp.ones((H, W), jnp.int8).at[1:-1, 1:-1].set(0) if grid is None else grid
    g = g.at[goal[0], goal[1]].set(2)
    return SearchState(
        grid=g, agent_pos=jnp.array(agent, jnp.int32), goal_pos=jnp.array(goal, jnp.int32),
        visit_counts=jnp.zeros((H, W), jnp.int32),
        energy=jnp.float32(cfg.b0 if energy is None else energy),
        memory_buffer=-jnp.ones((cfg.memory_k, 2), jnp.int32),
        mem_head=jnp.int32(0), mem_count=jnp.int32(0), mem_cursor=jnp.int32(0),
        heading=jnp.int32(heading),
        goal_stack=-jnp.ones((cfg.goal_stack_size, 2), jnp.int32), goal_depth=jnp.int32(0),
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


# --------------------------------------------------------------------------- memory P6/P7/P9
def test_p6_write_stores_current_position():
    s = _state((3, 4), (8, 8))
    ns = P.apply_primitive(s, jnp.int32(P.P6_WRITE), jnp.int32(0), cfg)
    assert int(ns.mem_count) == 1 and int(ns.mem_head) == 1
    assert tuple(np.asarray(ns.memory_buffer[0])) == (3, 4)   # bookmarked the position
    assert tuple(np.asarray(ns.agent_pos)) == (3, 4)          # write does not move
    assert float(s.energy - ns.energy) == float(P.PRIMITIVE_COST[P.P6_WRITE])


def test_p9_backtrack_teleports_to_written_waypoint():
    s = _state((3, 4), (8, 8))
    s = P.apply_primitive(s, jnp.int32(P.P6_WRITE), jnp.int32(0), cfg)   # bookmark (3,4)
    s = P.apply_primitive(s, jnp.int32(P.P1_MOTOR), jnp.int32(3), cfg)   # wander right -> (3,5)
    assert tuple(np.asarray(s.agent_pos)) == (3, 5)
    s = P.apply_primitive(s, jnp.int32(P.P9_BACKTRACK), jnp.int32(0), cfg)  # jump back
    assert tuple(np.asarray(s.agent_pos)) == (3, 4)


def test_p9_backtrack_is_noop_without_memory():
    s = _state((5, 5), (8, 8))
    ns = P.apply_primitive(s, jnp.int32(P.P9_BACKTRACK), jnp.int32(0), cfg)
    assert tuple(np.asarray(ns.agent_pos)) == (5, 5)          # no memory -> stays put


def test_p7_read_cursor_cycles_to_older_waypoint():
    s = _state((1, 1), (8, 8))
    s = P.apply_primitive(s, jnp.int32(P.P6_WRITE), jnp.int32(0), cfg)   # write A=(1,1)
    s = P.apply_primitive(s, jnp.int32(P.P1_MOTOR), jnp.int32(3), cfg)   # ->(1,2)
    s = P.apply_primitive(s, jnp.int32(P.P6_WRITE), jnp.int32(0), cfg)   # write B=(1,2), cursor=0->B
    from genesis.state import selected_waypoint
    sel0, _ = selected_waypoint(s, cfg.memory_k)
    assert tuple(np.asarray(sel0)) == (1, 2)                  # cursor 0 -> newest (B)
    s = P.apply_primitive(s, jnp.int32(P.P7_READ), jnp.int32(0), cfg)    # advance cursor
    sel1, _ = selected_waypoint(s, cfg.memory_k)
    assert tuple(np.asarray(sel1)) == (1, 1)                  # now older (A)


# --------------------------------------------------------------------------- P2 wall-follow
def test_p2_turns_right_in_open_space_and_updates_heading():
    s = _state((5, 5), (8, 8), heading=0)                    # facing up; right-of-up = east
    ns = P.apply_primitive(s, jnp.int32(P.P2_WALLFOLLOW), jnp.int32(0), cfg)
    assert tuple(np.asarray(ns.agent_pos)) == (5, 6)         # stepped east (right turn)
    assert int(ns.heading) == 3                              # now facing right
    assert float(s.energy - ns.energy) == float(P.PRIMITIVE_COST[P.P2_WALLFOLLOW])


def test_p2_goes_straight_when_right_is_blocked():
    g = jnp.ones((H, W), jnp.int8).at[1:-1, 1:-1].set(0)
    g = g.at[5, 6].set(1)                                    # wall to the east (the "right" of up)
    s = _state((5, 5), (8, 8), grid=g, heading=0)
    ns = P.apply_primitive(s, jnp.int32(P.P2_WALLFOLLOW), jnp.int32(0), cfg)
    assert tuple(np.asarray(ns.agent_pos)) == (4, 5)         # can't turn right -> go straight (up)
    assert int(ns.heading) == 0                              # still facing up


def test_p2_is_implemented_in_mask():
    s = _state((5, 5), (8, 8))
    assert bool(np.asarray(P.action_mask(s, cfg))[P.P2_WALLFOLLOW])


# --------------------------------------------------------------------------- P12 subgoal
def test_p12_pushes_subgoal_in_chosen_direction():
    from genesis.state import current_target
    s = _state((5, 5), (1, 1))                                       # final goal up-left
    ns = P.apply_primitive(s, jnp.int32(P.P12_SUBGOAL), jnp.int32(3), cfg)  # direction = right
    assert int(ns.goal_depth) == 1
    expected = (5, min(5 + cfg.subgoal_stride, W - 1))
    assert tuple(np.asarray(current_target(ns))) == expected
    assert float(s.energy - ns.energy) == float(P.PRIMITIVE_COST[P.P12_SUBGOAL])


def test_p5_targets_subgoal_after_p12():
    s = _state((5, 5), (1, 1))                                       # P5 alone -> up/left toward goal
    s = P.apply_primitive(s, jnp.int32(P.P12_SUBGOAL), jnp.int32(3), cfg)   # subgoal to the right
    ns = P.apply_primitive(s, jnp.int32(P.P5_GRADIENT), jnp.int32(0), cfg)
    assert tuple(np.asarray(ns.agent_pos)) == (5, 6)                 # greedy now heads to the subgoal


def test_reaching_subgoal_pops_it():
    s = _state((5, 5), (1, 1)).replace(
        goal_stack=jnp.array([[5, 6], [-1, -1], [-1, -1], [-1, -1]], jnp.int32),
        goal_depth=jnp.int32(1),
    )
    ns, _, done = P.step_primitive(s, jnp.int32(P.P1_MOTOR), jnp.int32(3), cfg)  # step onto subgoal
    assert tuple(np.asarray(ns.agent_pos)) == (5, 6)
    assert int(ns.goal_depth) == 0                                   # subgoal popped
    assert not bool(done)                                            # not the final goal -> continues
