"""SearchState — the live, per-episode snapshot that flows through reset/step.

Contrast with EnvConfig:
  - EnvConfig  = STATIC config, a frozen (hashable) dataclass, fixed at trace time.
  - SearchState = DYNAMIC state, a `chex.dataclass` registered as a JAX *pytree*. Being a pytree is
    what lets `jit` trace it, `vmap` add a batch axis to every field at once (1 state -> N parallel
    episodes), and `lax.scan` carry it through a rollout. JAX is functional, so we never mutate in
    place: `state.replace(field=new_value)` returns a NEW SearchState (that is how step advances).

Grid cell encoding (int8):  0 = empty, 1 = wall, 2 = goal.
Position convention: (row, col) i.e. (y, x), so it indexes directly as grid[row, col].

H and W are NOT stored here — they are static in EnvConfig and implied by the array shapes.
Stage scoping: `visit_counts` is tracked now but the traversal COST it feeds arrives in Stage 2.
"""

from __future__ import annotations

import chex


@chex.dataclass
class SearchState:
    grid: chex.Array          # (H, W) int8   — map: {0 empty, 1 wall, 2 goal}
    agent_pos: chex.Array     # (2,)  int32   — (row, col) of the agent
    goal_pos: chex.Array      # (2,)  int32   — (row, col) of the goal (cheap "reached?" check)
    visit_counts: chex.Array  # (H, W) int32  — entries per cell; feeds dynamic cost (1+visit_count)
    energy: chex.Array        # ()    float32 — remaining budget B_t (init B0); exhaustion ends episode
    step_count: chex.Array    # ()    int32   — steps taken this episode (drives max_steps timeout)
    done: chex.Array          # ()    bool_   — episode finished (reached goal or timed out)
    key: chex.Array           # (2,)  uint32  — per-episode PRNG key (reproducible, vmap-independent)
