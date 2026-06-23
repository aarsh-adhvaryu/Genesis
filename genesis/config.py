"""EnvConfig — the immutable bag of constants that defines one GENESIS world type.

Why a frozen dataclass (and not a chex.dataclass):
  - chex.dataclass = a *pytree* of traced JAX arrays (used for SearchState, which flows
    through jit/vmap and changes every step).
  - EnvConfig = *static* configuration. `height`/`width` set array SHAPES, and JAX compiles
    per-shape, so they must be plain Python ints fixed at trace time. `frozen=True` makes the
    whole object hashable, so it can be passed as a `static_argname` to `jax.jit`.

Stage scoping (per CLAUDE.md): Stage 1 uses only the structural + reward + generation fields.
The energy fields (`base_cost`, `b0`) are DECLARED now for a stable struct but are not consumed
until Stage 2 wires in the budget/cost system. Keeping them here avoids reshaping config later.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    # --- Structural (STATIC to JAX: these determine array shapes) ---
    height: int = 32          # grid rows (H)
    width: int = 32           # grid cols (W)
    obs_patch: int = 5        # side length K of the KxK local patch in the minimal obs (odd)
    max_steps: int = 256      # episode horizon; ~4x the 32-grid Manhattan diameter (~62) for slack

    # --- Reward (Stage 1 uses r_goal + time_penalty only) ---
    r_goal: float = 50.0          # one-time reward on reaching the goal
    time_penalty: float = 0.01    # small per-step cost; pushes toward shorter routes

    # --- Grid generation: domain-randomization ranges (adversary-driven in Stage 6) ---
    # Stage 1 starts near-empty rooms; BFS flood-fill guarantees solvability regardless of density.
    wall_density_min: float = 0.0
    wall_density_max: float = 0.25

    # --- Energy budget + dynamic traversal cost (Stage 2) ---
    base_cost: float = 1.0       # base traversal cost -> dynamic cost = base_cost * (1 + visit_count)
    b0: float = 100.0            # initial energy budget B0
    lambda_budget: float = 0.0   # weight on the per-step budget-preservation reward lambda*(B_t/B0)

    # --- Working memory (P6/P7/P9): K-slot FIFO ring of bookmarked waypoints ---
    memory_k: int = 16           # number of waypoint slots

    def __post_init__(self) -> None:
        # Shapes must be positive.
        assert self.height > 0 and self.width > 0, "grid dims must be positive"
        # The local obs patch is centered on the agent, so its side must be odd.
        assert self.obs_patch % 2 == 1, "obs_patch must be odd (centered on the agent)"
        assert self.max_steps > 0, "max_steps must be positive"
        # Generation ranges must be a valid sub-interval of [0, 1).
        assert 0.0 <= self.wall_density_min <= self.wall_density_max < 1.0, "bad wall-density range"

        # Reward-scaling invariant (roadmap section 6): the goal reward must dwarf the worst
        # possible accumulated cost, or the agent could profit by failing on purpose.
        #   max_episode_cost = max_steps * time_penalty   (Stage 1 has no energy cost yet)
        #   R_goal must be 10-50x that.
        max_episode_cost = self.max_steps * self.time_penalty
        ratio = self.r_goal / max_episode_cost
        assert 10.0 <= ratio <= 50.0, (
            f"R_goal/max_episode_cost = {ratio:.1f}x, must be in [10, 50] "
            f"(R_goal={self.r_goal}, max_episode_cost={max_episode_cost})"
        )
