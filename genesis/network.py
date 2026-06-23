"""network.py — the minimal actor-critic controller (Equinox MLP).

One shared trunk, two heads:
  actor  : obs -> logits over the N discrete actions  (a Categorical policy pi(a|s))
  critic : obs -> scalar value V(s)                    (the baseline for low-variance advantages)

Operates on a SINGLE observation vector; batch with jax.vmap. This is the Phase 1-4 controller
(MLP, ~2 layers); the Transformer upgrade is Phase 5+. Action space is the Stage-1 motor (4 moves)
and widens to the 13 primitives later without changing this file's shape contract.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


class ActorCritic(eqx.Module):
    """Factored actor-critic: shared trunk -> primitive head (13) + direction head (4) + critic.

    The direction head parameterizes P1 (Motor); other primitives ignore it. Joint policy factorizes
    as pi(prim, dir | s) = pi_prim(prim|s) * pi_dir(dir|s), so log-probs and entropies simply add.
    """

    trunk: list
    prim_head: eqx.nn.Linear
    dir_head: eqx.nn.Linear
    critic_head: eqx.nn.Linear

    def __init__(self, obs_dim: int, n_primitives: int, n_directions: int, hidden: int, *, key):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.trunk = [
            eqx.nn.Linear(obs_dim, hidden, key=k1),
            eqx.nn.Linear(hidden, hidden, key=k2),
        ]
        self.prim_head = eqx.nn.Linear(hidden, n_primitives, key=k3)
        self.dir_head = eqx.nn.Linear(hidden, n_directions, key=k4)
        self.critic_head = eqx.nn.Linear(hidden, 1, key=k5)

    def __call__(self, obs: jax.Array):
        """obs: (obs_dim,) -> (prim_logits (13,), dir_logits (4,), value scalar)."""
        x = obs
        for layer in self.trunk:
            x = jnp.tanh(layer(x))
        return self.prim_head(x), self.dir_head(x), self.critic_head(x)[0]
