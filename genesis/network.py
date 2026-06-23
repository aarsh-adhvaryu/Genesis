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
    trunk: list
    actor_head: eqx.nn.Linear
    critic_head: eqx.nn.Linear

    def __init__(self, obs_dim: int, n_actions: int, hidden: int, *, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.trunk = [
            eqx.nn.Linear(obs_dim, hidden, key=k1),
            eqx.nn.Linear(hidden, hidden, key=k2),
        ]
        self.actor_head = eqx.nn.Linear(hidden, n_actions, key=k3)
        self.critic_head = eqx.nn.Linear(hidden, 1, key=k4)

    def __call__(self, obs: jax.Array):
        """obs: (obs_dim,) -> (logits (n_actions,), value scalar)."""
        x = obs
        for layer in self.trunk:
            x = jnp.tanh(layer(x))
        logits = self.actor_head(x)
        value = self.critic_head(x)[0]  # (1,) -> scalar
        return logits, value
