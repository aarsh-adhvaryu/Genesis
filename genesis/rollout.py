"""rollout.py — collect on-policy experience (with auto-reset) and compute GAE advantages.

Two pieces of the PPO slice:
  collect_rollout : run the policy in N vmapped envs for T steps via lax.scan, auto-resetting any
                    env that finishes, logging a Transition per step.
  compute_gae     : turn (values, rewards, dones, bootstrap) into advantages + returns.

GAE math (per env, computed backward over t = T-1 .. 0):
    delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
    A_t     = delta_t + gamma * lam * (1 - done_t) * A_{t+1}
    return_t = A_t + V(s_t)
The (1 - done_t) factor severs advantage flow across episode boundaries.
"""

from __future__ import annotations

from typing import NamedTuple

import distrax
import jax
import jax.numpy as jnp
from jax import lax

from genesis.config import EnvConfig
from genesis.env import obs as obs_fn
from genesis.env import reset, step
from genesis.network import ActorCritic
from genesis.state import SearchState


class Transition(NamedTuple):
    """One step of experience, batched over envs (leading axis N after the rollout scan)."""

    obs: jax.Array       # (obs_dim,) the observation we acted on
    action: jax.Array    # ()   sampled action
    log_prob: jax.Array  # ()   log pi(a|s) at sample time (for the PPO ratio)
    value: jax.Array     # ()   V(s) critic estimate
    reward: jax.Array    # ()   reward received
    done: jax.Array      # ()   episode-end flag


def _reset_if_done(state: SearchState, done: jax.Array, key: jax.Array, cfg: EnvConfig) -> SearchState:
    """Single-env auto-reset: if done, swap in a fresh episode; else keep the state."""
    fresh = reset(key, cfg)
    return jax.tree_util.tree_map(lambda cur, new: jnp.where(done, new, cur), state, fresh)


def collect_rollout(net: ActorCritic, states: SearchState, key: jax.Array, cfg: EnvConfig, horizon: int):
    """Run `net` in the envs for `horizon` steps. Returns (transitions, final_states, last_value, key).

    transitions has leading shape (horizon, N); last_value (N,) bootstraps GAE.
    """
    n_envs = states.step_count.shape[0]

    def body(carry, _):
        states, key = carry
        key, k_act, k_reset = jax.random.split(key, 3)

        obs_b = jax.vmap(obs_fn, in_axes=(0, None))(states, cfg)      # (N, obs_dim)
        logits, values = jax.vmap(net)(obs_b)                         # (N, A), (N,)
        dist = distrax.Categorical(logits=logits)
        actions = dist.sample(seed=k_act)                            # (N,)
        log_probs = dist.log_prob(actions)                           # (N,)

        next_states, rewards, dones = jax.vmap(step, in_axes=(0, 0, None))(states, actions, cfg)
        # auto-reset finished envs so the scan keeps flowing
        next_states = jax.vmap(_reset_if_done, in_axes=(0, 0, 0, None))(
            next_states, dones, jax.random.split(k_reset, n_envs), cfg
        )

        transition = Transition(obs_b, actions, log_probs, values, rewards, dones)
        return (next_states, key), transition

    (final_states, key), transitions = lax.scan(body, (states, key), None, length=horizon)

    # bootstrap value for the step after the rollout's end
    final_obs = jax.vmap(obs_fn, in_axes=(0, None))(final_states, cfg)
    _, last_value = jax.vmap(net)(final_obs)                          # (N,)
    return transitions, final_states, last_value, key


def compute_gae(transitions: Transition, last_value: jax.Array, gamma: float, lam: float):
    """GAE over a rollout. Inputs are (T, N); returns advantages (T, N) and returns (T, N)."""

    def body(carry, x):
        gae, next_value = carry
        value, reward, done = x
        not_done = 1.0 - done.astype(jnp.float32)
        delta = reward + gamma * next_value * not_done - value
        gae = delta + gamma * lam * not_done * gae
        return (gae, value), gae

    init = (jnp.zeros_like(last_value), last_value)
    xs = (transitions.value, transitions.reward, transitions.done)
    _, advantages = lax.scan(body, init, xs, reverse=True)           # processes t = T-1 .. 0
    returns = advantages + transitions.value
    return advantages, returns
