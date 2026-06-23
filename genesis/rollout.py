"""rollout.py — on-policy experience collection (factored primitive actions) + GAE.

collect_rollout : run the factored policy in N vmapped envs for T steps via lax.scan, auto-resetting
                  finished envs, logging a Transition per step. The primitive head is budget-masked
                  (illegal primitives -> -inf) before sampling; the direction head is unmasked.
compute_gae     : (values, rewards, dones, bootstrap) -> advantages + returns.

Joint policy pi(prim,dir|s) = pi_prim * pi_dir  =>  log_prob = logp_prim + logp_dir.
GAE (backward over t):  delta_t = r_t + gamma V(s_{t+1})(1-done_t) - V(s_t);
                        A_t = delta_t + gamma lam (1-done_t) A_{t+1};  return_t = A_t + V(s_t).
"""

from __future__ import annotations

from typing import NamedTuple

import distrax
import jax
import jax.numpy as jnp
from jax import lax

from genesis.config import EnvConfig
from genesis.env import obs as obs_fn
from genesis.env import reset
from genesis.network import ActorCritic
from genesis.primitives import action_mask, step_primitive
from genesis.state import SearchState

_NEG_INF = -1e9  # masked-primitive logit


class Transition(NamedTuple):
    """One factored-action step, batched over envs (leading axis N after the rollout scan)."""

    obs: jax.Array        # (obs_dim,)
    primitive: jax.Array  # () chosen primitive id
    direction: jax.Array  # () chosen direction
    log_prob: jax.Array   # () logp_prim + logp_dir at sample time
    value: jax.Array      # () V(s)
    reward: jax.Array      # ()
    done: jax.Array        # ()
    prim_mask: jax.Array  # (N_PRIMITIVES,) legal-primitive mask used at sample time


def _reset_if_done(state: SearchState, done: jax.Array, key: jax.Array, cfg: EnvConfig) -> SearchState:
    fresh = reset(key, cfg)
    return jax.tree_util.tree_map(lambda cur, new: jnp.where(done, new, cur), state, fresh)


def _policy(net, state, cfg):
    """Per-env: mask primitive logits, build the two Categoricals. Returns (prim_dist, dir_dist, value, mask)."""
    o = obs_fn(state, cfg)
    prim_logits, dir_logits, value = net(o)
    mask = action_mask(state, cfg)
    prim_logits = jnp.where(mask, prim_logits, _NEG_INF)
    return distrax.Categorical(logits=prim_logits), distrax.Categorical(logits=dir_logits), value, mask


def collect_rollout(net: ActorCritic, states: SearchState, key: jax.Array, cfg: EnvConfig, horizon: int):
    """Run `net` for `horizon` steps. Returns (transitions[T,N], final_states, last_value[N], key)."""
    n_envs = states.step_count.shape[0]

    def body(carry, _):
        states, key = carry
        key, k_prim, k_dir, k_reset = jax.random.split(key, 4)

        obs_b = jax.vmap(obs_fn, in_axes=(0, None))(states, cfg)
        prim_dist, dir_dist, values, masks = jax.vmap(_policy, in_axes=(None, 0, None))(net, states, cfg)
        primitives = prim_dist.sample(seed=k_prim)
        directions = dir_dist.sample(seed=k_dir)
        log_probs = prim_dist.log_prob(primitives) + dir_dist.log_prob(directions)

        next_states, rewards, dones = jax.vmap(step_primitive, in_axes=(0, 0, 0, None))(
            states, primitives, directions, cfg
        )
        next_states = jax.vmap(_reset_if_done, in_axes=(0, 0, 0, None))(
            next_states, dones, jax.random.split(k_reset, n_envs), cfg
        )

        transition = Transition(obs_b, primitives, directions, log_probs, values, rewards, dones, masks)
        return (next_states, key), transition

    (final_states, key), transitions = lax.scan(body, (states, key), None, length=horizon)

    final_obs = jax.vmap(obs_fn, in_axes=(0, None))(final_states, cfg)
    _, _, last_value = jax.vmap(net)(final_obs)
    return transitions, final_states, last_value, key


def compute_gae(transitions: Transition, last_value: jax.Array, gamma: float, lam: float):
    """GAE over a rollout. Inputs (T,N); returns advantages (T,N) and returns (T,N)."""

    def body(carry, x):
        gae, next_value = carry
        value, reward, done = x
        not_done = 1.0 - done.astype(jnp.float32)
        delta = reward + gamma * next_value * not_done - value
        gae = delta + gamma * lam * not_done * gae
        return (gae, value), gae

    init = (jnp.zeros_like(last_value), last_value)
    xs = (transitions.value, transitions.reward, transitions.done)
    _, advantages = lax.scan(body, init, xs, reverse=True)
    returns = advantages + transitions.value
    return advantages, returns
