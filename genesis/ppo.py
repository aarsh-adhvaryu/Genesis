"""ppo.py — the PPO update: clipped surrogate + value loss + entropy bonus (Optax + Equinox).

Loss (per minibatch; advantages normalized):
    ratio   = exp(logp_new - logp_old)
    L_clip  = mean( min( ratio*A , clip(ratio, 1-eps, 1+eps)*A ) )      # policy gain (maximize)
    L_value = mean( (V_new - return)^2 )
    total   = -L_clip + vf_coef*0.5*L_value - ent_coef*entropy

make_update returns a function that runs n_epochs over n_minibatches of shuffled data, applying an
Optax step (global-norm clip -> Adam) each minibatch. Equinox convention: params = the array leaves
of the module; grads come from eqx.filter_value_and_grad.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import lax


@dataclass(frozen=True)
class PPOConfig:
    lr: float = 2.5e-4
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    n_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95


class Batch(NamedTuple):
    """Flattened rollout data (leading axis = T*N), the input to the PPO update (factored action)."""

    obs: jax.Array
    primitive: jax.Array  # chosen primitive id
    direction: jax.Array  # chosen direction
    log_prob: jax.Array   # logp_old = logp_prim + logp_dir (at action time)
    value: jax.Array      # V_old (kept for value-clipping if added later)
    advantage: jax.Array
    ret: jax.Array        # GAE return target
    prim_mask: jax.Array  # (N_PRIMITIVES,) legal-primitive mask at action time


class Metrics(NamedTuple):
    loss: jax.Array
    pg_loss: jax.Array
    v_loss: jax.Array
    entropy: jax.Array
    approx_kl: jax.Array
    clip_frac: jax.Array


def make_optimizer(ppo: PPOConfig) -> optax.GradientTransformation:
    """Global-norm gradient clipping followed by Adam (standard PPO optimizer)."""
    return optax.chain(optax.clip_by_global_norm(ppo.max_grad_norm), optax.adam(ppo.lr))


def ppo_loss(net, batch: Batch, ppo: PPOConfig):
    prim_logits, dir_logits, value = jax.vmap(net)(batch.obs)
    # re-apply the SAME budget mask used at action time so the ratio is consistent
    prim_logits = jnp.where(batch.prim_mask, prim_logits, -1e9)
    prim_dist = distrax.Categorical(logits=prim_logits)
    dir_dist = distrax.Categorical(logits=dir_logits)
    log_prob = prim_dist.log_prob(batch.primitive) + dir_dist.log_prob(batch.direction)

    # normalize advantages within the minibatch (stabilizes the policy gradient scale)
    adv = batch.advantage
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    ratio = jnp.exp(log_prob - batch.log_prob)
    pg_unclipped = ratio * adv
    pg_clipped = jnp.clip(ratio, 1.0 - ppo.clip_eps, 1.0 + ppo.clip_eps) * adv
    pg_loss = -jnp.minimum(pg_unclipped, pg_clipped).mean()          # maximize gain -> minimize -gain

    v_loss = 0.5 * jnp.mean((value - batch.ret) ** 2)
    entropy = (prim_dist.entropy() + dir_dist.entropy()).mean()      # factored policy: entropies add
    total = pg_loss + ppo.vf_coef * v_loss - ppo.ent_coef * entropy

    # diagnostics (not differentiated): how far the policy moved, how often clipping bound
    approx_kl = jnp.mean(batch.log_prob - log_prob)
    clip_frac = jnp.mean((jnp.abs(ratio - 1.0) > ppo.clip_eps).astype(jnp.float32))
    return total, Metrics(total, pg_loss, v_loss, entropy, approx_kl, clip_frac)


def make_update(optimizer: optax.GradientTransformation, ppo: PPOConfig):
    """Return update(net, opt_state, batch, key) -> (net, opt_state, mean_metrics)."""

    def loss_fn(net, b):
        return ppo_loss(net, b, ppo)

    def update(net, opt_state, batch: Batch, key):
        n_samples = batch.obs.shape[0]
        mb_size = n_samples // ppo.n_minibatches

        def epoch(carry, epoch_key):
            net, opt_state = carry
            perm = jax.random.permutation(epoch_key, n_samples)
            shuffled = jax.tree_util.tree_map(lambda x: x[perm], batch)
            minibatches = jax.tree_util.tree_map(
                lambda x: x.reshape((ppo.n_minibatches, mb_size) + x.shape[1:]), shuffled
            )

            def mb_step(carry, mb):
                net, opt_state = carry
                (_, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(net, mb)
                updates, opt_state = optimizer.update(grads, opt_state)
                net = eqx.apply_updates(net, updates)
                return (net, opt_state), metrics

            (net, opt_state), metrics = lax.scan(mb_step, (net, opt_state), minibatches)
            return (net, opt_state), metrics

        (net, opt_state), metrics = lax.scan(
            epoch, (net, opt_state), jax.random.split(key, ppo.n_epochs)
        )
        mean_metrics = jax.tree_util.tree_map(jnp.mean, metrics)  # over (n_epochs, n_minibatches)
        return net, opt_state, mean_metrics

    return update
