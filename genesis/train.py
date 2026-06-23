"""train.py — the watchable PPO slice. Run: `python -m genesis.train`.

Loops rollout -> GAE -> PPO update on a small/easy env config and tracks a true success-rate via
periodic GREEDY evaluation (full deterministic episodes to completion) — an unbiased signal we can
WATCH climb (the CLAUDE.md "FIRST PRIORITY" milestone). Saves a learning curve + a final greedy
trajectory to runs/. Per-update compute is a single eqx.filter_jit call; the outer loop stays in
Python for logging/rendering.
"""

from __future__ import annotations

import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

from genesis.config import EnvConfig
from genesis.env import obs as obs_fn
from genesis.env import obs_size, reset
from genesis.network import ActorCritic
from genesis.ppo import Batch, PPOConfig, make_optimizer, make_update
from genesis.primitives import N_DIRECTIONS, N_PRIMITIVES, action_mask, step_primitive
from genesis.rollout import collect_rollout, compute_gae

_NEG_INF = -1e9


def _greedy_action(net, state, cfg):
    """Deterministic factored action: masked-argmax primitive + argmax direction."""
    prim_logits, dir_logits, _ = net(obs_fn(state, cfg))
    prim_logits = jnp.where(action_mask(state, cfg), prim_logits, _NEG_INF)
    return jnp.argmax(prim_logits), jnp.argmax(dir_logits)

RUNS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs")


def make_train_step(env_cfg: EnvConfig, ppo_cfg: PPOConfig, horizon: int, optimizer):
    """Build the jitted per-update step: rollout -> GAE -> flatten -> PPO update -> diagnostics."""
    update = make_update(optimizer, ppo_cfg)

    @eqx.filter_jit
    def train_step(net, opt_state, states, key):
        key, k_roll, k_upd = jax.random.split(key, 3)
        trans, final_states, last_value, _ = collect_rollout(net, states, k_roll, env_cfg, horizon)
        adv, ret = compute_gae(trans, last_value, ppo_cfg.gamma, ppo_cfg.gae_lambda)

        flat = lambda x: x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])  # (T,N,..)->(T*N,..)
        batch = Batch(
            obs=flat(trans.obs), primitive=flat(trans.primitive), direction=flat(trans.direction),
            log_prob=flat(trans.log_prob), value=flat(trans.value), advantage=flat(adv),
            ret=flat(ret), prim_mask=flat(trans.prim_mask),
        )
        net, opt_state, metrics = update(net, opt_state, batch, k_upd)
        diag = {"mean_step_reward": trans.reward.mean(), "entropy": metrics.entropy,
                "approx_kl": metrics.approx_kl, "loss": metrics.loss}
        return net, opt_state, final_states, key, diag
    return train_step


def make_greedy_eval(env_cfg: EnvConfig, n_eval: int):
    """Build a jitted unbiased success-rate evaluator: run n_eval greedy episodes to completion."""

    @eqx.filter_jit
    def evaluate(net, key):
        states = jax.vmap(reset, in_axes=(0, None))(jax.random.split(key, n_eval), env_cfg)
        solved0 = jax.vmap(lambda s: jnp.all(s.agent_pos == s.goal_pos))(states)  # degenerate starts

        def body(carry, _):
            states, done_mask, succ_mask = carry
            prims, dirs = jax.vmap(_greedy_action, in_axes=(None, 0, None))(net, states, env_cfg)
            states, reward, done = jax.vmap(step_primitive, in_axes=(0, 0, 0, None))(
                states, prims, dirs, env_cfg)
            newly_done = done & (~done_mask)
            succ_mask = succ_mask | (newly_done & (reward > 0.0))       # count each episode once
            done_mask = done_mask | done
            return (states, done_mask, succ_mask), None

        init = (states, solved0, solved0)
        (_, _, succ_mask), _ = lax.scan(body, init, None, length=env_cfg.max_steps)
        return succ_mask.mean()
    return evaluate


def run_greedy_episode(net, env_cfg: EnvConfig, key, max_steps: int):
    """Roll a single deterministic (argmax) episode; return (final_state, n_steps, reached)."""
    state = reset(key, env_cfg)
    reached = bool(jnp.all(state.agent_pos == state.goal_pos))
    steps = 0
    for _ in range(max_steps):
        if reached:
            break
        prim, direction = _greedy_action(net, state, env_cfg)
        state, reward, done = step_primitive(state, prim, direction, env_cfg)
        steps += 1
        if bool(done):
            reached = bool(reward > 0.0)
            break
    return state, steps, reached


def train_run(env_cfg: EnvConfig, ppo_cfg: PPOConfig = PPOConfig(), *, n_envs=256, horizon=32,
              n_updates=300, eval_every=10, n_eval=512, tag="slice", seed=0):
    """Run the PPO loop on `env_cfg`; log greedy success; save runs/<tag>_curve.png + trajectory.

    Returns (net, hist_upd, hist_succ). Reusable across configs (slice, scaled, primitives, ...).
    """
    os.makedirs(RUNS_DIR, exist_ok=True)
    key = jax.random.PRNGKey(seed)
    key, k_net, k_reset = jax.random.split(key, 3)
    net = ActorCritic(obs_size(env_cfg), N_PRIMITIVES, N_DIRECTIONS, hidden=64, key=k_net)
    optimizer = make_optimizer(ppo_cfg)
    opt_state = optimizer.init(eqx.filter(net, eqx.is_array))
    states = jax.vmap(reset, in_axes=(0, None))(jax.random.split(k_reset, n_envs), env_cfg)

    train_step = make_train_step(env_cfg, ppo_cfg, horizon, optimizer)
    evaluate = make_greedy_eval(env_cfg, n_eval=n_eval)

    print(f"GENESIS PPO [{tag}] | env {env_cfg.height}x{env_cfg.width} | {n_envs} envs x T{horizon} "
          f"| {n_updates} updates | device {jax.devices()[0].platform}")
    print(f"{'upd':>4} | {'greedy success%':>15} | {'mean_r':>8} | {'entropy':>7} | {'kl':>7} | {'sec':>5}")
    print("-" * 64)

    hist_upd, hist_succ = [], []
    t0 = time.perf_counter()
    for i in range(1, n_updates + 1):
        net, opt_state, states, key, diag = train_step(net, opt_state, states, key)
        if i % eval_every == 0 or i == 1:
            key, k_eval = jax.random.split(key)
            succ = float(evaluate(net, k_eval))
            hist_upd.append(i); hist_succ.append(succ)
            print(f"{i:>4} | {succ*100:>14.1f}% | {float(diag['mean_step_reward']):>8.3f} | "
                  f"{float(diag['entropy']):>7.3f} | {float(diag['approx_kl']):>7.4f} | "
                  f"{time.perf_counter()-t0:>5.1f}")

    # --- learning curve (greedy success %) ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist_upd, [s * 100 for s in hist_succ], "-o", color="#2ca02c", ms=3)
    ax.set_xlabel("update"); ax.set_ylabel("greedy success %"); ax.set_ylim(0, 100)
    ax.set_title(f"GENESIS PPO [{tag}] — learning curve"); ax.grid(alpha=0.3)
    curve_path = os.path.join(RUNS_DIR, f"{tag}_curve.png")
    fig.tight_layout(); fig.savefig(curve_path, dpi=110); plt.close(fig)

    # --- a final greedy episode, with its path drawn ---
    from genesis.render import save_grid
    fstate, nsteps, reached = run_greedy_episode(net, env_cfg, jax.random.PRNGKey(999), env_cfg.max_steps)
    traj_path = os.path.join(RUNS_DIR, f"{tag}_episode.png")
    save_grid(fstate, traj_path, title=f"[{tag}] greedy: {'REACHED' if reached else 'failed'} in {nsteps} steps",
              show_trail=True)

    print("-" * 64)
    print(f"greedy success%: start {hist_succ[0]*100:.1f} -> final {hist_succ[-1]*100:.1f}")
    print(f"saved: {curve_path}  |  {traj_path}")
    return net, hist_upd, hist_succ


def main():
    """The original watchable slice: small/easy 11x11 maps, fast visible learning."""
    env_cfg = EnvConfig(
        height=11, width=11, max_steps=50, r_goal=20.0, time_penalty=0.02, wall_density_max=0.10
    )
    train_run(env_cfg, n_updates=300, tag="slice")


if __name__ == "__main__":
    main()
