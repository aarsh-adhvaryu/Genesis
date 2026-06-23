"""bench.py — throughput benchmark for the Stage-1 env. Run: `python -m genesis.bench`.

Measures env steps/sec = (N * T) / wall_time with warmup (XLA compile) excluded and the result
blocked-on inside the timed region (JAX dispatch is async). Also times vmapped reset (the heavier
kernel: it runs the H*W-iteration flood-fill). Reference target: 50k steps/sec.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
from jax import lax

from genesis.config import EnvConfig
from genesis.env import N_ACTIONS, reset, step


def _make_rollout(cfg: EnvConfig, n_envs: int, horizon: int):
    """Compiled fn: reset N envs, then lax.scan `horizon` random-action steps. cfg/N/T baked in."""
    reset_v = jax.vmap(reset, in_axes=(0, None))
    step_v = jax.vmap(step, in_axes=(0, 0, None))

    @jax.jit
    def run(key):
        k_reset, k_act = jax.random.split(key)
        states = reset_v(jax.random.split(k_reset, n_envs), cfg)

        def body(states, akey):
            actions = jax.random.randint(akey, (n_envs,), 0, N_ACTIONS)
            next_states, reward, _ = step_v(states, actions, cfg)
            return next_states, reward

        _, rewards = lax.scan(body, states, jax.random.split(k_act, horizon))
        return rewards  # (T, N) — returned so XLA can't prune the loop

    return run


def _time(fn, key, repeats: int = 5) -> float:
    """Best wall-time over `repeats` runs (warmup already done by caller)."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(key))
        best = min(best, time.perf_counter() - t0)
    return best


def bench_steps(cfg: EnvConfig, n_envs: int, horizon: int, key) -> float:
    run = _make_rollout(cfg, n_envs, horizon)
    jax.block_until_ready(run(key))           # warmup: compile + first exec, excluded from timing
    dt = _time(run, key)
    return (n_envs * horizon) / dt            # steps/sec


def bench_reset(cfg: EnvConfig, n_envs: int, key) -> float:
    reset_v = jax.jit(jax.vmap(reset, in_axes=(0, None)), static_argnums=1)
    keys = jax.random.split(key, n_envs)
    jax.block_until_ready(reset_v(keys, cfg))  # warmup
    best = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        jax.block_until_ready(reset_v(keys, cfg))
        best = min(best, time.perf_counter() - t0)
    return n_envs / best                        # resets/sec


def main():
    cfg = EnvConfig()
    dev = jax.devices()[0]
    horizon = 128
    key = jax.random.PRNGKey(0)

    print(f"device: {dev}  ({dev.platform})   grid: {cfg.height}x{cfg.width}   horizon T={horizon}")
    print(f"{'N envs':>8} | {'resets/sec':>14} | {'steps/sec':>16} | {'vs 50k target':>14}")
    print("-" * 64)
    for n_envs in (256, 512, 1024):
        rps = bench_reset(cfg, n_envs, key)
        sps = bench_steps(cfg, n_envs, horizon, key)
        print(f"{n_envs:>8} | {rps:>14,.0f} | {sps:>16,.0f} | {sps / 50_000:>12.1f}x")


if __name__ == "__main__":
    main()
