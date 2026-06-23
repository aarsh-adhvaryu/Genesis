"""render.py — minimal grid visualization (headless / WSL-friendly: saves PNG files).

Renders a SearchState as an image: empty / wall / goal / agent, with an optional visit-count
heat overlay. The trajectory/GIF replay for whole episodes builds on this in the PPO slice.
Interactive windows don't work well under WSL, so everything writes to a file you open in VSCode.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend: render to file, no display needed

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from genesis.state import SearchState

# index -> color: 0 empty, 1 wall, 2 goal, 3 agent, 4 trail (visited cell)
_CMAP = ListedColormap(["#ffffff", "#333333", "#2ca02c", "#d62728", "#aad4ff"])


def draw_grid(state: SearchState, ax, title: str | None = None, show_trail: bool = False) -> None:
    """Draw one (unbatched) SearchState onto a matplotlib axis.

    show_trail=True shades cells with visit_count>0 (the agent's path) in light blue.
    """
    g = np.asarray(state.grid).astype(int).copy()
    if show_trail:
        visited = np.asarray(state.visit_counts) > 0
        g[(visited) & (g == 0)] = 4  # trail only over empty cells (don't hide walls/goal)
    ar, ac = (int(x) for x in np.asarray(state.agent_pos))
    g[ar, ac] = 3  # agent drawn on top (overrides empty/goal cell underneath)
    ax.imshow(g, cmap=_CMAP, vmin=0, vmax=4, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def save_grid(state: SearchState, path: str, title: str | None = None, show_trail: bool = False) -> str:
    """Render a single state to `path` (PNG). Returns the path."""
    fig, ax = plt.subplots(figsize=(4, 4))
    draw_grid(state, ax, title, show_trail=show_trail)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def save_grid_panel(states: SearchState, path: str, n: int = 4, titles=None) -> str:
    """Render the first `n` states of a BATCHED SearchState (leading axis) in a row. Returns path."""
    import jax

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for i in range(n):
        one = jax.tree_util.tree_map(lambda x, i=i: x[i], states)
        draw_grid(one, axes[i], titles[i] if titles else f"map {i}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
