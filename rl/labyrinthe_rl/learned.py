"""The learned explorer at play: checkpoints, and batched explorations with the network as policy.

A checkpoint is one .npz file: the network's parameters (`params/...`), optionally the optimizer
state (`opt/...`), and a JSON `meta` record (network shape, training config, counters). Arrays are
stored under their pytree path, so a checkpoint loads without pickles and the parameters can be read
by any tool (the browser export of P4 included).

`explore_mazes` plays a batch of mazes to the end, greedily by default (the most likely probe), in a
work queue of parallel games like planner.explore_mazes; `Explorer(path)` wraps a checkpoint for the
bench.

    python -m labyrinthe_rl.learned SRC DST    copy a checkpoint without its optimizer state
"""

import json
import os
import sys
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import network
from .engine import init, legal_mask, step
from .lattice import observe

DEFAULT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


def default_checkpoint(size: int) -> Path:
    """Where the committed explorer of a board size lives (rl/checkpoints/explorer-NxN.npz)."""
    return DEFAULT_DIR / f"explorer-{size}x{size}.npz"


# ------------------------------------------------------------------ checkpoints


def _path_name(path) -> str:
    parts = []
    for p in path:
        if isinstance(p, jax.tree_util.DictKey):
            parts.append(str(p.key))
        elif isinstance(p, jax.tree_util.SequenceKey):
            parts.append(str(p.idx))
        elif isinstance(p, jax.tree_util.GetAttrKey):
            parts.append(p.name)
        else:
            parts.append(str(p).strip(".[]'"))
    return "/".join(parts)


def flatten(tree, prefix: str) -> dict:
    """A pytree as {"prefix/path/to/leaf": numpy array}."""
    leaves = jax.tree_util.tree_flatten_with_path(tree)[0]
    return {f"{prefix}/{_path_name(path)}": np.asarray(leaf) for path, leaf in leaves}


def unflatten(template, arrays: dict, prefix: str):
    """Fill `template`'s structure with the arrays saved by `flatten` (shapes must match)."""
    paths, treedef = jax.tree_util.tree_flatten_with_path(template)
    leaves = []
    for path, leaf in paths:
        name = f"{prefix}/{_path_name(path)}"
        if name not in arrays:
            raise KeyError(f"checkpoint has no {name}")
        a = arrays[name]
        if a.shape != np.shape(leaf):
            raise ValueError(f"{name}: checkpoint shape {a.shape}, expected {np.shape(leaf)}")
        leaves.append(jnp.asarray(a, dtype=jnp.asarray(leaf).dtype))
    return jax.tree_util.tree_unflatten(treedef, leaves)


def net_config(meta: dict) -> network.NetConfig:
    n = meta["net"]
    return network.NetConfig(channels=n["channels"], blocks=n["blocks"], dilations=tuple(n["dilations"]), plan=n.get("plan", False))


def save(path, params, meta: dict, opt_state=None) -> None:
    """Write a checkpoint atomically (a crash mid-write leaves the previous file intact)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = flatten(params, "params")
    if opt_state is not None:
        arrays.update(flatten(opt_state, "opt"))
    arrays["meta"] = np.array(json.dumps(meta))
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def load(path) -> tuple:
    """(params, meta, arrays) of a checkpoint; `arrays` still holds the optimizer state, if any."""
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(str(arrays["meta"]))
    return unflatten(template(meta), arrays, "params"), meta, arrays


def template(meta: dict) -> dict:
    """Parameters shaped like a checkpoint's network: the explorer's, or the designer's (P3)."""
    cfg = net_config(meta)
    if meta.get("kind") == "designer":
        from . import designer

        return designer.init_net(jax.random.key(0), cfg)
    return network.init(jax.random.key(0), cfg)


# ------------------------------------------------------------------ playing


def policy_logits(params, states, dilations):
    obs = jax.vmap(observe)(states)
    legal = jax.vmap(legal_mask)(states)
    return network.apply(params, obs, legal, dilations).logits


def explore_mazes(params, walls, entrance, exit, key, dilations=network.NetConfig().dilations, greedy: bool = True, slots: int = 1024) -> np.ndarray:
    """Walls hit by the network on each maze of a stacked batch (walls[count, E], entrance, exit).

    `greedy`: always the most likely probe; otherwise probes are sampled from the policy.
    """
    walls, entrance, exit = (jnp.asarray(x) for x in (walls, entrance, exit))
    if walls.shape[0] == 0:
        return np.zeros(0, dtype=np.int32)
    out = _explore_queue(params, walls, entrance, exit, key, tuple(dilations), greedy, min(slots, walls.shape[0]))
    return np.asarray(out)


def random_mazes(walls, entrance, exit, key, slots: int = 4096) -> np.ndarray:
    """Walls hit by an explorer that probes uniformly among legal sides: the floor any learner must
    clear."""
    walls, entrance, exit = (jnp.asarray(x) for x in (walls, entrance, exit))
    return np.asarray(_explore_queue(None, walls, entrance, exit, key, (), False, min(slots, walls.shape[0])))


@partial(jax.jit, static_argnames=("dilations", "greedy", "slots"))
def _explore_queue(params, walls, entrance, exit, key, dilations, greedy: bool, slots: int):
    total = walls.shape[0]
    load_games = jax.vmap(init)

    def fresh(idx):
        i = jnp.clip(idx, 0, total - 1)
        return load_games(walls[i], entrance[i], exit[i])

    def body(c):
        s, maze, handed, hits, finished, k = c
        k, sub = jax.random.split(k)
        if params is None:
            logits = jnp.where(jax.vmap(legal_mask)(s), 0.0, network.NEG)
        else:
            logits = policy_logits(params, s, dilations)
        edges = jnp.argmax(logits, axis=-1) if greedy else jax.random.categorical(sub, logits)
        s, _ = jax.vmap(step)(s, edges)
        ended = s.done & (maze >= 0)
        hits = hits.at[jnp.where(ended, maze, total)].set(s.hits, mode="drop")
        nxt = handed + jnp.cumsum(ended) - 1
        gets = ended & (nxt < total)
        new = fresh(nxt)
        s = jax.tree.map(lambda f, o: jnp.where(gets.reshape((-1,) + (1,) * (o.ndim - 1)), f, o), new, s)
        maze = jnp.where(gets, nxt, jnp.where(ended, -1, maze))
        return s, maze, handed + jnp.sum(ended), hits, finished + jnp.sum(ended), k

    maze = jnp.arange(slots)
    init_c = (fresh(maze), maze, jnp.int32(slots), jnp.zeros(total, jnp.int32), jnp.int32(0), key)
    out = jax.lax.while_loop(lambda c: c[4] < total, body, init_c)
    return out[3]


class Explorer:
    """A loaded checkpoint, ready to explore (any board size: the network is fully convolutional)."""

    def __init__(self, path, greedy: bool = True):
        self.path = Path(path)
        self.params, self.meta, _ = load(path)
        self.dilations = net_config(self.meta).dilations
        self.greedy = greedy

    def explore(self, walls, entrance, exit, key) -> np.ndarray:
        return explore_mazes(self.params, walls, entrance, exit, key, self.dilations, self.greedy)


def export(src, dst) -> dict:
    """Write `src` to `dst` without the optimizer state (what the bench and a browser need)."""
    params, meta, _ = load(src)
    save(dst, params, meta)
    return meta


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python -m labyrinthe_rl.learned SRC DST")
    m = export(sys.argv[1], sys.argv[2])
    print(f"{sys.argv[2]}: {m['size']}×{m['size']} {m.get('kind', 'explorer')}, iteration {m['iteration']}, {m['env_steps'] / 1e6:.2f} M steps")
