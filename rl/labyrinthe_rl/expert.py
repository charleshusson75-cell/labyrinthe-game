"""The expert explorer (src/bot/expert.js, DECISIONS.md D21), played by Node through
scripts/expert-solve.mjs: P2's yardstick in the evaluator, not ported to JAX.

Each maze is explored with a fixed particle count (no time budget) and its own seed, drawn from the
JAX key, so the walls hit depend on the mazes and the key only: not on the machine, or on how the
mazes are split between the Node processes that run in parallel.

Speed (Apple M1, 200 particles): ≈ 0.16 s per 5×5 maze on one core.
"""

import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .geometry import geometry, size_from_edges

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "expert-solve.mjs"
PARTICLES = 200  # as scripts/benchmark.mjs, which measured D21


def node() -> str | None:
    """Path of the Node executable, or None when it isn't installed."""
    return shutil.which("node")


def explore_mazes(walls, entrance, exit, key, particles: int = PARTICLES, workers: int | None = None, options=None) -> np.ndarray:
    """Walls hit by the expert on each maze of a stacked batch (walls[count, E], entrance, exit).

    `workers` Node processes share the mazes (default: one per CPU core); `options` are passed to
    expert.js as is (EXPERT_DEFAULTS, e.g. {"search": "greedy"}).
    """
    exe = node()
    if exe is None:
        raise RuntimeError("the expert explorer needs Node.js (node) on the PATH")
    walls, entrance, exit = np.asarray(walls, dtype=bool), np.asarray(entrance), np.asarray(exit)
    count = walls.shape[0]
    if count == 0:
        return np.zeros(0, dtype=np.int32)
    size = size_from_edges(walls.shape[1])
    g = geometry(size)
    seeds = np.asarray(jax.random.bits(key, (count,), jnp.uint32))
    mazes = [
        {"walls": [g.keys[e] for e in np.flatnonzero(w)], "entrance": int(s), "exit": int(x), "seed": int(seed)}
        for w, s, x, seed in zip(walls, entrance, exit, seeds)
    ]
    workers = max(1, min(workers or os.cpu_count() or 1, count))
    bounds = np.linspace(0, count, workers + 1).round().astype(int)
    expert = {"maxParticles": particles, **(options or {})}

    def run(lo, hi):
        request = json.dumps({"size": size, "mazes": mazes[lo:hi], "expert": expert})
        out = subprocess.run([exe, str(SCRIPT)], input=request, capture_output=True, text=True)
        if out.returncode:
            raise RuntimeError(f"{SCRIPT.name} failed:\n{out.stderr}")
        return json.loads(out.stdout)["hits"]

    with ThreadPoolExecutor(workers) as pool:
        hits = [h for part in pool.map(run, bounds[:-1], bounds[1:]) for h in part]
    failed = [i for i, h in enumerate(hits) if h is None]
    if failed:
        raise ValueError(f"the expert could not finish mazes {failed[:10]} (exit unreachable?)")
    return np.array(hits, dtype=np.int32)
