"""Today's bots in JAX, batched: the optimistic-planner explorer and the best-of-k designers.

Same decisions as brain.js (and baselines.py), drawn from JAX random numbers instead of the JS
generator, so they match the JS bots in distribution rather than draw for draw:
- `probe_scores` gives every (cell, direction) probe its rankProbes score (doubled, as an integer);
- `choose` applies chooseProbe's noise: easy picks any legal probe 80% of the time, normal one of
  the 4 first in rankProbes order 25% of the time, otherwise the best (ties uniformly);
- `explore` runs whole explorations for a batch of games; `design` keeps, per maze, the random
  candidate in which the hard explorer hits the most walls (candidates drawn by mazes.random_maze).
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .baselines import design_candidates, design_runs
from .engine import State, _tables, board_size, init, neighbor_grid, side_grid, step
from .mazes import random_maze, stack_mazes

LEVEL_INDEX = {"easy": 0, "normal": 1, "hard": 2}
EASY, NORMAL, HARD = 0, 1, 2
FAR = 1 << 20  # "infinite" distance (no path to the exit without crossing a known wall)
ILLEGAL_SCORE = 1 << 30


def optimistic_distances(state: State) -> jax.Array:
    """int32[N²]: fewest unknown sides on a path to the exit through no known wall (brain.js's
    0-1 BFS), FAR when there is none.

    Known passages always lie inside the reached region, so that region is contracted to one node
    and every remaining side costs 1: plain BFS sweeps from the exit.
    """
    size = board_size(state)
    passable = side_grid(~state.known_wall, False)  # [4, N, N]
    reached = state.reached.reshape(size, size)

    def contract(d):
        return jnp.where(reached, jnp.min(jnp.where(reached, d, FAR)), d)

    start = jnp.where(jnp.arange(size * size) == state.exit, 0, FAR).astype(jnp.int32).reshape(size, size)

    def sweep(carry):
        d, _ = carry
        new = contract(jnp.minimum(d, jnp.min(jnp.where(passable, neighbor_grid(d, FAR) + 1, FAR), axis=0)))
        return new, jnp.any(new != d)

    dist, _ = jax.lax.while_loop(lambda c: c[1], sweep, (contract(start), jnp.bool_(True)))
    return dist.reshape(-1)


def _rows(cells) -> jax.Array:
    """bool[N, N] → uint32[N]: one bitmask per row, bit c = column c."""
    bits = jnp.left_shift(jnp.uint32(1), jnp.arange(cells.shape[-1], dtype=jnp.uint32))
    return jnp.sum(jnp.where(cells, bits, jnp.uint32(0)), axis=-1, dtype=jnp.uint32)


def _cells(rows, size: int) -> jax.Array:
    """uint32[N] → bool[N, N]."""
    return (jnp.right_shift(rows[:, None], jnp.arange(size, dtype=jnp.uint32)) & 1).astype(bool)


def _region_layers(state: State) -> tuple:
    """BFS from the exit (reached region contracted) on row bitmasks, stopped when it touches the
    region. Returns (D, cells at distance ≤ D−1, cells at distance ≤ D): D is the region's distance
    (FAR if the exit can't be reached), and every frontier cell is at D−1, D or D+1."""
    size = board_size(state)
    pu, pr, pd, pl = (_rows(p) for p in side_grid(~state.known_wall, False))
    region = _rows(state.reached.reshape(size, size))
    start = _rows((jnp.arange(size * size) == state.exit).reshape(size, size))
    zero = jnp.zeros(1, dtype=jnp.uint32)

    def sweep(c):
        v, _, k, _, _ = c
        grown = (
            v
            | (jnp.right_shift(v, 1) & pr)  # the cell on the right is in
            | (jnp.left_shift(v, 1) & pl)
            | (jnp.concatenate([v[1:], zero]) & pd)  # the cell below is in
            | (jnp.concatenate([zero, v[:-1]]) & pu)
        )
        hit = jnp.any((grown & region) != 0)
        grown = jnp.where(hit, grown | region, grown)
        return grown, v, k + 1, hit, jnp.any(grown != v)

    init = (start, start, jnp.int32(0), jnp.any((start & region) != 0), jnp.bool_(True))
    cur, prev, k, hit, _ = jax.lax.while_loop(lambda c: ~c[3] & c[4], sweep, init)
    return jnp.where(hit, k, FAR), _cells(prev, size), _cells(cur, size)


def probe_scores(state: State) -> jax.Array:
    """int32[N², 4]: 2 × the rankProbes score of probing side d of cell c, ILLEGAL_SCORE when
    illegal. FAR stands for JS's Infinity (no path to the exit), where every probe ties."""
    size = board_size(state)
    reached = state.reached.reshape(size, size)
    unknown = side_grid(~state.known_open & ~state.known_wall, False)  # [4, N, N], False on the border
    legal = reached[None] & unknown & ~state.done
    d, within_prev, within = _region_layers(state)
    to_region = neighbor_grid(reached, False)
    to_prev = neighbor_grid(within_prev, False)
    to_within = neighbor_grid(within, False)
    dist_to = jnp.where(to_region | (to_within & ~to_prev), d, jnp.where(to_prev, d - 1, d + 1))
    score2 = jnp.where(d < FAR, 2 + 2 * dist_to + to_region.astype(jnp.int32), FAR)
    return jnp.where(legal, score2, ILLEGAL_SCORE).reshape(4, -1).T


def pick_uniform(mask, u) -> jax.Array:
    """Index of a uniformly chosen True entry of `mask`, from one uniform draw `u` in [0, 1)."""
    target = jnp.floor(u * jnp.sum(mask)).astype(jnp.int32)
    return jnp.argmax(jnp.cumsum(mask) > target)


def candidates(state: State) -> tuple:
    """Flat bool[N²·4] masks over (cell, side): every legal probe, the best ones (lowest score),
    and the first four in rankProbes order (score, then cells in the order they were reached, then
    U, R, D, L)."""
    s = probe_scores(state).reshape(-1)
    legal = s < ILLEGAL_SCORE
    best = legal & (s == jnp.min(s))
    # Real scores are ≤ 2N²+3 and moves ≤ E; every infinite score ties at 4095, as in JS.
    n = s.shape[0]
    at = jnp.repeat(jnp.minimum(state.reached_at, 1023), 4)
    big = jnp.iinfo(jnp.int32).max
    order = jnp.where(legal, (jnp.minimum(s, 4095) * 1024 + at) * 4 + jnp.arange(n) % 4, big)
    fourth = jnp.min(order)  # the keys are distinct: four successive minimums give the first four
    for _ in range(3):
        fourth = jnp.min(jnp.where(order > fourth, order, big))
    return legal, best, legal & (order <= fourth)


def choose(state: State, u, level=HARD) -> jax.Array:
    """The planner's next probe as an edge (−1 when there is none).

    `u`: two uniform draws in [0, 1), for chooseProbe's noise roll and for the pick among equals.
    `level`: EASY, NORMAL or HARD.
    """
    _, _, cell_edges, _ = _tables(state.walls.shape[-1])
    legal, best, first4 = candidates(state)
    roll = u[0]
    mask = jnp.where((level == EASY) & (roll < 0.8), legal, jnp.where((level == NORMAL) & (roll < 0.25), first4, best))
    edge = cell_edges.reshape(-1)[pick_uniform(mask, u[1])]
    return jnp.where(jnp.any(legal), edge, -1)


@jax.jit
def explore(states: State, key, level) -> State:
    """Finish every exploration of a batch with the planner (`level` per game or for all)."""
    b = states.done.shape[0]
    levels = jnp.broadcast_to(jnp.asarray(level, dtype=jnp.int32), (b,))
    max_steps = states.walls.shape[-1] + 1  # every legal probe reveals a side

    def cond(c):
        s, _, i = c
        return jnp.any(~s.done) & (i < max_steps)

    def body(c):
        s, k, i = c
        k, sub = jax.random.split(k)
        edges = jax.vmap(choose)(s, jax.random.uniform(sub, (b, 2)), levels)
        s, _ = jax.vmap(step)(s, edges)
        return s, k, i + 1

    final, _, _ = jax.lax.while_loop(cond, body, (states, key, jnp.int32(0)))
    return final


def explore_mazes(walls, entrance, exit, key, level, slots: int = 4096) -> np.ndarray:
    """Walls hit by the planner on each maze of a stacked batch (any number of mazes).

    The mazes are a work queue for `slots` parallel games: a slot whose game ends records its hits
    and loads the next maze, so no slot idles while one long exploration finishes.
    """
    walls, entrance, exit = (jnp.asarray(x) for x in (walls, entrance, exit))
    return np.asarray(_explore_queue(walls, entrance, exit, key, jnp.int32(level), min(slots, walls.shape[0])))


@partial(jax.jit, static_argnames="slots")
def _explore_queue(walls, entrance, exit, key, level, slots: int):
    total = walls.shape[0]
    load = jax.vmap(init)

    def fresh(idx):
        i = jnp.clip(idx, 0, total - 1)
        return load(walls[i], entrance[i], exit[i])

    def body(c):
        s, maze, handed, hits, finished, k = c
        k, sub = jax.random.split(k)
        edges = jax.vmap(choose, in_axes=(0, 0, None))(s, jax.random.uniform(sub, (slots, 2)), level)
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


def design(size: int, walls: int, level: str, count: int, rng: np.random.Generator, key) -> tuple:
    """`count` mazes from a designer level, stacked (walls[count, E], entrance[count], exit[count])."""
    k = design_candidates(size, level)
    cands = stack_mazes([random_maze(size, walls, rng.random) for _ in range(count * k)])
    if k == 1:
        return cands
    runs = design_runs(level)
    hits = explore_mazes(*(np.repeat(x, runs, axis=0) for x in cands), key, HARD)
    best = np.argmax(hits.reshape(count, k, runs).sum(axis=2), axis=1)  # first best, like brain.js
    pick = np.arange(count) * k + best
    return tuple(x[pick] for x in cands)

