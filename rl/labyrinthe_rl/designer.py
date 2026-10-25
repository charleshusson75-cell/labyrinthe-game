"""The learned maze designer (RL P3, DECISIONS.md D17, D19, D25): an autoregressive policy on the
same (2N+1)×(2N+1) lattice as the explorer. It places the entrance, then the exit, then the walls one
at a time.

Episode for a wall budget W (`steps(W)` = W + 2 decisions):
    step 0      E, a cell pixel;
    step 1      S, a cell pixel other than E;
    steps 2…    one wall per step, a side pixel, until W walls are placed.
Validity is enforced by masking, so every finished maze is playable: a wall can't go where there
already is one, nor on a side whose closing would cut S off from E (`cut_mask`). Masking never runs
out of legal walls while W ≤ (N−1)²: a simple E→S path uses at most N²−1 sides, so more than that
many open sides always leave one off the path. The game's default budgets ± 2 are all below.

Actions: one index space, the N² cells first, then the E edges (geometry.py order). Steps after the
maze is complete (a batch mixes budgets) are padding: only action 0 is legal and they're masked out
of the losses (`DBatch.valid`).

Observation planes (channels last, float32[2N+1, 2N+1, N_D_PLANES]):

  WALL      walls placed so far, plus the outer border
  OPEN      inner sides without a wall
  CUT       open sides whose closing would cut S off from E (they can't take a wall)
  ENTRANCE  the entrance cell, once placed
  EXIT      the exit cell, once placed
  CELL      every cell pixel
  DIST_E    (distance from E through open sides + 1) / N² on the cells reachable from E, else 0
  DIST_S    the same from S
  PHASE_E   constant: 1 while placing E
  PHASE_S   constant: 1 while placing S
  W_TOTAL   constant: the budget / inner sides (as the explorer's W_TOTAL)
  W_LEFT    constant: walls still to place / inner sides
  PATH      constant: shortest E→S distance / N² (0 before S is placed)
  SIZE      constant: N / 10

The network is the explorer's tower (network.trunk) with these planes as input; a 1×1 policy head
read at cell and side pixels; and a value (the walls the explorer will hit) = W · sigmoid(dense(mean
and max pooled features)).
"""

from collections import Counter
from functools import lru_cache, partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import network
from .engine import neighbor_grid, side_grid
from .geometry import MAX_SIZE, geometry, n_edges, size_from_edges
from .lattice import _side_grids, action_pixels, cell_pixels, compose, lattice_size
from .mazes import distances_from

D_PLANES = ("WALL", "OPEN", "CUT", "ENTRANCE", "EXIT", "CELL", "DIST_E", "DIST_S", "PHASE_E", "PHASE_S", "W_TOTAL", "W_LEFT", "PATH", "SIZE")
(WALL, OPEN, CUT, ENTRANCE, EXIT, CELL, DIST_E, DIST_S, PHASE_E, PHASE_S, W_TOTAL, W_LEFT, PATH, SIZE) = range(len(D_PLANES))
N_D_PLANES = len(D_PLANES)


def steps(budget: int) -> int:
    """Decisions in an episode: E, S, then the walls."""
    return int(budget) + 2


def n_actions(size: int) -> int:
    return size * size + n_edges(size)


# ------------------------------------------------------------------ the designer's game


class DState(NamedTuple):
    walls: jax.Array  # bool[E]   walls placed so far
    entrance: jax.Array  # int32[]   −1 until placed
    exit: jax.Array  # int32[]   −1 until placed
    placed: jax.Array  # int32[]   walls placed
    budget: jax.Array  # int32[]   walls to place
    t: jax.Array  # int32[]   decisions taken
    done: jax.Array  # bool[]    the maze is complete


def init(budget, size: int) -> DState:
    return DState(
        walls=jnp.zeros(n_edges(size), dtype=bool),
        entrance=jnp.int32(-1),
        exit=jnp.int32(-1),
        placed=jnp.int32(0),
        budget=jnp.asarray(budget, dtype=jnp.int32),
        t=jnp.int32(0),
        done=jnp.bool_(False),
    )


def _rows(cells) -> jax.Array:
    """bool[..., N, N] → uint32[..., N]: one bitmask per row, bit c = column c."""
    bits = jnp.left_shift(jnp.uint32(1), jnp.arange(cells.shape[-1], dtype=jnp.uint32))
    return jnp.sum(jnp.where(cells, bits, jnp.uint32(0)), axis=-1, dtype=jnp.uint32)


@lru_cache(maxsize=None)
def _closing_bits(size: int) -> np.ndarray:
    """uint32[E, 4, N]: the bits each edge clears in the row masks of open sides (U, R, D, L)."""
    g = geometry(size)
    out = np.zeros((g.n_edges, 4, size), dtype=np.uint32)
    for e, (a, b) in enumerate(g.edge_cells):
        ra, ca = divmod(int(a), size)
        rb, cb = divmod(int(b), size)
        if b == a + 1:  # a's right side is b's left side
            out[e, 1, ra] |= np.uint32(1 << ca)
            out[e, 3, rb] |= np.uint32(1 << cb)
        else:  # a's down side is b's up side
            out[e, 2, ra] |= np.uint32(1 << ca)
            out[e, 0, rb] |= np.uint32(1 << cb)
    return out


def _flood(masks, start) -> jax.Array:
    """Cells reachable from `start` (uint32[..., N] row bitmasks) through the open sides `masks`
    (uint32[..., 4, N]: U, R, D, L), as row bitmasks."""
    pu, pr, pd, pl = (masks[..., i, :] for i in range(4))

    def grow(c):
        v, _ = c
        zero = jnp.zeros_like(v[..., :1])
        new = (
            v
            | (jnp.right_shift(v, 1) & pr)  # the cell on the right is in
            | (jnp.left_shift(v, 1) & pl)
            | (jnp.concatenate([v[..., 1:], zero], axis=-1) & pd)  # the cell below is in
            | (jnp.concatenate([zero, v[..., :-1]], axis=-1) & pu)
        )
        return new, jnp.any(new != v)

    v, _ = jax.lax.while_loop(lambda c: c[1], grow, (start, jnp.bool_(True)))
    return v


def cut_mask(walls, entrance, exit, size: int) -> jax.Array:
    """bool[E]: the open sides whose closing would leave S unreachable from E (one flood fill per
    side, all at once on row bitmasks)."""
    open_edges = ~walls
    masks = jnp.stack([_rows(p) for p in side_grid(open_edges, False)])  # [4, N]
    closed = masks[None] & ~jnp.asarray(_closing_bits(size))  # [E, 4, N]
    start = _rows((jnp.arange(size * size) == entrance).reshape(size, size))
    reached = _flood(closed, jnp.broadcast_to(start, (open_edges.shape[0], size)))
    row, col = exit // size, exit % size
    gets_there = (jnp.right_shift(reached[:, row], col.astype(jnp.uint32)) & 1) == 1
    return open_edges & ~gets_there


def distances(open_edges, start, size: int) -> jax.Array:
    """int32[N²]: BFS distance from `start` through `open_edges`, −1 where unreachable."""
    side_open = side_grid(open_edges, False)
    first = (jnp.arange(size * size) == start).reshape(size, size)

    def grow(c):
        r, d, k, _ = c
        new = r | jnp.any(side_open & neighbor_grid(r, False), axis=0)
        return new, jnp.where(new & ~r, k + 1, d), k + 1, jnp.any(new != r)

    _, d, _, _ = jax.lax.while_loop(lambda c: c[3], grow, (first, jnp.where(first, 0, -1), jnp.int32(0), jnp.bool_(True)))
    return d.reshape(-1)


def observe(state: DState, size: int) -> tuple:
    """(obs float32[2N+1, 2N+1, N_D_PLANES], legal bool[N² + E]) of one designer state."""
    f = jnp.float32
    n_c, n_e = size * size, n_edges(size)
    cells = jnp.arange(n_c)
    has_e, has_s = state.entrance >= 0, state.exit >= 0
    e, s = jnp.maximum(state.entrance, 0), jnp.maximum(state.exit, 0)
    open_edges = ~state.walls
    walling = (state.t >= 2) & ~state.done
    cut = cut_mask(state.walls, e, s, size) & walling
    right, down = _side_grids(jnp.stack([state.walls, open_edges, cut], axis=-1).astype(f), [1, 0, 0])  # border = wall
    side_planes = compose(jnp.zeros((size, size, 3), f), right, down)
    dist_e, dist_s = distances(open_edges, e, size), distances(open_edges, s, size)
    de = jnp.where(has_e & (dist_e >= 0), (dist_e + 1) / n_c, 0.0)
    ds = jnp.where(has_s & (dist_s >= 0), (dist_s + 1) / n_c, 0.0)
    cell_planes = jnp.stack([has_e & (cells == e), has_s & (cells == s), jnp.ones(n_c, bool), de, ds], axis=-1).astype(f)
    cell_lattice = compose(cell_planes.reshape(size, size, 5), jnp.zeros((size, size + 1, 5), f), jnp.zeros((size + 1, size, 5), f))
    path = jnp.where(has_e & has_s, jnp.maximum(dist_e[s], 0) / n_c, 0.0)
    scalars = jnp.stack([state.t == 0, state.t == 1, state.budget / n_e, (state.budget - state.placed) / n_e, path, size / MAX_SIZE]).astype(f)
    n = lattice_size(size)
    obs = jnp.concatenate([side_planes, cell_lattice, jnp.broadcast_to(scalars, (n, n, 6))], axis=-1)
    cell_legal = (state.t == 0) | ((state.t == 1) & (cells != e))
    legal = jnp.concatenate([cell_legal & (state.t < 2), walling & open_edges & ~cut])
    legal = jnp.where(state.done, jnp.arange(n_c + n_e) == 0, legal)  # padding: a dummy action
    return obs, legal


def step(state: DState, action, size: int) -> DState:
    n_c = size * size
    action = jnp.asarray(action, dtype=jnp.int32)
    place = (state.t >= 2) & ~state.done
    walls = state.walls | ((jnp.arange(n_edges(size)) == action - n_c) & place)
    placed = state.placed + place.astype(jnp.int32)
    t = state.t + 1
    return DState(
        walls=walls,
        entrance=jnp.where(state.t == 0, action, state.entrance),
        exit=jnp.where(state.t == 1, action, state.exit),
        placed=placed,
        budget=state.budget,
        t=t,
        done=(t >= 2) & (placed >= state.budget),
    )


# ------------------------------------------------------------------ network


def init_net(key, cfg: network.NetConfig) -> dict:
    params = network.init(key, cfg, in_planes=N_D_PLANES, head_out=1)
    params["value"] = {"w": jnp.zeros((2 * cfg.channels, 1), jnp.float32), "b": jnp.zeros(1, jnp.float32)}
    return params


def apply(params: dict, obs, legal, dilations=network.NetConfig().dilations) -> tuple:
    """(logits float32[B, N² + E] masked to legal actions, value float32[B] = expected walls hit)."""
    size = (obs.shape[1] - 1) // 2
    h = network.trunk(params, obs, dilations)
    maps = network._conv(h, params["head"])[..., 0]  # [B, L, L]
    cx, px = cell_pixels(size), action_pixels(size)
    logits = jnp.concatenate([maps[:, cx[:, 0], cx[:, 1]], maps[:, px[:, 0], px[:, 1]]], axis=-1)
    pooled = jnp.concatenate([jnp.mean(h, axis=(1, 2)), jnp.max(h, axis=(1, 2))], axis=-1)
    v = (pooled @ params["value"]["w"] + params["value"]["b"])[:, 0]
    value = obs[:, 0, 0, W_TOTAL] * n_edges(size) * jax.nn.sigmoid(v)
    return jnp.where(legal, logits, network.NEG), value


# ------------------------------------------------------------------ generating mazes


class DBatch(NamedTuple):
    obs: jax.Array  # [T, B, L, L, C]
    legal: jax.Array  # [T, B, A]
    action: jax.Array  # [T, B]
    logp: jax.Array  # [T, B]
    value: jax.Array  # [T, B]  expected walls hit
    valid: jax.Array  # [T, B]  a real decision (not padding after the maze is complete)
    mag_logp: jax.Array = None  # [T, B, A] the magnet policy's log-probabilities (league.py)


@partial(jax.jit, static_argnames=("size", "n_steps", "dilations"))
def rollout(params, budgets, key, size: int, n_steps: int, dilations, magnet=None) -> tuple:
    """One maze per budget with the sampled policy, every decision recorded: (final DState, DBatch)."""
    states = jax.vmap(partial(init, size=size))(budgets)

    def one(s, k):
        obs, legal = jax.vmap(partial(observe, size=size))(s)
        logits, value = apply(params, obs, legal, dilations)
        action = jax.random.categorical(k, logits)
        logp = jnp.take_along_axis(jax.nn.log_softmax(logits), action[:, None], axis=1)[:, 0]
        mag = None if magnet is None else jax.nn.log_softmax(apply(magnet, obs, legal, dilations)[0])
        rec = DBatch(obs=obs, legal=legal, action=action, logp=logp, value=value, valid=~s.done, mag_logp=mag)
        return jax.vmap(partial(step, size=size))(s, action), rec

    return jax.lax.scan(one, states, jax.random.split(key, n_steps))


@partial(jax.jit, static_argnames=("size", "n_steps", "dilations", "greedy"))
def _sample(params, budgets, key, size: int, n_steps: int, dilations, greedy: bool):
    states = jax.vmap(partial(init, size=size))(budgets)

    def one(s, k):
        obs, legal = jax.vmap(partial(observe, size=size))(s)
        logits, _ = apply(params, obs, legal, dilations)
        action = jnp.argmax(logits, axis=-1) if greedy else jax.random.categorical(k, logits)
        return jax.vmap(partial(step, size=size))(s, action), None

    final, _ = jax.lax.scan(one, states, jax.random.split(key, n_steps))
    return final.walls, final.entrance, final.exit


def sample_mazes(params, size: int, budgets, key, dilations=network.NetConfig().dilations, greedy: bool = False, chunk: int = 4096) -> tuple:
    """Mazes from the designer, one per budget: (walls bool[B, E], entrance int32[B], exit int32[B])."""
    budgets = np.asarray(budgets, dtype=np.int32)
    n_steps = steps(budgets.max())
    parts = []
    for i, lo in enumerate(range(0, len(budgets), chunk)):
        b = budgets[lo : lo + chunk]
        pad = chunk - len(b) if len(budgets) > chunk else 0  # same shape for every chunk: one compile
        b = np.concatenate([b, np.full(pad, b[0], np.int32)])
        w, e, s = _sample(params, jnp.asarray(b), jax.random.fold_in(key, i), size, n_steps, tuple(dilations), greedy)
        parts.append(tuple(np.asarray(x)[: len(b) - pad] for x in (w, e, s)))
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))


# ------------------------------------------------------------------ PPO


def advantages(batch: DBatch, reward, lam: float) -> tuple:
    """GAE(λ), undiscounted, the reward (walls hit) arriving at each maze's last decision.
    Returns (advantages, returns), 0 on padding steps."""
    valid = batch.valid.astype(jnp.float32)
    last = valid * (1.0 - jnp.concatenate([valid[1:], jnp.zeros_like(valid[:1])], axis=0))  # the final decision

    def back(carry, x):
        adv_next, v_next = carry
        v, ok, end = x
        delta = end * reward + (1.0 - end) * v_next - v
        adv = ok * (delta + lam * (1.0 - end) * adv_next)
        return (adv, v), adv

    _, adv = jax.lax.scan(back, (jnp.zeros_like(reward), jnp.zeros_like(reward)), (batch.value, valid, last), reverse=True)
    return adv, (adv + batch.value) * valid


class PPOConfig(NamedTuple):
    epochs: int = 3
    minibatch: int = 2048
    lr: float = 3e-4
    clip: float = 0.2
    gae_lambda: float = 0.95
    entropy: float = 0.01
    value_coef: float = 0.1
    magnet: float = 0.0  # weight of KL(π ‖ magnet), when the batch carries the magnet's log-probs
    max_grad_norm: float = 1.0


def loss_fn(params, mb: dict, cfg: PPOConfig, dilations):
    logits, value = apply(params, mb["obs"], mb["legal"], dilations)
    w = mb["valid"].astype(jnp.float32)
    n = jnp.maximum(jnp.sum(w), 1.0)
    logp_all = jax.nn.log_softmax(logits)
    logp = jnp.take_along_axis(logp_all, mb["action"][:, None], axis=1)[:, 0]
    ratio = jnp.exp(logp - mb["logp"])
    mean = jnp.sum(w * mb["adv"]) / n
    std = jnp.sqrt(jnp.sum(w * (mb["adv"] - mean) ** 2) / n)
    adv = (mb["adv"] - mean) / (std + 1e-8)
    pg = -jnp.sum(w * jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv)) / n
    v_loss = 0.5 * jnp.sum(w * (value - mb["ret"]) ** 2) / n
    p = jnp.exp(logp_all)
    entropy = -jnp.sum(w * jnp.sum(jnp.where(mb["legal"], p * logp_all, 0.0), axis=-1)) / n
    total = pg + cfg.value_coef * v_loss - cfg.entropy * entropy
    stats = {"pg": pg, "value": v_loss, "entropy": entropy, "kl": jnp.sum(w * (mb["logp"] - logp)) / n,
             "clipped": jnp.sum(w * (jnp.abs(ratio - 1) > cfg.clip)) / n}
    if "mag_logp" in mb:
        kl_mag = jnp.sum(w * jnp.sum(jnp.where(mb["legal"], p * (logp_all - mb["mag_logp"]), 0.0), axis=-1)) / n
        total = total + cfg.magnet * kl_mag
        stats["magnet_kl"] = kl_mag
    return total, stats


def make_optimizer(cfg: PPOConfig):
    return optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.lr))


def make_update(cfg: PPOConfig, dilations):
    """(optimizer, update): `update(params, opt_state, batch, reward, key)` runs the PPO epochs on the
    valid decisions of one batch of mazes (a Python loop around one compiled gradient step, as for
    the explorer)."""
    opt = make_optimizer(cfg)
    grad_fn = jax.grad(loss_fn, has_aux=True)

    @jax.jit
    def prepare(batch: DBatch, reward):
        adv, ret = advantages(batch, reward, cfg.gae_lambda)
        flat = {"obs": batch.obs, "legal": batch.legal, "action": batch.action, "logp": batch.logp,
                "valid": batch.valid, "adv": adv, "ret": ret}
        if batch.mag_logp is not None:
            flat["mag_logp"] = batch.mag_logp
        flat = {k: v.reshape((-1,) + v.shape[2:]) for k, v in flat.items()}
        w = flat["valid"].astype(jnp.float32)
        target, pred = flat["ret"], batch.value.reshape(-1)
        mean = jnp.sum(w * target) / jnp.sum(w)
        var = jnp.sum(w * (target - mean) ** 2)
        explained = 1 - jnp.sum(w * (target - pred) ** 2) / jnp.maximum(var, 1e-8)
        return flat, explained

    @jax.jit
    def sgd_step(params, opt_state, flat, idx):
        mb = {k: v[idx] for k, v in flat.items()}
        grads, stats = grad_fn(params, mb, cfg, dilations)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, stats

    def update(params, opt_state, batch: DBatch, reward, key):
        flat, explained = prepare(batch, jnp.asarray(reward, jnp.float32))
        valid = np.flatnonzero(np.asarray(flat["valid"]))  # train on real decisions only
        n_mb = max(1, len(valid) // cfg.minibatch)
        size = min(cfg.minibatch, len(valid))
        all_stats = []
        rng = np.random.default_rng(np.asarray(jax.random.key_data(key)).ravel())
        for _ in range(cfg.epochs):
            perm = rng.permutation(valid)
            for i in range(n_mb):
                params, opt_state, stats = sgd_step(params, opt_state, flat, jnp.asarray(perm[i * size : (i + 1) * size]))
                all_stats.append(stats)
        stats = jax.tree.map(lambda *x: jnp.mean(jnp.stack(x)), *all_stats)
        stats["explained_var"] = explained
        return params, opt_state, stats

    return opt, update


# ------------------------------------------------------------------ diversity


def diversity(walls, entrance, exit) -> dict:
    """How varied a sample of mazes is.

    distinct       share of distinct mazes in the sample
    eff_mazes      1 / P(two draws are the same maze), from the pairs that collide: exp of the
                   Rényi-2 entropy. With no collision it is reported as > the number of pairs.
    es_pairs       distinct (E, S) pairs, and es_entropy their empirical entropy (nats)
    hamming        mean share of walls that differ between two random mazes of the sample
                   (|a xor b| / (|a| + |b|)): 0 = all identical, 1 = no wall in common
    path           mean length (steps) of the shortest E→S path; reachable: share of cells
                   reachable from E
    """
    walls, entrance, exit = np.asarray(walls, bool), np.asarray(entrance), np.asarray(exit)
    n = len(entrance)
    keys = [w.tobytes() + bytes([int(e), int(s)]) for w, e, s in zip(np.packbits(walls, axis=1), entrance, exit)]
    counts = np.array(list(Counter(keys).values()))
    pairs = n * (n - 1) / 2
    same = float(np.sum(counts * (counts - 1) / 2))
    es = np.array(list(Counter(zip(entrance.tolist(), exit.tolist())).values()), dtype=float) / n
    perm = np.random.default_rng(0).permutation(n)
    sizes = walls.sum(axis=1)
    ham = np.logical_xor(walls, walls[perm]).sum(axis=1) / np.maximum(sizes + sizes[perm], 1)
    g = geometry(size_from_edges(walls.shape[1]))
    paths, reach = [], []
    for w, e, s in zip(walls[:512], entrance[:512], exit[:512]):
        d = distances_from(g, w, int(e))
        paths.append(d[int(s)])
        reach.append(np.mean(np.array(d) >= 0))
    return {
        "n": n,
        "distinct": len(counts) / n,
        "eff_mazes": pairs / same if same else float("inf"),
        "pairs": pairs,
        "es_pairs": len(es),
        "es_entropy": float(-np.sum(es * np.log(es))),
        "hamming": float(ham[perm != np.arange(n)].mean()) if n > 1 else 0.0,
        "path": float(np.mean(paths)),
        "reachable": float(np.mean(reach)),
    }


def diversity_line(d: dict) -> str:
    eff = f"> {d['pairs']:.0f}" if d["eff_mazes"] == float("inf") else f"{d['eff_mazes']:.0f}"
    return (f"distinct {100 * d['distinct']:.1f}%, effective mazes {eff}, E/S pairs {d['es_pairs']} "
            f"(H {d['es_entropy']:.2f}), hamming {d['hamming']:.2f}, path {d['path']:.1f}, reachable {d['reachable']:.2f}")


# ------------------------------------------------------------------ at play


class Designer:
    """A designer checkpoint, ready to make mazes (any board size: the network is convolutional)."""

    def __init__(self, path, greedy: bool = False):
        from . import learned

        self.params, self.meta, _ = learned.load(path)
        self.dilations = tuple(self.meta["net"]["dilations"])
        self.greedy = greedy

    def design(self, size: int, walls: int, count: int, rng, key) -> tuple:
        return sample_mazes(self.params, size, np.full(count, walls), key, self.dilations, self.greedy)
