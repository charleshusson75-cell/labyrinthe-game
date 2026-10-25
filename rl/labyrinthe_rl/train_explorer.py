"""Train the learned explorer (RL P2, docs/RL_DESIGN.md §5, DECISIONS.md D17, D19, D23).

    python -m labyrinthe_rl.train_explorer --size 4 --run rl/runs/explorer-4x4
    python -m labyrinthe_rl.train_explorer --run rl/runs/explorer-4x4 --resume [--iterations 2000]

PPO on the explorer's side of a round, against a mix of designers: every finished game is replaced
by a maze drawn from the mix (`--mix`). Designers: "uniform" (among valid mazes), "random"
(randomMaze), the planner's "easy", "normal" and "hard", and "adversary": each iteration it keeps,
out of `--adv-candidates` fresh randomMaze and uniform mazes, the one the current greedy policy hits
the most walls in (the pick of D21's exploitability test, aimed at the learner itself). The wall
budget of each maze is drawn from a band around the game's default (`--walls`, D15); the network
reads it from its input planes.

- Policy: the network's probe logits, masked to legal sides; reward −1 per wall hit, undiscounted.
- Value: expected walls still to hit (the sum of the hit head, see network.py), with GAE(λ).
- Belief: P(wall) for every unknown side, trained by cross-entropy on the true maze (privileged
  information in simulation). It is an auxiliary loss: it teaches the trunk what designs look like.

Designed mazes ("normal", "hard") are slow to make (≈ 150 hard mazes/s on 5×5), so each is drawn
once into a pool, cached on disk and shared by runs; the cheap samplers are refreshed every
iteration. Every `--eval-every` iterations the greedy policy explores fixed held-out mazes of each
designer at the default budget, next to the hard explorer, a uniformly random explorer and (if Node
is installed) the expert on the same mazes; and the worst of 32 candidates per round
(`--eval-worst` rounds), a proxy for how exploitable the policy is.

Outputs in the run directory: `checkpoint.npz` (latest, with the optimizer state: `--resume`
continues from it), `snap-<iteration>.npz` (parameters only, kept for later leagues), `log.jsonl`
(one record per iteration) and `refs.json` (the reference explorers' scores on the eval mazes).
"""

import argparse
import json
import os
import signal
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import exploit, expert, learned, network, planner
from .engine import WALL, State, init, legal_mask, step
from .geometry import default_wall_budget, max_wall_budget
from .lattice import UNKNOWN, action_pixels, observe
from .mazes import sample_mazes

CHEAP = ("uniform", "random")  # samplers fast enough to refresh every iteration
DESIGNED = ("easy", "normal", "hard")  # planner.design levels: pooled and cached
ADVERSARY = "adversary"  # the worst of a few fresh mazes for the current policy
SEED_OFFSET = {k: i for i, k in enumerate(("uniform", "random", "normal", "hard", ADVERSARY, "easy"))}  # pools' seeds
EVAL_KINDS = ("uniform", "random", "normal", "hard")
RUNS = Path(__file__).resolve().parents[1] / "runs"


@dataclass
class Config:
    size: int = 5
    walls: str = ""  # wall budgets for training: "12" or a band "10-14" (default: game default ± 2)
    mix: str = "uniform:0.2,random:0.15,normal:0.15,hard:0.3,adversary:0.2"
    cheap_pool: int = 16384  # mazes per cheap sampler; `refresh` of them are replaced every iteration
    refresh: int = 2048
    designed_pool: int = 40000  # mazes per designed pool (cached in pool_dir)
    pool_dir: str = str(RUNS / "pools")
    adv_pool: int = 4096  # adversarial mazes kept; `adv_refresh` are re-picked every iteration
    adv_refresh: int = 128
    adv_candidates: int = 8  # each pick keeps the worst of this many fresh mazes (same budget)
    workers: int = 0  # processes that design the pools (0: one per CPU core)
    channels: int = 32
    blocks: int = 4
    plan: bool = False  # belief planner's scores as a policy prior (network.plan_scores)
    envs: int = 256  # games played in parallel
    horizon: int = 32  # steps per game per iteration (envs × horizon samples per update)
    epochs: int = 3
    minibatch: int = 512
    lr: float = 3e-4
    clip: float = 0.2
    gae_lambda: float = 0.95
    entropy: float = 0.01
    value_coef: float = 0.05  # the value sums ~E per-side terms: its gradient is large (D23)
    belief_coef: float = 1.0
    max_grad_norm: float = 1.0
    iterations: int = 1000
    eval_every: int = 25
    eval_mazes: int = 512  # held-out mazes per designer, at the default budget
    expert_ref: int = 256  # of those, how many the expert explores once as a reference (needs Node)
    eval_worst: int = 128  # rounds of the worst-of-32 pick (exploit.py) in each evaluation; 0: none
    save_every: int = 25
    snapshot_every: int = 100
    seed: int = 0


def wall_band(cfg: Config) -> list:
    """The wall budgets mazes are drawn from during training."""
    d, hi = default_wall_budget(cfg.size), max_wall_budget(cfg.size)
    spec = cfg.walls or f"{max(1, d - 2)}-{min(hi, d + 2)}"
    lo, _, top = spec.partition("-")
    ws = list(range(int(lo), int(top or lo) + 1))
    if not ws or ws[0] < 1 or ws[-1] > hi:
        raise ValueError(f"walls must be within 1..{hi} on a {cfg.size}×{cfg.size} board, got {spec!r}")
    return ws


def parse_mix(spec: str) -> dict:
    mix = {}
    for part in spec.split(","):
        name, _, weight = part.partition(":")
        if name not in CHEAP + DESIGNED + (ADVERSARY,):
            raise ValueError(f"unknown designer {name!r} in the mix ({', '.join(CHEAP + DESIGNED + (ADVERSARY,))})")
        mix[name] = float(weight or 1)
    total = sum(mix.values())
    return {k: v / total for k, v in mix.items() if v > 0}


# ------------------------------------------------------------------ maze pools


def _design_chunk(task) -> tuple:
    size, walls, level, count, seed = task
    rng = np.random.default_rng(seed)
    return planner.design(size, walls, level, count, rng, jax.random.key(seed))


def _concat(parts) -> tuple:
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))


def _split(count: int, ws: list) -> list:
    """count mazes spread evenly over the budgets ws."""
    base, extra = divmod(count, len(ws))
    return [(w, base + (i < extra)) for i, w in enumerate(ws)]


def designed_mazes(level: str, size: int, ws: list, count: int, seed: int, workers: int, cache_dir) -> tuple:
    """`count` mazes from a planner designer, spread over the budgets `ws`, cached on disk."""
    cache = Path(cache_dir) / f"{level}-{size}x{size}-w{ws[0]}-{ws[-1]}-n{count}-s{seed}.npz"
    if cache.exists():
        with np.load(cache) as z:
            return z["walls"], z["entrance"], z["exit"]
    chunk = 256
    tasks = []
    for w, n in _split(count, ws):
        for lo in range(0, n, chunk):
            tasks.append((size, w, level, min(chunk, n - lo), int(np.random.SeedSequence([seed, w, lo]).generate_state(1)[0])))
    workers = workers or os.cpu_count() or 1
    t0 = time.perf_counter()
    if workers == 1:
        parts = [_design_chunk(t) for t in tasks]
    else:
        # Fresh processes, each single-threaded: JAX's own threads would only compete here.
        old = os.environ.get("XLA_FLAGS")
        os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
        try:
            with ProcessPoolExecutor(min(workers, len(tasks)), mp_context=get_context("spawn")) as pool:
                parts = list(pool.map(_design_chunk, tasks))
        finally:
            if old is None:
                os.environ.pop("XLA_FLAGS")
            else:
                os.environ["XLA_FLAGS"] = old
    mazes = _concat(parts)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, walls=mazes[0], entrance=mazes[1], exit=mazes[2])
    os.replace(tmp, cache)
    print(f"  designed {count} {level} mazes in {time.perf_counter() - t0:.0f} s → {cache}")
    return mazes


def cheap_mazes(kind: str, size: int, ws: list, count: int, seed: int) -> tuple:
    return _concat([sample_mazes(kind, size, w, n, seed * 1000 + w) for w, n in _split(count, ws) if n])


class Pool:
    """The training mazes of every designer of the mix, concatenated, with the mix's weights.
    A reset draws a designer from the mix, then one of its mazes uniformly."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.mix = parse_mix(cfg.mix)
        self.kinds = list(self.mix)
        self.ws = wall_band(cfg)
        parts, self.counts = [], []
        for kind in self.kinds:
            seed = cfg.seed * 100 + SEED_OFFSET[kind]  # by name: the same cached pools whatever the mix
            if kind in CHEAP:
                parts.append(cheap_mazes(kind, cfg.size, self.ws, cfg.cheap_pool, seed))
            elif kind == ADVERSARY:  # unpicked candidates, until the first picks replace them
                half = cfg.adv_pool // 2
                parts.append(_concat([cheap_mazes("random", cfg.size, self.ws, half, seed), cheap_mazes("uniform", cfg.size, self.ws, cfg.adv_pool - half, seed)]))
            else:
                parts.append(designed_mazes(kind, cfg.size, self.ws, cfg.designed_pool, seed, cfg.workers, cfg.pool_dir))
            self.counts.append(len(parts[-1][1]))
        self.walls, self.entrance, self.exit = (np.array(x) for x in _concat(parts))
        self.offsets = np.concatenate([[0], np.cumsum(self.counts)[:-1]])
        self._cursor = {k: 0 for k in self.kinds}

    def refresh(self, iteration: int, params=None, dilations=None) -> dict:
        """Replace `refresh` mazes of each cheap sampler with new ones, and `adv_refresh` adversarial
        mazes with new picks against `params` (rolling windows). Returns the adversary's stats."""
        stats = {}
        for i, kind in enumerate(self.kinds):
            seed = int(np.random.SeedSequence([self.cfg.seed, iteration, i]).generate_state(1)[0] % 2**31)
            if kind in CHEAP and self.cfg.refresh > 0:
                new = cheap_mazes(kind, self.cfg.size, self.ws, min(self.cfg.refresh, self.counts[i]), seed)
            elif kind == ADVERSARY and self.cfg.adv_refresh > 0 and params is not None:
                new, stats = self._pick(min(self.cfg.adv_refresh, self.counts[i]), seed, params, dilations)
            else:
                continue
            n = len(new[1])
            at = self.offsets[i] + (self._cursor[kind] + np.arange(n)) % self.counts[i]
            self.walls[at], self.entrance[at], self.exit[at] = new
            self._cursor[kind] = (self._cursor[kind] + n) % self.counts[i]
        return stats

    def _pick(self, n: int, seed: int, params, dilations) -> tuple:
        """n adversarial mazes: per budget, the worst of k fresh candidates for the greedy policy."""
        k, parts, worst, mean = self.cfg.adv_candidates, [], [], []
        for w, m in _split(n, self.ws):
            if not m:
                continue
            cands = exploit.candidates(self.cfg.size, w, m, k, seed + w)
            hits = learned.explore_mazes(params, *cands, jax.random.key(0), dilations).reshape(m, k)
            best = np.arange(m) * k + np.argmax(hits, axis=1)
            parts.append(tuple(x[best] for x in cands))
            worst.append(hits.max(axis=1))
            mean.append(hits.mean(axis=1))
        return _concat(parts), {"adv_pick": float(np.concatenate(worst).mean()), "adv_candidates": float(np.concatenate(mean).mean())}

    def device(self) -> "PoolArrays":
        probs = np.array([self.mix[k] for k in self.kinds], dtype=np.float32)
        return PoolArrays(
            jnp.asarray(self.walls), jnp.asarray(self.entrance, jnp.int32), jnp.asarray(self.exit, jnp.int32),
            jnp.asarray(self.offsets, jnp.int32), jnp.asarray(self.counts, jnp.int32), jnp.log(jnp.asarray(probs)),
        )


class PoolArrays(NamedTuple):
    walls: jax.Array
    entrance: jax.Array
    exit: jax.Array
    offsets: jax.Array
    counts: jax.Array
    log_probs: jax.Array


def draw(pool: PoolArrays, key, n: int) -> tuple:
    """n fresh games: (states, designer index of each)."""
    k1, k2 = jax.random.split(key)
    kind = jax.random.categorical(k1, pool.log_probs, shape=(n,))
    u = jax.random.uniform(k2, (n,))
    idx = pool.offsets[kind] + jnp.minimum((u * pool.counts[kind]).astype(jnp.int32), pool.counts[kind] - 1)
    return jax.vmap(init)(pool.walls[idx], pool.entrance[idx], pool.exit[idx]), kind


# ------------------------------------------------------------------ rollouts and PPO


class Envs(NamedTuple):
    states: State
    kind: jax.Array  # designer of each game's maze


class Batch(NamedTuple):
    obs: jax.Array  # [T, B, L, L, C]
    legal: jax.Array  # [T, B, E]
    action: jax.Array  # [T, B]
    logp: jax.Array
    value: jax.Array  # −(expected walls still to hit): the critic of rewards −1 per wall
    reward: jax.Array
    done: jax.Array
    walls: jax.Array  # [T, B, E] the true maze: the belief target
    ended_hits: jax.Array  # [T, B] walls hit by a game that ended at this step, −1 otherwise
    ended_kind: jax.Array


def _where_games(mask, new, old):
    return jax.tree.map(lambda f, o: jnp.where(mask.reshape((-1,) + (1,) * (o.ndim - 1)), f, o), new, old)


@partial(jax.jit, static_argnames=("horizon", "dilations"))
def rollout(params, envs: Envs, pool: PoolArrays, key, horizon: int, dilations):
    """`horizon` steps of every game with the sampled policy; ended games restart on new mazes."""
    n = envs.kind.shape[0]

    def one(carry, k):
        e = carry
        k_act, k_new = jax.random.split(k)
        obs = jax.vmap(observe)(e.states)
        legal = jax.vmap(legal_mask)(e.states)
        out = network.apply(params, obs, legal, dilations)
        action = jax.random.categorical(k_act, out.logits)
        logp = jnp.take_along_axis(jax.nn.log_softmax(out.logits), action[:, None], axis=1)[:, 0]
        states, result = jax.vmap(step)(e.states, action)
        done = states.done
        new_states, new_kind = draw(pool, k_new, n)
        rec = Batch(
            obs=obs, legal=legal, action=action, logp=logp, value=-out.value,
            reward=-(result == WALL).astype(jnp.float32), done=done, walls=e.states.walls,
            ended_hits=jnp.where(done, states.hits, -1), ended_kind=e.kind,
        )
        nxt = Envs(_where_games(done, new_states, states), jnp.where(done, new_kind, e.kind))
        return nxt, rec

    envs, batch = jax.lax.scan(one, envs, jax.random.split(key, horizon))
    obs = jax.vmap(observe)(envs.states)
    last_value = -network.apply(params, obs, jax.vmap(legal_mask)(envs.states), dilations).value
    return envs, batch, last_value


def gae(batch: Batch, last_value, lam: float) -> tuple:
    """Advantages and λ-returns, undiscounted (every game ends within E probes)."""

    def back(carry, x):
        adv_next, v_next = carry
        r, v, done = x
        live = 1.0 - done.astype(jnp.float32)
        delta = r + live * v_next - v
        adv = delta + lam * live * adv_next
        return (adv, v), adv

    _, adv = jax.lax.scan(back, (jnp.zeros_like(last_value), last_value), (batch.reward, batch.value, batch.done), reverse=True)
    return adv, adv + batch.value


def loss_fn(params, mb: dict, cfg: Config, dilations):
    out = network.apply(params, mb["obs"], mb["legal"], dilations)
    logp_all = jax.nn.log_softmax(out.logits)
    logp = jnp.take_along_axis(logp_all, mb["action"][:, None], axis=1)[:, 0]
    ratio = jnp.exp(logp - mb["logp"])
    adv = (mb["adv"] - mb["adv"].mean()) / (mb["adv"].std() + 1e-8)
    pg = -jnp.mean(jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv))
    v_loss = 0.5 * jnp.mean((-out.value - mb["ret"]) ** 2)  # returns count walls as −1
    entropy = -jnp.mean(jnp.sum(jnp.where(mb["legal"], jnp.exp(logp_all) * logp_all, 0.0), axis=-1))
    size = (mb["obs"].shape[1] - 1) // 2
    px = action_pixels(size)
    unknown = mb["obs"][:, px[:, 0], px[:, 1], UNKNOWN]
    bce = optax.sigmoid_binary_cross_entropy(out.belief, mb["walls"].astype(jnp.float32))
    b_loss = jnp.sum(bce * unknown) / jnp.maximum(jnp.sum(unknown), 1.0)
    total = pg + cfg.value_coef * v_loss - cfg.entropy * entropy + cfg.belief_coef * b_loss
    stats = {
        "pg": pg, "value": v_loss, "entropy": entropy, "belief": b_loss,
        "kl": jnp.mean(mb["logp"] - logp), "clipped": jnp.mean(jnp.abs(ratio - 1) > cfg.clip),
    }
    return total, stats


def make_optimizer(cfg: Config):
    return optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.lr))


def make_update(cfg: Config, dilations):
    """(optimizer, update): `update(params, opt_state, batch, last_value, key)` runs the PPO epochs.

    The minibatch loop is Python around one compiled gradient step: XLA:CPU runs convolution
    gradients ~40× slower inside a `lax.scan` than outside one.
    """
    opt = make_optimizer(cfg)
    grad_fn = jax.grad(loss_fn, has_aux=True)

    @jax.jit
    def prepare(batch: Batch, last_value):
        adv, ret = gae(batch, last_value, cfg.gae_lambda)
        flat = {
            "obs": batch.obs, "legal": batch.legal, "action": batch.action, "logp": batch.logp,
            "walls": batch.walls, "adv": adv, "ret": ret,
        }
        flat = {k: v.reshape((-1,) + v.shape[2:]) for k, v in flat.items()}
        var = jnp.var(ret)
        return flat, 1 - jnp.var(ret - batch.value) / jnp.maximum(var, 1e-8)

    @jax.jit
    def sgd_step(params, opt_state, flat, idx):
        mb = {k: v[idx] for k, v in flat.items()}
        grads, stats = grad_fn(params, mb, cfg, dilations)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, stats

    def update(params, opt_state, batch: Batch, last_value, key):
        flat, explained = prepare(batch, last_value)
        n = flat["action"].shape[0]
        n_mb = max(1, n // cfg.minibatch)
        size = n // n_mb
        all_stats = []
        for k in jax.random.split(key, cfg.epochs):
            perm = jax.random.permutation(k, n)[: n_mb * size].reshape(n_mb, size)
            for i in range(n_mb):
                params, opt_state, stats = sgd_step(params, opt_state, flat, perm[i])
                all_stats.append(stats)
        stats = jax.tree.map(lambda *x: jnp.mean(jnp.stack(x)), *all_stats)
        stats["explained_var"] = explained
        return params, opt_state, stats

    return opt, update


# ------------------------------------------------------------------ evaluation


class EvalSets:
    """Held-out mazes of each designer at the default budget, and the reference explorers' walls hit
    on them (computed once per run, kept in refs.json)."""

    def __init__(self, cfg: Config, run: Path):
        self.cfg = cfg
        w = default_wall_budget(cfg.size)
        self.walls = w
        self.sets = {}
        for i, kind in enumerate(EVAL_KINDS):
            seed = 10_000 + cfg.seed * 100 + i  # never a training pool's seed
            if kind in CHEAP:
                self.sets[kind] = cheap_mazes(kind, cfg.size, [w], cfg.eval_mazes, seed)
            else:
                self.sets[kind] = designed_mazes(kind, cfg.size, [w], cfg.eval_mazes, seed, cfg.workers, cfg.pool_dir)
        self.key = jax.random.key(10_000 + cfg.seed)
        self.worst = exploit.candidates(cfg.size, w, cfg.eval_worst, 32, 20_000 + cfg.seed) if cfg.eval_worst else None
        path = run / "refs.json"
        self.refs = json.loads(path.read_text()) if path.exists() else {}
        if self._references():
            path.write_text(json.dumps(self.refs, indent=1))

    def _references(self) -> bool:
        """Fill in the missing reference scores; True if any was added."""
        added = False
        for kind, mazes in self.sets.items():
            if kind in self.refs:
                continue
            r = {
                "hard": float(planner.explore_mazes(*mazes, self.key, planner.HARD).mean()),
                "random": float(learned.random_mazes(*mazes, self.key).mean()),
            }
            n = min(self.cfg.expert_ref, len(mazes[1]))
            if n and expert.node():
                r["expert"] = float(expert.explore_mazes(*(x[:n] for x in mazes), self.key).mean())
                r["expert_mazes"] = n
            self.refs[kind], added = r, True
        if self.worst is not None and "worst32" not in self.refs:
            self.refs["worst32"] = {
                "hard": self._worst(planner.explore_mazes(*self.worst, self.key, planner.HARD)),
                "random": self._worst(learned.random_mazes(*self.worst, self.key)),
            }
            added = True
        return added

    def _worst(self, hits) -> float:
        return float(np.asarray(hits).reshape(-1, 32).max(axis=1).mean())

    def evaluate(self, params, dilations) -> dict:
        out = {}
        for kind, mazes in self.sets.items():
            hits = learned.explore_mazes(params, *mazes, self.key, dilations)
            out[kind] = float(hits.mean())
            n = self.refs[kind].get("expert_mazes")
            if n:
                out[kind + "@expert"] = float(hits[:n].mean())  # on the mazes the expert explored
        if self.worst is not None:
            out["worst32"] = self._worst(learned.explore_mazes(params, *self.worst, self.key, dilations))
        return out

    def line(self, scores: dict) -> str:
        parts = []
        for kind in self.sets:
            r = self.refs[kind]
            ref = f"hard {r['hard']:.2f}, rnd {r['random']:.2f}"
            if "expert" in r:
                ref += f", expert {r['expert']:.2f} vs {scores[kind + '@expert']:.2f}"
            parts.append(f"{kind} {scores[kind]:.2f} ({ref})")
        if "worst32" in scores:
            r = self.refs["worst32"]
            parts.append(f"worst of 32 {scores['worst32']:.2f} (hard {r['hard']:.2f}, rnd {r['random']:.2f})")
        return " | ".join(parts)


# ------------------------------------------------------------------ the loop


def _meta(cfg: Config, dilations, iteration: int, env_steps: int, episodes: int, key, elapsed: float, last_eval) -> dict:
    return {
        "version": 1,
        "kind": "explorer",
        "size": cfg.size,
        "net": {"channels": cfg.channels, "blocks": cfg.blocks, "dilations": list(dilations), "plan": cfg.plan},
        "config": asdict(cfg),
        "iteration": iteration,
        "env_steps": env_steps,
        "episodes": episodes,
        "key": np.asarray(jax.random.key_data(key)).tolist(),
        "elapsed": elapsed,
        "eval": last_eval,
    }


def train(cfg: Config, run, resume: bool = False, init_from=None, log=print) -> dict:
    """Run (or continue) training until `cfg.iterations`; returns the last checkpoint's meta."""
    run = Path(run)
    ckpt = run / "checkpoint.npz"
    dilations = network.NetConfig().dilations
    if ckpt.exists() and not resume:
        raise FileExistsError(f"{ckpt} exists: pass --resume, or choose another --run")
    run.mkdir(parents=True, exist_ok=True)
    opt = make_optimizer(cfg)
    if resume and ckpt.exists():
        params, meta, arrays = learned.load(ckpt)
        dilations = tuple(meta["net"]["dilations"])
        opt_state = learned.unflatten(opt.init(params), arrays, "opt")
        iteration, env_steps, episodes = meta["iteration"], meta["env_steps"], meta["episodes"]
        key = jax.random.wrap_key_data(jnp.asarray(meta["key"], dtype=jnp.uint32))
        elapsed, last_eval = meta["elapsed"], meta["eval"]
        log(f"resuming {ckpt} at iteration {iteration} ({env_steps / 1e6:.2f} M steps)")
    else:
        key = jax.random.key(cfg.seed)
        key, k_init = jax.random.split(key)
        params = network.init(k_init, network.NetConfig(cfg.channels, cfg.blocks, dilations, cfg.plan))
        if init_from:
            params, src, _ = learned.load(init_from)  # fully convolutional: any board size
            shape = (src["net"]["channels"], src["net"]["blocks"], src["net"].get("plan", False))
            if shape != (cfg.channels, cfg.blocks, cfg.plan):
                raise ValueError(f"{init_from} has network {shape}, expected {(cfg.channels, cfg.blocks, cfg.plan)}")
            log(f"initialized from {init_from} ({src['size']}×{src['size']}, iteration {src['iteration']})")
        opt_state = opt.init(params)
        iteration = env_steps = episodes = 0
        elapsed, last_eval = 0.0, None
    (run / "config.json").write_text(json.dumps(asdict(cfg), indent=1))  # as of the latest (re)start
    _, update = make_update(cfg, dilations)

    log(f"{cfg.size}×{cfg.size}, walls {wall_band(cfg)}, mix {parse_mix(cfg.mix)}, net {cfg.channels}×{cfg.blocks}{' + planner' if cfg.plan else ''} "
        f"({network.n_params(params):,} parameters), {cfg.envs} games × {cfg.horizon} steps per update")
    pool = Pool(cfg)
    evals = EvalSets(cfg, run)
    log("references (walls hit, default budget): " + ", ".join(
        f"{k}: " + "/".join(f"{n} {v:.2f}" for n, v in r.items() if n != "expert_mazes") for k, r in evals.refs.items()))

    key, k_env = jax.random.split(key)
    envs = Envs(*draw(pool.device(), k_env, cfg.envs))

    def save(path, with_opt=True):
        meta = _meta(cfg, dilations, iteration, env_steps, episodes, key, elapsed, last_eval)
        learned.save(path, params, meta, opt_state if with_opt else None)
        return meta

    stop = {"now": False}

    def on_term(signum, frame):
        stop["now"] = True

    old_term = signal.signal(signal.SIGTERM, on_term)
    log_file = open(run / "log.jsonl", "a")
    try:
        while iteration < cfg.iterations and not stop["now"]:
            t0 = time.perf_counter()
            adv = pool.refresh(iteration, params, dilations)
            key, k_roll, k_upd = jax.random.split(key, 3)
            envs, batch, last_value = rollout(params, envs, pool.device(), k_roll, cfg.horizon, dilations)
            params, opt_state, stats = update(params, opt_state, batch, last_value, k_upd)
            stats = {k: float(v) for k, v in stats.items()}
            iteration += 1
            steps = cfg.envs * cfg.horizon
            env_steps += steps
            ended = np.asarray(batch.ended_hits) >= 0
            hits, kinds = np.asarray(batch.ended_hits)[ended], np.asarray(batch.ended_kind)[ended]
            episodes += int(ended.sum())
            train_hits = {k: float(hits[kinds == i].mean()) for i, k in enumerate(pool.kinds) if np.any(kinds == i)}
            dt = time.perf_counter() - t0
            elapsed += dt
            record = {"iteration": iteration, "env_steps": env_steps, "steps_per_s": steps / dt, "train_hits": train_hits, **stats, **adv}
            if iteration % cfg.eval_every == 0 or iteration == cfg.iterations:
                last_eval = {"iteration": iteration, **evals.evaluate(params, dilations)}
                record["eval"] = last_eval
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
            log(f"it {iteration:5d} {env_steps / 1e6:7.2f}M {steps / dt:6.0f}/s | train hits "
                + " ".join(f"{k} {v:.2f}" for k, v in train_hits.items())
                + (f" (picks {adv['adv_pick']:.2f} of candidates {adv['adv_candidates']:.2f})" if adv else "")
                + f" | pg {stats['pg']:+.3f} v {stats['value']:.3f} ent {stats['entropy']:.2f} bel {stats['belief']:.3f}"
                f" kl {stats['kl']:.4f} ev {stats['explained_var']:.2f}")
            if "eval" in record:
                log(f"  eval (greedy, {evals.walls} walls): " + evals.line(last_eval))
            if iteration % cfg.save_every == 0:
                save(ckpt)
            if cfg.snapshot_every and iteration % cfg.snapshot_every == 0:
                save(run / f"snap-{iteration:06d}.npz", with_opt=False)
    except KeyboardInterrupt:
        log("interrupted")
    finally:
        signal.signal(signal.SIGTERM, old_term)
        log_file.close()
    meta = save(ckpt)
    log(f"saved {ckpt} (iteration {iteration}, {env_steps / 1e6:.2f} M steps, {elapsed / 60:.1f} min)")
    return meta


def config_from_args(argv=None) -> tuple:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory (checkpoints, logs)")
    ap.add_argument("--resume", action="store_true", help="continue from the run's checkpoint.npz")
    ap.add_argument("--init", default=None, help="start from another checkpoint's parameters (e.g. a 4×4 run)")
    for f in fields(Config):
        kind = type(f.default)
        if kind is bool:
            kind = lambda v: {"1": True, "true": True, "yes": True, "0": False, "false": False, "no": False}[v.lower()]
        ap.add_argument("--" + f.name.replace("_", "-"), type=kind, default=None)
    args = ap.parse_args(argv)
    given = {f.name: getattr(args, f.name) for f in fields(Config) if getattr(args, f.name) is not None}
    ckpt = Path(args.run) / "checkpoint.npz"
    if args.resume and ckpt.exists():
        with np.load(ckpt) as z:
            saved = json.loads(str(z["meta"]))["config"]
        free = ("iterations", "eval_every", "save_every", "snapshot_every", "mix")  # a curriculum may change the mix
        changed = sorted(k for k in given if k not in free and given[k] != saved.get(k))
        if changed:
            ap.error(f"--resume keeps the run's config; can't change {', '.join(changed)}")
        cfg = Config(**{**asdict(Config()), **saved, **given})
    else:
        cfg = Config(**given)
    return cfg, args


def main(argv=None):
    cfg, args = config_from_args(argv)
    train(cfg, args.run, resume=args.resume, init_from=args.init)


if __name__ == "__main__":
    main()
