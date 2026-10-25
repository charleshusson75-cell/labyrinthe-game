"""League self-play: the designer and the explorer learn together (RL P3, DECISIONS.md D19, D24, D25).

    python -m labyrinthe_rl.league --run rl/runs/league-5x5 --explorer-init STAGE0.npz [--hours 10.5]
    python -m labyrinthe_rl.league --run rl/runs/league-5x5 --resume [--hours 10.5]
    python -m labyrinthe_rl.league --run rl/runs/league-5x5 --worker      exploiters, on another GPU
    python -m labyrinthe_rl.league --run rl/runs/designer-vs-p2 --designer-vs p2 [--hours 0.5]

Agents:
- the main explorer and the main designer (designer.py), both learning;
- frozen snapshots of both, taken every `--gen-every` iterations (one generation);
- exploiters: a fresh explorer trained only against one generation's frozen designer, and a fresh
  designer trained only against its frozen explorer. What they gain over the main agents measures
  how exploitable the generation is; then they join the league as frozen members;
- fixed anchors: the designers of P2's mix (`--mix`: uniform, randomMaze, normal, hard, adversary),
  and the explorers of `--opponents` (P2's checkpoint, greedy as in bench; the hard planner).

One iteration:
1. The main designer makes `--d-mazes` mazes (budgets drawn from the band, as the explorer's). Each
   maze is explored by one opponent: the main explorer for a share `--d-self-share`, otherwise a
   league explorer drawn by PFSP, weight ∝ prior · (floor + (1 − x)^p) with x its walls hit / W on the
   designer's mazes (an EMA): the explorers that handle its mazes best. Reward = walls hit, minus the
   batch mean of that maze's opponent (a baseline independent of the designer's actions: it removes
   the opponent's luck of the draw). PPO update of the designer.
2. Those mazes enter the "self" pool. The main explorer plays `--envs` games × `--horizon` probes,
   each game on a "self" maze for a share `--self-share`, otherwise on a maze of a league designer
   drawn by PFSP, weight ∝ prior · (floor + x^p) with x its walls hit / W on that designer's mazes:
   the designers it handles worst. PPO update of the explorer (P2's losses).
3. Magnets (MMD / R-NaD): each role's loss also has coef · KL(π ‖ magnet) (`--magnet`, `--d-magnet`),
   the magnet being an exponential moving average of the role's parameters (`--magnet-rate`), so
   training settles toward a regularized equilibrium instead of cycling. The entropy bonuses pull
   toward uniform play on top of that (the designer's keeps its mazes varied).

Outputs in `--run`: `explorer.npz` / `designer.npz` (parameters, optimizer state; the bench loads
them), their magnets, `league.json` (counters, members, PFSP statistics, exploiters' results),
`league/gen-NNNN/` (frozen agents and exploiters of every generation), `log.jsonl`, `refs.json`.
Checkpoints are written every `--save-minutes`, at every generation and on exit (SIGTERM included);
`--resume` continues from them, `--hours` stops cleanly.

Exploiters run `--exploiters inline` (at each generation, in this process), `worker` (another process,
typically on the second GPU: `--worker`, which picks up new generations as they appear) or `off`.

`--designer-vs PATH` (or `p2`: P2's committed explorer) trains a fresh designer against that one frozen
explorer (its sampled policy), with the same PPO, and evaluates it like the league's: how exploitable
that explorer is to a learned designer, and a designer to start a league from (`--designer-init`).
"""

import argparse
import json
import os
import signal
import time
from dataclasses import asdict, dataclass, fields, replace
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import designer, learned, network, planner, train_explorer
from .exploit import candidates as adversarial_candidates
from .geometry import default_wall_budget
from .train_explorer import ADVERSARY, CHEAP, DESIGNED, Envs, EvalSets, PoolArrays, cheap_mazes, designed_mazes, draw, parse_mix, wall_band

MAX_SLOTS = 64  # explorer pool slots (designers); fixed so the rollout compiles once
POOL_ROWS = 32768  # the pool's arrays are padded to a multiple of this many mazes, for the same reason


@dataclass
class Config(train_explorer.Config):
    # explorer: P2's fields (train_explorer.Config) plus
    magnet: float = 0.02
    magnet_rate: float = 0.02  # magnet ← magnet + rate · (params − magnet), every iteration, both roles
    explorer_init: str = ""  # start the main explorer from this checkpoint (stage 0); same network shape
    # designer
    d_channels: int = 32
    d_blocks: int = 4
    d_mazes: int = 512  # mazes per iteration
    d_epochs: int = 3
    d_minibatch: int = 2048
    d_lr: float = 1e-3  # 3e-4 barely moved the designer (one noisy reward per ~14 decisions)
    d_entropy: float = 0.01
    d_value_coef: float = 0.1
    d_magnet: float = 0.02
    designer_init: str = ""
    # league
    self_share: float = 0.4  # explorer games on the main designer's newest mazes
    d_self_share: float = 0.5  # designer mazes explored by the main explorer
    opponents: str = "p2,hard"  # fixed explorers in the designer's league
    p2_checkpoint: str = ""  # default: rl/checkpoints/explorer-NxN.npz
    pfsp_power: float = 2.0
    pfsp_floor: float = 0.1
    pfsp_rate: float = 0.1  # EMA rate of the PFSP statistics
    self_pool: int = 8192
    member_pool: int = 4096  # mazes kept per frozen designer
    member_refresh: int = 512  # of them, re-sampled per iteration (one member at a time, in turn)
    gen_every: int = 100
    max_members: int = 12  # frozen snapshots kept per role (and as many exploiters)
    exploiters: str = "inline"  # inline | worker | off
    x_exploit_iters: int = 100
    d_exploit_iters: int = 100
    exploit_init: str = "scratch"  # scratch | main
    exploit_mazes: int = 1024  # held-out mazes to measure an exploiter's gain
    d_eval_mazes: int = 1024
    precision: str = "fp32"


def _ema(magnet, params, rate: float):
    return jax.tree.map(lambda m, p: m + rate * (p - m), magnet, params)


ema = jax.jit(_ema)


def pfsp(xs: list, priors: list, power: float, floor: float, hard_is_high: bool) -> np.ndarray:
    """Prioritized fictitious self-play weights (sum 1). x: walls hit / W of the main agent against each
    member (None: not measured yet, taken as 0.5). The explorer favours the designers it hits most
    walls on (`hard_is_high`); the designer, the explorers that hit fewest on its mazes."""
    x = np.array([0.5 if v is None else v for v in xs], dtype=float)
    d = x if hard_is_high else 1.0 - x
    w = np.asarray(priors, dtype=float) * (floor + np.clip(d, 0, 1) ** power)
    return w / w.sum() if w.sum() > 0 else np.full(len(w), 1 / len(w))


def _pow2(n: int) -> int:
    return 1 << max(0, int(n - 1).bit_length())


# ------------------------------------------------------------------ the league's members


class MazeSource:
    """A designer the explorer trains against: a pool of its mazes (and its network, if any)."""

    def __init__(self, name: str, kind: str, mazes: tuple, params=None, dilations=None, prior: float = 1.0, path: str = ""):
        self.name, self.kind, self.params, self.dilations, self.prior, self.path = name, kind, params, dilations, prior, path
        self.walls, self.entrance, self.exit = (np.array(x) for x in mazes)
        self.cursor = 0
        self.x = None  # EMA of the main explorer's walls hit / W on these mazes
        self.active = True

    def replace(self, new: tuple) -> None:
        """Overwrite the oldest mazes (a rolling window)."""
        n = min(len(new[1]), len(self.entrance))
        at = (self.cursor + np.arange(n)) % len(self.entrance)
        self.walls[at], self.entrance[at], self.exit[at] = (x[:n] for x in new)
        self.cursor = (self.cursor + n) % len(self.entrance)


class Opponent:
    """An explorer the designer trains against."""

    def __init__(self, name: str, kind: str, explore, params=None, prior: float = 1.0, path: str = ""):
        self.name, self.kind, self.explore, self.params, self.prior, self.path = name, kind, explore, params, prior, path
        self.x = None  # EMA of its walls hit / W on the main designer's mazes
        self.active = True


def network_opponent(name: str, kind: str, params, dilations, greedy: bool = False, path: str = "") -> Opponent:
    def explore(w, e, s, key):
        return learned.explore_mazes(params, w, e, s, key, dilations, greedy=greedy)

    return Opponent(name, kind, explore, params, path=path)


def hard_opponent() -> Opponent:
    return Opponent("hard", "anchor", lambda w, e, s, key: planner.explore_mazes(w, e, s, key, planner.HARD))


class League:
    """Both sides of the league, with PFSP sampling."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ws = wall_band(cfg)
        self.sources: list = []  # explorer pool slots (a pruned member's slot is reused)
        self.opponents: list = []
        self.rr = 0  # round robin over network sources for refreshes
        self.born = 0  # members added so far: their age, for pruning

    # --- explorer side

    def add_source(self, src: MazeSource) -> None:
        """A new pool slot, or the slot of a pruned member (games still in flight on its mazes are
        then counted for the newcomer once: harmless for the statistics)."""
        src.born, self.born = self.born, self.born + 1
        free = next((i for i, s in enumerate(self.sources) if not s.active), None)
        if free is not None:
            self.sources[free] = src
        elif len(self.sources) >= MAX_SLOTS:
            raise RuntimeError(f"more than {MAX_SLOTS} maze sources")
        else:
            self.sources.append(src)

    def source(self, name: str):
        return next((s for s in self.sources if s.name == name), None)

    def source_weights(self) -> np.ndarray:
        """Per slot, the probability that a new game starts on one of its mazes."""
        cfg = self.cfg
        w = np.zeros(len(self.sources))
        live = [i for i, s in enumerate(self.sources) if s.active and s.name != "self" and len(s.entrance)]
        if live:
            w[live] = (1 - cfg.self_share) * pfsp([self.sources[i].x for i in live], [self.sources[i].prior for i in live], cfg.pfsp_power, cfg.pfsp_floor, True)
        me = self.source("self")
        w[self.sources.index(me)] = cfg.self_share if live else 1.0
        return w

    def device(self) -> PoolArrays:
        parts = [(s.walls, s.entrance, s.exit) if s.active else (s.walls[:0], s.entrance[:0], s.exit[:0]) for s in self.sources]
        counts = np.array([len(p[1]) for p in parts] + [0] * (MAX_SLOTS - len(parts)), dtype=np.int32)
        offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int32)
        walls = np.concatenate([p[0] for p in parts])
        rows = -(-len(walls) // POOL_ROWS) * POOL_ROWS
        pad = rows - len(walls)
        walls = np.concatenate([walls, np.zeros((pad, walls.shape[1]), bool)])
        ent = np.concatenate([p[1] for p in parts] + [np.zeros(pad, np.int32)]).astype(np.int32)
        ext = np.concatenate([p[2] for p in parts] + [np.ones(pad, np.int32)]).astype(np.int32)
        w = np.zeros(MAX_SLOTS)
        w[: len(self.sources)] = self.source_weights()
        with np.errstate(divide="ignore"):
            logp = np.log(w)
        return PoolArrays(jnp.asarray(walls), jnp.asarray(ent), jnp.asarray(ext), jnp.asarray(offsets), jnp.asarray(counts), jnp.asarray(logp, jnp.float32))

    def observe_games(self, batch) -> dict:
        """Update the explorer-side PFSP statistics from the games that ended in a rollout; returns the
        mean walls hit per source."""
        hits = np.asarray(batch.ended_hits)
        ended = hits >= 0
        kinds = np.asarray(batch.ended_kind)[ended]
        budgets = np.asarray(jnp.sum(batch.walls, axis=-1))[ended]
        hits = hits[ended]
        out = {}
        for i, s in enumerate(self.sources):
            m = kinds == i
            if m.any():
                frac = float(np.mean(hits[m] / budgets[m]))
                s.x = frac if s.x is None else (1 - self.cfg.pfsp_rate) * s.x + self.cfg.pfsp_rate * frac
                out[s.name] = float(hits[m].mean())
        return out

    def refresh(self, iteration: int, xparams, dilations) -> dict:
        """New mazes for the cheap anchors, adversarial picks against the main explorer, and a
        re-sample of one frozen designer's pool."""
        cfg, stats = self.cfg, {}
        for i, s in enumerate(self.sources):
            if not s.active:
                continue
            seed = int(np.random.SeedSequence([cfg.seed, iteration, i]).generate_state(1)[0] % 2**31)
            if s.name in CHEAP and cfg.refresh:
                s.replace(cheap_mazes(s.name, cfg.size, self.ws, min(cfg.refresh, len(s.entrance)), seed))
            elif s.name == ADVERSARY and cfg.adv_refresh:
                new, stats = adversary_picks(cfg, self.ws, min(cfg.adv_refresh, len(s.entrance)), seed, xparams, dilations)
                s.replace(new)
        nets = [s for s in self.sources if s.active and s.params is not None and s.name != "self"]
        if nets and cfg.member_refresh:
            s = nets[self.rr % len(nets)]
            self.rr += 1
            budgets = np.random.default_rng([cfg.seed, iteration, 7]).choice(self.ws, cfg.member_refresh)
            s.replace(designer.sample_mazes(s.params, cfg.size, budgets, jax.random.key(iteration), s.dilations))
        return stats

    # --- designer side

    def add_opponent(self, opp: Opponent) -> None:
        opp.born, self.born = self.born, self.born + 1
        self.opponents.append(opp)

    def opponent_weights(self) -> np.ndarray:
        cfg = self.cfg
        live = [i for i, o in enumerate(self.opponents) if o.active and o.name != "self"]
        w = np.zeros(len(self.opponents))
        if live:
            w[live] = (1 - cfg.d_self_share) * pfsp([self.opponents[i].x for i in live], [self.opponents[i].prior for i in live], cfg.pfsp_power, cfg.pfsp_floor, False)
        w[[o.name for o in self.opponents].index("self")] = cfg.d_self_share if live else 1.0
        return w

    def explore(self, mazes: tuple, key, rng) -> tuple:
        """Each maze explored by one opponent drawn by weight (in chunks, so the explorations compile
        for a few shapes only). Returns (walls hit per maze, opponent index per maze, per-opponent
        means)."""
        n = len(mazes[1])
        chunk = max(1, n // 16)
        weights = self.opponent_weights()
        counts = rng.multinomial(n // chunk, weights) * chunk
        counts[[o.name for o in self.opponents].index("self")] += n - counts.sum()
        order = rng.permutation(n)
        hits = np.zeros(n, dtype=np.int32)
        who = np.zeros(n, dtype=np.int32)
        budgets = mazes[0].sum(axis=1)
        means, pos = {}, 0
        for i, (opp, c) in enumerate(zip(self.opponents, counts)):
            if not c:
                continue
            idx = order[pos : pos + c]
            pos += c
            padded = np.concatenate([idx, np.resize(idx, _pow2(c) - c)])  # a power-of-two batch
            h = np.asarray(opp.explore(*(x[padded] for x in mazes), jax.random.fold_in(key, i)))[:c]
            hits[idx], who[idx] = h, i
            frac = float(np.mean(h / budgets[idx]))
            opp.x = frac if opp.x is None else (1 - self.cfg.pfsp_rate) * opp.x + self.cfg.pfsp_rate * frac
            means[opp.name] = float(h.mean())
        return hits, who, means

    # --- bookkeeping

    def state(self) -> dict:
        return {
            "sources": [{"name": s.name, "kind": s.kind, "x": s.x, "active": s.active, "path": s.path, "prior": s.prior, "born": s.born} for s in sorted(self.sources, key=lambda m: m.born)],
            "opponents": [{"name": o.name, "kind": o.kind, "x": o.x, "active": o.active, "path": o.path, "prior": o.prior, "born": o.born} for o in self.opponents],
            "rr": self.rr,
        }

    def prune(self, kind: str) -> None:
        """Keep the newest `max_members` frozen members of a kind on each side."""
        for members in (self.sources, self.opponents):
            live = sorted((m for m in members if m.kind == kind and m.active), key=lambda m: m.born)
            for m in live[: max(0, len(live) - self.cfg.max_members)]:
                m.active = False
                if isinstance(m, MazeSource):  # free the memory; the next new member takes the slot
                    m.walls, m.entrance, m.exit = m.walls[:0], m.entrance[:0], m.exit[:0]


def adversary_picks(cfg: Config, ws: list, n: int, seed: int, params, dilations) -> tuple:
    """n adversarial mazes: per budget, the worst of `adv_candidates` fresh ones for the greedy policy
    (as train_explorer.Pool._pick)."""
    k, parts, worst, mean = cfg.adv_candidates, [], [], []
    for w, m in train_explorer._split(n, ws):
        if not m:
            continue
        cands = adversarial_candidates(cfg.size, w, m, k, seed + w)
        hits = learned.explore_mazes(params, *cands, jax.random.key(0), dilations).reshape(m, k)
        best = np.arange(m) * k + np.argmax(hits, axis=1)
        parts.append(tuple(x[best] for x in cands))
        worst.append(hits.max(axis=1))
        mean.append(hits.mean(axis=1))
    new = tuple(np.concatenate([p[i] for p in parts]) for i in range(3))
    return new, {"adv_pick": float(np.concatenate(worst).mean()), "adv_candidates": float(np.concatenate(mean).mean())}


def anchor_sources(cfg: Config, ws: list) -> list:
    """P2's designers as fixed league members (pools as in train_explorer.Pool)."""
    mix = parse_mix(cfg.mix)
    out = []
    for kind, weight in mix.items():
        seed = cfg.seed * 100 + train_explorer.SEED_OFFSET[kind]
        if kind in CHEAP:
            mazes = cheap_mazes(kind, cfg.size, ws, cfg.cheap_pool, seed)
        elif kind == ADVERSARY:
            half = cfg.adv_pool // 2
            mazes = train_explorer._concat([cheap_mazes("random", cfg.size, ws, half, seed), cheap_mazes("uniform", cfg.size, ws, cfg.adv_pool - half, seed)])
        else:
            mazes = designed_mazes(kind, cfg.size, ws, cfg.designed_pool, seed, cfg.workers, cfg.pool_dir)
        out.append(MazeSource(kind, "anchor", mazes, prior=weight * len(mix)))
    return out


def p2_path(cfg: Config) -> str:
    return cfg.p2_checkpoint or str(learned.default_checkpoint(cfg.size))


# ------------------------------------------------------------------ learners


class ExplorerLearner:
    """PPO on the explorer (train_explorer's rollout and losses), with an optional magnet."""

    def __init__(self, cfg: Config, params, opt_state=None, magnet=None):
        self.cfg, self.dilations = cfg, network.NetConfig().dilations
        self.opt, self.update = train_explorer.make_update(cfg, self.dilations)
        self.params = params
        self.opt_state = self.opt.init(params) if opt_state is None else opt_state
        self.magnet = magnet if cfg.magnet else None
        self.envs = None

    def iterate(self, pool: PoolArrays, key) -> tuple:
        k0, k1, k2 = jax.random.split(key, 3)
        if self.envs is None:
            self.envs = Envs(*draw(pool, k0, self.cfg.envs))
        self.envs, batch, last = train_explorer.rollout(self.params, self.envs, pool, k1, self.cfg.horizon, self.dilations, self.magnet)
        self.params, self.opt_state, stats = self.update(self.params, self.opt_state, batch, last, k2)
        if self.magnet is not None:
            self.magnet = ema(self.magnet, self.params, self.cfg.magnet_rate)
        return {k: float(v) for k, v in stats.items()}, batch


class DesignerLearner:
    """PPO on the designer, with an optional magnet."""

    def __init__(self, cfg: Config, params, opt_state=None, magnet=None):
        self.cfg, self.dilations = cfg, network.NetConfig().dilations
        self.pcfg = designer.PPOConfig(
            epochs=cfg.d_epochs, minibatch=cfg.d_minibatch, lr=cfg.d_lr, clip=cfg.clip, gae_lambda=cfg.gae_lambda,
            entropy=cfg.d_entropy, value_coef=cfg.d_value_coef, magnet=cfg.d_magnet if magnet is not None else 0.0,
            max_grad_norm=cfg.max_grad_norm,
        )
        self.opt, self.update = designer.make_update(self.pcfg, self.dilations)
        self.params = params
        self.opt_state = self.opt.init(params) if opt_state is None else opt_state
        self.magnet = magnet if cfg.d_magnet else None
        self.n_steps = designer.steps(max(wall_band(cfg)))

    def generate(self, budgets, key) -> tuple:
        final, batch = designer.rollout(self.params, jnp.asarray(budgets, jnp.int32), key, self.cfg.size, self.n_steps, self.dilations, self.magnet)
        mazes = (np.asarray(final.walls), np.asarray(final.entrance), np.asarray(final.exit))
        return mazes, batch

    def learn(self, batch, reward, key) -> dict:
        self.params, self.opt_state, stats = self.update(self.params, self.opt_state, batch, reward, key)
        if self.magnet is not None:
            self.magnet = ema(self.magnet, self.params, self.cfg.magnet_rate)
        return {k: float(v) for k, v in stats.items()}


def seq_entropy(batch) -> float:
    """Mean −log p(decision sequence) of the sampled mazes: an upper bound of the maze distribution's
    entropy (a maze has W! wall orders), nats."""
    return float(-jnp.mean(jnp.sum(jnp.where(batch.valid, batch.logp, 0.0), axis=0)))


def opponent_baseline(hits, who) -> np.ndarray:
    """Walls hit minus the batch mean of the maze's opponent, plus the overall mean."""
    out = hits.astype(np.float32).copy()
    for i in np.unique(who):
        m = who == i
        out[m] -= hits[m].mean()
    return out + hits.mean()


# ------------------------------------------------------------------ exploiters


def _net_meta(cfg: Config, kind: str, channels: int, blocks: int, **extra) -> dict:
    return {"version": 1, "kind": kind, "size": cfg.size, "net": {"channels": channels, "blocks": blocks, "dilations": list(network.NetConfig().dilations), "plan": False},
            "config": asdict(cfg), **extra}


def exploit_explorer(cfg: Config, d_params, d_dil, x_main, key, deadline: float = 0.0, log=print) -> tuple:
    """A fresh explorer trained only against a frozen designer. Returns (params, results) or
    (None, None) past the deadline. Gain = main explorer's walls hit − the exploiter's, on held-out
    mazes of that designer at the default budget (sampled policies; greedy too)."""
    ws = wall_band(cfg)
    k_init, k_pool, k_eval, key = jax.random.split(key, 4)
    xc = replace(cfg, magnet=0.0)
    dil = network.NetConfig().dilations
    init = x_main if cfg.exploit_init == "main" else network.init(k_init, network.NetConfig(cfg.channels, cfg.blocks, dil))
    learner = ExplorerLearner(xc, init)
    rng = np.random.default_rng(int(jax.random.randint(k_pool, (), 0, 2**31 - 1)))
    src = MazeSource("target", "target", designer.sample_mazes(d_params, cfg.size, rng.choice(ws, cfg.member_pool), k_pool, d_dil))
    t0 = time.time()
    for it in range(cfg.x_exploit_iters):
        if deadline and time.time() > deadline:
            return None, None
        src.replace(designer.sample_mazes(d_params, cfg.size, rng.choice(ws, cfg.member_refresh), jax.random.fold_in(k_pool, it + 1), d_dil))
        key, k = jax.random.split(key)
        learner.iterate(single_pool(src), k)
    w = default_wall_budget(cfg.size)
    held = designer.sample_mazes(d_params, cfg.size, np.full(cfg.exploit_mazes, w), k_eval, d_dil)
    res = {}
    for name, p in (("main", x_main), ("exploiter", learner.params)):
        for greedy in (False, True):
            res[f"{name}_{'greedy' if greedy else 'sampled'}"] = float(learned.explore_mazes(p, *held, jax.random.fold_in(k_eval, 1), dil, greedy=greedy).mean())
    res["gain"] = res["main_sampled"] - res["exploiter_sampled"]
    res["gain_greedy"] = res["main_greedy"] - res["exploiter_greedy"]
    res["iterations"] = cfg.x_exploit_iters
    res["minutes"] = (time.time() - t0) / 60
    log(f"  explorer exploiter: {res['exploiter_sampled']:.2f} walls vs main {res['main_sampled']:.2f} on the frozen designer's mazes (gain {res['gain']:+.2f}; greedy {res['gain_greedy']:+.2f}), {res['minutes']:.1f} min")
    return learner.params, res


def single_pool(src: MazeSource) -> PoolArrays:
    """A pool with one source (exploiters)."""
    counts = np.zeros(MAX_SLOTS, np.int32)
    counts[0] = len(src.entrance)
    logp = np.full(MAX_SLOTS, -np.inf, np.float32)
    logp[0] = 0.0
    return PoolArrays(jnp.asarray(src.walls), jnp.asarray(src.entrance, jnp.int32), jnp.asarray(src.exit, jnp.int32),
                      jnp.zeros(MAX_SLOTS, jnp.int32), jnp.asarray(counts), jnp.asarray(logp))


def exploit_designer(cfg: Config, x_params, x_dil, d_main, d_dil, key, deadline: float = 0.0, log=print) -> tuple:
    """A fresh designer trained only against a frozen explorer (its sampled policy). Gain = the frozen
    explorer's walls hit on the exploiter's mazes − on the main designer's, default budget."""
    ws = wall_band(cfg)
    k_init, k_eval, key = jax.random.split(key, 3)
    dc = replace(cfg, d_magnet=0.0)
    init = d_main if cfg.exploit_init == "main" else designer.init_net(k_init, network.NetConfig(cfg.d_channels, cfg.d_blocks, network.NetConfig().dilations))
    learner = DesignerLearner(dc, init)
    rng = np.random.default_rng(int(jax.random.randint(k_init, (), 0, 2**31 - 1)))
    t0 = time.time()
    for it in range(cfg.d_exploit_iters):
        if deadline and time.time() > deadline:
            return None, None
        key, k1, k2, k3 = jax.random.split(key, 4)
        mazes, batch = learner.generate(rng.choice(ws, cfg.d_mazes), k1)
        hits = learned.explore_mazes(x_params, *mazes, k2, x_dil, greedy=False)
        learner.learn(batch, hits.astype(np.float32), k3)
    w = default_wall_budget(cfg.size)
    res = {}
    for name, p in (("main", d_main), ("exploiter", learner.params)):
        mazes = designer.sample_mazes(p, cfg.size, np.full(cfg.exploit_mazes, w), jax.random.fold_in(k_eval, 0), d_dil)
        for greedy in (False, True):
            res[f"{name}_{'greedy' if greedy else 'sampled'}"] = float(learned.explore_mazes(x_params, *mazes, jax.random.fold_in(k_eval, 1), x_dil, greedy=greedy).mean())
        d = designer.diversity(*mazes)
        res[f"{name}_distinct"] = d["distinct"]
        res[f"{name}_path"] = d["path"]
    res["gain"] = res["exploiter_sampled"] - res["main_sampled"]
    res["gain_greedy"] = res["exploiter_greedy"] - res["main_greedy"]
    res["iterations"] = cfg.d_exploit_iters
    res["minutes"] = (time.time() - t0) / 60
    log(f"  designer exploiter: the frozen explorer hits {res['exploiter_sampled']:.2f} walls on its mazes vs {res['main_sampled']:.2f} on the main designer's (gain {res['gain']:+.2f}; greedy explorer {res['gain_greedy']:+.2f}), {res['minutes']:.1f} min")
    return learner.params, res


def designer_vs(cfg: Config, run, target: str, log=print) -> None:
    """A fresh designer against one frozen explorer (see the module docstring)."""
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    path = p2_path(cfg) if target == "p2" else target
    xp, xmeta, _ = learned.load(path)
    x_dil = tuple(xmeta["net"]["dilations"])
    ckpt = run / "designer.npz"
    dc = replace(cfg, d_magnet=0.0)
    dil = network.NetConfig().dilations
    if ckpt.exists():
        dp, meta, arrays = learned.load(ckpt)
        learner = DesignerLearner(dc, dp)
        learner.opt_state = learned.unflatten(learner.opt.init(dp), arrays, "opt")
        it, total, elapsed = meta["iteration"], meta["d_mazes"], meta["elapsed"]
        key = jax.random.wrap_key_data(jnp.asarray(meta["key"], dtype=jnp.uint32))
        log(f"resuming {ckpt} at iteration {it}")
    else:
        key = jax.random.key(cfg.seed)
        learner = DesignerLearner(dc, designer.init_net(jax.random.fold_in(key, 1), network.NetConfig(cfg.d_channels, cfg.d_blocks, dil)))
        it = total = 0
        elapsed = 0.0
    (run / "config.json").write_text(json.dumps({**asdict(cfg), "designer_vs": path}, indent=1))
    ws, w = wall_band(cfg), default_wall_budget(cfg.size)
    eval_key = jax.random.key(30_000 + cfg.seed)
    log(f"designer {cfg.d_channels}×{cfg.d_blocks} ({network.n_params(learner.params):,} parameters) against {path} (sampled), {cfg.d_mazes} mazes per iteration")

    def save():
        meta = _net_meta(cfg, "designer", cfg.d_channels, cfg.d_blocks, iteration=it, d_mazes=total, elapsed=elapsed,
                         key=np.asarray(jax.random.key_data(key)).tolist(), target=path)
        learned.save(ckpt, learner.params, meta, learner.opt_state)

    stop = {"now": False}
    old = signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    started = last_save = time.time()
    log_file = open(run / "log.jsonl", "a")
    try:
        while it < cfg.iterations and not stop["now"] and not (cfg.hours and time.time() - started > cfg.hours * 3600):
            t0 = time.perf_counter()
            key, k1, k2, k3, k4 = jax.random.split(key, 5)
            rng = np.random.default_rng(np.asarray(jax.random.key_data(k4)).ravel())
            mazes, batch = learner.generate(rng.choice(ws, cfg.d_mazes), k1)
            hits = learned.explore_mazes(xp, *mazes, k2, x_dil, greedy=False)
            stats = learner.learn(batch, hits.astype(np.float32), k3)
            stats["seq_entropy"] = seq_entropy(batch)
            it += 1
            total += cfg.d_mazes
            dt = time.perf_counter() - t0
            elapsed += dt
            rec = {"iteration": it, "d_mazes": total, "mazes_per_s": cfg.d_mazes / dt, "hits": float(hits.mean()), **stats}
            if it % cfg.eval_every == 0:
                ev = designer.sample_mazes(learner.params, cfg.size, np.full(cfg.d_eval_mazes, w), eval_key, dil)
                k = jax.random.fold_in(eval_key, 1)
                rec["eval"] = {
                    "target_sampled": float(learned.explore_mazes(xp, *ev, k, x_dil, greedy=False).mean()),
                    "target_greedy": float(learned.explore_mazes(xp, *ev, k, x_dil).mean()),
                    "hard": float(planner.explore_mazes(*ev, k, planner.HARD).mean()),
                    **designer.diversity(*ev),
                }
            log_file.write(json.dumps(rec) + "\n")
            log_file.flush()
            log(f"it {it:5d} {total / 1e6:6.2f}M mazes {rec['mazes_per_s']:5.0f}/s | hits {rec['hits']:.2f} | ent {stats['entropy']:.2f} seq {stats['seq_entropy']:.1f} kl {stats['kl']:.3f} ev {stats['explained_var']:.2f}")
            if "eval" in rec:
                e = rec["eval"]
                log(f"  eval ({cfg.d_eval_mazes} mazes, {w} walls): target {e['target_greedy']:.2f} greedy / {e['target_sampled']:.2f} sampled, hard {e['hard']:.2f} | {designer.diversity_line(e)}")
            if cfg.save_minutes and time.time() - last_save > cfg.save_minutes * 60:
                save()
                last_save = time.time()
    except KeyboardInterrupt:
        log("interrupted")
    finally:
        signal.signal(signal.SIGTERM, old)
        log_file.close()
    save()
    log(f"saved {ckpt} (iteration {it}, {total / 1e6:.2f} M mazes, {elapsed / 60:.1f} min)")


def gen_dir(run: Path, g: int) -> Path:
    return Path(run) / "league" / f"gen-{g:04d}"


def exploit_generation(cfg: Config, run: Path, g: int, deadline: float = 0.0, log=print) -> dict | None:
    """Train both exploiters of generation g (frozen agents in league/gen-g/) and save them there."""
    d = gen_dir(run, g)
    xp, xmeta, _ = learned.load(d / "explorer.npz")
    dp, dmeta, _ = learned.load(d / "designer.npz")
    x_dil, d_dil = tuple(xmeta["net"]["dilations"]), tuple(dmeta["net"]["dilations"])
    log(f"exploiting generation {g} ({cfg.exploit_init} init, {cfg.x_exploit_iters} + {cfg.d_exploit_iters} iterations)")
    key = jax.random.key(cfg.seed * 7919 + g)
    xe, rx = exploit_explorer(cfg, dp, d_dil, xp, jax.random.fold_in(key, 0), deadline, log)
    if xe is None:
        return None
    de, rd = exploit_designer(cfg, xp, x_dil, dp, d_dil, jax.random.fold_in(key, 1), deadline, log)
    if de is None:
        return None
    learned.save(d / "exploit-explorer.npz", xe, _net_meta(cfg, "explorer", cfg.channels, cfg.blocks, generation=g, exploiter=True))
    learned.save(d / "exploit-designer.npz", de, _net_meta(cfg, "designer", cfg.d_channels, cfg.d_blocks, generation=g, exploiter=True))
    result = {"generation": g, "explorer": rx, "designer": rd, "exploitability": rx["gain"] + rd["gain"], "time": time.time()}
    tmp = d / "exploit.json.tmp"
    tmp.write_text(json.dumps(result, indent=1))
    os.replace(tmp, d / "exploit.json")
    return result


def worker(cfg: Config, run, log=print) -> None:
    """Exploiters in their own process: the newest generation that has none yet, again and again,
    until the main process writes league/STOP or the time budget runs out. (A STOP older than this
    worker is a previous session's.)"""
    run = Path(run)
    started = time.time()
    deadline = started + cfg.hours * 3600 if cfg.hours else 0.0
    stop = run / "league" / "STOP"
    log(f"exploiter worker on {jax.devices()[0].device_kind}, run {run}")
    while not (stop.exists() and stop.stat().st_mtime > started) and not (deadline and time.time() > deadline):
        ready = sorted(int(p.parent.name[4:]) for p in (run / "league").glob("gen-*/ready"))
        todo = [g for g in ready if not (gen_dir(run, g) / "exploit.json").exists()]
        if not todo:
            time.sleep(20)
            continue
        result = exploit_generation(cfg, run, todo[-1], deadline, log)
        if result:
            log(f"generation {todo[-1]} exploited: exploitability {result['exploitability']:+.2f} walls")
    log("exploiter worker done")


# ------------------------------------------------------------------ the main loop


class Trainer:
    def __init__(self, cfg: Config, run, resume: bool = False, log=print):
        self.cfg, self.run, self.log = cfg, Path(run), log
        self.dil = network.NetConfig().dilations
        self.ws = wall_band(cfg)
        run = self.run
        if (run / "league.json").exists() and not resume:
            raise FileExistsError(f"{run / 'league.json'} exists: pass --resume, or choose another --run")
        run.mkdir(parents=True, exist_ok=True)
        (run / "league").mkdir(exist_ok=True)
        (run / "league" / "STOP").unlink(missing_ok=True)
        self.league = League(cfg)
        state = json.loads((run / "league.json").read_text()) if resume and (run / "league.json").exists() else None
        if state:
            self._load(state)
        else:
            self._fresh()
        (run / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
        self._populate(state)
        self.evals = EvalSets(cfg, run)
        self.d_eval_key = jax.random.key(30_000 + cfg.seed)

    # --- set-up

    def _fresh(self):
        cfg = self.cfg
        key = jax.random.key(cfg.seed)
        key, kx, kd = jax.random.split(key, 3)
        if cfg.explorer_init:
            xp, src, _ = learned.load(cfg.explorer_init)
            if (src["net"]["channels"], src["net"]["blocks"]) != (cfg.channels, cfg.blocks):
                raise ValueError(f"{cfg.explorer_init} is {src['net']['channels']}×{src['net']['blocks']}, expected {cfg.channels}×{cfg.blocks}")
            self.log(f"explorer from {cfg.explorer_init} ({src.get('env_steps', 0) / 1e6:.1f} M probes)")
        else:
            xp = network.init(kx, network.NetConfig(cfg.channels, cfg.blocks, self.dil))
        dp = learned.load(cfg.designer_init)[0] if cfg.designer_init else designer.init_net(kd, network.NetConfig(cfg.d_channels, cfg.d_blocks, self.dil))
        self.x = ExplorerLearner(cfg, xp, magnet=xp)
        self.d = DesignerLearner(cfg, dp, magnet=dp)
        self.key = key
        self.it = self.env_steps = self.d_total = self.gens = 0
        self.elapsed = 0.0
        self.absorbed = []
        self.exploits = []
        self.last_eval = None

    def _load(self, state):
        cfg, run = self.cfg, self.run
        xp, _, xa = learned.load(run / "explorer.npz")
        dp, _, da = learned.load(run / "designer.npz")
        xm = learned.load(run / "explorer-magnet.npz")[0]
        dm = learned.load(run / "designer-magnet.npz")[0]
        self.x = ExplorerLearner(cfg, xp, magnet=xm)
        self.x.opt_state = learned.unflatten(self.x.opt.init(xp), xa, "opt")
        self.d = DesignerLearner(cfg, dp, magnet=dm)
        self.d.opt_state = learned.unflatten(self.d.opt.init(dp), da, "opt")
        self.key = jax.random.wrap_key_data(jnp.asarray(state["key"], dtype=jnp.uint32))
        self.it, self.env_steps, self.d_total, self.gens = state["iteration"], state["env_steps"], state["d_mazes"], state["generations"]
        self.elapsed, self.absorbed, self.exploits, self.last_eval = state["elapsed"], state["absorbed"], state["exploits"], state.get("eval")
        self.log(f"resuming {run} at iteration {self.it} ({self.env_steps / 1e6:.2f} M probes, {self.gens} generations)")

    def _populate(self, state):
        """Members: anchors, self, frozen snapshots and exploiters (from league.json on resume)."""
        cfg, lg = self.cfg, self.league
        for src in anchor_sources(cfg, self.ws):
            lg.add_source(src)
        rng = np.random.default_rng([cfg.seed, self.it, 1])
        lg.add_source(MazeSource("self", "self", designer.sample_mazes(self.d.params, cfg.size, rng.choice(self.ws, cfg.self_pool), jax.random.key(self.it), self.dil)))
        lg.add_opponent(Opponent("self", "self", lambda w, e, s, key: learned.explore_mazes(self.x.params, w, e, s, key, self.dil, greedy=False)))
        for name in [n.strip() for n in cfg.opponents.split(",") if n.strip()]:
            if name == "hard":
                lg.add_opponent(hard_opponent())
            elif name == "p2":
                path = p2_path(cfg)
                p2 = learned.Explorer(path)
                lg.add_opponent(network_opponent("p2", "anchor", p2.params, p2.dilations, greedy=True, path=path))
            else:
                raise ValueError(f"unknown fixed opponent {name!r} (p2, hard)")
        if state:
            saved = {s["name"]: s for s in state["league"]["sources"]}
            for s in state["league"]["sources"]:
                if s["kind"] in ("snapshot", "exploiter") and s["active"]:
                    self._add_designer_member(s["name"], s["kind"], s["path"])
            for s in lg.sources:
                if s.name in saved:
                    s.x = saved[s.name]["x"]
            for o in state["league"]["opponents"]:
                if o["kind"] in ("snapshot", "exploiter") and o["active"]:
                    self._add_explorer_member(o["name"], o["kind"], o["path"])
            xs = {o["name"]: o["x"] for o in state["league"]["opponents"]}
            for o in lg.opponents:
                o.x = xs.get(o.name, o.x)
            lg.rr = state["league"]["rr"]

    def _add_designer_member(self, name: str, kind: str, path: str) -> None:
        cfg = self.cfg
        params, meta, _ = learned.load(self.run / path)
        dil = tuple(meta["net"]["dilations"])
        rng = np.random.default_rng([cfg.seed, len(self.league.sources)])
        mazes = designer.sample_mazes(params, cfg.size, rng.choice(self.ws, cfg.member_pool), jax.random.key(len(self.league.sources)), dil)
        self.league.add_source(MazeSource(name, kind, mazes, params, dil, path=path))

    def _add_explorer_member(self, name: str, kind: str, path: str) -> None:
        params, meta, _ = learned.load(self.run / path)
        self.league.add_opponent(network_opponent(name, kind, params, tuple(meta["net"]["dilations"]), greedy=False, path=path))

    # --- checkpoints

    def _meta(self, kind: str) -> dict:
        c = self.cfg
        shape = (c.channels, c.blocks) if kind == "explorer" else (c.d_channels, c.d_blocks)
        return _net_meta(c, kind, *shape, iteration=self.it, env_steps=self.env_steps, d_mazes=self.d_total, elapsed=self.elapsed, eval=self.last_eval)

    def save(self):
        run = self.run
        learned.save(run / "explorer.npz", self.x.params, self._meta("explorer"), self.x.opt_state)
        learned.save(run / "explorer-magnet.npz", self.x.magnet if self.x.magnet is not None else self.x.params, self._meta("explorer"))
        learned.save(run / "designer.npz", self.d.params, self._meta("designer"), self.d.opt_state)
        learned.save(run / "designer-magnet.npz", self.d.magnet if self.d.magnet is not None else self.d.params, self._meta("designer"))
        state = {
            "iteration": self.it, "env_steps": self.env_steps, "d_mazes": self.d_total, "generations": self.gens,
            "elapsed": self.elapsed, "key": np.asarray(jax.random.key_data(self.key)).tolist(),
            "absorbed": self.absorbed, "exploits": self.exploits, "eval": self.last_eval, "league": self.league.state(),
        }
        tmp = run / "league.json.tmp"
        tmp.write_text(json.dumps(state, indent=1))
        os.replace(tmp, run / "league.json")

    # --- generations and exploiters

    def new_generation(self):
        cfg = self.cfg
        self.gens += 1
        g = self.gens
        d = gen_dir(self.run, g)
        d.mkdir(parents=True, exist_ok=True)
        learned.save(d / "explorer.npz", self.x.params, {**self._meta("explorer"), "generation": g})
        learned.save(d / "designer.npz", self.d.params, {**self._meta("designer"), "generation": g})
        (d / "ready").write_text(str(self.it))
        rel = lambda name: str((d / name).relative_to(self.run))
        self._add_designer_member(f"d:g{g}", "snapshot", rel("designer.npz"))
        self._add_explorer_member(f"x:g{g}", "snapshot", rel("explorer.npz"))
        self.league.prune("snapshot")
        self.log(f"generation {g} frozen at iteration {self.it}")
        if cfg.exploiters == "inline":
            exploit_generation(cfg, self.run, g, log=self.log)
        self.absorb()

    def absorb(self):
        """Add the exploiters that finished (in this process or the worker's) to the league."""
        for p in sorted((self.run / "league").glob("gen-*/exploit.json")):
            g = int(p.parent.name[4:])
            if g in self.absorbed:
                continue
            result = json.loads(p.read_text())
            rel = lambda name: str((p.parent / name).relative_to(self.run))
            self._add_designer_member(f"d:xg{g}", "exploiter", rel("exploit-designer.npz"))
            self._add_explorer_member(f"x:xg{g}", "exploiter", rel("exploit-explorer.npz"))
            self.league.prune("exploiter")
            self.absorbed.append(g)
            self.exploits.append({"generation": g, "iteration": self.it, "explorer_gain": result["explorer"]["gain"],
                                  "designer_gain": result["designer"]["gain"], "exploitability": result["exploitability"]})
            self.log(f"exploiters of generation {g} joined: explorer gain {result['explorer']['gain']:+.2f}, designer gain {result['designer']['gain']:+.2f} walls")

    # --- evaluation

    def evaluate(self) -> dict:
        cfg = self.cfg
        out = {"iteration": self.it, "explorer": self.evals.evaluate(self.x.params, self.dil)}
        w = default_wall_budget(cfg.size)
        mazes = designer.sample_mazes(self.d.params, cfg.size, np.full(cfg.d_eval_mazes, w), self.d_eval_key, self.dil)
        k = jax.random.fold_in(self.d_eval_key, 1)
        dz = {
            "main_greedy": float(learned.explore_mazes(self.x.params, *mazes, k, self.dil).mean()),
            "main_sampled": float(learned.explore_mazes(self.x.params, *mazes, k, self.dil, greedy=False).mean()),
            "hard": float(planner.explore_mazes(*mazes, k, planner.HARD).mean()),
        }
        p2 = next((o for o in self.league.opponents if o.name == "p2"), None)
        if p2 is not None:
            dz["p2"] = float(np.asarray(p2.explore(*mazes, k)).mean())
        dz.update(designer.diversity(*mazes))
        out["designer"] = dz
        return out

    def eval_line(self, ev: dict) -> str:
        d = ev["designer"]
        line = f"  designer mazes ({self.cfg.d_eval_mazes}, default budget): main explorer {d['main_greedy']:.2f} greedy / {d['main_sampled']:.2f} sampled"
        if "p2" in d:
            line += f", P2 {d['p2']:.2f}"
        line += f", hard {d['hard']:.2f} | {designer.diversity_line(d)}"
        return "  explorer (greedy): " + self.evals.line(ev["explorer"]) + "\n" + line

    # --- the loop

    def iteration(self) -> dict:
        cfg, lg = self.cfg, self.league
        t0 = time.perf_counter()
        self.key, kd, ko, ku, kx = jax.random.split(self.key, 5)
        rng = np.random.default_rng(np.asarray(jax.random.key_data(ko)).ravel())
        # 1. designer
        mazes, dbatch = self.d.generate(rng.choice(self.ws, cfg.d_mazes), kd)
        hits, who, opp_hits = lg.explore(mazes, ko, rng)
        dstats = self.d.learn(dbatch, opponent_baseline(hits, who), ku)
        dstats["seq_entropy"] = seq_entropy(dbatch)
        lg.source("self").replace(mazes)
        t1 = time.perf_counter()
        # 2. explorer
        adv = lg.refresh(self.it, self.x.params, self.dil)
        xstats, batch = self.x.iterate(lg.device(), kx)
        x_hits = lg.observe_games(batch)
        t2 = time.perf_counter()
        self.it += 1
        steps = cfg.envs * cfg.horizon
        self.env_steps += steps
        self.d_total += cfg.d_mazes
        self.elapsed += t2 - t0
        return {
            "iteration": self.it, "env_steps": self.env_steps, "d_mazes": self.d_total,
            "probes_per_s": steps / (t2 - t1), "mazes_per_s": cfg.d_mazes / (t1 - t0), "seconds": t2 - t0,
            "x_hits": x_hits, "x": xstats, "d_hits": float(hits.mean()), "d_opp_hits": opp_hits, "d": dstats, **adv,
            "x_weights": {s.name: round(float(w), 4) for s, w in zip(lg.sources, lg.source_weights()) if w > 0},
            "d_weights": {o.name: round(float(w), 4) for o, w in zip(lg.opponents, lg.opponent_weights()) if w > 0},
        }

    def train(self) -> None:
        cfg, run = self.cfg, self.run
        stop = {"now": False}
        old = signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
        started = last_save = time.time()
        log_file = open(run / "log.jsonl", "a")
        self.log(f"{cfg.size}×{cfg.size}, walls {self.ws}; explorer {cfg.channels}×{cfg.blocks} ({network.n_params(self.x.params):,} parameters), "
                 f"designer {cfg.d_channels}×{cfg.d_blocks} ({network.n_params(self.d.params):,}); {cfg.envs} games × {cfg.horizon} probes "
                 f"and {cfg.d_mazes} mazes per iteration; exploiters {cfg.exploiters}")
        try:
            while self.it < cfg.iterations and not stop["now"]:
                if cfg.hours and time.time() - started > cfg.hours * 3600:
                    self.log(f"time budget of {cfg.hours} h reached")
                    break
                rec = self.iteration()
                if self.it % cfg.eval_every == 0 or self.it == cfg.iterations:
                    self.last_eval = rec["eval"] = self.evaluate()
                log_file.write(json.dumps(rec) + "\n")
                log_file.flush()
                x, d = rec["x"], rec["d"]
                self.log(f"it {self.it:5d} {self.env_steps / 1e6:7.2f}M {rec['probes_per_s']:6.0f} probes/s {rec['mazes_per_s']:5.0f} mazes/s | X "
                         + " ".join(f"{k} {v:.1f}" for k, v in rec["x_hits"].items())
                         + f" | ent {x['entropy']:.2f} kl {x['kl']:.3f} mag {x.get('magnet_kl', 0):.3f} bel {x['belief']:.3f}"
                         + f" | D {rec['d_hits']:.2f} (" + " ".join(f"{k} {v:.1f}" for k, v in rec["d_opp_hits"].items())
                         + f") ent {d['entropy']:.2f} seq {d['seq_entropy']:.1f} kl {d['kl']:.3f} mag {d.get('magnet_kl', 0):.3f} ev {d['explained_var']:.2f}")
                if "eval" in rec:
                    self.log(self.eval_line(rec["eval"]))
                if cfg.exploiters == "worker" and self.it % 5 == 0:
                    self.absorb()
                if cfg.gen_every and self.it % cfg.gen_every == 0:
                    self.new_generation()
                    self.save()
                    last_save = time.time()
                elif cfg.save_minutes and time.time() - last_save > cfg.save_minutes * 60:
                    self.save()
                    last_save = time.time()
        except KeyboardInterrupt:
            self.log("interrupted")
        finally:
            signal.signal(signal.SIGTERM, old)
            log_file.close()
            (run / "league" / "STOP").write_text(str(self.it))
        self.save()
        self.log(f"saved {run} (iteration {self.it}, {self.env_steps / 1e6:.2f} M probes, {self.d_total / 1e6:.2f} M mazes, {self.elapsed / 60:.1f} min)")


def train(cfg: Config, run, resume: bool = False, log=print) -> Trainer:
    t = Trainer(cfg, run, resume, log)
    t.train()
    return t


def config_from_args(argv=None) -> tuple:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--worker", action="store_true", help="train exploiters for the run's new generations (another process)")
    ap.add_argument("--designer-vs", default=None, help="train a fresh designer against this explorer checkpoint (or p2)")
    for f in fields(Config):
        kind = type(f.default)
        if kind is bool:
            kind = lambda v: {"1": True, "true": True, "yes": True, "0": False, "false": False, "no": False}[v.lower()]
        ap.add_argument("--" + f.name.replace("_", "-"), type=kind, default=None)
    args = ap.parse_args(argv)
    given = {f.name: getattr(args, f.name) for f in fields(Config) if getattr(args, f.name) is not None}
    saved_path = Path(args.run) / "config.json"
    if (args.resume or args.worker) and saved_path.exists():
        saved = json.loads(saved_path.read_text())
        cfg = Config(**{**asdict(Config()), **{k: v for k, v in saved.items() if k in {f.name for f in fields(Config)}}, **given})
    else:
        cfg = Config(**given)
    return cfg, args


def main(argv=None):
    cfg, args = config_from_args(argv)
    if cfg.precision != "fp32":
        network.COMPUTE_DTYPE = {"fp16": jnp.float16, "bf16": jnp.bfloat16}[cfg.precision]
    if args.worker:
        worker(cfg, args.run)
    elif args.designer_vs:
        designer_vs(cfg, args.run, args.designer_vs)
    else:
        train(cfg, args.run, resume=args.resume)


if __name__ == "__main__":
    main()
