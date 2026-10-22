"""Duplicate-format evaluation of bots (docs/RL_DESIGN.md §3.6, DECISIONS.md D15).

    python -m labyrinthe_rl.bench [--size 5] [--walls W] [--deals 500] [--seed 0]
                                  [--pairs hard:easy,hard:normal,normal:easy,hard:hard]
                                  [--mazes bots|random|uniform] [--particles 200]

Each deal is a pair of mazes (a, b): designed by bots A and B (`--mazes bots`, as in a real round),
or dealt by a sampler. Both bots explore both mazes, i.e. play both seats:
    seat 1: A explores b while B explores a   (the round as played, when the bots designed them)
    seat 2: A explores a while B explores b
A player finishes on turn (walls hit + 1) and the starter wins ties, so with each bot starting once
a seat is worth 1 to whoever hit fewer walls and ½ each when equal: a score that doesn't depend on
who starts. Explorations of the same maze share their random key, and each bot's mazes are made
once and reused in every pairing.

Bots: easy, normal, hard (src/bot/brain.js, batched in planner.py) and expert (src/bot/expert.js run
by Node, see expert.py; it designs like hard, as in the game).

Columns, per pairing A:B:
    hits A, hits B    walls hit as explorer (mean over both seats) and as a share of the budget W
    explorer edge     B's hits − A's hits on the same mazes (> 0: A explores better), mean ± s.e.
    designer edge     hits on a − hits on b, same two explorers (> 0: A's mazes are harder)
    score (dup)       A's starter-independent score over both seats: explorers on identical mazes
    score (round)     the same on seat 1 alone: the round as played
"""

import argparse
import time
from dataclasses import dataclass
from typing import Callable

import jax
import numpy as np

from . import expert, planner
from .geometry import default_wall_budget, max_wall_budget
from .mazes import sample_mazes


@dataclass(frozen=True)
class Bot:
    name: str
    # (size, walls, count, numpy rng, jax key) -> stacked mazes (walls[count, E], entrance, exit)
    design: Callable
    # (walls, entrance, exit, jax key) -> walls hit on each maze, int[count]
    explore: Callable


def heuristic(level: str) -> Bot:
    """Today's bot at `level` (src/bot/brain.js): designer and explorer of that level."""
    return Bot(
        name=level,
        design=lambda size, walls, count, rng, key: planner.design(size, walls, level, count, rng, key),
        explore=lambda w, e, s, key: planner.explore_mazes(w, e, s, key, planner.LEVEL_INDEX[level]),
    )


def expert_bot(particles: int = expert.PARTICLES) -> Bot:
    """The expert level (src/bot/expert.js, D21): hard's designer, the Bayesian explorer through Node."""
    return Bot(
        name="expert",
        design=BOTS["hard"].design,
        explore=lambda w, e, s, key: expert.explore_mazes(w, e, s, key, particles),
    )


BOTS = {level: heuristic(level) for level in planner.LEVEL_INDEX}
BOTS["expert"] = expert_bot()


def seat_score(mine, theirs) -> np.ndarray:
    """Per seat: 1 if I hit fewer walls, ½ if equal (each starter wins once), 0 otherwise."""
    return (mine < theirs) + 0.5 * (mine == theirs)


def mean_se(x) -> tuple:
    x = np.asarray(x, dtype=float)
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


class Deals:
    """The maze sets of one evaluation, made once and shared by every pairing: per bot and seat
    letter when the bots design them (bot X's "A" mazes are the same in every pairing where X is
    A), or two sets from a sampler. Each set carries the random key its explorations use."""

    def __init__(self, size: int, walls: int, count: int, mazes: str = "bots", seed: int = 0):
        self.size, self.walls, self.count, self.mazes, self.seed = size, walls, count, mazes, seed
        self._sets = {}

    def get(self, bot: Bot, letter: str) -> tuple:
        name = (bot.name if self.mazes == "bots" else self.mazes, letter)
        if name not in self._sets:
            i = len(self._sets)
            key = jax.random.fold_in(jax.random.key(self.seed), i)
            if self.mazes == "bots":
                rng = np.random.default_rng([self.seed, i])
                maze = bot.design(self.size, self.walls, self.count, rng, jax.random.fold_in(key, 1))
            else:
                maze = sample_mazes(self.mazes, self.size, self.walls, self.count, self.seed * 1000 + i)
            self._sets[name] = (maze, key)
        return self._sets[name]


def duplicate_match(a: Bot, b: Bot, deals: Deals) -> dict:
    """Both bots explore both mazes of every deal; returns per-deal hits and the summary statistics."""
    maze_a, key_a = deals.get(a, "A")
    maze_b, key_b = deals.get(b, "B")
    h = {
        "A on a": a.explore(*maze_a, key_a),
        "A on b": a.explore(*maze_b, key_b),
        "B on a": b.explore(*maze_a, key_a),
        "B on b": b.explore(*maze_b, key_b),
    }
    hits_a = np.concatenate([h["A on b"], h["A on a"]])
    hits_b = np.concatenate([h["B on a"], h["B on b"]])
    seat1 = seat_score(h["A on b"], h["B on a"])
    seat2 = seat_score(h["A on a"], h["B on b"])
    return {
        "hits": h,
        "hits A": mean_se(hits_a),
        "hits B": mean_se(hits_b),
        "explorer edge": mean_se((h["B on a"] + h["B on b"] - h["A on a"] - h["A on b"]) / 2),
        "designer edge": mean_se((h["A on a"] + h["B on a"] - h["A on b"] - h["B on b"]) / 2),
        "score (dup)": mean_se((seat1 + seat2) / 2),
        "score (round)": mean_se(seat1),
    }


def _pm(x) -> str:
    return f"{x[0]:+.2f} ± {x[1]:.2f}"


def _pct(x) -> str:
    return f"{100 * x[0]:.0f}% ± {100 * x[1]:.0f}"


def _hits(x, walls) -> str:
    return f"{x[0]:.2f} ({x[0] / walls:.2f}W)"


def report(results: dict, walls: int, mazes: str) -> str:
    cols = ["hits A", "hits B", "explorer edge", "designer edge", "score (dup)", "score (round)"]
    widths = [14, 14, 15, 15, 13, 14]
    lines = [f"{'pairing':<16}" + "".join(f" {c:>{w}}" for c, w in zip(cols, widths))]
    for (na, nb), r in results.items():
        cells = [
            _hits(r["hits A"], walls),
            _hits(r["hits B"], walls),
            _pm(r["explorer edge"]),
            _pm(r["designer edge"]) if mazes == "bots" else "—",
            _pct(r["score (dup)"]),
            _pct(r["score (round)"]),
        ]
        lines.append(f"{na + ' vs ' + nb:<16}" + "".join(f" {c:>{w}}" for c, w in zip(cells, widths)))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=5)
    ap.add_argument("--walls", type=int, default=None, help="wall budget (default: the game's default)")
    ap.add_argument("--deals", type=int, default=500)
    ap.add_argument("--pairs", default="hard:easy,hard:normal,normal:easy,hard:hard")
    ap.add_argument("--mazes", default="bots", choices=["bots", "random", "uniform"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--particles", type=int, default=expert.PARTICLES, help="particles of the expert explorer")
    args = ap.parse_args(argv)
    bots = {**BOTS, "expert": expert_bot(args.particles)}
    pairs = [tuple(pair.split(":")) for pair in args.pairs.split(",")]
    if any(len(p) != 2 or not set(p) <= bots.keys() for p in pairs):
        ap.error(f"--pairs is a list of A:B, with bots among {', '.join(bots)}")
    walls = args.walls or default_wall_budget(args.size)
    if not 1 <= walls <= max_wall_budget(args.size):
        ap.error(f"walls must be 1..{max_wall_budget(args.size)} on a {args.size}×{args.size} board")
    source = {"bots": "designed by the bots", "random": "from randomMaze", "uniform": "uniform among valid mazes"}[args.mazes]
    print(f"{args.size}×{args.size}, {walls} walls, {args.deals} deals (×2 seats), mazes {source}\n")
    deals = Deals(args.size, walls, args.deals, args.mazes, args.seed)
    results = {}
    t0 = time.perf_counter()
    for na, nb in pairs:
        results[(na, nb)] = duplicate_match(bots[na], bots[nb], deals)
    print(report(results, walls, args.mazes))
    print(f"\n({time.perf_counter() - t0:.1f} s)")


if __name__ == "__main__":
    main()
