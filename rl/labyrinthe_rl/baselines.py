"""Today's bots (src/bot/brain.js), ported line by line to plain Python.

With a `SeededRng` they make exactly the same draws as the JS, so the fixtures can check them
probe for probe. They're the reference; planner.py is the batched JAX version used for evaluation.

- Explorer: the optimistic planner. Unknown sides are assumed open; the bot probes a side on a
  shortest such path to S (rankProbes), with noise by level (chooseProbe).
- Designers: easy = one random maze; normal / hard = best of 6 / 30 random mazes by walls hit by
  the hard explorer (2 / 4 runs each), fewer candidates on big boards (designMaze).
The 'expert' level isn't ported here.
"""

import math
from collections import deque
from dataclasses import dataclass, field

from .geometry import geometry
from .jsrandom import pick
from .mazes import Maze, random_maze

LEVELS = ("easy", "normal", "hard")
INF = float("inf")


@dataclass
class Progress:
    """What an explorer knows (match.js progressOf), cells in the order they were reached."""

    size: int
    visited: list
    open: set = field(default_factory=set)  # edges
    walls: set = field(default_factory=set)  # edges
    hits: int = 0
    moves: int = 0

    @classmethod
    def start(cls, size: int, entrance: int) -> "Progress":
        return cls(size, [entrance])

    def record(self, cell: int, d: int, is_open: bool) -> None:
        n, e = geometry(self.size).adj[cell][d]
        self.moves += 1
        if is_open:
            self.open.add(e)
            if n not in self.visited:
                self.visited.append(n)
        else:
            self.walls.add(e)
            self.hits += 1


def optimistic_distances(progress: Progress, exit: int) -> list:
    """0-1 BFS from the exit: known passages cost 0, unknown sides 1, known walls are impassable."""
    g = geometry(progress.size)
    dist = [INF] * g.n_cells
    dist[exit] = 0
    dq = deque([exit])
    while dq:
        c = dq.popleft()
        for n, e in g.adj[c]:
            if n < 0 or e in progress.walls:
                continue
            w = 0 if e in progress.open else 1
            if dist[c] + w < dist[n]:
                dist[n] = dist[c] + w
                if w == 0:
                    dq.appendleft(n)
                else:
                    dq.append(n)
    return dist


def rank_probes(progress: Progress, exit: int) -> list:
    """Every legal probe as (cell, dir, score), best (lowest) first, in the JS order."""
    g = geometry(progress.size)
    dist = optimistic_distances(progress, exit)
    visited = set(progress.visited)
    out = []
    for cell in progress.visited:
        for d, (to, e) in enumerate(g.adj[cell]):
            if e < 0 or e in progress.open or e in progress.walls:
                continue
            # Prefer probes that reach new cells; probing between two known cells only closes a loop.
            out.append((cell, d, 1 + dist[to] + (0.5 if to in visited else 0)))
    out.sort(key=lambda p: p[2])  # stable, like Array.prototype.sort
    return out


def choose_probe(progress: Progress, exit: int, level: str, rng):
    """brain.js chooseProbe for easy / normal / hard: (cell, dir) or None."""
    ranked = rank_probes(progress, exit)
    if not ranked:
        return None
    best = [p for p in ranked if p[2] == ranked[0][2]]
    roll = rng()
    if level == "easy" and roll < 0.8:
        p = pick(rng, ranked)
    elif level == "normal" and roll < 0.25:
        p = pick(rng, ranked[:4])
    else:
        p = pick(rng, best)
    return p[0], p[1]


@dataclass
class Solve:
    hits: float  # inf if the explorer got stuck (never happens with these rules)
    moves: int
    probes: list  # (cell, dir, open)


def simulate_solve(maze: Maze, level: str, rng) -> Solve:
    """brain.js simulateSolve: one whole exploration of `maze`."""
    g = geometry(maze.size)
    progress = Progress.start(maze.size, maze.entrance)
    walls = maze.walls.tolist()
    probes = []
    for _ in range(4 * g.n_cells):
        if maze.exit in progress.visited:
            return Solve(progress.hits, progress.moves, probes)
        p = choose_probe(progress, maze.exit, level, rng)
        if p is None:
            break
        is_open = not walls[g.adj[p[0]][p[1]][1]]
        progress.record(p[0], p[1], is_open)
        probes.append((p[0], p[1], is_open))
    return Solve(INF, len(probes), probes)


def design_candidates(size: int, level: str) -> int:
    """How many random mazes a designer tries (scaled down on big boards, as in brain.js)."""
    scale = min(1, 25 / size**2)
    return max(1, math.floor(0.5 + {"easy": 1, "normal": 6, "hard": 30}.get(level, 6) * scale))


def design_runs(level: str) -> int:
    return 4 if level == "hard" else 2


def design_maze(size: int, walls: int, level: str, rng) -> Maze:
    """brain.js designMaze: the candidate the hard explorer hits the most walls in."""
    best, best_score = None, -1
    for _ in range(design_candidates(size, level)):
        maze = random_maze(size, walls, rng)
        total = sum(simulate_solve(maze, "hard", rng).hits for _ in range(design_runs(level)))
        if total > best_score:
            best, best_score = maze, total
    return best

