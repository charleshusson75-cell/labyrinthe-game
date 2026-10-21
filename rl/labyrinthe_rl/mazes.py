"""Mazes on the host side (numpy): validity, BFS, and the two samplers.

- `random_maze` reproduces src/game/maze.js randomMaze draw for draw (given a `SeededRng`): a perfect
  maze, E/S far apart, then walls knocked down or added to hit the budget without opening E→S
  shortcuts. It's what the game's "random" button and every JS bot design start from.
- `uniform_maze` / `uniform_batch` sample uniformly among *all* valid mazes: random walls, random
  distinct E/S, rejected until S is reachable. Pockets, loops and E/S next to each other all happen.

A maze is `Maze(size, walls, entrance, exit)` with `walls` a bool mask over edges (geometry.py order).
"""

import math
from dataclasses import dataclass

import numpy as np

from .geometry import Geometry, default_wall_budget, geometry, max_wall_budget
from .jsrandom import pick, rand_int, shuffle


@dataclass(frozen=True, eq=False)
class Maze:
    size: int
    walls: np.ndarray  # bool[E]
    entrance: int
    exit: int

    @property
    def wall_count(self) -> int:
        return int(self.walls.sum())

    def to_js(self) -> dict:
        """The JS maze model, canonical form (sorted keys)."""
        return {"walls": geometry(self.size).keys_of(self.walls), "entrance": self.entrance, "exit": self.exit}

    @classmethod
    def from_js(cls, size: int, maze: dict) -> "Maze":
        walls = geometry(size).walls_from_keys(maze["walls"])
        return cls(size, walls, maze["entrance"], maze["exit"])

    def same_as(self, other: "Maze") -> bool:
        return (
            self.size == other.size
            and self.entrance == other.entrance
            and self.exit == other.exit
            and np.array_equal(self.walls, other.walls)
        )


# ------------------------------------------------------------------ BFS (maze.js order: U, R, D, L)


# Plain Python lists inside the loops: indexing numpy scalars one at a time is several times slower.


def _plain(walls) -> list:
    return walls.tolist() if isinstance(walls, np.ndarray) else walls


def distances_from(g: Geometry, walls, start: int) -> list:
    """BFS distances from `start` through open sides (−1 = unreachable)."""
    walls = _plain(walls)
    dist = [-1] * g.n_cells
    dist[start] = 0
    queue = [start]
    for c in queue:
        for n, e in g.adj[c]:
            if n < 0 or dist[n] >= 0 or walls[e]:
                continue
            dist[n] = dist[c] + 1
            queue.append(n)
    return dist


def shortest_path(g: Geometry, walls, start: int, goal: int):
    """Cells of a shortest path start → goal (inclusive), as maze.js picks it, or None."""
    walls = _plain(walls)
    prev = [-1] * g.n_cells
    seen = [False] * g.n_cells
    seen[start] = True
    queue = [start]
    for c in queue:
        if c == goal:
            break
        for n, e in g.adj[c]:
            if n < 0 or seen[n] or walls[e]:
                continue
            seen[n] = True
            prev[n] = c
            queue.append(n)
    if not seen[goal]:
        return None
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    return path[::-1]


def _path_length(g: Geometry, walls: list, start: int, goal: int) -> int:
    """Cells on a shortest path start → goal (len(shortest_path)), assuming there is one."""
    dist = [-1] * g.n_cells
    dist[start] = 0
    queue = [start]
    for c in queue:
        if c == goal:
            return dist[c] + 1
        for n, e in g.adj[c]:
            if n >= 0 and dist[n] < 0 and not walls[e]:
                dist[n] = dist[c] + 1
                queue.append(n)
    raise ValueError("no path")


def reachable_batch(g: Geometry, walls: np.ndarray, start: np.ndarray) -> np.ndarray:
    """bool[B, N²]: cells reachable from start[b] through the open sides of walls[b] (bool[B, E])."""
    walls = np.asarray(walls, dtype=bool)
    b = walls.shape[0]
    border = g.cell_edges < 0
    side_open = ~walls[:, np.maximum(g.cell_edges, 0)] & ~border  # [B, N², 4]
    nbr = np.maximum(g.neighbors, 0)
    reached = np.zeros((b, g.n_cells), dtype=bool)
    reached[np.arange(b), start] = True
    while True:
        grown = reached | (side_open & reached[:, nbr]).any(axis=2)
        if np.array_equal(grown, reached):
            return reached
        reached = grown


def maze_issues(size: int, walls: np.ndarray, entrance, exit, budget: int) -> list:
    """Rule violations, with the codes of maze.js mazeIssues ('bad-walls' can't occur for a mask)."""
    g = geometry(size)
    issues = []
    count = int(np.sum(walls))
    if count < budget:
        issues.append("walls-left")
    if count > budget:
        issues.append("too-many-walls")
    has_e = isinstance(entrance, (int, np.integer)) and 0 <= entrance < g.n_cells
    has_s = isinstance(exit, (int, np.integer)) and 0 <= exit < g.n_cells
    if not has_e:
        issues.append("no-entrance")
    if not has_s:
        issues.append("no-exit")
    if has_e and has_s:
        if entrance == exit:
            issues.append("same-cell")
        elif shortest_path(g, np.asarray(walls, dtype=bool), int(entrance), int(exit)) is None:
            issues.append("no-path")
    return issues


# ------------------------------------------------------------------ randomMaze (maze.js port)


def normalize_walls(size: int, walls=None) -> int:
    """The wall budget as normalizeSettings clamps it (None/0 → the default)."""
    w = default_wall_budget(size) if not walls else int(np.floor(walls + 0.5))
    return min(max_wall_budget(size), max(1, w))


def _spanning_tree(g: Geometry, rng) -> list:
    """Recursive backtracker; open edges in the order they were carved (JS Set order)."""
    opened = []
    seen = [False] * g.n_cells
    start = rand_int(rng, g.n_cells)
    stack = [start]
    seen[start] = True
    while stack:
        c = stack[-1]
        nxt = [(n, e) for n, e in g.adj[c] if n >= 0 and not seen[n]]
        if not nxt:
            stack.pop()
            continue
        n, e = pick(rng, nxt)
        seen[n] = True
        opened.append(e)
        stack.append(n)
    return opened


def _try_random_maze(g: Geometry, budget: int, rng, far_ratio: float):
    opened = _spanning_tree(g, rng)
    walls = [True] * g.n_edges
    for e in opened:
        walls[e] = False
    entrance = rand_int(rng, g.n_cells)
    dist = distances_from(g, walls, entrance)
    threshold = math.ceil(max(dist) * far_ratio)
    far = [cell for cell, d in enumerate(dist) if d > 0 and d >= threshold]
    exit = pick(rng, far)

    count = g.n_edges - len(opened)
    if count > budget:
        # Knock walls down (creates loops), skipping those that shorten E→S unless nothing else is left.
        target = _path_length(g, walls, entrance, exit)
        shortcuts = []
        for e in shuffle(rng, [e for e, w in enumerate(walls) if w]):
            if count <= budget:
                break
            walls[e] = False
            count -= 1
            if _path_length(g, walls, entrance, exit) < target:
                walls[e] = True
                count += 1
                shortcuts.append(e)
        for e in shortcuts:
            if count <= budget:
                break
            walls[e] = False
            count -= 1
    elif count < budget:
        # Close passages off the E→S path (the path is unique in a perfect maze).
        path = shortest_path(g, walls, entrance, exit)
        on_path = {g.edge_between(a, b) for a, b in zip(path, path[1:])}
        for e in shuffle(rng, [e for e in opened if e not in on_path]):
            if count >= budget:
                break
            walls[e] = True
            count += 1
        if count < budget:
            return None
    return Maze(g.size, np.array(walls, dtype=bool), entrance, exit)


def random_maze(size: int, walls=None, rng=None) -> Maze:
    """maze.js randomMaze: identical output to the JS for the same `SeededRng` seed."""
    g = geometry(size)
    budget = normalize_walls(size, walls)
    for attempt in range(60):
        maze = _try_random_maze(g, budget, rng, 0.7 if attempt < 30 else 0.3)
        if maze is not None:
            return maze
    raise RuntimeError(f"Could not generate a {size}×{size} maze with {budget} walls")


# ------------------------------------------------------------------ uniform sampler


def uniform_batch(size: int, walls: int, count: int, rng: np.random.Generator, max_rounds: int = 1000):
    """`count` mazes uniform among all valid ones: (walls bool[count, E], entrance int[count], exit int[count])."""
    g = geometry(size)
    if not 1 <= walls <= g.n_edges:
        raise ValueError(f"{walls} walls don't fit a {size}×{size} board")
    got_w, got_e, got_s = [], [], []
    have = 0
    for _ in range(max_rounds):
        if have >= count:
            break
        k = max(64, 2 * (count - have))
        order = np.argsort(rng.random((k, g.n_edges)), axis=1)
        w = np.zeros((k, g.n_edges), dtype=bool)
        np.put_along_axis(w, order[:, :walls], True, axis=1)
        ent = rng.integers(0, g.n_cells, size=k)
        ext = (ent + rng.integers(1, g.n_cells, size=k)) % g.n_cells  # uniform among the other cells
        ok = reachable_batch(g, w, ent)[np.arange(k), ext]
        got_w.append(w[ok])
        got_e.append(ent[ok])
        got_s.append(ext[ok])
        have += int(ok.sum())
    else:
        raise RuntimeError(f"rejection sampling failed for {size}×{size} with {walls} walls")
    return (
        np.concatenate(got_w)[:count],
        np.concatenate(got_e)[:count].astype(np.int32),
        np.concatenate(got_s)[:count].astype(np.int32),
    )


def uniform_maze(size: int, walls: int, rng: np.random.Generator) -> Maze:
    w, e, s = uniform_batch(size, walls, 1, rng)
    return Maze(size, w[0], int(e[0]), int(s[0]))


def stack_mazes(mazes) -> tuple:
    """A list of Maze → (walls bool[B, E], entrance int32[B], exit int32[B]) for the batched engine."""
    return (
        np.stack([m.walls for m in mazes]),
        np.array([m.entrance for m in mazes], dtype=np.int32),
        np.array([m.exit for m in mazes], dtype=np.int32),
    )


def sample_mazes(kind: str, size: int, walls: int, count: int, seed: int = 0) -> tuple:
    """`count` mazes from a named sampler ('random' = randomMaze, 'uniform'), stacked for the engine."""
    if kind == "uniform":
        return uniform_batch(size, walls, count, np.random.default_rng(seed))
    if kind == "random":
        gen = np.random.default_rng(seed)
        return stack_mazes([random_maze(size, walls, gen.random) for _ in range(count)])
    raise ValueError(f"unknown maze sampler {kind!r} (random, uniform)")
