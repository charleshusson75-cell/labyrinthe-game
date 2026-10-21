"""Board geometry shared by every module, matching src/game/grid.js.

A cell is an integer, row * size + col (row 0 = top). An edge (a side shared by two adjacent cells)
is an integer too: its position in grid.js `allEdges(size)`, i.e. row-major over cells and, for each
cell, its right edge then its down edge. The JS key of an edge is the string "a-b" with a < b.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

DIRS = ("U", "R", "D", "L")
DELTA = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIR_INDEX = {d: i for i, d in enumerate(DIRS)}

MIN_SIZE = 4  # the game's limits (src/game/maze.js); 2-3 are allowed here for the exact-solver checks
MAX_SIZE = 10


def n_edges(size: int) -> int:
    return 2 * size * (size - 1)


@dataclass(frozen=True, eq=False)
class Geometry:
    size: int
    n_cells: int
    n_edges: int
    edge_cells: np.ndarray  # int32[E, 2]: the two cells (a < b) of every edge
    cell_edges: np.ndarray  # int32[N², 4]: edge on each side (U, R, D, L) of a cell, -1 on the border
    neighbors: np.ndarray  # int32[N², 4]: neighbouring cell on each side, -1 on the border
    right_edges: np.ndarray  # int32[N, N−1]: edge on the right of cell (r, c)
    down_edges: np.ndarray  # int32[N−1, N]: edge below cell (r, c)
    adj: tuple  # per cell, per side (U, R, D, L): (neighbour, edge) as plain ints, (-1, -1) on the border
    keys: tuple  # JS key "a-b" of every edge
    key_index: dict  # JS key -> edge

    def edge_of(self, cell: int, d: int) -> int:
        """Edge on side `d` of `cell`, or -1 on the border or for an invalid cell / direction."""
        if not (0 <= cell < self.n_cells and 0 <= d < 4):
            return -1
        return int(self.cell_edges[cell, d])

    def edge_between(self, a: int, b: int) -> int:
        """Edge between two adjacent cells."""
        return self.key_index[f"{min(a, b)}-{max(a, b)}"]

    def walls_from_keys(self, keys) -> np.ndarray:
        walls = np.zeros(self.n_edges, dtype=bool)
        walls[[self.key_index[k] for k in keys]] = True
        return walls

    def keys_of(self, walls: np.ndarray) -> list:
        """Sorted JS keys of a wall mask (canonicalMaze sorts the strings)."""
        return sorted(self.keys[e] for e in np.flatnonzero(walls))


@lru_cache(maxsize=None)
def geometry(size: int) -> Geometry:
    if not 2 <= size <= MAX_SIZE:
        raise ValueError(f"board size must be 2..{MAX_SIZE}, got {size}")
    cells = size * size
    edge_cells = []
    for r in range(size):
        for c in range(size):
            cell = r * size + c
            if c < size - 1:
                edge_cells.append((cell, cell + 1))
            if r < size - 1:
                edge_cells.append((cell, cell + size))
    index = {pair: e for e, pair in enumerate(edge_cells)}
    neighbors = np.full((cells, 4), -1, dtype=np.int32)
    cell_edges = np.full((cells, 4), -1, dtype=np.int32)
    for cell in range(cells):
        r, c = divmod(cell, size)
        for d, (dr, dc) in enumerate(DELTA):
            rr, cc = r + dr, c + dc
            if 0 <= rr < size and 0 <= cc < size:
                n = rr * size + cc
                neighbors[cell, d] = n
                cell_edges[cell, d] = index[(min(cell, n), max(cell, n))]
    keys = tuple(f"{a}-{b}" for a, b in edge_cells)
    return Geometry(
        size=size,
        n_cells=cells,
        n_edges=len(edge_cells),
        edge_cells=np.array(edge_cells, dtype=np.int32).reshape(-1, 2),
        cell_edges=cell_edges,
        neighbors=neighbors,
        right_edges=cell_edges[:, 1].reshape(size, size)[:, :-1].copy(),
        down_edges=cell_edges[:, 2].reshape(size, size)[:-1, :].copy(),
        adj=tuple(tuple(zip(map(int, neighbors[c]), map(int, cell_edges[c]))) for c in range(cells)),
        keys=keys,
        key_index={k: e for e, k in enumerate(keys)},
    )


@lru_cache(maxsize=None)
def size_from_edges(edges: int) -> int:
    """Board size from an edge count (2·N·(N−1) is one-to-one)."""
    for size in range(2, MAX_SIZE + 1):
        if n_edges(size) == edges:
            return size
    raise ValueError(f"no board has {edges} edges")


def default_wall_budget(size: int) -> int:
    """≈ ¾ of (size−1)², as in src/game/maze.js (5×5 → 12). JS rounds halves up."""
    return int(np.floor(0.75 * (size - 1) ** 2 + 0.5))


def max_wall_budget(size: int) -> int:
    return (size - 1) ** 2 + size
