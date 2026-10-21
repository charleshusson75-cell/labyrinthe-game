"""The maze as a (2N+1)×(2N+1) image (DECISIONS.md D17), for fully convolutional networks.

Pixel (i, j) of the lattice is:
- a cell when i and j are both odd: cell (r, c) sits at (2r+1, 2c+1);
- a side when exactly one is odd: the right side of (r, c) at (2r+1, 2c+2), its down side at
  (2r+2, 2c+1); the outer border sides are on the frame (i or j = 0 or 2N);
- a corner when both are even (always 0).

Observation planes (channels last, float32[2N+1, 2N+1, C]):

  OPEN      sides probed and found open
  WALL      sides known to be walls: walls hit, plus the outer border (always closed)
  UNKNOWN   inner sides not probed yet
  REACHED   cells reached
  ENTRANCE  the entrance cell
  EXIT      the exit cell
  CELL      every cell pixel: tells cells from corners, and the board from zero padding
  W_TOTAL   constant: walls in the maze / inner sides (the wall density)
  W_FOUND   constant: walls hit / inner sides   (so W_TOTAL − W_FOUND = walls left, same scale)
  SIZE      constant: N / 10

An action (an edge, geometry.py order) is a side pixel: `action_pixels` and `pixel_actions` map
between the two, and `actions_from_lattice` reads per-action values (e.g. policy logits) off a
lattice-shaped network output.
"""

import math
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from .engine import State, legal_mask
from .geometry import MAX_SIZE, geometry, size_from_edges

PLANES = ("OPEN", "WALL", "UNKNOWN", "REACHED", "ENTRANCE", "EXIT", "CELL", "W_TOTAL", "W_FOUND", "SIZE")
(OPEN, WALL, UNKNOWN, REACHED, ENTRANCE, EXIT, CELL, W_TOTAL, W_FOUND, SIZE) = range(len(PLANES))
N_PLANES = len(PLANES)


def lattice_size(size: int) -> int:
    return 2 * size + 1


@lru_cache(maxsize=None)
def action_pixels(size: int) -> np.ndarray:
    """int32[E, 2]: the (row, col) lattice pixel of every edge."""
    g = geometry(size)
    a, b = g.edge_cells[:, 0], g.edge_cells[:, 1]
    ra, ca = np.divmod(a, size)
    right = b == a + 1
    rows = np.where(right, 2 * ra + 1, 2 * ra + 2)
    cols = np.where(right, 2 * ca + 2, 2 * ca + 1)
    out = np.stack([rows, cols], axis=1).astype(np.int32)
    out.flags.writeable = False
    return out


@lru_cache(maxsize=None)
def pixel_actions(size: int) -> np.ndarray:
    """int32[2N+1, 2N+1]: the edge at each side pixel, −1 elsewhere (cells, corners, border)."""
    lat = np.full((lattice_size(size),) * 2, -1, dtype=np.int32)
    px = action_pixels(size)
    lat[px[:, 0], px[:, 1]] = np.arange(len(px))
    lat.flags.writeable = False
    return lat


@lru_cache(maxsize=None)
def cell_pixels(size: int) -> np.ndarray:
    """int32[N², 2]: the (row, col) lattice pixel of every cell."""
    r, c = np.divmod(np.arange(size * size), size)
    out = np.stack([2 * r + 1, 2 * c + 1], axis=1).astype(np.int32)
    out.flags.writeable = False
    return out


@lru_cache(maxsize=None)
def border_mask(size: int) -> np.ndarray:
    """bool[2N+1, 2N+1]: the outer border sides."""
    n = lattice_size(size)
    m = np.zeros((n, n), dtype=bool)
    odd = np.arange(1, n, 2)
    m[0, odd] = m[n - 1, odd] = m[odd, 0] = m[odd, n - 1] = True
    m.flags.writeable = False
    return m


def compose(cells, right, down) -> jax.Array:
    """Interleave grids into a lattice (pure reshapes, no gathers: fast when batched on CPU).

    cells [N, N, ...]; right [N, N+1, ...]: the vertical sides of each cell row, left border
    first; down [N+1, N, ...]: the horizontal sides of each column, top border first. Trailing
    axes (channels) are kept. Corners are 0. Returns [2N+1, 2N+1, ...].
    """
    n = cells.shape[0]
    rest = cells.shape[2:]
    odd = jnp.stack([cells, right[:, 1:]], axis=2).reshape((n, 2 * n) + rest)
    odd = jnp.concatenate([right[:, :1], odd], axis=1)  # cell rows: border, cell, side, cell, ..., border
    corner = jnp.zeros_like(down)
    even = jnp.stack([corner, down], axis=2).reshape((n + 1, 2 * n) + rest)
    even = jnp.concatenate([even, corner[:, :1]], axis=1)  # side rows: corner, side, corner, ..., corner
    rows = jnp.stack([even[:n], odd], axis=1).reshape((2 * n, 2 * n + 1) + rest)
    return jnp.concatenate([rows, even[n:]], axis=0)


def _side_grids(values, border) -> tuple:
    """Per-edge values [E, ...] → (right [N, N+1, ...], down [N+1, N, ...]), the outer border set to
    `border` (broadcast over the trailing axes)."""
    g = geometry(size_from_edges(values.shape[0]))
    n, rest = g.size, values.shape[1:]
    col = jnp.broadcast_to(jnp.asarray(border, dtype=values.dtype), (n, 1) + rest)
    row = jnp.broadcast_to(jnp.asarray(border, dtype=values.dtype), (1, n) + rest)
    right = jnp.concatenate([col, values[g.right_edges], col], axis=1)
    down = jnp.concatenate([row, values[g.down_edges], row], axis=0)
    return right, down


def edges_to_lattice(values) -> jax.Array:
    """Per-edge values (shape [E, ...]) on their side pixels; 0 elsewhere."""
    right, down = _side_grids(values, 0)
    size = size_from_edges(values.shape[0])
    return compose(jnp.zeros((size, size) + values.shape[1:], dtype=values.dtype), right, down)


def cells_to_lattice(values) -> jax.Array:
    """Per-cell values (shape [N², ...]) on their cell pixels; 0 elsewhere."""
    size, rest = math.isqrt(values.shape[0]), values.shape[1:]
    cells = values.reshape((size, size) + rest)
    return compose(cells, jnp.zeros((size, size + 1) + rest, values.dtype), jnp.zeros((size + 1, size) + rest, values.dtype))


def actions_from_lattice(x) -> jax.Array:
    """Read per-action values off a lattice: x[..., 2N+1, 2N+1] → [..., E]."""
    size = (x.shape[-1] - 1) // 2
    px = action_pixels(size)
    return x[..., px[:, 0], px[:, 1]]


def observe(state: State) -> jax.Array:
    """The explorer's observation, float32[2N+1, 2N+1, N_PLANES] (see the module docstring)."""
    n_e = state.walls.shape[-1]
    size = size_from_edges(n_e)
    f = jnp.float32
    unknown = ~state.known_open & ~state.known_wall
    sides = jnp.stack([state.known_open, state.known_wall, unknown], axis=-1).astype(f)  # [E, 3]
    right, down = _side_grids(sides, [0, 1, 0])  # the outer border is a known wall
    cells = jnp.arange(size * size)
    cell_planes = jnp.stack([state.reached, cells == state.entrance, cells == state.exit, jnp.ones_like(state.reached)], -1)
    cell_planes = cell_planes.astype(f).reshape(size, size, 4)
    zeros3 = jnp.zeros((size, size, 3), f)
    side_planes = compose(zeros3, right, down)
    cell_lattice = compose(cell_planes, jnp.zeros((size, size + 1, 4), f), jnp.zeros((size + 1, size, 4), f))
    n = lattice_size(size)
    scalars = jnp.stack([jnp.sum(state.walls) / n_e, state.hits / n_e, jnp.float32(size / MAX_SIZE)]).astype(f)
    constants = jnp.broadcast_to(scalars, (n, n, 3))
    return jnp.concatenate([side_planes, cell_lattice, constants], axis=-1)


def true_maze(state: State) -> jax.Array:
    """Target of the belief head, float32[2N+1, 2N+1]: 1 on the inner sides that are walls."""
    return edges_to_lattice(state.walls.astype(jnp.float32))


def belief_mask(state: State) -> jax.Array:
    """bool[2N+1, 2N+1]: where the belief loss applies (inner sides still unknown)."""
    return edges_to_lattice(~state.known_open & ~state.known_wall)


def legal_lattice(state: State) -> jax.Array:
    """bool[2N+1, 2N+1]: legal probes as side pixels."""
    return edges_to_lattice(legal_mask(state))


observe_batch = jax.jit(jax.vmap(observe))
