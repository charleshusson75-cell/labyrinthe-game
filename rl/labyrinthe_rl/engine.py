"""The explorer's side of a round, in JAX: one hidden maze, probed until the exit is reached.

Same rules as the JS reference (src/game/match.js, src/net/session.js):
- a probe is a side of a cell already reached; it must still be unknown (probeError);
- open → the side becomes a known passage and the cell behind it is reached; wall → it's noted, and
  in the real game the turn passes (one hit);
- the round is over for this explorer once the exit is reached; later probes are refused.

Every function works on one game and is pure, so `jax.jit` and `jax.vmap` apply directly. The board
size is read from the array shapes (static under `jit`), so one compiled function serves one size.
Actions are edges, numbered as in grid.js `allEdges()` (see geometry.py).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .geometry import DIR_INDEX, geometry, size_from_edges

OPEN = 1  # step results
WALL = 0
ILLEGAL = -1

NOT_REACHED = np.iinfo(np.int32).max  # reached_at of a cell not reached yet


class State(NamedTuple):
    walls: jax.Array  # bool[E]   the hidden maze
    entrance: jax.Array  # int32[]
    exit: jax.Array  # int32[]
    known_open: jax.Array  # bool[E]   sides probed and found open
    known_wall: jax.Array  # bool[E]   sides probed and found to be walls
    reached: jax.Array  # bool[N²]  cells reached (connected to E by known passages)
    reached_at: jax.Array  # int32[N²] move that first reached each cell (E: 0), the order JS lists them in
    hits: jax.Array  # int32[]   walls hit so far
    moves: jax.Array  # int32[]   legal probes so far
    done: jax.Array  # bool[]    exit reached


def _tables(n_edges: int):
    """Geometry of the board with `n_edges` edges, as constants for tracing."""
    g = geometry(size_from_edges(n_edges))
    return g, jnp.asarray(g.edge_cells), jnp.asarray(g.cell_edges), jnp.asarray(g.neighbors)


def board_size(state: State) -> int:
    return size_from_edges(state.walls.shape[-1])


def init(walls, entrance, exit) -> State:
    """Start of an exploration: only the entrance is reached."""
    walls = jnp.asarray(walls, dtype=bool)
    g = geometry(size_from_edges(walls.shape[-1]))
    entrance = jnp.asarray(entrance, dtype=jnp.int32)
    exit = jnp.asarray(exit, dtype=jnp.int32)
    cells = jnp.arange(g.n_cells)
    reached = cells == entrance
    return State(
        walls=walls,
        entrance=entrance,
        exit=exit,
        known_open=jnp.zeros(g.n_edges, dtype=bool),
        known_wall=jnp.zeros(g.n_edges, dtype=bool),
        reached=reached,
        reached_at=jnp.where(reached, 0, NOT_REACHED).astype(jnp.int32),
        hits=jnp.int32(0),
        moves=jnp.int32(0),
        done=entrance == exit,
    )


def legal_mask(state: State) -> jax.Array:
    """bool[E]: the edges `step` accepts (unknown and touching a reached cell, game not over)."""
    _, edge_cells, _, _ = _tables(state.walls.shape[-1])
    touches = state.reached[edge_cells[:, 0]] | state.reached[edge_cells[:, 1]]
    return touches & ~state.known_open & ~state.known_wall & ~state.done


def step(state: State, edge) -> tuple:
    """Probe `edge`. Returns (new state, result): OPEN, WALL, or ILLEGAL (state unchanged)."""
    n_e = state.walls.shape[-1]
    g, edge_cells, _, _ = _tables(n_e)
    edge = jnp.asarray(edge, dtype=jnp.int32)
    hot = jnp.arange(n_e) == edge  # all-false when edge is out of range
    a, b = edge_cells[jnp.clip(edge, 0, n_e - 1)]
    unknown = ~jnp.any(hot & (state.known_open | state.known_wall))
    legal = jnp.any(hot) & unknown & ~state.done & (state.reached[a] | state.reached[b])
    is_wall = jnp.any(hot & state.walls)
    opened = legal & ~is_wall
    hit = legal & is_wall

    behind = jnp.where(state.reached[a], b, a)  # the cell this probe would enter
    entered = (jnp.arange(g.n_cells) == behind) & opened
    new_cell = entered & ~state.reached
    # Known passages always join two reached cells, so the flood fill from E over known passages
    # (match.js progressOf) only ever grows by the cell behind a new opening.
    reached = state.reached | entered
    moves = state.moves + legal.astype(jnp.int32)
    new_state = State(
        walls=state.walls,
        entrance=state.entrance,
        exit=state.exit,
        known_open=state.known_open | (hot & opened),
        known_wall=state.known_wall | (hot & hit),
        reached=reached,
        reached_at=jnp.where(new_cell, moves, state.reached_at),
        hits=state.hits + hit.astype(jnp.int32),
        moves=moves,
        done=state.done | jnp.any(entered & (jnp.arange(g.n_cells) == state.exit)),
    )
    result = jnp.where(legal, jnp.where(is_wall, WALL, OPEN), ILLEGAL).astype(jnp.int32)
    return new_state, result


def probe(state: State, cell, d) -> tuple:
    """Probe side `d` (0..3 = U, R, D, L) of `cell`, legal exactly when match.js probeError is null
    and the round isn't over. Same results as `step`."""
    n_e = state.walls.shape[-1]
    g, _, cell_edges, _ = _tables(n_e)
    cell = jnp.asarray(cell, dtype=jnp.int32)
    d = jnp.asarray(d, dtype=jnp.int32)
    valid = (cell >= 0) & (cell < g.n_cells) & (d >= 0) & (d < 4)
    c = jnp.clip(cell, 0, g.n_cells - 1)
    edge = cell_edges[c, jnp.clip(d, 0, 3)]
    ok = valid & state.reached[c] & (edge >= 0)
    return step(state, jnp.where(ok, edge, -1))


def side_grid(values, fill) -> jax.Array:
    """Per-edge values (shape [E]) → [4, N, N]: the value on side U, R, D, L of every cell, `fill`
    on the outer border."""
    g = geometry(size_from_edges(values.shape[-1]))
    right, down = values[g.right_edges], values[g.down_edges]
    col = jnp.full((g.size, 1), fill, dtype=values.dtype)
    row = jnp.full((1, g.size), fill, dtype=values.dtype)
    up = jnp.concatenate([row, down], axis=0)  # the down side of the cell above
    left = jnp.concatenate([col, right], axis=1)  # the right side of the cell on the left
    return jnp.stack([up, jnp.concatenate([right, col], axis=1), jnp.concatenate([down, row], axis=0), left])


def neighbor_grid(x, fill) -> jax.Array:
    """x[N, N] → [4, N, N]: every cell's neighbour value on side U, R, D, L, `fill` off the board.
    (Shifted slices: much cheaper than a gather when batched on CPU.)"""
    n = x.shape[0]
    col = jnp.full((n, 1), fill, dtype=x.dtype)
    row = jnp.full((1, n), fill, dtype=x.dtype)
    return jnp.stack(
        [
            jnp.concatenate([row, x[:-1]], axis=0),
            jnp.concatenate([x[:, 1:], col], axis=1),
            jnp.concatenate([x[1:], row], axis=0),
            jnp.concatenate([col, x[:, :-1]], axis=1),
        ]
    )


def reachable(open_edges, start) -> jax.Array:
    """bool[N²]: cells reachable from `start` through `open_edges` (bool[E]), by flood fill."""
    g = geometry(size_from_edges(open_edges.shape[-1]))
    side_open = side_grid(open_edges, False)

    def grow(carry):
        r, _ = carry
        new = r | jnp.any(side_open & neighbor_grid(r, False), axis=0)
        return new, jnp.any(new != r)

    start_set = (jnp.arange(g.n_cells) == start).reshape(g.size, g.size)
    reached, _ = jax.lax.while_loop(lambda c: c[1], grow, (start_set, jnp.bool_(True)))
    return reached.reshape(-1)


def maze_valid(walls, entrance, exit, n_walls) -> jax.Array:
    """A playable maze: exactly `n_walls` walls, E ≠ S on the board, S reachable from E."""
    walls = jnp.asarray(walls, dtype=bool)
    n_cells = geometry(size_from_edges(walls.shape[-1])).n_cells
    on_board = (entrance >= 0) & (entrance < n_cells) & (exit >= 0) & (exit < n_cells)
    e = jnp.clip(entrance, 0, n_cells - 1)
    s = jnp.clip(exit, 0, n_cells - 1)
    return (jnp.sum(walls) == n_walls) & on_board & (e != s) & reachable(~walls, e)[s]


# ------------------------------------------------------------------ host-side helpers (numpy)


def probe_error(state: State, cell, d) -> str | None:
    """The JS error code for a probe (None = legal): probeError's codes, plus 'over' once the exit is
    reached (the session refuses moves when there is a winner)."""
    size = board_size(state)
    g = geometry(size)
    if bool(state.done):
        return "over"
    if isinstance(d, str):
        d = DIR_INDEX.get(d, -1)
    if not (isinstance(cell, (int, np.integer)) and 0 <= cell < g.n_cells and 0 <= d < 4):
        return "invalid"
    if not bool(state.reached[cell]):
        return "not-reached"
    edge = g.cell_edges[cell, d]
    if edge < 0:
        return "border"
    if bool(state.known_open[edge]) or bool(state.known_wall[edge]):
        return "known"
    return None


def visited_order(state: State) -> list:
    """Reached cells in the order they were reached (the iteration order of JS `progress.visited`)."""
    at = np.asarray(state.reached_at)
    cells = np.flatnonzero(np.asarray(state.reached))
    return [int(c) for c in cells[np.argsort(at[cells], kind="stable")]]


init_batch = jax.jit(jax.vmap(init))
step_batch = jax.jit(jax.vmap(step))
legal_batch = jax.jit(jax.vmap(legal_mask))
valid_batch = jax.jit(jax.vmap(maze_valid, in_axes=(0, 0, 0, None)))
