import jax
import jax.numpy as jnp
import numpy as np
import pytest

from labyrinthe_rl import engine, lattice
from labyrinthe_rl.geometry import geometry
from labyrinthe_rl.mazes import sample_mazes


@pytest.mark.parametrize("size", [3, 4, 5, 10])
def test_action_pixels(size):
    g = geometry(size)
    px = lattice.action_pixels(size)
    assert len({tuple(p) for p in px}) == g.n_edges
    assert ((px[:, 0] % 2) != (px[:, 1] % 2)).all(), "sides have exactly one odd coordinate"
    lat = lattice.pixel_actions(size)
    assert np.array_equal(lat[px[:, 0], px[:, 1]], np.arange(g.n_edges))
    assert (lat >= 0).sum() == g.n_edges
    # the side pixel sits between the two cell pixels of its edge
    cells = lattice.cell_pixels(size)
    a, b = g.edge_cells[:, 0], g.edge_cells[:, 1]
    assert np.array_equal(2 * px, cells[a] + cells[b])
    assert lattice.border_mask(size).sum() == 4 * size


def test_round_trips():
    size = 6
    g = geometry(size)
    x = jnp.arange(g.n_edges, dtype=jnp.float32) + 1
    lat = lattice.edges_to_lattice(x)
    assert lat.shape == (13, 13)
    assert np.array_equal(lattice.actions_from_lattice(lat), x)
    assert float(lat.sum()) == float(x.sum())
    cells = lattice.cells_to_lattice(jnp.arange(size * size, dtype=jnp.float32))
    px = lattice.cell_pixels(size)
    assert np.array_equal(np.asarray(cells)[px[:, 0], px[:, 1]], np.arange(size * size))
    # batched reads: [..., L, L] → [..., E]
    assert lattice.actions_from_lattice(jnp.stack([lat, 2 * lat])).shape == (2, g.n_edges)


@pytest.mark.parametrize("size", [4, 5, 10])
def test_observation_planes(size):
    g = geometry(size)
    w, e, s = sample_mazes("uniform", size, 2 * size, 1, seed=size)
    state = engine.init(w[0], e[0], s[0])
    rng = np.random.default_rng(size)
    for _ in range(3 * size):
        legal = np.flatnonzero(np.asarray(engine.legal_mask(state)))
        if not len(legal):
            break
        state, _ = engine.step(state, int(rng.choice(legal)))
    obs = np.asarray(lattice.observe(state))
    n = 2 * size + 1
    assert obs.shape == (n, n, lattice.N_PLANES) and obs.dtype == np.float32
    px, cells = lattice.action_pixels(size), lattice.cell_pixels(size)
    at_sides = obs[px[:, 0], px[:, 1]]
    at_cells = obs[cells[:, 0], cells[:, 1]]
    ko, kw = np.asarray(state.known_open), np.asarray(state.known_wall)
    assert np.array_equal(at_sides[:, lattice.OPEN], ko)
    assert np.array_equal(at_sides[:, lattice.WALL], kw)
    assert np.array_equal(at_sides[:, lattice.UNKNOWN], ~ko & ~kw)
    border = lattice.border_mask(size)
    assert (obs[border][:, lattice.WALL] == 1).all() and (obs[border][:, lattice.OPEN] == 0).all()
    assert np.array_equal(at_cells[:, lattice.REACHED], np.asarray(state.reached))
    assert at_cells[:, lattice.ENTRANCE].argmax() == int(state.entrance) and obs[..., lattice.ENTRANCE].sum() == 1
    assert at_cells[:, lattice.EXIT].argmax() == int(state.exit) and obs[..., lattice.EXIT].sum() == 1
    assert obs[..., lattice.CELL].sum() == size * size and (at_cells[:, lattice.CELL] == 1).all()
    corners = obs[::2, ::2]
    assert (corners[..., : lattice.W_TOTAL] == 0).all()
    assert np.allclose(obs[..., lattice.W_TOTAL], 2 * size / g.n_edges)
    assert np.allclose(obs[..., lattice.W_FOUND], int(state.hits) / g.n_edges)
    assert np.allclose(obs[..., lattice.SIZE], size / 10)
    # belief target and masks
    assert np.array_equal(np.asarray(lattice.actions_from_lattice(lattice.true_maze(state))), w[0])
    assert np.asarray(lattice.true_maze(state)).sum() == 2 * size
    assert np.array_equal(lattice.actions_from_lattice(lattice.belief_mask(state)), ~ko & ~kw)
    assert np.array_equal(lattice.actions_from_lattice(lattice.legal_lattice(state)), engine.legal_mask(state))
    # batched
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), state)
    assert np.array_equal(np.asarray(lattice.observe_batch(batch))[1], obs)
