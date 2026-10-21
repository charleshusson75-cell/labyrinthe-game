import jax
import jax.numpy as jnp
import numpy as np
import pytest

from labyrinthe_rl import engine
from labyrinthe_rl.geometry import geometry, size_from_edges
from labyrinthe_rl.mazes import reachable_batch, sample_mazes


def test_edges_in_grid_js_order():
    # grid.js allEdges(3): row-major over cells, right edge then down edge.
    assert geometry(3).keys == ("0-1", "0-3", "1-2", "1-4", "2-5", "3-4", "3-6", "4-5", "4-7", "5-8", "6-7", "7-8")
    for size in range(2, 11):
        g = geometry(size)
        assert g.n_edges == 2 * size * (size - 1) and size_from_edges(g.n_edges) == size
        for cell in range(g.n_cells):
            for d in range(4):
                n, e = g.neighbors[cell, d], g.cell_edges[cell, d]
                assert (n < 0) == (e < 0)
                if e >= 0:
                    assert sorted(g.edge_cells[e]) == sorted((cell, n))
                    assert g.adj[cell][d] == (n, e)
        assert np.array_equal(g.right_edges, g.cell_edges[:, 1].reshape(size, size)[:, :-1])


def corridor():
    """4×4: E = cell 0, S = cell 2, walls on 0-4 and 1-5 (and nothing else)."""
    g = geometry(4)
    return g, g.walls_from_keys(["0-4", "1-5"]), 0, 2


def test_step_rules():
    g, walls, e, s = corridor()
    state = engine.init(walls, e, s)
    assert engine.legal_mask(state).sum() == 2  # 0-1 and 0-4
    state, r = engine.step(state, g.key_index["0-4"])
    assert int(r) == engine.WALL and int(state.hits) == 1 and int(state.moves) == 1
    state, r = engine.step(state, g.key_index["0-4"])
    assert int(r) == engine.ILLEGAL and int(state.moves) == 1, "a known side can't be probed again"
    state, r = engine.step(state, g.key_index["2-3"])
    assert int(r) == engine.ILLEGAL, "the side must touch a reached cell"
    state, r = engine.step(state, g.n_edges)
    assert int(r) == engine.ILLEGAL, "out of range"
    state, r = engine.step(state, g.key_index["0-1"])
    assert int(r) == engine.OPEN and bool(state.reached[1]) and not bool(state.done)
    state, r = engine.step(state, g.key_index["1-2"])
    assert int(r) == engine.OPEN and bool(state.done)
    assert engine.visited_order(state) == [0, 1, 2]
    assert [int(state.reached_at[c]) for c in (0, 1, 2)] == [0, 2, 3]
    assert not engine.legal_mask(state).any(), "nothing is legal once the exit is reached"
    assert int(engine.step(state, g.key_index["2-3"])[1]) == engine.ILLEGAL


def test_probe_by_cell_and_direction():
    g, walls, e, s = corridor()
    state = engine.init(walls, e, s)
    assert engine.probe_error(state, 0, "U") == "border"
    assert engine.probe_error(state, 0, "X") == "invalid"
    assert engine.probe_error(state, 16, "R") == "invalid"
    assert engine.probe_error(state, 1, "R") == "not-reached"
    assert int(engine.probe(state, 0, 0)[1]) == engine.ILLEGAL
    assert int(engine.probe(state, 1, 3)[1]) == engine.ILLEGAL, "from an unreached cell, even toward E"
    state, r = engine.probe(state, 0, 1)
    assert int(r) == engine.OPEN
    assert engine.probe_error(state, 1, "L") == "known"


def random_rollout(walls, entrance, exit, key, steps):
    """Uniformly random legal probes, batched; returns every intermediate state."""
    states = engine.init_batch(walls, entrance, exit)
    seen = []
    for t in range(steps):
        legal = engine.legal_batch(states)
        logits = jnp.where(legal, 0.0, -jnp.inf)
        edges = jax.random.categorical(jax.random.fold_in(key, t), logits)
        edges = jnp.where(legal.any(axis=1), edges, -1)
        states, _ = engine.step_batch(states, edges)
        seen.append(states)
    return seen


@pytest.mark.parametrize("size", [4, 5, 7, 10])
def test_invariants_under_random_play(size):
    walls, entrance, exit = sample_mazes("uniform", size, 2 * size, 64, seed=size)
    for s in random_rollout(walls, entrance, exit, jax.random.key(size), 3 * size * size)[:: size]:
        s = jax.tree.map(np.asarray, s)
        assert not (s.known_open & s.walls).any() and not (s.known_wall & ~s.walls).any()
        assert np.array_equal(s.hits, s.known_wall.sum(1))
        assert np.array_equal(s.moves, s.hits + s.known_open.sum(1))
        assert np.array_equal(s.done, s.reached[np.arange(64), s.exit])
        # reached = flood fill from E over known passages
        fill = jax.vmap(engine.reachable)(jnp.asarray(s.known_open), jnp.asarray(s.entrance))
        assert np.array_equal(np.asarray(fill), s.reached)
        g = geometry(size)
        touch = s.reached[:, g.edge_cells[:, 0]] | s.reached[:, g.edge_cells[:, 1]]
        legal = np.asarray(engine.legal_batch(jax.tree.map(jnp.asarray, s)))
        assert np.array_equal(legal, touch & ~s.known_open & ~s.known_wall & ~s.done[:, None])


def test_batched_equals_single():
    walls, entrance, exit = sample_mazes("random", 6, 19, 8, seed=1)
    batch = engine.init_batch(walls, entrance, exit)
    edges = jnp.array([0, 3, 7, 11, 20, 30, 40, 59], dtype=jnp.int32)
    out, res = engine.step_batch(batch, edges)
    for i in range(8):
        one, r = jax.jit(engine.step)(engine.init(walls[i], entrance[i], exit[i]), edges[i])
        assert int(r) == int(res[i])
        assert all(np.array_equal(a, b[i]) for a, b in zip(one, out))


def test_maze_valid():
    for size in (4, 5, 8, 10):
        for kind in ("random", "uniform"):
            w, e, s = sample_mazes(kind, size, size * 3, 32, seed=size)
            assert np.asarray(engine.valid_batch(w, e, s, size * 3)).all()
            assert not np.asarray(engine.valid_batch(w, e, s, size * 3 + 1)).any(), "wrong wall count"
            assert not np.asarray(engine.valid_batch(w, e, e, size * 3)).any(), "E = S"
    g, walls, e, s = corridor()
    closed = walls.copy()
    closed[g.key_index["1-2"]] = closed[g.key_index["0-1"]] = True
    assert not bool(engine.maze_valid(closed, e, s, 4)), "S unreachable"
    assert bool(engine.maze_valid(walls, e, s, 2))
    assert not bool(engine.maze_valid(walls, e, 16, 2)), "S off the board"


def test_reachable_matches_numpy():
    rng = np.random.default_rng(0)
    for size in (3, 6, 10):
        g = geometry(size)
        walls = rng.random((50, g.n_edges)) < 0.45
        start = rng.integers(0, g.n_cells, 50)
        ours = jax.vmap(engine.reachable)(jnp.asarray(~walls), jnp.asarray(start))
        assert np.array_equal(np.asarray(ours), reachable_batch(g, walls, start))
