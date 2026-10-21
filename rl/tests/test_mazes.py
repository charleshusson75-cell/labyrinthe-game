import itertools

import numpy as np
import pytest

from labyrinthe_rl import engine
from labyrinthe_rl.geometry import default_wall_budget, geometry, max_wall_budget
from labyrinthe_rl.jsrandom import SeededRng, shuffle
from labyrinthe_rl.mazes import Maze, maze_issues, random_maze, sample_mazes, uniform_batch


def test_seeded_rng_known_values():
    # seededRng(1) in src/game/random.js
    r = SeededRng(1)
    assert [r() for _ in range(3)] == [0.6270739405881613, 0.002735721180215478, 0.5274470399599522]
    assert sorted(shuffle(SeededRng(7), range(10))) == list(range(10))


@pytest.mark.parametrize("size", range(4, 11))
def test_random_maze_is_valid(size):
    g = geometry(size)
    for walls in (1, default_wall_budget(size), (size - 1) ** 2, max_wall_budget(size)):
        for seed in range(3):
            m = random_maze(size, walls, SeededRng(seed))
            assert m.wall_count == walls
            assert maze_issues(size, m.walls, m.entrance, m.exit, walls) == []
            assert Maze.from_js(size, m.to_js()).same_as(m)
    # the JS clamps the budget the same way
    assert random_maze(size, 10_000, SeededRng(0)).wall_count == max_wall_budget(size)
    assert random_maze(size, None, SeededRng(0)).wall_count == default_wall_budget(size)
    assert g.n_edges > max_wall_budget(size)


def test_uniform_sampler_is_uniform():
    """On a 2×2 board every valid (walls, E, S) must come up equally often."""
    size, walls = 2, 2
    g = geometry(size)
    valid = []
    for ws in itertools.combinations(range(g.n_edges), walls):
        mask = np.zeros(g.n_edges, dtype=bool)
        mask[list(ws)] = True
        for e, s in itertools.permutations(range(g.n_cells), 2):
            if not maze_issues(size, mask, e, s, walls):
                valid.append((ws, e, s))
    assert 0 < len(valid) < 6 * 12  # some combinations cut S off
    index = {v: i for i, v in enumerate(valid)}
    n = 40_000
    w, e, s = uniform_batch(size, walls, n, np.random.default_rng(0))
    counts = np.zeros(len(valid))
    for row, a, b in zip(w, e, s):
        counts[index[(tuple(np.flatnonzero(row)), int(a), int(b))]] += 1
    k = len(valid)
    chi2 = ((counts - n / k) ** 2 / (n / k)).sum()
    assert chi2 < (k - 1) + 6 * np.sqrt(2 * (k - 1))


@pytest.mark.parametrize("kind", ["random", "uniform"])
def test_sample_mazes(kind):
    for size in (4, 7, 10):
        walls = default_wall_budget(size)
        w, e, s = sample_mazes(kind, size, walls, 20, seed=3)
        assert w.shape == (20, geometry(size).n_edges) and (w.sum(1) == walls).all()
        assert np.asarray(engine.valid_batch(w, e, s, walls)).all()
    with pytest.raises(ValueError):
        sample_mazes("nope", 5, 12, 1)


def test_issue_codes():
    m = random_maze(4, 7, SeededRng(3))
    assert maze_issues(4, m.walls, m.entrance, m.exit, 8) == ["walls-left"]
    assert maze_issues(4, m.walls, m.entrance, m.exit, 6) == ["too-many-walls"]
    assert maze_issues(4, m.walls, m.entrance, m.entrance, 7) == ["same-cell"]
    assert maze_issues(4, m.walls, None, m.exit, 7) == ["no-entrance"]
    assert maze_issues(4, m.walls, m.entrance, 16, 7) == ["no-exit"]
