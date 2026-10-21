"""The Python port must reproduce the JS reference exactly (fixtures from scripts/export-fixtures.mjs)."""

import json
from functools import lru_cache
from pathlib import Path

import jax
import numpy as np
import pytest

from labyrinthe_rl import baselines, engine, lattice, planner
from labyrinthe_rl.geometry import DIR_INDEX, DIRS, geometry
from labyrinthe_rl.jsrandom import SeededRng
from labyrinthe_rl.mazes import Maze, maze_issues, random_maze

FIXTURES = Path(__file__).parent / "fixtures"


def load(pattern):
    files = sorted(FIXTURES.glob(pattern))
    assert files, f"no fixtures {pattern}: run node scripts/export-fixtures.mjs"
    return [pytest.param(json.loads(f.read_text()), id=f.stem) for f in files]


@lru_cache(maxsize=None)
def jitted():
    planner_view = jax.jit(lambda s: (planner.candidates(s), planner.probe_scores(s)))
    return jax.jit(engine.probe), jax.jit(engine.step), planner_view, jax.jit(engine.legal_mask)


def test_fixture_count():
    assert len(list(FIXTURES.glob("*.json"))) >= 200


@pytest.mark.parametrize("fx", load("probes-*.json"))
def test_probe_sequence(fx):
    """Every probe: legality and error code, answer, newly reached cell, hits, moves, end flag."""
    probe, step, planner_view, legal_of = jitted()
    size = fx["size"]
    g = geometry(size)
    maze = Maze.from_js(size, fx["maze"])
    assert maze_issues(size, maze.walls, maze.entrance, maze.exit, fx["budget"]) == fx["issues"] == []
    if fx["mazeSeed"] is not None:
        assert random_maze(size, fx["budget"], SeededRng(fx["mazeSeed"])).same_as(maze)

    state = engine.init(maze.walls, maze.entrance, maze.exit)
    visited = [maze.entrance]
    snapshots = {r["at"]: r["ranked"] for r in fx["ranks"]}
    progress = baselines.Progress.start(size, maze.entrance)
    for i, p in enumerate(fx["probes"]):
        if i in snapshots:
            check_ranks(snapshots[i], state, progress, maze.exit, planner_view)
        d = DIR_INDEX.get(p["dir"], -1)
        assert engine.probe_error(state, p["from"], p["dir"]) == p["error"], f"probe {i}"
        new, result = probe(state, p["from"], d)
        if p["error"] is None:
            assert int(result) == (engine.OPEN if p["open"] else engine.WALL), f"probe {i}"
            # The same probe as an action (an edge) gives the same result.
            via_edge, r2 = step(state, g.cell_edges[p["from"], d])
            assert int(r2) == int(result)
            assert all(np.array_equal(a, b) for a, b in zip(via_edge, new))
            progress.record(p["from"], d, p["open"])
        else:
            assert int(result) == engine.ILLEGAL, f"probe {i}"
            assert all(np.array_equal(a, b) for a, b in zip(new, state)), "an illegal probe changes nothing"
            edge = g.edge_of(p["from"], d)
            if p["error"] in ("invalid", "border", "known", "over"):
                assert edge < 0 or not legal_of(state)[edge]
                assert int(step(state, edge)[1]) == engine.ILLEGAL
        if p["reached"] is not None:
            visited.append(p["reached"])
        state = new
        assert engine.visited_order(state) == visited, f"probe {i}"
        assert (int(state.hits), int(state.moves), bool(state.done)) == (p["hits"], p["moves"], p["done"]), f"probe {i}"

    # The flood fill from E over known passages gives the same reached set.
    assert np.array_equal(engine.reachable(state.known_open, state.entrance), state.reached)
    # The lattice shows exactly what the explorer knows.
    obs = np.asarray(lattice.observe(state))
    assert obs[..., lattice.OPEN].sum() == len(progress.open)
    assert obs[..., lattice.WALL].sum() == len(progress.walls) + 4 * size
    assert obs[..., lattice.REACHED].sum() == len(visited)
    assert obs[0, 0, lattice.W_FOUND] == pytest.approx(fx["probes"][-1]["hits"] / g.n_edges)


def check_ranks(expected, state, progress, exit, planner_view):
    # Python port: same list, same order, same scores.
    assert [list(p) for p in baselines.rank_probes(progress, exit)] == expected
    # JAX planner: same score for every legal (cell, side), nothing else legal; the same best
    # probes and the same first four (what 'normal' picks among when it errs).
    (legal, best, first4), scores = jax.tree.map(np.asarray, planner_view(state))
    assert legal.sum() == len(expected)
    for cell, d, score in expected:
        assert scores[cell, d] == 2 * score
    flat = lambda entries: {4 * c + d for c, d, _ in entries}
    assert set(np.flatnonzero(best)) == flat([p for p in expected if p[2] == expected[0][2]])
    assert set(np.flatnonzero(first4)) == flat(expected[:4])


@pytest.mark.parametrize("fx", load("mazes-*.json"))
def test_random_maze_and_issues(fx):
    size = fx["size"]
    for r in fx["random"]:
        got = random_maze(size, r["budget"], SeededRng(r["seed"]))
        assert got.to_js() == r["maze"], f"seed {r['seed']}, budget {r['budget']}"
    for c in fx["crafted"]:
        m = c["maze"]
        walls = geometry(size).walls_from_keys(m["walls"])
        assert maze_issues(size, walls, m["entrance"], m["exit"], fx["budget"]) == c["issues"], c["name"]
        if m["entrance"] is not None and m["exit"] is not None:
            valid = bool(engine.maze_valid(walls, m["entrance"], m["exit"], fx["budget"]))
            assert valid == (c["issues"] == []), c["name"]


@pytest.mark.parametrize("fx", load("solve-*.json"))
def test_solve_trace(fx):
    """The ported explorer makes the same probes as the JS bot for the same seed."""
    for run in fx["runs"]:
        maze = Maze.from_js(fx["size"], run["maze"])
        assert random_maze(fx["size"], run["budget"], SeededRng(run["mazeSeed"])).same_as(maze)
        solve = baselines.simulate_solve(maze, fx["level"], SeededRng(run["seed"]))
        assert [list(p) for p in solve.probes] == run["probes"]
        assert (solve.hits, solve.moves) == (run["hits"], run["moves"])


@pytest.mark.parametrize("fx", load("design-*.json"))
def test_design(fx):
    """The ported designers pick the same maze as the JS bot for the same seed."""
    maze = baselines.design_maze(fx["size"], fx["budget"], fx["level"], SeededRng(fx["seed"]))
    assert maze.to_js() == fx["maze"]


def test_dirs_match_js():
    assert DIRS == ("U", "R", "D", "L")
