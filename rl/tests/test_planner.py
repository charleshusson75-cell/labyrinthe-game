"""The batched JAX bots against the exact Python port (itself checked against JS in test_parity)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from labyrinthe_rl import baselines, engine, planner
from labyrinthe_rl.geometry import default_wall_budget
from labyrinthe_rl.mazes import Maze, sample_mazes


def test_full_distances_match_port():
    size = 7
    w, e, s = sample_mazes("uniform", size, 20, 32, seed=0)
    rng = np.random.default_rng(0)
    dist = jax.jit(planner.optimistic_distances)
    for i in range(32):
        state = engine.init(w[i], e[i], s[i])
        progress = baselines.Progress.start(size, int(e[i]))
        for _ in range(rng.integers(0, 30)):
            ranked = baselines.rank_probes(progress, int(s[i]))
            if not ranked or bool(state.done):
                break
            cell, d, _ = ranked[rng.integers(len(ranked))]
            state, r = engine.probe(state, cell, d)
            progress.record(cell, d, int(r) == engine.OPEN)
        ref = np.array(baselines.optimistic_distances(progress, int(s[i])))
        ours = np.asarray(dist(state)).astype(float)
        ours[ours >= planner.FAR] = np.inf
        assert np.array_equal(ours, ref)


@pytest.mark.parametrize("level", ["easy", "normal", "hard"])
def test_same_hits_as_port_in_distribution(level):
    size, walls, n = 5, default_wall_budget(5), 600
    w, e, s = sample_mazes("random", size, walls, n, seed=11)
    ours = planner.explore_mazes(w, e, s, jax.random.key(0), planner.LEVEL_INDEX[level], slots=256)
    rng = np.random.default_rng(1)
    ref = np.array([baselines.simulate_solve(Maze(size, w[i], int(e[i]), int(s[i])), level, rng.random).hits for i in range(n)])
    se = np.sqrt((ours.var() + ref.var()) / n)
    assert abs(ours.mean() - ref.mean()) < 4 * se + 0.05, (ours.mean(), ref.mean())


def test_explore_queue_matches_batch_finish():
    w, e, s = sample_mazes("uniform", 6, 19, 300, seed=2)
    queue = planner.explore_mazes(w, e, s, jax.random.key(3), planner.HARD, slots=64)
    batch = np.asarray(planner.explore(engine.init_batch(w, e, s), jax.random.key(4), planner.HARD).hits)
    assert queue.shape == (300,) and (queue >= 0).all() and (queue <= 19).all()
    se = np.sqrt((queue.var() + batch.var()) / 300)
    assert abs(queue.mean() - batch.mean()) < 4 * se + 0.05
    # every game really finished
    final = planner.explore(engine.init_batch(w, e, s), jax.random.key(4), planner.HARD)
    assert np.asarray(final.done).all()


def test_planner_only_plays_legal_probes():
    w, e, s = sample_mazes("random", 8, 37, 16, seed=5)
    states = engine.init_batch(w, e, s)
    choose = jax.jit(jax.vmap(planner.choose, in_axes=(0, 0, None)))
    for t in range(60):
        u = jax.random.uniform(jax.random.key(t), (16, 2))
        edges = choose(states, u, planner.EASY if t % 2 else planner.NORMAL)
        legal = engine.legal_batch(states)
        ok = np.where(np.asarray(states.done), np.asarray(edges) == -1, np.asarray(legal)[np.arange(16), np.maximum(np.asarray(edges), 0)])
        assert ok.all()
        states, _ = engine.step_batch(states, edges)


def test_design_candidates_scale_like_js():
    # brain.js: max(1, round(base · min(1, 25 / size²))) with base 1 / 6 / 30
    expected = {4: (1, 6, 30), 5: (1, 6, 30), 6: (1, 4, 21), 7: (1, 3, 15), 8: (1, 2, 12), 9: (1, 2, 9), 10: (1, 2, 8)}
    for size, counts in expected.items():
        assert tuple(baselines.design_candidates(size, lvl) for lvl in baselines.LEVELS) == counts
    assert baselines.design_runs("hard") == 4 and baselines.design_runs("normal") == 2


def test_designers_make_valid_mazes():
    for level in baselines.LEVELS:
        w, e, s = planner.design(5, 12, level, 20, np.random.default_rng(0), jax.random.key(0))
        assert np.asarray(engine.valid_batch(w, e, s, 12)).all()
    # the hard designer's mazes cost the hard explorer more walls than random ones
    hard = planner.design(5, 12, "hard", 300, np.random.default_rng(1), jax.random.key(1))
    easy = planner.design(5, 12, "easy", 300, np.random.default_rng(2), jax.random.key(2))
    hits = [planner.explore_mazes(*m, jax.random.key(9), planner.HARD).mean() for m in (hard, easy)]
    assert hits[0] > hits[1] + 0.5
