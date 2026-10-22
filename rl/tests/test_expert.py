import jax
import numpy as np
import pytest

from labyrinthe_rl import bench, expert, planner

pytestmark = pytest.mark.skipif(expert.node() is None, reason="the expert explorer runs in Node.js")


def test_explore_is_deterministic_and_split_independent():
    mazes = planner.design(4, 7, "hard", 4, np.random.default_rng(0), jax.random.key(0))
    mazes = tuple(np.concatenate([x, x]) for x in mazes)  # same entrances and exits again, new seeds
    key = jax.random.key(1)
    hits = expert.explore_mazes(*mazes, key, particles=40, workers=1)
    assert hits.shape == (8,) and hits.dtype == np.int32
    assert np.all((0 <= hits) & (hits <= 7))
    # a maze's result depends on the maze and its seed only, not on the mazes its Node process saw before
    assert hits.tolist() == expert.explore_mazes(*mazes, key, particles=40, workers=8).tolist()
    assert len(expert.explore_mazes(*(x[:0] for x in mazes), key)) == 0


def test_expert_beats_hard():
    # D21: on 4×4 (7 walls) the expert hits ≈ 1.9 walls on hard's mazes, hard ≈ 6.9
    deals = bench.Deals(4, 7, 8, "bots", seed=0)
    r = bench.duplicate_match(bench.expert_bot(particles=60), bench.BOTS["hard"], deals)
    assert r["hits A"][0] < 3.5 < 5.5 < r["hits B"][0]
    assert r["score (round)"][0] > 0.9


def test_cli(capsys):
    bench.main(["--size", "4", "--deals", "4", "--pairs", "expert:hard", "--particles", "40"])
    assert "expert vs hard" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        bench.main(["--pairs", "expert:nobody"])
