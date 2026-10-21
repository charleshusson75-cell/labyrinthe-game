import numpy as np

from labyrinthe_rl import bench


def test_seat_score_is_starter_independent():
    mine, theirs = np.array([3, 5, 4]), np.array([4, 5, 2])
    assert bench.seat_score(mine, theirs).tolist() == [1.0, 0.5, 0.0]


def test_duplicate_match():
    deals = bench.Deals(4, 7, 24, "bots", seed=0)
    r = bench.duplicate_match(bench.BOTS["hard"], bench.BOTS["easy"], deals)
    assert all(len(h) == 24 for h in r["hits"].values())
    assert 0 <= r["score (dup)"][0] <= 1 and 0 <= r["score (round)"][0] <= 1
    # the same bot against itself on shared mazes and keys: exactly even
    same = bench.duplicate_match(bench.BOTS["normal"], bench.BOTS["normal"], deals)
    assert same["explorer edge"][0] == 0 and same["score (dup)"][0] == 0.5
    # maze sets are made once per bot and seat letter
    assert deals.get(bench.BOTS["hard"], "A")[0] is deals.get(bench.BOTS["hard"], "A")[0]


def test_cli(capsys):
    bench.main(["--size", "4", "--deals", "16", "--pairs", "hard:easy,normal:normal", "--mazes", "uniform"])
    out = capsys.readouterr().out
    assert "hard vs easy" in out and "score (dup)" in out and "4×4, 7 walls, 16 deals" in out
