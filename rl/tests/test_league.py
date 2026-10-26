import json
import sys
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from labyrinthe_rl import bench, designer, exploit, league, learned, network
from labyrinthe_rl.engine import reachable
from labyrinthe_rl.geometry import geometry, n_edges
from labyrinthe_rl.mazes import distances_from, reachable_batch


def random_states(size, count, seed):
    """Designer states in the wall phase: E, S placed and some valid walls (via the env itself)."""
    rng = np.random.default_rng(seed)
    params = designer.init_net(jax.random.key(seed), network.NetConfig(4, 1))
    budgets = rng.integers(1, (size - 1) ** 2 + 1, count)
    walls, entrance, exit = designer.sample_mazes(params, size, budgets, jax.random.key(seed))
    return walls, entrance, exit


def partial_states(size, count, seed):
    """Designer states partway through the walls (a random policy, stopped at a random step)."""
    rng = np.random.default_rng(seed)
    states = []
    for i in range(count):
        budget = int(rng.integers(1, (size - 1) ** 2 + 1))
        s = designer.init(budget, size)
        for _ in range(int(rng.integers(2, budget + 2))):
            _, legal = designer.observe(s, size)
            s = designer.step(s, int(rng.choice(np.flatnonzero(np.asarray(legal)))), size)
        states.append(s)
    return states


def test_next_wall_mask_matches_brute_force():
    """On 3×3, side e can take the next wall exactly when some completion (the walls left among the
    sides after e) keeps S reachable from E."""
    from itertools import combinations

    size = 3
    g = geometry(size)
    checked = 0
    for s in partial_states(size, 40, seed=1):
        if bool(s.done):
            continue
        left = int(s.budget - s.placed)
        mask = np.asarray(designer.next_wall_mask(s.walls, s.last, s.entrance, s.exit, left, size))
        walls = np.asarray(s.walls)
        for e in range(g.n_edges):
            ok = False
            if e > int(s.last) and not walls[e]:
                for rest in combinations(range(e + 1, g.n_edges), left - 1):
                    w = walls.copy()
                    w[[e, *rest]] = True
                    if distances_from(g, w, int(s.entrance))[int(s.exit)] >= 0:
                        ok = True
                        break
            assert mask[e] == ok
            checked += 1
    assert checked > 100


def test_distances_match_bfs():
    size = 5
    g = geometry(size)
    walls, entrance, _ = random_states(size, 16, seed=1)
    for w, e in zip(walls, entrance):
        d = np.asarray(designer.distances(jnp.asarray(~w), int(e), size))
        assert d.tolist() == distances_from(g, w, int(e))


@pytest.mark.parametrize("size", [4, 5])
def test_designed_mazes_are_valid(size):
    budgets = np.array([1, 3, (size - 1) ** 2, 6] * 16)
    walls, entrance, exit = random_states(size, 64, seed=7)
    params = designer.init_net(jax.random.key(0), network.NetConfig(8, 2))
    walls, entrance, exit = designer.sample_mazes(params, size, budgets, jax.random.key(3))
    assert np.array_equal(walls.sum(axis=1), budgets)
    assert np.all(entrance != exit) and np.all((entrance >= 0) & (exit >= 0))
    assert reachable_batch(geometry(size), walls, entrance)[np.arange(len(exit)), exit].all()
    # greedy design is deterministic
    again = designer.sample_mazes(params, size, budgets, jax.random.key(4), greedy=True)
    assert np.array_equal(again[0], designer.sample_mazes(params, size, budgets, jax.random.key(5), greedy=True)[0])


def test_observation_and_legal_moves():
    size = 4
    s = designer.init(5, size)
    obs, legal = designer.observe(s, size)
    assert obs.shape == (9, 9, designer.N_D_PLANES)
    assert bool(legal[: size * size].all()) and not bool(legal[size * size :].any())  # placing E
    assert float(obs[0, 0, designer.PHASE_E]) == 1
    s = designer.step(s, 5, size)
    _, legal = designer.observe(s, size)
    assert not bool(legal[5]) and int(legal[: size * size].sum()) == size * size - 1  # placing S ≠ E
    s = designer.step(s, 6, size)  # S next to E
    obs, legal = designer.observe(s, size)
    assert int(legal[: size * size].sum()) == 0 and float(obs[1, 1, designer.PATH]) == pytest.approx(1 / 16)
    # 5 walls to place among 24 sides: the last 4 sides can't come first (no room for the rest after them)
    assert np.flatnonzero(np.asarray(legal[size * size :])).tolist() == list(range(n_edges(size) - 4))
    assert float(obs[..., designer.LEGAL].sum()) == n_edges(size) - 4
    order = []
    for _ in range(5):
        _, legal = designer.observe(s, size)
        a = int(np.flatnonzero(np.asarray(legal))[-1])  # the latest side allowed: walls end up packed at the end
        order.append(a - size * size)
        s = designer.step(s, a, size)
    assert bool(s.done) and int(s.walls.sum()) == 5 and order == sorted(order) and int(s.last) == order[-1]
    _, legal = designer.observe(s, size)
    assert np.flatnonzero(np.asarray(legal)).tolist() == [0]  # padding


def test_advantages_reward_at_the_last_decision():
    size = 4
    params = designer.init_net(jax.random.key(0), network.NetConfig(8, 2))
    budgets = jnp.array([3, 5, 7, 5])
    _, batch = designer.rollout(params, budgets, jax.random.key(1), size, designer.steps(7), network.NetConfig().dilations)
    assert np.asarray(batch.valid).sum(axis=0).tolist() == [5, 7, 9, 7]
    reward = jnp.array([1.0, 2.0, 3.0, 4.0])
    adv, ret = designer.advantages(batch, reward, 1.0)  # λ = 1: Monte Carlo
    valid = np.asarray(batch.valid)
    assert np.allclose(np.asarray(ret)[valid], np.broadcast_to(np.asarray(reward), valid.shape)[valid], atol=1e-5)
    assert np.all(np.asarray(adv)[~valid] == 0)


def test_designer_learns_against_a_fixed_explorer():
    """A tiny PPO run raises the walls the (fixed) hard planner hits: the gradient has the right sign."""
    from labyrinthe_rl import planner

    cfg = league.Config(size=4, walls="7", d_channels=8, d_blocks=2, d_mazes=128, d_minibatch=256, d_lr=3e-3, d_entropy=0.0)
    learner = league.DesignerLearner(cfg, designer.init_net(jax.random.key(0), network.NetConfig(8, 2)))
    key = jax.random.key(1)
    first = []
    for it in range(12):
        key, k1, k2, k3 = jax.random.split(key, 4)
        mazes, batch = learner.generate(np.full(128, 7), k1)
        hits = planner.explore_mazes(*mazes, k2, planner.HARD)
        learner.learn(batch, hits.astype(np.float32), k3)
        first.append(hits.mean())
    assert np.mean(first[-3:]) > np.mean(first[:3])


def test_entropy_target_raises_the_coefficient():
    """With a target the wall decisions can't reach, the entropy bonus's coefficient goes up after an
    update; with a target they exceed, it stays at its floor (the configured coefficient)."""
    for target, rises in ((50.0, True), (1e-3, False)):
        cfg = league.Config(size=4, walls="7", d_channels=8, d_blocks=2, d_mazes=64, d_minibatch=128, d_entropy=0.01, d_entropy_target=target)
        learner = league.DesignerLearner(cfg, designer.init_net(jax.random.key(0), network.NetConfig(8, 2)))
        mazes, batch = learner.generate(np.full(64, 7), jax.random.key(1))
        stats = learner.learn(batch, np.ones(64, np.float32), jax.random.key(2))
        assert 0 < stats["wall_entropy"] < stats["entropy"] + 1 and stats["entropy_coef"] == 0.01
        assert (learner.ent_coef > 0.01) == rises and learner.ent_coef >= 0.01
    # the maze entropy is the sequence entropy: a batch of identical sequences would score 0
    assert league.seq_entropy(batch) > 0


def test_diversity_counts_repeats():
    walls, entrance, exit = random_states(4, 40, seed=2)
    d = designer.diversity(walls, entrance, exit)
    assert d["distinct"] == 1.0 and d["eff_mazes"] == float("inf")
    rep = designer.diversity(np.repeat(walls[:4], 10, axis=0), np.repeat(entrance[:4], 10), np.repeat(exit[:4], 10))
    assert rep["distinct"] == pytest.approx(0.1)
    assert rep["eff_mazes"] == pytest.approx(4 * 780 / (4 * 45 * 4), rel=0.5)  # ≈ 4 equally likely mazes


def test_pfsp_weights():
    w = league.pfsp([0.9, 0.1, None], [1, 1, 1], power=2, floor=0.1, hard_is_high=True)
    assert w.sum() == pytest.approx(1) and w[0] > w[2] > w[1]
    w = league.pfsp([0.9, 0.1], [1, 1], power=2, floor=0.1, hard_is_high=False)
    assert w[1] > w[0]
    assert league.opponent_baseline(np.array([1, 3, 10, 12]), np.array([0, 0, 1, 1])).tolist() == [5.5, 7.5, 5.5, 7.5]


def tiny_league(tmp_path, **kw) -> league.Config:
    base = dict(
        size=4, walls="6-7", channels=8, blocks=2, d_channels=8, d_blocks=2, envs=16, horizon=8, epochs=1, minibatch=64,
        cheap_pool=32, refresh=8, designed_pool=16, adv_pool=16, adv_refresh=4, adv_candidates=4, workers=1,
        eval_mazes=16, expert_ref=0, eval_worst=2, d_eval_mazes=32, d_mazes=32, d_minibatch=64, d_epochs=1,
        self_pool=64, member_pool=32, member_refresh=8, gen_every=2, exploiters="inline", x_exploit_iters=1,
        d_exploit_iters=1, exploit_mazes=16, iterations=4, eval_every=2, save_minutes=0, max_members=1,
        pool_dir=str(tmp_path / "pools"), mix="uniform:0.5,hard:0.3,adversary:0.2",
    )
    return league.Config(**{**base, **kw})


def test_league_runs_resumes_and_plays(tmp_path, capsys):
    run = tmp_path / "league"
    cfg = tiny_league(tmp_path)
    t = league.train(cfg, run, log=lambda *_: None)
    assert t.it == 4 and t.gens == 2 and t.absorbed == [1, 2]
    state = json.loads((run / "league.json").read_text())
    names = {s["name"]: s for s in state["league"]["sources"]}
    assert {"self", "uniform", "hard", "adversary", "d:g2", "d:xg2"} <= set(names)
    assert not names["d:g1"]["active"] if "d:g1" in names else True  # pruned (max_members=1)
    assert {o["name"] for o in state["league"]["opponents"]} >= {"self", "p2", "hard", "x:g2", "x:xg2"}
    exploit_res = json.loads((run / "league" / "gen-0002" / "exploit.json").read_text())
    assert {"gain", "main_sampled", "exploiter_sampled"} <= set(exploit_res["explorer"]) and "gain" in exploit_res["designer"]
    records = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
    assert len(records) == 4 and "eval" in records[1] and "distinct" in records[1]["eval"]["designer"]
    assert all(np.isfinite(r["x"]["magnet_kl"]) and np.isfinite(r["d"]["magnet_kl"]) for r in records)
    assert sum(records[-1]["x_weights"].values()) == pytest.approx(1, abs=1e-3)

    with pytest.raises(FileExistsError):
        league.Trainer(cfg, run, log=lambda *_: None)
    cfg2, args = league.config_from_args(["--run", str(run), "--resume", "--iterations", "5"])
    assert cfg2 == tiny_league(tmp_path, iterations=5)
    t2 = league.train(cfg2, run, resume=True, log=lambda *_: None)
    assert t2.it == 5 and t2.gens == 2 and {s.name for s in t2.league.sources if s.active} >= {"d:g2", "d:xg2"}

    # the run's agents play in the evaluator
    bench.main(["--size", "4", "--deals", "8", "--pairs", "p3:hard", "--p3-run", str(run)])
    assert "p3 vs hard" in capsys.readouterr().out
    bench.main(["--size", "4", "--deals", "8", "--designers", "p3,hard,uniform", "--explorers", "hard,p3,p3-sampled", "--p3-run", str(run)])
    out = capsys.readouterr().out
    assert "p3" in out and "uniform" in out and "distinct" in out
    exploit.main(["--size", "4", "--rounds", "4", "--bots", "p3,hard", "--p3-run", str(run)])
    assert "p3" in capsys.readouterr().out
    from labyrinthe_rl import report

    report.main([str(run), "--every", "2"])
    out = capsys.readouterr().out
    assert "| generation |" in out and "| 4 |" in out and "d:xg2" in out


def test_worker_and_designer_vs(tmp_path):
    run = tmp_path / "league"
    cfg = tiny_league(tmp_path, exploiters="worker", iterations=2)
    league.train(cfg, run, log=lambda *_: None)
    assert (run / "league" / "gen-0001" / "ready").exists() and (run / "league" / "STOP").exists()
    assert not (run / "league" / "gen-0001" / "exploit.json").exists()
    # the worker exploits the new generation (the main process's STOP is older than the worker: ignored)
    done = []

    def log(msg):
        done.append(msg)
        if (run / "league" / "gen-0001" / "exploit.json").exists():
            (run / "league" / "STOP").write_text("")

    league.worker(cfg, run, log=log)
    assert (run / "league" / "gen-0001" / "exploit.json").exists()

    vs = tmp_path / "vs"
    league.designer_vs(league.Config(**{**vars(cfg), "iterations": 2, "eval_every": 2}), vs, "p2", log=lambda *_: None)
    params, meta, _ = learned.load(vs / "designer.npz")
    assert meta["kind"] == "designer" and meta["iteration"] == 2
    rec = [json.loads(line) for line in (vs / "log.jsonl").read_text().splitlines()]
    assert "target_greedy" in rec[-1]["eval"]


def test_kaggle_jobs_parse():
    """Every command drive.py would run on Kaggle parses with its module's own options."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
    import drive
    from labyrinthe_rl import calibrate, train_explorer

    class A:
        net, precision, run, resume, explorer_init, designer_init, extra = "64x6", "fp32", "league-1", False, "runs/stage0/checkpoint.npz", "", ["--gen-every", "200"]

    for name in ("calibrate", "calibrate-alone", "stage0", "league"):
        for cmd in (c for job in drive.jobs(name, A) for c in job.get("cmds", [job.get("cmd")])):
            cmd = [c.replace("{hours}", "1.5") for c in cmd]
            module, argv = cmd[2], cmd[3:]
            if module == "labyrinthe_rl.calibrate":
                assert calibrate.parser().parse_args(argv).iterations == 3
            elif module == "labyrinthe_rl.train_explorer":
                cfg, _ = train_explorer.config_from_args(argv)
                assert cfg.hours == 1.5 and cfg.channels == 64
            else:
                cfg, args = league.config_from_args(argv)
                assert cfg.hours == 1.5
    assert "__SPEC__" in (Path(drive.__file__).parent / "kernel.py").read_text()
