import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from labyrinthe_rl import bench, learned, network, train_explorer
from labyrinthe_rl.engine import init_batch, legal_batch
from labyrinthe_rl.geometry import n_edges
from labyrinthe_rl.lattice import observe_batch
from labyrinthe_rl.mazes import sample_mazes


def tiny_config(tmp_path, **kw) -> train_explorer.Config:
    base = dict(
        size=4, walls="7", channels=8, blocks=2, envs=16, horizon=8, epochs=1, minibatch=64,
        cheap_pool=32, refresh=8, designed_pool=16, adv_pool=16, adv_refresh=4, adv_candidates=4,
        eval_mazes=16, expert_ref=0, eval_worst=2, workers=1,
        iterations=2, eval_every=2, save_every=1, snapshot_every=2, pool_dir=str(tmp_path / "pools"),
    )
    return train_explorer.Config(**{**base, **kw})


def start(size, walls, count, seed):
    walls, entrance, exit = sample_mazes("uniform", size, walls, count, seed=seed)
    return walls, init_batch(jnp.asarray(walls), jnp.asarray(entrance), jnp.asarray(exit))


@pytest.mark.parametrize("plan", [False, True])
def test_network_on_any_board(plan):
    params = network.init(jax.random.key(0), network.NetConfig(8, 2, plan=plan))
    apply = jax.jit(network.apply)
    for size in (4, 5):
        _, states = start(size, 6, 3, seed=size)
        legal = legal_batch(states)
        out = apply(params, observe_batch(states), legal)
        assert out.logits.shape == out.belief.shape == (3, n_edges(size))
        assert bool(jnp.all((out.logits > network.NEG / 2) == legal))
        # the value sums a probability per unknown side: between 0 and the number of unknown sides
        assert bool(jnp.all((out.value >= 0) & (out.value <= n_edges(size))))


def test_plan_scores_follow_the_belief():
    walls, states = start(5, 12, 64, seed=3)
    obs, legal = observe_batch(states), legal_batch(states)
    # a belief that knows the maze: some first probe leads to the exit without a wall
    q = network.plan_scores(obs, jnp.asarray(walls, jnp.float32))
    assert np.allclose(jnp.min(jnp.where(legal, q, np.inf), axis=1), 0)
    # each probe's score is its own cost plus the cheapest rest of the way, so never below its cost
    p = jnp.full(walls.shape, 0.3)
    q = network.plan_scores(obs, p)
    assert bool(jnp.all(jnp.where(legal, q >= 0.3 - 1e-6, True)))


def test_checkpoint_round_trip(tmp_path):
    cfg = network.NetConfig(8, 2)
    params = network.init(jax.random.key(3), cfg)
    opt = train_explorer.make_optimizer(train_explorer.Config())
    opt_state = opt.init(params)
    meta = {"net": {"channels": 8, "blocks": 2, "dilations": list(cfg.dilations)}, "iteration": 7}
    path = tmp_path / "c.npz"
    learned.save(path, params, meta, opt_state)
    loaded, meta2, arrays = learned.load(path)
    assert meta2 == meta
    assert all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(loaded)))
    state2 = learned.unflatten(opt.init(loaded), arrays, "opt")
    assert all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(opt_state), jax.tree.leaves(state2)))


def test_train_resume_and_bench(tmp_path, capsys):
    run = tmp_path / "run"
    cfg = tiny_config(tmp_path)
    meta = train_explorer.train(cfg, run, log=lambda *_: None)
    assert meta["iteration"] == 2 and meta["env_steps"] == 2 * 16 * 8
    assert set(meta["eval"]) >= set(train_explorer.EVAL_KINDS)
    assert (run / "snap-000002.npz").exists() and (run / "refs.json").exists()
    records = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
    assert len(records) == 2 and all(np.isfinite(r["pg"]) and np.isfinite(r["belief"]) for r in records)
    # the adversary keeps the worst of its candidates for the current policy
    assert all(r["adv_pick"] >= r["adv_candidates"] for r in records)
    assert "worst32" in meta["eval"] and "worst32" in json.loads((run / "refs.json").read_text())

    with pytest.raises(FileExistsError):
        train_explorer.train(cfg, run, log=lambda *_: None)

    before, _, _ = learned.load(run / "checkpoint.npz")
    resumed, args = train_explorer.config_from_args(["--run", str(run), "--resume", "--iterations", "3"])
    assert resumed == tiny_config(tmp_path, iterations=3)
    meta = train_explorer.train(resumed, run, resume=True, log=lambda *_: None)
    assert meta["iteration"] == 3 and meta["env_steps"] == 3 * 16 * 8
    after, _, _ = learned.load(run / "checkpoint.npz")
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after)))
    with pytest.raises(SystemExit):
        train_explorer.config_from_args(["--run", str(run), "--resume", "--envs", "8"])

    # the checkpoint plays in the evaluator
    bench.main(["--size", "4", "--deals", "8", "--pairs", "learned:hard", "--mazes", "uniform", "--checkpoint", str(run / "checkpoint.npz")])
    assert "learned vs hard" in capsys.readouterr().out


def test_random_explorer_and_greedy_rollouts():
    walls, entrance, exit = sample_mazes("uniform", 4, 7, 40, seed=2)
    hits = learned.random_mazes(walls, entrance, exit, jax.random.key(0))
    assert hits.shape == (40,) and hits.min() >= 0 and hits.max() <= 7
    params = network.init(jax.random.key(1), network.NetConfig(8, 2))
    greedy = learned.explore_mazes(params, walls, entrance, exit, jax.random.key(0), slots=16)
    again = learned.explore_mazes(params, walls, entrance, exit, jax.random.key(5), slots=7)
    assert np.array_equal(greedy, again)  # greedy play depends on the maze only


def test_wall_band_and_mix():
    assert train_explorer.wall_band(train_explorer.Config(size=5)) == [10, 11, 12, 13, 14]
    assert train_explorer.wall_band(train_explorer.Config(size=4, walls="7")) == [7]
    with pytest.raises(ValueError):
        train_explorer.wall_band(train_explorer.Config(size=4, walls="3-40"))
    mix = train_explorer.parse_mix("uniform:1,hard:3")
    assert mix == {"uniform": 0.25, "hard": 0.75}
    assert set(train_explorer.parse_mix(train_explorer.Config().mix)) == {"uniform", "random", "normal", "hard", "adversary"}
    with pytest.raises(ValueError):
        train_explorer.parse_mix("nope:1")


def test_adversarial_pick():
    from labyrinthe_rl import exploit
    from labyrinthe_rl.engine import valid_batch

    mazes = exploit.candidates(4, 7, rounds=3, k=4, seed=0)
    assert mazes[0].shape == (12, n_edges(4))
    assert bool(jnp.all(valid_batch(*(jnp.asarray(x) for x in mazes), 7)))
    r = exploit.adversarial(bench.BOTS["hard"], mazes, 3, 4, jax.random.key(0))
    assert r["worst"][0] >= r["mean"][0]
