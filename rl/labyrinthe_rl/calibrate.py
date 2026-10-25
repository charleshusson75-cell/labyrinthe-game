"""Training throughput on this machine (P3 calibration, DECISIONS.md D24): how many probes per second
the explorer's PPO loop (train_explorer.py, P2's recipe) trains, per network size.

    python -m labyrinthe_rl.calibrate [--size 5] [--nets 32x4,64x6,96x8] [--envs 1024]
                                      [--minibatch 2048] [--precision fp32,fp16] [--iterations 4]
                                      [--designer] [--out calibration.json]

One iteration is what train_explorer does: the adversary's picks, `envs` games × `horizon` probes
with the sampled policy, then the PPO epochs. Probes/s = envs × horizon / iteration time, measured
after two warm-up iterations (compilation). The maze sources don't change the cost, so the pool is
made of cheap mazes (uniform, randomMaze) and the adversary only. Every (net, envs, minibatch,
precision) combination is measured; `--designer` also measures the designer's side of a league
iteration (league.py): generate mazes, explore them with the explorer, PPO on the designer.

fp16 runs the convolutions in half precision with float32 accumulation (network.COMPUTE_DTYPE), which
uses the tensor cores of a T4. Everything else stays float32.
"""

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import designer, learned, network, train_explorer
from .geometry import default_wall_budget


def explorer_rate(size: int, channels: int, blocks: int, envs: int, minibatch: int, iterations: int, seed: int = 0) -> dict:
    cfg = train_explorer.Config(
        size=size, mix="uniform:0.3,random:0.5,adversary:0.2", channels=channels, blocks=blocks, envs=envs,
        minibatch=minibatch, cheap_pool=8192, refresh=1024, adv_pool=2048, seed=seed,
    )
    dil = network.NetConfig().dilations
    key = jax.random.key(seed)
    params = network.init(key, network.NetConfig(channels, blocks, dil))
    opt, update = train_explorer.make_update(cfg, dil)
    opt_state = opt.init(params)
    pool = train_explorer.Pool(cfg)
    envs_state = train_explorer.Envs(*train_explorer.draw(pool.device(), key, cfg.envs))
    times = []
    for it in range(iterations + 2):
        t0 = time.perf_counter()
        pool.refresh(it, params, dil)
        key, k1, k2 = jax.random.split(key, 3)
        envs_state, batch, last_value = train_explorer.rollout(params, envs_state, pool.device(), k1, cfg.horizon, dil)
        params, opt_state, stats = update(params, opt_state, batch, last_value, k2)
        float(stats["pg"])  # wait for the device
        times.append(time.perf_counter() - t0)
    best, mean = min(times[2:]), float(np.mean(times[2:]))
    probes = cfg.envs * cfg.horizon
    return {"probes_per_s": probes / mean, "best_probes_per_s": probes / best, "iteration_s": mean,
            "compile_s": times[0] - mean, "params": network.n_params(params)}


def designer_rate(size: int, channels: int, blocks: int, mazes: int, minibatch: int, iterations: int, seed: int = 0) -> dict:
    dil = network.NetConfig().dilations
    key = jax.random.key(seed)
    dparams = designer.init_net(key, network.NetConfig(channels, blocks, dil))
    xparams = network.init(jax.random.fold_in(key, 1), network.NetConfig(channels, blocks, dil))
    pcfg = designer.PPOConfig(minibatch=minibatch)
    opt, update = designer.make_update(pcfg, dil)
    opt_state = opt.init(dparams)
    w = default_wall_budget(size)
    budgets = jnp.asarray(np.random.default_rng(seed).integers(w - 2, w + 3, mazes), jnp.int32)
    n_steps = designer.steps(w + 2)
    times = []
    for it in range(iterations + 2):
        t0 = time.perf_counter()
        key, k1, k2, k3 = jax.random.split(key, 4)
        final, batch = designer.rollout(dparams, budgets, k1, size, n_steps, dil)
        hits = learned.explore_mazes(xparams, final.walls, final.entrance, final.exit, k2, dil, greedy=False)
        dparams, opt_state, stats = update(dparams, opt_state, batch, hits, k3)
        float(stats["pg"])
        times.append(time.perf_counter() - t0)
    mean = float(np.mean(times[2:]))
    return {"mazes_per_s": mazes / mean, "iteration_s": mean, "compile_s": times[0] - mean}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=5)
    ap.add_argument("--nets", default="32x4,64x6,96x8")
    ap.add_argument("--envs", default="1024")
    ap.add_argument("--minibatch", default="2048")
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--designer", action="store_true", help="also measure the designer's side of a league iteration")
    ap.add_argument("--d-mazes", type=int, default=1024)
    ap.add_argument("--out", default=None, help="append the results (JSON lines) to this file")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    dev = jax.devices()[0]
    print(f"{dev.platform} {dev.device_kind} ×{len(jax.devices())}, JAX {jax.__version__}, {args.size}×{args.size}\n")
    print(f"{'net':<7} {'prec':<5} {'envs':>5} {'minibatch':>9} {'params':>9} {'probes/s':>10} {'iter s':>7} {'compile s':>9}")
    results = []
    for prec in args.precision.split(","):
        network.COMPUTE_DTYPE = {"fp32": jnp.float32, "fp16": jnp.float16, "bf16": jnp.bfloat16}[prec]
        for net in args.nets.split(","):
            c, b = map(int, net.split("x"))
            for envs in map(int, args.envs.split(",")):
                for mb in map(int, args.minibatch.split(",")):
                    try:
                        r = explorer_rate(args.size, c, b, envs, mb, args.iterations)
                    except Exception as e:  # e.g. out of device memory: record it and go on
                        print(f"{net:<7} {prec:<5} {envs:>5} {mb:>9}  failed: {str(e).splitlines()[0][:100]}", flush=True)
                        continue
                    r.update(kind="explorer", net=net, precision=prec, envs=envs, minibatch=mb, device=dev.device_kind, size=args.size)
                    results.append(r)
                    print(f"{net:<7} {prec:<5} {envs:>5} {mb:>9} {r['params']:>9,} {r['probes_per_s']:>10,.0f} {r['iteration_s']:>7.2f} {r['compile_s']:>9.1f}", flush=True)
                    if args.out:
                        with open(args.out, "a") as f:
                            f.write(json.dumps(r) + "\n")
            if args.designer:
                d = designer_rate(args.size, c, b, args.d_mazes, int(args.minibatch.split(",")[0]), args.iterations)
                d.update(kind="designer", net=net, precision=prec, mazes=args.d_mazes, device=dev.device_kind, size=args.size)
                results.append(d)
                print(f"{net:<7} {prec:<5} designer: {d['mazes_per_s']:,.0f} mazes/s ({args.d_mazes} per iteration, {d['iteration_s']:.2f} s, compile {d['compile_s']:.0f} s)", flush=True)
                if args.out:
                    with open(args.out, "a") as f:
                        f.write(json.dumps(d) + "\n")
    return results


if __name__ == "__main__":
    main()
