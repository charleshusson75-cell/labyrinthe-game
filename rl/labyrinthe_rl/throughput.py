"""Batched explorer throughput on this machine (P1 go/no-go: ≥ 10⁵ steps/s on 5×5, CPU).

    python -m labyrinthe_rl.throughput [--sizes 5,10] [--batch 1024,4096,16384] [--steps 200]
                                       [--policy random,planner] [--obs] [--devices 1]

A step is one probe in one game. Every slot plays games back to back: when its game ends it starts
the next maze from a pool (auto-reset), as a training loop does. Policies: 'random' (uniform among
legal probes: the engine's own cost) or 'planner' (today's hard bot, planner.py). With --obs the
lattice observation (lattice.py) is also computed after every step. With --devices K the batch is
split over K CPU devices (the script sets XLA's host device count itself).
"""

import argparse
import os
import sys
import time


def _parse(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="5,10")
    ap.add_argument("--batch", default="1024,4096,16384")
    ap.add_argument("--steps", type=int, default=200, help="steps per timed run")
    ap.add_argument("--policy", default="random,planner")
    ap.add_argument("--obs", action="store_true", help="also compute the lattice observation every step")
    ap.add_argument("--devices", type=int, default=1, help="split the batch over this many CPU devices")
    ap.add_argument("--repeats", type=int, default=3, help="timed runs per setting (the best is kept)")
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse(sys.argv[1:] if argv is None else argv)
    if args.devices > 1:  # must happen before JAX starts
        os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count={args.devices}"

    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec

    from . import engine, lattice, planner
    from .geometry import default_wall_budget
    from .mazes import sample_mazes

    def make_run(policy: str, with_obs: bool, steps: int):
        @jax.jit
        def run(states, pool, cursor, key):
            b = states.done.shape[0]
            total = pool[0].shape[0]

            def one(carry, _):
                s, cursor, key, acc = carry
                key, k = jax.random.split(key)
                u = jax.random.uniform(k, (b, 2))
                if policy == "random":
                    edges = jax.vmap(lambda st, x: planner.pick_uniform(engine.legal_mask(st), x))(s, u[:, 0])
                else:
                    edges = jax.vmap(planner.choose, in_axes=(0, 0, None))(s, u, planner.HARD)
                s, _ = jax.vmap(engine.step)(s, edges)
                if with_obs:  # carried to the next step, so it is really written out (not fused away)
                    acc = jax.vmap(lattice.observe)(s)
                ended = s.done
                hits = jnp.sum(jnp.where(ended, s.hits, 0))
                idx = (cursor + jnp.cumsum(ended) - 1) % total
                fresh = jax.vmap(engine.init)(pool[0][idx], pool[1][idx], pool[2][idx])
                s = jax.tree.map(lambda f, o: jnp.where(ended.reshape((-1,) + (1,) * (o.ndim - 1)), f, o), fresh, s)
                return (s, cursor + jnp.sum(ended), key, acc), (jnp.sum(ended), hits)

            obs0 = jax.vmap(lattice.observe)(states) if with_obs else jnp.float32(0)
            (s, cursor, key, last), (ends, hits) = jax.lax.scan(one, (states, cursor, key, obs0), None, length=steps)
            return s, cursor, jnp.sum(last), jnp.sum(ends), jnp.sum(hits)

        return run

    devices = jax.devices()[: args.devices]
    shard = NamedSharding(jax.sharding.Mesh(devices, ("b",)), PartitionSpec("b")) if len(devices) > 1 else None
    print(f"{jax.devices()[0].device_kind} ×{len(devices)}, JAX {jax.__version__}, {args.steps} steps per run, "
          f"observation {'on' if args.obs else 'off'}\n")
    print(f"{'board':<7} {'policy':<8} {'batch':>6} {'steps/s':>12} {'games/s':>10} {'hits/game':>10}")
    for size in map(int, args.sizes.split(",")):
        walls = default_wall_budget(size)
        pool = tuple(jnp.asarray(x) for x in sample_mazes("random", size, walls, 8192, seed=size))
        for policy in args.policy.split(","):
            run = make_run(policy, args.obs, args.steps)
            for b in map(int, args.batch.split(",")):
                idx = jnp.arange(b) % pool[0].shape[0]
                states = jax.vmap(engine.init)(pool[0][idx], pool[1][idx], pool[2][idx])
                if shard is not None:
                    states = jax.device_put(states, shard)
                key = jax.random.key(0)
                out = run(states, pool, jnp.int32(b), key)  # compile + warm up
                jax.block_until_ready(out)
                best = float("inf")
                for r in range(args.repeats):
                    t0 = time.perf_counter()
                    out = run(out[0], pool, out[1], jax.random.fold_in(key, r))
                    jax.block_until_ready(out)
                    best = min(best, time.perf_counter() - t0)
                games, hits = int(out[3]), int(out[4])
                print(f"{size}×{size:<5} {policy:<8} {b:>6} {b * args.steps / best:>12,.0f} {games / best:>10,.0f}"
                      f" {hits / max(games, 1):>10.2f}")


if __name__ == "__main__":
    main()
