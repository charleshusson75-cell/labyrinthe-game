"""Summary of a league run (league.py) as Markdown tables: learning curves, the designer's variety,
the exploiters' gains per generation, and the league at the end.

    python -m labyrinthe_rl.report RUN [--every 250]

Columns of the curve (evaluations every `eval_every` iterations, one row per `--every`):
    explorer   greedy walls hit on P2's held-out mazes (uniform, randomMaze, normal, hard) at the
               default budget, and the worst of 32 candidates (exploit.py's pick)
    designer   walls hit on the main designer's mazes (default budget) by the main explorer
               (greedy / sampled), P2's explorer and the hard planner; share of distinct mazes,
               E/S pairs, mean E→S distance; maze entropy (nats per maze, estimated on the training
               batch) and entropy per wall decision
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load(run) -> tuple:
    run = Path(run)
    records = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
    state = json.loads((run / "league.json").read_text())
    config = json.loads((run / "config.json").read_text())
    exploits = []
    for p in sorted((run / "league").glob("gen-*/exploit.json")):
        exploits.append(json.loads(p.read_text()))
    return records, state, config, exploits


def curve(records: list, every: int) -> str:
    evals = [r for r in records if "eval" in r]
    rows = [r for r in evals if r["iteration"] % every == 0 or r is evals[0]]
    if evals and rows[-1] is not evals[-1]:
        rows.append(evals[-1])
    lines = [
        "| iteration | probes | mazes | uniform | random | normal | hard | worst of 32 | D: main greedy / sampled | D: P2 | D: hard | distinct | E/S pairs | E→S | H (nats) | H / wall |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    by_it = {r["iteration"]: r for r in records}
    for r in rows:
        x, d = r["eval"]["explorer"], r["eval"]["designer"]
        # the designer's entropy around this iteration (training batches), averaged over 10 iterations
        near = [by_it[i]["d"] for i in range(r["iteration"] - 9, r["iteration"] + 1) if i in by_it]
        h = np.mean([n["seq_entropy"] for n in near])
        hw = np.mean([n["wall_entropy"] for n in near])
        lines.append(
            f"| {r['iteration']} | {r['env_steps'] / 1e6:.1f} M | {r['d_mazes'] / 1e6:.2f} M | {x['uniform']:.2f} | {x['random']:.2f} | "
            f"{x['normal']:.2f} | {x['hard']:.2f} | {x.get('worst32', float('nan')):.2f} | {d['main_greedy']:.2f} / {d['main_sampled']:.2f} | "
            f"{d.get('p2', float('nan')):.2f} | {d['hard']:.2f} | {100 * d['distinct']:.0f}% | {d['es_pairs']} | {d['path']:.1f} | {h:.1f} | {hw:.2f} |"
        )
    return "\n".join(lines)


def exploiter_table(exploits: list) -> str:
    lines = [
        "| generation | explorer exploiter: walls, main → exploiter (gain) | designer exploiter: walls, main → exploiter (gain) | exploitability |",
        "|---|---|---|---|",
    ]
    for e in exploits:
        x, d = e["explorer"], e["designer"]
        lines.append(
            f"| {e['generation']} | {x['main_sampled']:.2f} → {x['exploiter_sampled']:.2f} ({x['gain']:+.2f}) | "
            f"{d['main_sampled']:.2f} → {d['exploiter_sampled']:.2f} ({d['gain']:+.2f}) | {e['exploitability']:+.2f} |"
        )
    return "\n".join(lines)


def league_table(state: dict) -> str:
    lines = ["| side | member | kind | x (walls hit / W of the main agent) |", "|---|---|---|---|"]
    for side, key in (("designers (explorer's games)", "sources"), ("explorers (designer's mazes)", "opponents")):
        for m in state["league"][key]:
            if m["active"]:
                x = "—" if m["x"] is None else f"{m['x']:.2f}"
                lines.append(f"| {side} | {m['name']} | {m['kind']} | {x} |")
    return "\n".join(lines)


def summary(records: list, state: dict, config: dict) -> str:
    last = records[-1]
    probes = np.median([r["probes_per_s"] for r in records])
    mazes = np.median([r["mazes_per_s"] for r in records])
    return (f"{config['size']}×{config['size']}, explorer {config['channels']}×{config['blocks']}, designer {config['d_channels']}×{config['d_blocks']}; "
            f"{last['iteration']} iterations, {last['env_steps'] / 1e6:.1f} M probes, {last['d_mazes'] / 1e6:.2f} M mazes, "
            f"{state['generations']} generations, {state['elapsed'] / 3600:.2f} h of training "
            f"(median {probes:,.0f} probes/s on the explorer's side, {mazes:,.0f} mazes/s on the designer's); "
            f"designer entropy coefficient now {state.get('d_entropy_coef', float('nan')):.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--every", type=int, default=250)
    args = ap.parse_args(argv)
    records, state, config, exploits = load(args.run)
    print(summary(records, state, config) + "\n")
    print(curve(records, args.every) + "\n")
    if exploits:
        print(exploiter_table(exploits) + "\n")
    print(league_table(state))


if __name__ == "__main__":
    main()
