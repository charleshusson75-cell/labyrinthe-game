# rl/: the self-play training stack

Python + JAX tooling for a self-improving Labyrinthe bot ([design](../docs/RL_DESIGN.md), decisions
D15–D22 in [DECISIONS.md](../docs/DECISIONS.md)). This is **phase P1**: the game engine, encodings,
maze samplers, today's bots and the evaluator. Nothing learns yet; P2 (explorer learning) builds on
these pieces.

The JavaScript game stays the reference. `scripts/export-fixtures.mjs` records what it does, and the
tests here must reproduce it exactly.

## Setup

Python 3.12, CPU only (no GPU needed for 4×4/5×5). Node.js (≥ 18, as for the game) too: it
exports the fixtures, and the evaluator runs the expert bot in it.

```bash
python3.12 -m venv rl/.venv
rl/.venv/bin/pip install -e "rl[test]"
```

## Commands

Run from the repository root.

```bash
rl/.venv/bin/pytest rl
```

```bash
rl/.venv/bin/python -m labyrinthe_rl.bench --size 5 --deals 1000
```

```bash
rl/.venv/bin/python -m labyrinthe_rl.bench --size 5 --deals 1000 --pairs expert:hard
```

```bash
rl/.venv/bin/python -m labyrinthe_rl.throughput --sizes 5,10
```

```bash
node scripts/export-fixtures.mjs
```

- `pytest rl`: parity with the JS engine plus unit tests (about a minute).
- `bench`: duplicate-format evaluation of two bots. Options: `--pairs hard:normal,...`,
  `--mazes bots|random|uniform`, `--walls`, `--seed`. Bots: `easy`, `normal`, `hard`, and `expert`
  (P2's bar, run by Node; `--particles`, default 200).
- `throughput`: batched explorer steps/s on this machine. Options: `--obs` also computes the lattice
  observation, `--devices 8` splits the batch over 8 CPU devices.
- `export-fixtures.mjs`: regenerates `rl/tests/fixtures/` after a change to the JS rules or bots;
  `--check` (run in CI) fails if they're out of date.

## What's inside (`labyrinthe_rl/`)

| module | what |
|---|---|
| `geometry.py` | cells and edges, numbered exactly like `src/game/grid.js` `allEdges()`: row-major over cells, right edge then down edge |
| `engine.py` | the rules in JAX, one game per call, `jit`/`vmap`-ready, any size 2–10 (the game uses 4–10). `init`, `step(state, edge)`, `probe(state, cell, dir)`, `legal_mask`, `reachable`, `maze_valid`, plus host helpers (`probe_error` returns the JS error codes) |
| `lattice.py` | the (2N+1)×(2N+1) image of D17: observation planes, belief target and mask, action ↔ side-pixel maps, legal mask as an image |
| `mazes.py` | `random_maze` (the JS `randomMaze`, identical draw for draw with a `SeededRng`), `uniform_batch` (uniform among all valid mazes, by rejection), validity and issue codes |
| `jsrandom.py` | bit-exact port of `src/game/random.js` (`seededRng`, `randInt`, `pick`, `shuffle`) |
| `baselines.py` | today's bots ported line by line from `src/bot/brain.js` (planner explorer and easy/normal/hard designers). Same moves as the JS for the same seed |
| `planner.py` | the same bots batched in JAX (same decisions, JAX randomness): `choose`, `explore_mazes` (a work queue of games), `design` |
| `bench.py` | duplicate-format evaluator |
| `expert.py` | the expert explorer (`src/bot/expert.js`, D21) as a bench bot, run by Node through `scripts/expert-solve.mjs`. Not ported: see below |
| `throughput.py` | steps/s benchmark |
| `env.py` | a Gymnasium wrapper (`Labyrinthe-Explorer-v0`) for standard tooling. Training will use the batched functions directly |

### Engine state

`State` holds, per game:
- `walls` (bool[E], the hidden maze), `entrance`, `exit`;
- `known_open`, `known_wall` (bool[E]), `reached` (bool[N²]);
- `reached_at`: when each cell was reached. That is the order JS lists cells in, which the
  planner's tie-breaks need.
- `hits`, `moves`, `done`.

A probe is legal when the side touches a reached cell, is still unknown, and the exit isn't reached
yet. Otherwise `step` returns `ILLEGAL` and changes nothing. `probe(cell, dir)` adds `probeError`'s
own rule that `cell` itself must be reached.

### Observation planes (`lattice.observe`)

Shape `float32[2N+1, 2N+1, 10]`. Cells sit at (odd, odd), sides at (odd, even)/(even, odd), corners
at (even, even).

| plane | content |
|---|---|
| `OPEN` | known open sides |
| `WALL` | known walls, plus the outer border (always closed) |
| `UNKNOWN` | inner sides not probed yet |
| `REACHED`, `ENTRANCE`, `EXIT` | cells |
| `CELL` | every cell pixel: tells cells from corners, and the board from zero padding |
| `W_TOTAL`, `W_FOUND` | constant planes: walls in the maze / inner sides, walls hit / inner sides |
| `SIZE` | constant plane: N / 10 |

The belief head's target is `true_maze` (1 on inner walls), trained on `belief_mask` (the unknown
sides). An action is an edge, and `action_pixels`, `pixel_actions` and `actions_from_lattice` convert
between edges and side pixels.

### Evaluation (`bench`)

Each deal is a maze pair (a, b), designed by bots A and B or dealt by a sampler. Both bots explore
both mazes, so each plays both seats on the same pair:
- **Score (dup)** compares explorers on identical mazes. Each seat is worth 1 to whoever hits fewer
  walls, ½ each when equal. That's the starter-independent value of a round in which each player
  starts once.
- **Score (round)** is the same on the round as played: A explores b, B explores a.
- **Explorer edge** and **designer edge** split the difference in walls hit between the two roles.

### The expert bot (`expert.py`)

The expert explorer is not ported: `bench` runs `src/bot/expert.js` itself, in Node ([D22](../docs/DECISIONS.md)).
`scripts/expert-solve.mjs` reads mazes as JSON on stdin and writes the walls hit on each:
- a fixed particle count (`--particles`, default 200) replaces the 150 ms time budget, so results
  don't depend on the machine;
- each maze's seed comes from the bench's JAX key, and each maze gets a fresh copy of expert.js. Its
  results don't depend on which mazes a Node process explored before;
- one Node process per CPU core; about 0.16 s per 5×5 exploration on one core.

`expert.explore_mazes(walls, entrance, exit, key, particles=200, workers=None, options=None)` takes
the same stacked mazes as the planner. `options` go to expert.js as they are, e.g.
`{"search": "greedy"}`.

## Parity with the JS engine

`rl/tests/fixtures/` holds 203 JSON files covering sizes 4–10:
- 154 probe sequences on `randomMaze` and uniform mazes: 6,720 probes, including 1,238 illegal
  ones (not reached, border, known side, invalid, after the end). Every answer, newly reached cell,
  hit count, move count and end flag is recorded, plus 1,135 `rankProbes` snapshots.
- `randomMaze` outputs for many seeds and budgets.
- `mazeIssues` on crafted mazes.
- 84 full bot explorations (4,080 probes).
- 21 designer choices.

The tests check:
- the JAX engine (both `probe` and `step`);
- the ported planner and designers: same lists, scores and moves as JS;
- the JAX planner: same scores, same best probes and same first four for every snapshot.

## Measured (P1, 2026-10-22, Apple M1, CPU, JAX 0.12)

Batched explorer steps/s (`throughput`). One step is one probe in one game; games restart from a
maze pool as they end. One device is one CPU core; 8 devices split the batch (`--devices 8`). Best
batch size kept (1,024 on one device, 8,192 on eight).

| board | policy | 1 device | 8 devices |
|---|---|---|---|
| 5×5 | uniform among legal probes | 2.5 M | 4.7 M |
| 5×5 | the hard planner | 0.46 M | 1.5 M |
| 5×5 | uniform + lattice observation every step | 1.2 M | |
| 5×5 | the hard planner + lattice observation | 0.40 M | |
| 10×10 | uniform among legal probes | 0.59 M | 1.9 M |
| 10×10 | the hard planner | 0.11 M | 0.48 M |
| 10×10 | uniform + lattice observation every step | 0.32 M | |
| 10×10 | the hard planner + lattice observation | 0.089 M | |

**Go/no-go met:** the target was ≥ 10⁵ explorer steps/s on 5×5.

Today's bots in duplicate format (`bench --deals 1000`, 5×5, 12 walls, mazes designed by the bots;
± is one standard error):

| pairing | hits A | hits B | explorer edge | designer edge | score (dup) | score (round) |
|---|---|---|---|---|---|---|
| hard vs easy | 9.71 (0.81 W) | 11.25 (0.94 W) | +1.54 ± 0.04 | +2.39 ± 0.07 | 68% ± 1 | 96% ± 1 |
| hard vs normal | 11.04 (0.92 W) | 11.38 (0.95 W) | +0.33 ± 0.02 | +0.81 ± 0.04 | 59% ± 1 | 79% ± 1 |
| normal vs easy | 9.91 (0.83 W) | 11.17 (0.93 W) | +1.26 ± 0.04 | +1.68 ± 0.07 | 68% ± 1 | 88% ± 1 |
| hard vs hard | 11.56 (0.96 W) | 11.56 (0.96 W) | 0 | −0.03 ± 0.03 | 50% | 50% ± 1 |

The round scores match the JS benchmark (`scripts/benchmark.mjs`: hard beats normal 80%, normal
beats easy 86%), a check on the ports. The run takes about 15 s.

The expert, P2's bar (`--pairs expert:hard,expert:normal,expert:expert`, 1,000 deals):

| pairing | hits A | hits B | explorer edge | designer edge | score (dup) | score (round) |
|---|---|---|---|---|---|---|
| expert vs hard | 5.29 (0.44 W) | 11.54 (0.96 W) | +6.25 ± 0.05 | −0.01 ± 0.05 | 100% | 100% |
| expert vs normal | 5.31 (0.44 W) | 11.38 (0.95 W) | +6.06 ± 0.05 | +0.28 ± 0.05 | 99% | 100% |
| expert vs expert | 5.29 (0.44 W) | 5.29 (0.44 W) | 0 | −0.02 ± 0.09 | 50% | 48% ± 1 |

In the round as played the expert hits fewer walls than hard 99.7% of the time (equal 0.2%), as in
D21 (5.3 of 12 walls on hard's mazes, beats hard in 99% of rounds). The expert designs like hard, so
the designer edge against hard is ≈ 0. Expert vs hard takes 2 minutes.

## Notes for P2

- **On CPU, XLA is fast at elementwise work and slow at gathers inside loops.**
  - The planner's BFS runs on one bitmask per board row: 4× faster on 5×5, 9× on 10×10.
  - The lattice is built by interleaving reshapes: 14× faster than painting pixels by gathers, inside
    a rollout loop.
- **Batched rollouts are a work queue** (`planner.explore_mazes`): a slot whose game ends loads the
  next maze, instead of the batch waiting for its slowest game (2.5× faster on 5×5).
- **Swapping seats on the same maze pair cancels the designers exactly.** So "score (dup)" compares
  explorers only; compare designers with the designer edge and the round score.
- **The "expert" explorer** (`src/bot/expert.js`, D21) is P2's bar. `bench` plays it through Node
  (D22), which is fine for evaluation but too slow for training loops.
