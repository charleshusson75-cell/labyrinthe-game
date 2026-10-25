# rl/: the self-play training stack

Python + JAX tooling for a self-improving Labyrinthe bot ([design](../docs/RL_DESIGN.md), decisions
D15–D23 in [DECISIONS.md](../docs/DECISIONS.md)).
- **Phase P1:** the game engine, encodings, maze samplers, today's bots and the evaluator.
- **Phase P2:** the learned explorer. A convolutional network with a belief head, trained by PPO
  against a mix of fixed designers (`train_explorer`), and played in the evaluator as the `learned`
  bot.
- **Phase P3 (in progress):** both roles learn. A designer network (`designer`) and a league of
  current, frozen, exploiter and fixed agents (`league`), trained on Kaggle GPUs (`rl/kaggle/`),
  played in the evaluator as the `p3` bot.

The JavaScript game stays the reference. `scripts/export-fixtures.mjs` records what it does, and the
tests here must reproduce it exactly.

## Setup

Python 3.12, CPU only (no GPU needed for 4×4/5×5). Node.js (≥ 18, as for the game) too: it
exports the fixtures, and the evaluator runs the expert bot in it. Dependencies: JAX, NumPy,
Gymnasium, and optax (Adam) for training; the network itself is plain JAX.

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
rl/.venv/bin/python -m labyrinthe_rl.train_explorer --size 4 --run rl/runs/explorer-4x4
```

```bash
rl/.venv/bin/python -m labyrinthe_rl.train_explorer --run rl/runs/explorer-4x4 --resume --iterations 2000
```

```bash
rl/.venv/bin/python -m labyrinthe_rl.bench --size 5 --deals 1000 --pairs learned:hard,learned:expert
```

```bash
rl/.venv/bin/python -m labyrinthe_rl.exploit --size 5 --bots learned,hard,expert
```

```bash
rl/.venv/bin/python -m labyrinthe_rl.throughput --sizes 5,10
```

```bash
node scripts/export-fixtures.mjs
```

P3 (league self-play, [D25](../docs/DECISIONS.md)):

```bash
rl/.venv/bin/python -m labyrinthe_rl.league --run rl/runs/league-4x4 --size 4 --exploiters inline
```

```bash
rl/.venv/bin/python -m labyrinthe_rl.league --run rl/runs/league-4x4 --resume --iterations 2000
```

```bash
rl/.venv/bin/python -m labyrinthe_rl.bench --designers p3,hard,uniform --explorers hard,expert,learned,p3
```

```bash
rl/.venv/bin/python -m labyrinthe_rl.calibrate --nets 32x4,64x6 --envs 256 --minibatch 512
```

```bash
rl/.venv/bin/python rl/kaggle/drive.py push league --hours 10.8 --net 64x6
```

- `pytest rl`: parity with the JS engine plus unit tests (about a minute).
- `train_explorer`: trains the learned explorer (see below). It writes checkpoints and logs into
  `--run` (`rl/runs/` is git-ignored); `--resume` continues a run; `--init` starts from another
  run's parameters, of any board size. Every field of `Config` is an option (`--envs`, `--lr`,
  `--mix`, `--walls`, ...).
- `bench`: duplicate-format evaluation of two bots. Options: `--pairs hard:normal,...`,
  `--mazes bots|random|uniform`, `--walls`, `--seed`. Bots: `easy`, `normal`, `hard`, `expert`
  (P2's bar, run by Node; `--particles`, default 200), and `learned` (`--checkpoint`, by default the
  committed `rl/checkpoints/explorer-NxN.npz`).
- `throughput`: batched explorer steps/s on this machine. Options: `--obs` also computes the lattice
  observation, `--devices 8` splits the batch over 8 CPU devices.
- `export-fixtures.mjs`: regenerates `rl/tests/fixtures/` after a change to the JS rules or bots;
  `--check` (run in CI) fails if they're out of date.
- `league`: trains the designer and the explorer together (below). `--worker` trains the exploiters
  of a run's new generations in another process (the second GPU on Kaggle); `--designer-vs PATH|p2`
  trains a fresh designer against one frozen explorer.
- `bench --designers`: scores designers (walls hit by each of `--explorers` on their mazes, and how
  varied the mazes are). Bots `p3` / `p3-sampled` are P3's agents (`--p3-run DIR` or
  `--p3-explorer` / `--p3-designer`, default `rl/checkpoints/p3-*-NxN.npz`); `exploit --bots p3`.
- `calibrate`: training probes/s of the explorer's PPO loop per network size (and the designer's
  mazes/s with `--designer`); `--precision fp16` runs the convolutions in half precision.
- `rl/kaggle/drive.py`: Kaggle sessions from the Mac (`upload`, `push`, `status`, `fetch`, `quota`;
  see below).

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
| `network.py` | the explorer's network (D17, D23): a residual convolutional tower over the lattice, policy / belief / hit heads, any board size. Plain JAX, parameters in a dict |
| `train_explorer.py` | PPO training of the explorer against a mix of designers: maze pools, batched rollouts, GAE, the belief loss, held-out evaluation, checkpoints |
| `learned.py` | checkpoints (`save`, `load`), and batched explorations with a network as policy (`explore_mazes`, greedy by default; `Explorer` for the bench) |
| `exploit.py` | an adversarial pick (the worst of 32 fresh mazes for each explorer), a cheap proxy for exploitability |
| `designer.py` | the learned designer (P3, D25): its game (E, S, then walls one at a time, cut sides masked), observation planes, network, batched sampling, PPO pieces, maze diversity statistics |
| `league.py` | league self-play for both roles: PFSP over frozen snapshots, exploiters and anchors, magnet regularization, generations, exploiters (inline or a worker process), checkpoints |
| `calibrate.py` | training throughput of the PPO loops per network size and precision |
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

### Training the explorer (`train_explorer.py`, [D23](../docs/DECISIONS.md))

PPO on the explorer's side of a round, against fixed designers (P2: the designer doesn't learn yet).
- **Games:** `--envs` (256) games run in parallel. Each iteration plays `--horizon` (32) probes in
  every game with the sampled policy, then runs `--epochs` (3) PPO epochs over those 8,192 samples
  in minibatches of `--minibatch` (512). A game that ends restarts on a new maze at once.
- **Mazes:** each new game draws a designer from `--mix` (default
  `uniform:0.2,random:0.15,normal:0.15,hard:0.3,adversary:0.2`), then one of its mazes. The wall budget is drawn from a band around the game's default (`--walls`,
  default ± 2: 5–9 on 4×4, 10–14 on 5×5; D15); the network reads it from its input planes.
  - Uniform and `randomMaze` mazes are cheap: a pool of 16,384 each, of which 2,048 are replaced
    every iteration.
  - Designed mazes (`normal`, `hard`: the best of 6 or 30 random mazes against the hard explorer)
    are slow, ≈ 150 hard mazes/s on 5×5. 40,000 per designer are made once, on every CPU core, and
    cached in `rl/runs/pools/` for later runs.
  - `adversary` aims at the learner itself: every iteration, 128 of its 4,096 mazes are replaced
    by new picks, each the worst of 8 fresh randomMaze and uniform mazes (same budget) for the
    current greedy policy.
- **Learning signal:** reward −1 per wall hit, undiscounted, GAE(λ = 0.95).
- **Losses:**
  - PPO's clipped surrogate (clip 0.2) and an entropy bonus (0.01);
  - the value's squared error (× 0.05: the value sums one term per side, so its gradient is large);
  - the belief's cross-entropy against the true maze, on the unknown sides (× 1).
  - Adam, learning rate 3·10⁻⁴, gradients clipped to norm 1.
- **Evaluation:** every `--eval-every` (25) iterations, the greedy policy explores 512 held-out mazes
  from each designer (uniform, randomMaze, normal, hard) at the default budget. The log line puts
  three references next to it, measured once per run on the same mazes (`refs.json`): the hard
  explorer, a uniformly random explorer, and the expert (on the first 256). It also reports the
  worst of 32 candidates per round over 128 rounds (`--eval-worst`, see `exploit` below).
- **Outputs** in `--run`:
  - `checkpoint.npz`: parameters, Adam state, counters and random key, written every
    `--save-every` (25) iterations and on exit, Ctrl-C or SIGTERM included;
  - `snap-NNNNNN.npz`: parameters only, every `--snapshot-every` (100) iterations, for later
    leagues;
  - `log.jsonl`: one record per iteration (losses, walls hit per designer, evals);
  - `refs.json` and `config.json`.
- **Resume:** `--resume` reloads the checkpoint and keeps the run's config. Only `--iterations`,
  the eval / save periods and `--mix` (a curriculum) may change. The games in flight restart on new
  mazes.
- **Planner prior** (`--plan true`, experimental, off by default): the policy logits also get
  −w · (expected walls on the cheapest way to the exit through each probe), planned on the belief
  head's P(wall) (`network.plan_scores`, the greedy planner of the expert's rollouts), with w learned.

`learned.Explorer(path)` loads a checkpoint for play. `bench` uses it for the `learned` bot, which
explores greedily (the most likely probe) and designs like hard, as the expert does. To make a run
the bench's default, drop the optimizer state into `rl/checkpoints/`:

```bash
rl/.venv/bin/python -m labyrinthe_rl.learned rl/runs/explorer-5x5/checkpoint.npz rl/checkpoints/explorer-5x5.npz
```

`exploit` measures how exploitable an explorer is, with D21's adversarial pick: each round a
designer keeps, out of 32 fresh mazes (16 randomMaze, 16 uniform), the one this explorer hits the
most walls in. Options: `--bots learned,hard,expert`, `--rounds`, `--candidates`, `--checkpoint`.

### P3: the designer and the league (`designer.py`, `league.py`, [D25](../docs/DECISIONS.md))

- **Designer:** places E, then S, then the walls one at a time on the explorer's lattice. A wall can't
  go on a wall or on a side whose closing would cut S off from E, so every maze is valid. 14 input
  planes (walls, open and cut sides, E, S, BFS distances from E and from S, phase, budget, walls
  left, E→S distance, size); policy logits at cell and side pixels, value = W · sigmoid(…).
- **One league iteration:**
  1. the main designer makes `--d-mazes` mazes; each is explored by one opponent (the main explorer
     half the time, otherwise a league explorer by PFSP), reward = walls hit minus that opponent's
     batch mean; PPO on the designer;
  2. the mazes join the "self" pool; the main explorer plays `--envs` × `--horizon` probes, 40% of
     its games on "self" mazes, the rest on league designers by PFSP; PPO on the explorer (P2's
     losses);
  3. both losses carry + coef · KL(π ‖ magnet), the magnet an EMA of the role's parameters.
- **League members:** anchors (P2's designers `--mix`; explorers `--opponents p2,hard`), frozen
  snapshots of both main agents every `--gen-every` iterations, and exploiters: a fresh explorer
  trained only against a generation's designer, a fresh designer only against its explorer. Their
  gains (walls) are the exploitability estimate; then they join. `--max-members` of each kind are
  kept.
- **Outputs** in `--run`: `explorer.npz`, `designer.npz` (+ optimizer state) and their magnets,
  `league.json`, `league/gen-NNNN/` (frozen agents, exploiters, `exploit.json`), `log.jsonl`
  (per iteration: walls hit per source and opponent, PFSP weights, losses; evaluations), `refs.json`.
  `--save-minutes` and `--hours` make it fit Kaggle sessions; `--resume` continues.
- **Evaluation** every `--eval-every` iterations: the explorer on P2's held-out sets and the
  worst-of-32 pick, the designer's mazes (default budget) explored by the main explorer, P2's and
  hard, with their diversity: distinct share, effective number of mazes (pair collisions), E/S pairs,
  wall Hamming distance, E→S distance.

### Kaggle (`rl/kaggle/`, D24, D25)

- `drive.py upload [--runs NAME,...]` packs `rl/runs/pools` and the named runs into `bundle.tar`,
  as a version of the private dataset `charleshusson/labyrinthe-p3` (created on first use).
- `drive.py push JOB --hours H` fills `kernel.py` with the job, the commit (default: `origin/main`,
  which must hold the code: the kernel clones the public repository) and the time budget, and pushes
  it as the private script kernel `labyrinthe-p3` on "GPU T4 x2" with internet. Jobs: `calibrate`,
  `stage0` (P2's recipe with a bigger net on GPU 0, a designer against P2's explorer on GPU 1),
  `league` (main training on GPU 0, the exploiter worker on GPU 1; `--resume` to continue).
- The kernel installs jax[cuda12], unpacks the bundle into `/kaggle/working`, runs one job per GPU
  with a deadline under the 12-hour cap, and leaves runs, `logs/` and `summary.json` as its output.
- `drive.py status` (check sparingly), `drive.py fetch` (output into `rl/runs/kaggle/<time>/`),
  `drive.py quota` (GPU hours used this week).

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

## Measured (P2, 2026-10-25, Apple M1, CPU)

The committed explorers `rl/checkpoints/explorer-4x4.npz` and `explorer-5x5.npz` (32 channels ×
4 blocks, default config). Each was trained by one `train_explorer` run from scratch. The 5×5 run
started on the mix without the adversary and switched to the default mix at iteration 40 (`--resume
--mix`).

**Bench, mazes designed by the bots** (`bench --deals 1000`; learned and expert design like hard, so
these are walls hit on hard's mazes):

| board | pairing | hits A | hits B | explorer edge | score (dup) |
|---|---|---|---|---|---|
| 4×4, 7 walls | learned vs hard | 1.71 (0.24 W) | 6.86 (0.98 W) | +5.15 ± 0.04 | 100% |
| 4×4, 7 walls | learned vs expert | 1.70 (0.24 W) | 1.88 (0.27 W) | +0.18 ± 0.03 | 56% ± 1 |
| 4×4, 7 walls | expert vs hard | 1.91 (0.27 W) | 6.87 (0.98 W) | +4.96 ± 0.03 | 100% |
| 5×5, 12 walls | learned vs hard | 4.09 (0.34 W) | 11.54 (0.96 W) | +7.45 ± 0.06 | 100% |
| 5×5, 12 walls | learned vs expert | 4.17 (0.35 W) | 5.25 (0.44 W) | +1.08 ± 0.06 | 64% ± 1 |
| 5×5, 12 walls | expert vs hard | 5.30 (0.44 W) | 11.54 (0.96 W) | +6.23 ± 0.05 | 100% |

The learned explorer hits fewer walls than the expert on hard's mazes: by 0.18 on 4×4 and 1.08 on
5×5. Expert vs hard reproduces D22 (5.29 vs 11.54). These three 5×5 pairings take 5 minutes (the
expert runs in Node).

**Robustness.** On uniform mazes (`--mazes uniform`), and under the adversarial pick (`exploit`,
200 rounds, the worst of 32 candidates for each explorer):

| board | uniform: learned / expert / hard | adversarial pick: learned / expert / hard |
|---|---|---|
| 4×4, 7 walls | 2.53 / 2.57 / 1.08 | 6.19 / 6.07 / 6.91 |
| 5×5, 12 walls | 3.20 / 2.65 / 1.58 | 9.55 / 9.49 / 11.55 |

Its weak spot is the expert's: what it assumes about designs. On uniform mazes, which have short
direct paths, hard's straight-line planner is far better. Under the adversarial pick it is pushed
about as high as the expert.

**Learning curve** (greedy evaluation in the loop, default budget, held-out mazes per designer; the
references in italics explored the same mazes):

| board | probes | uniform | randomMaze | normal | hard | worst of 32 |
|---|---|---|---|---|---|---|
| 4×4, 7 walls | 0.2 M | 2.52 | 2.77 | 2.71 | 2.42 | 6.36 |
| | 0.4 M | 2.47 | 2.64 | 2.52 | 2.22 | 6.38 |
| | 0.8 M | 2.75 | 2.47 | 2.12 | 1.88 | 6.23 |
| | 1.4 M | 2.47 | 2.31 | 1.82 | 1.70 | 6.12 |
| | *expert* | *2.58* | *2.32* | *1.86* | *1.94* | |
| | *hard* | *1.04* | *4.68* | *6.29* | *6.89* | *6.84* |
| | *random explorer* | *3.03* | *6.13* | *6.67* | *6.73* | *7.00* |
| 5×5, 12 walls | 0.2 M | 4.25 | 5.14 | 5.68 | 5.96 | |
| | 0.4 M | 4.09 | 4.58 | 4.86 | 4.85 | 10.12 |
| | 0.8 M | 3.76 | 4.63 | 4.77 | 4.52 | 10.22 |
| | 1.2 M | 3.76 | 4.23 | 4.41 | 4.15 | 9.65 |
| | 2.0 M | 3.18 | 4.39 | 4.28 | 4.12 | 9.52 |
| | *expert* | *2.45* | *5.24* | *5.37* | *5.16* | |
| | *hard* | *1.47* | *7.94* | *10.56* | *11.57* | *11.61* |
| | *random explorer* | *5.12* | *10.87* | *11.66* | *11.72* | *12.00* |

512 held-out mazes per designer (the expert explored the first 256 of them), 128 rounds of the
worst-of-32 pick. Wall time: 65 min for 4×4 and 141 min for 5×5, the two runs sharing the CPU most
of the time (≈ 200–450 probes/s each; ≈ 900 for 4×4 alone).

## Notes for P2

- **XLA:CPU runs convolution gradients ~40× slower inside `lax.scan` than outside.** So the PPO
  minibatch loop is Python around one compiled gradient step. Rollouts (forward passes only) are
  fine inside `scan`.
- **The network's gradient caps training speed:** ≈ 3,500 samples/s for the 85k-parameter net on
  the M1's CPU, whatever the batch size or the number of XLA devices; the forward pass runs at
  ≈ 13,000/s. A training step is ~90% gradients.
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
