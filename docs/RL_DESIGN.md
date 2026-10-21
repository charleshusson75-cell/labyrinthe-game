# Toward a self-improving bot — design notes

Status: research plan. Written 2026-10-21, decisions taken 2026-10-22. Implemented so far: the
exact solver (`research/exact_solve.mjs`), the benchmark (`scripts/benchmark.mjs`), P1: the
training stack's foundation in `rl/` (§5.1), and P2 step 0: the no-learning Bayesian explorer,
shipped as the "expert" level (§5.2, D21).

> **Decisions (2026-10-22, see [DECISIONS.md](DECISIONS.md) D15–D20)**
> - **Saturation:** walls-hit ≈ W is an artifact of a predictable explorer. We train on walls hit,
>   evaluate in duplicate format, randomize the wall budget during training, and re-pick default
>   budgets from self-play data.
> - **One pipeline, scaling to 10×10:** no enumeration; 4×4 → **5×5 first** → 8×8 → 10×10.
> - **The maze is an image:** a (2N+1)² lattice, with fully convolutional networks that work at any
>   size.
> - **Stack:** Python + JAX in `rl/`, the JS engine as reference with shared fixtures, ONNX in the
>   browser.
> - **Algorithm:** league self-play with regularized PPO (MMD/R-NaD-style) plus exploiters. The
>   explorer has a belief head; play-time search comes later.
> - **Human data:** only real game logs, after self-play works.

**TL;DR**

- The game is a **two-player zero-sum game with hidden information**, of the same family as
  Battleship and Stratego's secret set-up. The designer is a *hider* and the explorer a *searcher*.
- Solving 3×3 boards **exactly** gives three lessons that shape everything else:
  1. the best designer must **randomize** over dozens of mazes for the same E/S;
  2. today's bot is decent on average but **fully exploitable** (a designer who knows it makes it hit
     every wall);
  3. hand-made rules like "put S far from E" are **not** what optimal play does.
- **Version A, self-improvement (recommended):** a population-based self-play loop (PSRO-style).
  - Explorers get better by searching over beliefs about the hidden maze.
  - Designers get better by generating *diverse* mazes that hurt the current explorers.
  - The yardstick is **exploitability**: how much a fresh best-response bot could still beat it.
    Levels come from measured Elo, not hand-tuned knobs.
- **Version B, adapting to one opponent:** yes, it can work, because every round ends with a full
  reveal. We get the opponent's exact maze *and* their full move list, perfectly labelled. A
  population prior plus a per-player model, with **safe exploitation** (move from the equilibrium
  strategy toward a best response only as the evidence grows), stored in the player's own browser.

---

## 1. What kind of game is this?

**Formally**, each round is two independent copies of the same zero-sum game, raced against each
other:

- **Designer (hider):** picks a maze = exactly *W* walls + entrance E + exit S, with S reachable from
  E. It may pick at random (a *mixed strategy*).
- **Explorer (searcher):** is told E and S, then probes sides adaptively. An open side is free;
  a wall costs one turn. Payoff = **walls hit before reaching S** (the explorer minimizes it, the
  designer maximizes it).
- **The race:** whoever needs fewer turns wins; the starter wins ties. This couples the two copies
  (see §5, "race-aware play"), but to first order each player just wants to *hit few walls* and
  *make the other hit many*.

**Four structural facts make the game tractable:**

| Fact | Consequence |
|---|---|
| E and S are public | The game splits into one subgame per (E, S) pair. The designer picks the best pair, then mixes walls inside it. |
| After designing, only the explorer acts | The explorer's best response to a designer mix is a **single-agent search under uncertainty** (a POMDP), with no opponent left to model. The designer's best response is an **optimization over mazes**. |
| The wall count is known | The explorer can *count*: with 11 of 12 walls found, the last unknown sides are almost all open. Today's bot ignores this. Related: the Canadian Traveller Problem with a bounded number *k* of blocked edges ([Wikipedia](https://en.wikipedia.org/wiki/Canadian_traveller_problem), [complexity of variants](https://arxiv.org/pdf/1207.4710)). |
| A wall can't be hit twice | Payoff is bounded by *W*. Between strong players it saturates near *W* (see the benchmark in PLAN.md), so evaluation needs variance reduction (§3.6). |

**Closest relatives:**
- **Battleship:** hidden placement, then probing. It is now a standard benchmark for
  imperfect-information self-play ([NashPG 2025](https://arxiv.org/html/2510.18183); search on top of
  an R-NaD network in Battleships, [2023](https://web3.arxiv.org/pdf/2312.15220v3)), and was already
  used by [POMCP](https://neurips.cc/virtual/2010/poster/2145).
- **Stratego's deployment phase:** DeepNash learned the secret set-up end-to-end with
  [R-NaD](https://arxiv.org/pdf/2206.15378).
- **Search games** (hider vs searcher on a graph; [overview](https://en.wikipedia.org/wiki/Search_game);
  Alpern & Gal, *The Theory of Search Games and Rendezvous*, 2003).
- **Adversarial maze generation:** [PAIRED](https://arxiv.org/pdf/2012.02096) trains a maze-designing
  adversary against a navigating agent.

**How big is it?** Valid mazes per board size (counted by enumeration or sampling):

| board, walls | valid mazes | approach |
|---|---|---|
| 3×3, 3 | 14,768 | exact (done, §2) |
| 3×3, 4 | 28,804 | exact (done, §2) |
| 4×4, 7 | 7.3 × 10⁷ | exact designer side is borderline; learning |
| 5×5, 12 (default) | ≈ 3 × 10¹² | learning + search |
| 7×7, 27 | ≈ 1.5 × 10²⁵ | learning + search |

"Not many possibilities" holds for the *good* mazes, not all mazes. Finding that small good region,
and the right way to mix within it, is precisely what self-play is for.

## 2. Ground truth: the exact solution on 3×3

`node research/exact_solve.mjs 3 3` computes the equilibrium:
- **Explorer side:** exact Bayes-optimal search, by dynamic programming over the set of mazes still
  consistent with what the explorer has seen.
- **Designer side:** learns its mixture with multiplicative weights against those best responses.
- **Certificate:** upper and lower bounds on the equilibrium value. It runs in about 30 s.

| | 3×3, 3 walls | 3×3, 4 walls |
|---|---|---|
| **Equilibrium value** (walls hit) | **1.47 – 1.51** of 3 | **1.95 – 2.04** of 4 |
| Mazes the designer mixes over (best E/S) | ~59 of 194 | ~83 of 411 |
| Best E/S | E=C1, S=C3 | **E=B2 (centre), S=C1 (diagonal neighbour)** |
| Today's bot vs the equilibrium designer | 1.52 (+0.03) | 2.37 (+0.38, i.e. +19%) |
| Today's bot vs a designer who knows it | **3.00 (every wall)** | **4.00 (every wall)** |
| Equilibrium explorer, worst case | 1.51 | 2.04 |

**Lessons:**

1. **Randomization is not optional.** The optimal designer spreads its weight over 40–130 mazes per
   E/S pair. Any bot with a "favourite" maze, today's included (it keeps the best of N candidates), can
   be learned and explored at near-zero cost.
2. **Exploitability is the real weakness, not average strength.** Today's explorer is near-optimal
   against a random strong designer, but a designer who knows it can force it into *every* wall. The
   equilibrium explorer can't be pushed above about half the walls. Self-play should therefore be
   judged by *exploitability*, as in the imperfect-information literature (OpenSpiel's NashConv).
3. **Hand-written intuitions break.** "E and S far apart" is how our generator works, yet the best
   4-wall subgame puts S *diagonally next to* E, in the centre. Learning should own the E/S choice too.
4. **The method is validated on a game where we know the answer.** Any learned pipeline must
   reproduce ≈1.49 and ≈2.0 on these boards before we trust it on 5×5.

## 3. Version A — self-improvement (the main track)

### 3.1 The loop

Population-based self-play with an equilibrium meta-solver: PSRO (Lanctot et al. 2017; see also
[Self-Play PSRO](https://arxiv.org/pdf/2207.06541)). The exact solver above is the same idea with
exact oracles.

```
                ┌────────────── population ──────────────┐
                │  designers  D1 … Dk     explorers X1 … Xk │
                └───────────────┬─────────────────────────┘
       simulate every Di vs Xj  │  (cheap: thousands of rounds / s)
                                ▼
                    payoff matrix  →  meta-Nash  (σD, σX)
                                │
          ┌─────────────────────┴─────────────────────┐
          ▼                                           ▼
 explorer oracle: best search vs σD           designer oracle: mazes that beat σX
 (belief search, distilled into a network)    (diverse generator: GFlowNet / QD / RL)
          └──────────── add both to the population, repeat ───────┘
```

It stops when neither oracle finds a meaningful improvement. That gap *is* the exploitability, and
it's measured, not guessed.

### 3.2 The explorer oracle: searching a belief

- **Belief:** a sample of mazes ("particles") from the designer mix σD, kept only if consistent with
  the observations. The wall count and E→S connectivity are hard constraints, so counting comes for
  free. Sampling: a learned *maze predictor* proposes completions, and MCMC (swap a wall, keep
  validity) repairs them.
- **Planning:** POMCP-style Monte-Carlo search over action/observation histories
  ([Silver & Veness 2010](https://neurips.cc/virtual/2010/poster/2145)), whose rewards are already
  defined (−1 per wall).
- **Amortization (expert iteration):** a network learns to imitate the search's choices and
  to predict the value (expected remaining wall hits), so search gets cheaper and stronger each
  generation (AlphaZero-style, but over beliefs).
- **Network:**
  - *Inputs:* per-side planes (known open / known wall / unknown), reached cells, E, S, walls left to
    find, and later the race state.
  - *Outputs:* a probe policy masked to legal moves, a value, and an **auxiliary head predicting
    P(wall) for every side**, a cheap self-supervised "world model" that also feeds the particle
    sampler.
  - *Architecture:* convolutional or graph-based, so one network can generalize across board sizes.

### 3.3 The designer oracle: diverse, strong mazes

Lesson 1 says the designer must output a *distribution*, not a single best maze. Three options, in
order of ambition:

1. **Quality-diversity archive** (MAP-Elites style): keep the best maze per "niche" (E/S zone, path
   length, wall clustering) and let the meta-solver mix them. Simple and interpretable; a good first
   step.
2. **GFlowNet generator:** samples mazes with probability ∝ exp(β · hits against σX), i.e.
   strong *and* diverse by construction
   ([Bengio et al. 2021](https://arxiv.org/abs/2106.04399v1),
   [foundations](https://arxiv.org/pdf/2111.09266)). Walls are placed one at a time; invalid moves
   (too many walls, cutting S off) are masked.
3. **Policy-gradient generator with entropy regularization**, which is the DeepNash style.

### 3.4 The alternative: end-to-end regularized self-play

One network with a *design head* and an *explore head*, trained by self-play with a regularized
policy gradient that converges to equilibrium instead of cycling:
- [R-NaD/DeepNash](https://arxiv.org/pdf/2206.15378) in Stratego; also used for
  [Liar's Poker](https://arxiv.org/pdf/2511.03724).
- *Magnetic mirror descent* (MMD), which a 2025 study finds matches or beats specialized methods in
  [several imperfect-information games](https://arxiv.org/pdf/2502.08938).
- [NashPG (2025)](https://arxiv.org/html/2510.18183) reports gains over PSRO and R-NaD, including on
  Battleship (from the authors' own experiments).

Less structure to engineer, but it is sample-hungry and harder to debug. **Plan:** start with 3.1–3.3,
which exploits "only the explorer acts after design". Keep 3.4 as a comparison baseline once the
simulator is fast.

### 3.5 Compute and tooling

- **Simulator:** port the rules to a vectorized simulator (JAX in the style of
  [Pgx](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8f153093758af93861a74a1305dfdc18-Abstract.html):
  thousands of parallel games on one GPU, 10–100× faster than plain Python). Keep the JS engine as the
  reference: **shared JSON fixtures** (maze, probes → expected answers and turn order) checked by
  both test suites in CI, so the two rule sets can't drift.
- **Budget:** 4×4 and 5×5 prototypes on a laptop CPU; a single GPU (or Colab) for 7×7. The exact
  3×3 solver doubles as a unit test for the whole learning pipeline.
- **Shipping:** distilled networks are tiny (a 5×5 board is a few hundred numbers in, 100 possible probes out). Export
  to ONNX for the browser, or a hand-written forward pass. The game stays static-hosted.

### 3.6 Evaluation (lab-grade)

- **Exploitability:** periodically train a *fresh* explorer against the frozen designer, and run a
  fresh designer search against the frozen explorer. The gap between the two values is the headline
  metric (OpenSpiel calls it NashConv).
- **Head-to-head ratings:** Elo or Bradley–Terry against frozen references (today's easy/normal/hard,
  plus each past generation). **Levels become rating bands of checkpoints.**
- **Variance reduction:**
  - *duplicate format:* both bots play both seats on the same maze pairs, which cancels luck and the
    starter's advantage;
  - *control variates:* [AIVAT](https://www.arxiv.org/pdf/1612.06915) cut the games needed in poker
    by 10–44×.
- **Sanity:** the pipeline must reproduce the exact 3×3 values in §2.

## 4. Version B — adapting to one opponent

**Would it work?** Likely yes, and better than in most games, because of commit–reveal. After every
round we get two perfectly labelled datasets about the opponent:
- their **maze**, showing their designer habits;
- their **full move list on our maze**, showing their explorer habits.

Humans are predictable in exactly the ways §2 says are costly. Against a predictable explorer, a
designer can climb from about half the walls (equilibrium) toward *all* of them. Against a designer
who reuses or slightly edits their maze (our own UI keeps last round's maze as a draft!), an explorer
can walk to S almost for free.

**Model:**
- **Population prior.** "How people design and explore", from self-play plus (opt-in) human logs.
- **Per-player posterior,** updated after each round:
  - *Simple, data-efficient:* a parametric persona. For exploration: heads toward S, direction
    biases, wall-following, backtracking depth. For design: E/S zones, walls near S, symmetry,
    similarity to their previous maze. Bayesian updates over a few rounds.
  - *Neural:* a player embedding inferred from past rounds, which conditions a behaviour-cloning
    model. This is the [Machine Theory of Mind / ToMnet](https://arxiv.org/pdf/1802.07740) idea of
    predicting an agent from a few past episodes; recent work does adaptive modelling in
    imperfect-information games with transformers ([StratFormer, 2026](https://arxiv.org/pdf/2604.25796)).

**Exploiting safely.** A best response to a *wrong* model can be badly exploited, and humans adapt.
- The classic answer is the **restricted Nash response**: assume the opponent follows the model with
  probability *p*, and play an equilibrium of that modified game. It is Pareto-optimal between
  exploitation and safety (Johanson et al. 2007; refinements:
  [AAAI](https://ojs.aaai.org/index.php/AAAI/article/download/7981/7840),
  [safe exploitation by subgame refinement](https://proceedings.neurips.cc/paper_files/paper/2022/file/b12a1d1014e952e676f5d6931d03241a-Paper-Conference.pdf),
  [safe equilibrium](https://arxiv.org/pdf/2201.04266)).
- In practice: play **λ·(best response to the model) + (1−λ)·(equilibrium bot)**, with λ raised only
  as the model's predictions of held-out moves prove accurate. This is close to the recent
  [confidence-scheduled restricted responses](https://arxiv.org/pdf/2607.28520). Weight recent
  rounds more, because people learn too.

**Storage and privacy.** The bot only ever plays the person at the keyboard, so their profile lives
in their own browser: a few KB in `localStorage`, plus an optional "export my games" button.
Learning a *population* prior from everybody's games needs opt-in collection through a small
endpoint. That's a product decision, not a technical blocker.

**Evaluation.** First against synthetic "human-like" personas (biased explorers and designers):
measure how many rounds the adaptive bot needs to beat the equilibrium bot against each. Then A/B
with real players.

## 5. Roadmap (revised 2026-10-22)

| Phase | Deliverable | Go/no-go |
|---|---|---|
| **P0 (done)** | Exact 3×3 solver; bot benchmark; 4×4–10×10 boards in the game | Ground-truth values in §2 |
| **P1 ✅** (2026-10-22) | `rl/`: JAX engine (any size), lattice encoding, maze samplers, Python ports of today's bots, duplicate-format evaluator, JS→Python parity fixtures in CI (§5.1) | Parity green; ≥ 10⁵ explorer steps/s on 5×5 (CPU). **Met:** 203 fixtures reproduced exactly; 2.5 M steps/s on 5×5 on one CPU core (0.46 M with the hard planner as policy) |
| **P2** | Explorer learning vs fixed designer mixes (random + "hard"), 4×4 → 5×5, with belief head. **Step 0 ✅** (2026-10-22): Bayesian explorer without learning, the "expert" level (§5.2) | Fewer walls hit than the "hard" heuristic on 5×5 (step 0: 5.3 vs 7.8–11.5 of 12); now also fewer than "expert"; ≈ equilibrium values on 3×3 |
| **P3** | Full league self-play on 5×5 (both roles, MMD-regularized PPO, exploiters) | Exploitability falls generation after generation; beats "hard" in duplicate matches |
| **P4** | Ship the 5×5 bot to the site (ONNX); levels = rating bands of checkpoints | < 50 ms per move in the browser |
| **P5** | Scale: curriculum to 8×8 / 10×10 on a cloud GPU; re-pick default wall budgets (D15) | Near-equilibrium hits/W well below 1; still improving |
| **P6** | Play-time search with belief particles; race-aware explorer | Measurable gain over the raw network |
| **P7** | Version B (per-player adaptation) from real game logs only | Beats the equilibrium bot against regulars, never much worse |

The race-aware explorer (P6) uses the fact that you know your own maze and watch the opponent's
progress: you can predict when they'll finish, so optimal play takes more risk when behind.

### 5.1 P1 — engine, encodings, baselines, evaluation (done 2026-10-22)

`rl/` (Python 3.12 + JAX, CPU); setup, commands and details in [rl/README.md](../rl/README.md).

**What's there.**
- The rules in JAX for any size, compiled and batched.
- The (2N+1)² lattice of D17: observation planes, belief target, actions as side pixels.
- `randomMaze` ported draw for draw, plus a sampler uniform among all valid mazes.
- Today's bots: ported line by line, and batched in JAX.
- A duplicate-format evaluator, a throughput benchmark, and a Gymnasium wrapper.

**Parity.** `scripts/export-fixtures.mjs` exports 203 fixtures from the JS engine (sizes 4–10):
- 6,720 probes, 1,238 of them illegal;
- 1,135 `rankProbes` snapshots;
- seeded bot explorations and designs.

Python reproduces all of them exactly. CI checks both sides: the committed fixtures must be what the
JS produces today, and Python must reproduce them.

**Throughput** (Apple M1; explorer steps/s, one probe in one game, games restarted as they end):

| board | uniform legal probes | hard planner | uniform + lattice observation |
|---|---|---|---|
| 5×5, one CPU core | **2.5 M** | 0.46 M | 1.2 M |
| 5×5, 8 CPU devices | 4.7 M | 1.5 M | |
| 10×10, one CPU core | 0.59 M | 0.11 M | 0.32 M |
| 10×10, 8 CPU devices | 1.9 M | 0.48 M | |

**Today's bots in duplicate format** (5×5, 12 walls, 1,000 deals, both seats on each maze pair):

| | explorer edge (walls) | designer edge (walls) | score, both seats | score, round as played |
|---|---|---|---|---|
| hard vs normal | +0.33 ± 0.02 | +0.81 ± 0.04 | 59% | 79% |
| normal vs easy | +1.26 ± 0.04 | +1.68 ± 0.07 | 68% | 88% |

The round scores match the JS benchmark (80% and 86%).

**For P2.**
- **Swapping seats cancels the designers.** Swapping seats on the same maze pair removes the maze
  luck and the starter's advantage (D15), but it also cancels the designers exactly. So the duplicate
  score compares explorers only; designers are read from the designer edge (the same two explorers
  on both bots' mazes).
- **CPU performance.** On CPU, gathers inside compiled loops are the bottleneck, not arithmetic.
  - The planner's BFS runs on row bitmasks.
  - The lattice is built by reshapes (14× faster).
  - Rollouts are a work queue of games.
  The network's forward pass will now dominate the cost per step.
- **The expert explorer (§5.2) isn't ported to `rl/` yet.** It is the bar P2 must beat.

### 5.2 P2 step 0 — Bayesian explorer without learning (done 2026-10-22)

`src/bot/expert.js`, details and all numbers in [D21](DECISIONS.md).

**How it works.**
- **Belief:** particles = mazes with exactly W walls, S reachable from E, consistent with every
  answer. They are sampled by rejection, or by MCMC wall swaps that keep the maze valid.
- **Move:** one step of policy improvement with Monte-Carlo rollouts of a belief-cost greedy planner.
- **Prior:** a prior over *designs* that the explorer partly infers during the round.

**What it taught us.**

| | uniform prior | prior over designs (shipped) | "hard" |
|---|---|---|---|
| 3×3, random walls (Bayes-optimal: 0.59 / 0.72) | **0.57 / 0.73** | 1.77 / 2.08 | 0.59 / 0.72 |
| 3×3 vs equilibrium designer (value 1.47–1.51 / 1.95–2.04) | 1.46 / 2.40 | **1.49 / 2.15** | 1.52 / 2.37 |
| 5×5 "hard" mazes, walls hit of 12 | 11.6 | **5.3** | 11.5 |
| 5×5 adversarial pick (worst of 32 per round) | — | **7.6** | 10.9 |

(3×3 cells: 3 walls / 4 walls, best equilibrium subgame.)

- **Searching is the easy part; knowing what designs look like is the hard part.**
  - The search alone is Bayes-optimal on 3×3.
  - It gains nothing on 5×5 until the prior knows that designed mazes wind.
- **For P2/P3:** the belief head *is* that prior, so it must be trained against a mix of designers,
  not one.
- **The particle sampler** is reusable for P6 (play-time search). Keeping particles between moves
  (SMC) is the natural next improvement.

## 6. Open questions

- **Default wall budgets per board size:** to be re-picked from self-play data (D15).
- **Game rules for humans:** whether to add a two-round match mode with alternating starters, so
  ties stop favouring whoever started.

## References

- Self-play and equilibrium methods:
  - Perolat et al., *Mastering Stratego with model-free multiagent RL* (DeepNash, R-NaD),
    [arXiv:2206.15378](https://arxiv.org/pdf/2206.15378) ·
    [DeepMind blog](https://deepmind.google/discover/blog/mastering-stratego-the-classic-game-of-imperfect-information/)
  - McAleer et al., *Self-Play PSRO*, [arXiv:2207.06541](https://arxiv.org/pdf/2207.06541)
  - *Reevaluating Policy Gradient Methods for Imperfect-Information Games* (MMD and friends),
    [arXiv:2502.08938](https://arxiv.org/pdf/2502.08938)
  - *NashPG*, [arXiv:2510.18183](https://arxiv.org/html/2510.18183)
  - *Liar's Poker via self-play*, [arXiv:2511.03724](https://arxiv.org/pdf/2511.03724)
  - Survey of self-play methods, [arXiv:2408.01072](https://arxiv.org/pdf/2408.01072)
- Search and planning:
  - Silver & Veness, *Monte-Carlo Planning in Large POMDPs* (POMCP),
    [NeurIPS 2010](https://neurips.cc/virtual/2010/poster/2145)
  - Search in imperfect-information games (incl. Battleships), [arXiv:2312.15220](https://web3.arxiv.org/pdf/2312.15220v3)
  - Search games, [overview](https://en.wikipedia.org/wiki/Search_game); Canadian Traveller Problem,
    [overview](https://en.wikipedia.org/wiki/Canadian_traveller_problem) and
    [variants](https://arxiv.org/pdf/1207.4710)
  - McMahan, Gordon & Blum, *Planning in the presence of cost functions controlled by an adversary*
    (double oracle), ICML 2003
- Generating designs:
  - Dennis et al., *PAIRED*, [arXiv:2012.02096](https://arxiv.org/pdf/2012.02096)
  - Bengio et al., *GFlowNets*, [arXiv:2106.04399](https://arxiv.org/abs/2106.04399v1) ·
    [Foundations](https://arxiv.org/pdf/2111.09266)
- Opponent modelling and safe exploitation:
  - Rabinowitz et al., *Machine Theory of Mind*, [arXiv:1802.07740](https://arxiv.org/pdf/1802.07740)
  - *StratFormer*, [arXiv:2604.25796](https://arxiv.org/pdf/2604.25796)
  - Safe exploitation: [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/b12a1d1014e952e676f5d6931d03241a-Paper-Conference.pdf),
    [Safe equilibrium](https://arxiv.org/pdf/2201.04266),
    [AAAI](https://ojs.aaai.org/index.php/AAAI/article/download/7981/7840),
    [confidence-scheduled restricted responses](https://arxiv.org/pdf/2607.28520)
- Infrastructure and evaluation:
  - Koyamada et al., *Pgx*, [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8f153093758af93861a74a1305dfdc18-Abstract.html)
  - Burch et al., *AIVAT*, [arXiv:1612.06915](https://www.arxiv.org/pdf/1612.06915)
