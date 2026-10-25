# The bot and the learning plan, explained

A plain-language companion to [RL_DESIGN.md](RL_DESIGN.md) (the technical plan) and
[DECISIONS.md](DECISIONS.md) (the why). Written 2026-10-23 for someone with an ML background but
no reinforcement-learning (RL) experience; state of the code as of 2026-10-25.

**Short version:** what you play today is hand-built logic. Humans can beat it because of two
specific blind spots (§2), and the planned learning is aimed at exactly those. The first learned
explorer (P2, 2026-10-25) exists in the training code: it beats the expert on bot-made mazes, but it
isn't connected to the site yet.

---

## 1. What's live on the site today

| Level | How it explores *your* maze | How it designs *its* maze | Learns? |
|---|---|---|---|
| easy / normal / hard | **Optimistic planner:** assumes every side it hasn't tried is open, takes the shortest route to S, tries the first untested side on it. Easy and normal add randomness. | Draws random mazes and keeps the one its own optimistic planner struggles with most (best of 1 / 6 / 30). | No |
| **expert** | **Bayesian search** (below) | Same generator as "normal": best of 6 | No |

**The expert explorer** (`src/bot/expert.js`). It holds about 100–400 complete guesses of your maze.
Each guess is consistent with everything it has seen: exactly W walls, S reachable from E, every
answer so far respected. In ML terms, these are Monte Carlo samples from a posterior.
- **Choosing a move:** for each possible move, it plays out the rest of the exploration on every
  guess with a fast greedy planner and averages the walls hit. It picks the move with the lowest
  average.
- **The prior:** guesses are weighted by an assumption about designers, 90% "winding maze with a long
  E→S path" and 10% "random walls". During a round it updates how winding it thinks *your* maze is.
- **Counting:** every guess has exactly W walls, so after finding 11 of 12 it knows almost everything
  else is open.

Everything here is hand-set: the 90/10 prior, the planner, the scoring. Nothing is trained, and
nothing is remembered between rounds or between players.

> **Known discrepancy (found 2026-10-23, not fixed yet):** D21 says the expert *designs* like "hard"
> (best of 30). The code (`designMaze` in `src/bot/brain.js`) actually gives it the "normal"
> generator (best of 6).

**In the repo but not on the site:**
- **`research/exact_solve.mjs`:** the exact optimum on 3×3 boards (ground truth for perfect play).
- **`scripts/benchmark.mjs`:** plays the bots against each other.
- **`rl/`, the Python training engine (P1):**
  - game simulation at about 2.5 million moves per second on one core;
  - the maze encoded as an image (D17);
  - the current bots ported, plus a bridge that runs the real expert;
  - fair same-maze comparisons.

  No neural network exists yet.

## 2. Why a human can beat both

- **Their exploring leans toward S.** "Hard" assumes the straight line. The expert knows mazes tend to
  wind, but not *which way yours* winds. When it's unsure, its planner still prefers routes toward S,
  so a detour away from S fools it.
- **Their mazes are tuned against the wrong opponent.** The bots pick mazes that defeat a greedy,
  shortest-path explorer, which a human isn't. And the generator always builds the same kind of maze,
  so the pattern is readable.
- **No memory:** a player's habits are forgotten every round.

## 3. Reinforcement learning in five ideas, for an ML person

1. **No labels, only outcomes.** Instead of "the right answer", you get a score after acting: −1 per
   wall hit. The hard part is working out which decisions caused the score (credit assignment).
2. **Policy = a classifier over legal moves; value = a regression of the score still to come.**
   Training pushes up the probability of moves that turned out better than the value predicted
   (policy gradient; PPO is the standard stable version).
3. **Self-play: the data comes from the models playing each other.** As they improve, the data
   distribution shifts, so it's a moving target.
4. **"Best" means unexploitable.** With hidden information, any fixed habit can be countered, as in
   rock-paper-scissors. The ideal is a strategy that varies so that no counter-strategy gains anything
   on average (an equilibrium). The progress metric is **exploitability**: how much a specialist
   trained only to beat the bot could still win.
5. **Closest analogy: a GAN.** The designer is like the generator, and the explorer is its adversary.
   It has the same failure modes: mode collapse (the designer always makes one maze) and cycling. The
   fixes are similar too: regularization and a population of past versions.

## 4. Track A — machine-based self-improvement (the priority)

- **Data:** none from humans. The bots play each other millions of times in simulation.
- **Explorer network:**
  - *Input:* an image of what it knows (open sides, walls, explored cells, E, S, walls left to find).
  - *Outputs:* which side to probe (policy); expected walls still to hit (value); and P(wall) for
    every side.
  - *The P(wall) output is plain supervised learning:* in simulation we know the true maze, so it
    learns where walls tend to be. It's a **learned prior**, replacing the expert's hand-set 90/10
    guess.
- **Designer network:** generates a maze step by step (E, S, then one wall at a time, like
  autoregressive generation). Its reward is the walls the explorer hits, plus a bonus for variety, so
  it can't settle on one readable style.
- **The loop:**
  1. Train both against a "league" of their own past versions.
  2. Periodically train specialists whose only job is to find holes. They measure exploitability,
     then join the league.
  3. Repeat: 4×4 → **5×5** → 8×8 → 10×10.
  4. Check against the exact 3×3 answers before trusting anything on 5×5.
- **Status:** P0–P1 (foundation) and the non-learning expert are done. P2's first learned explorer
  beats the expert on mazes made by the bots, but not yet on uniformly random ones (§7). Next is P3.

## 5. Track B — human-based adaptation (later)

- **Data:** only real game logs, with consent (D20). Every round ends with the full reveal, so each
  round gives a perfectly labelled example of how *you* design and how *you* explore. Online games
  between two humans count too: both browsers see everything at the end.
- **Model:** a general "how people play" model plus a per-player profile. The profile captures habits
  like a favourite detour direction or reusing last round's maze.
- **Safe exploitation:** start from the unexploitable Track A bot, and lean toward "counter this
  player" only as the profile proves it predicts their moves. That way a wrong guess about someone
  can't make the bot much weaker.
- **Status:** not started. It needs a logging feature first.

## 6. What growth to expect (expectations, not promises)

| Stage | What changes for the player |
|---|---|
| **Now** | Strong explorer by bot standards: it beats "hard" 99% of the time and is near-optimal on 3×3 against the best possible designer. But it leans toward S, and its mazes are readable. A good human wins. |
| **Learned explorer (P2)** | Learns where walls really tend to be instead of being told. A modest gain on bot-made mazes. It's trained on bot designs, so human styles may still surprise it. |
| **Self-play designer (P3)** | **The biggest felt change:** varied mazes tuned against strong explorers, with no recognizable pattern. Exploring the bot's mazes gets much harder. |
| **Track B** | Targets a specific player's habits. The only stage aimed at beating *you* personally. |

**Unexploitable doesn't mean unbeatable.** Track A guarantees that no single trick beats the bot on
average, but a strong human still wins rounds through good exploration and some luck: with perfect
play on both sides, about half the walls get hit. Measuring how strong it really is against people
needs real games, another reason the logs matter.

---

## 7. How implementation starts

Each phase is a self-contained piece of work with a written brief, run as its own Claude session.
Each one ends with tests passing, the docs updated and a push, as P1 did. You decide what starts.

**Phase P2 (learned explorer): first runs done 2026-10-25** ([D23](DECISIONS.md)). It differs
from the plan below in three ways:
- the network is plain JAX + Optax (no Flax);
- the P(wall) head trains together with the move head instead of first;
- the designer mix includes an adversary that keeps the mazes the current explorer handles worst.

Planning on the learned P(wall) alone turned out weaker than the learned move head, so the "quick
check" in step 2 wasn't the useful lever. Go/no-go: met on bot-made 5×5 mazes (4.2 walls of 12 vs
5.3 for the expert, 1,000 deals). Not met on uniformly random mazes (3.2 vs 2.7 for the expert,
1.6 for hard).

The plan as written before the work:
1. **The network** (`rl/labyrinthe_rl/network.py`): a convolutional network on the maze image (Flax +
   Optax on top of JAX), with the three heads from §4 (move, expected walls, P(wall)).
2. **Teach it where walls are, by supervised learning first.**
   - *Data:* generate mazes with the existing designers plus a "winding" generator. Play partial
     explorations on them (the fast simulator does millions per minute), and train the P(wall) head
     to predict the true walls.
   - *Quick check:* plug that learned prior into today's expert in place of its hand-set 90/10 guess,
     and see whether the expert gets better.
3. **Then RL.** Train the move head with PPO against a fixed mix of designers. Reward −1 per wall,
   illegal moves masked out, value head as the baseline.
4. **Measure** in duplicate format (same mazes, both seats) against the expert, on 4×4 then 5×5. On
   3×3, check against the exact values.
5. **Go/no-go:** fewer walls hit than the expert on 5×5. If not, stop and diagnose before P3.

**What it needs:**
- *Compute:* this Mac is enough for 4×4/5×5 (hours, not days). A free cloud GPU (Colab or Kaggle)
  only from 8×8 up (P5).
- *Setup in this folder, once:*
  `cd rl && python3 -m venv .venv && .venv/bin/pip install -e ".[test]"`
- *Your input:* say "start P2". Optionally, decide whether RL work keeps landing directly on `main`
  (fine while the site doesn't use it) or goes through pull requests.

**Then P3–P5.** Each is the same pattern: brief → session → tests + measurements → push.
- **P3:** the designer network and the league, on 5×5.
- **P4:** shipping (§8).
- **P5:** bigger boards.

## 8. How each track ships on GitHub

**The principle:** GitHub hosts and checks, it doesn't train.
- *Training* runs on this Mac or a cloud GPU and produces a **model file** (a few MB). The
  free GitHub Actions runners have no GPU and limited time.
- *Playing* stays 100% in the browser: the static site loads the model file and runs it locally. No
  server, same free hosting as today.

**Track A, the self-improving bot:**

| What | Where on GitHub |
|---|---|
| Training code, configs, tests | `rl/` in the repo, as today |
| Trained model the site uses | `models/explorer-5x5-v1.onnx` (+ the designer), plain files in the repo, served by GitHub Pages like the rest of the site |
| Model catalogue | `models/manifest.json`: each model's level name, board sizes, measured rating, and the commit it was trained from. The level menu reads it. |
| Full checkpoints + benchmark report | A **GitHub Release** (e.g. tag `bot-v1`): an archive for reproducibility, not loaded by the site |
| Safety checks | CI loads every shipped model, plays a few games, and checks its moves match the Python reference on fixed positions |

- **In the game:** a new level (say "master") appears next to expert. Choosing it downloads the model
  once, using ONNX Runtime Web from a CDN, pinned and integrity-checked like PeerJS today. The model
  runs in a background thread so the page never freezes.
- **New versions:** a new version is a new file plus a manifest entry. Old versions can stay as lower
  levels, so levels become rating bands.
- **Why the site's files and not the releases:** web pages can't download release files directly (no
  CORS headers), so release files are only for keeping.

**Track B, adaptation:**
- **The population model** ("how people play") ships exactly like Track A: an ONNX file in `models/`.
- **The per-player profile** is computed and stored in the player's own browser, next to their saved
  games. It never leaves their device, so adaptation itself needs no server.
- **The only new piece outside GitHub is log collection,** needed to train the population model.
  Two steps:
  1. An **"Export my games"** button (a JSON file you can send manually). This is enough to start, at
     zero cost.
  2. Later, an opt-in **"Contribute my games"** switch posting anonymized logs to a small free
     endpoint (e.g. a Cloudflare Worker with storage). This comes with a privacy note in the README,
     and needs both players' consent for online games.

**Releases as milestones:**
- **`bot-v1`:** first learned explorer.
- **`bot-v2`:** self-play designer.
- **`bot-v3`:** 8×8/10×10.
- **`adapt-v1`:** Track B.

Each release's notes contain the benchmark tables, so progress is visible on the repo's Releases page.

**End-state repository layout:**

```
index.html, styles.css, src/     the game (static site); src/bot/ = hand-written levels + model runner
models/                           shipped models + manifest.json (served by GitHub Pages)
rl/                               Python/JAX training code, configs, tests
research/                         exact solver, experiments
docs/                             decisions, plan, RL design, this explainer
.github/workflows/                CI: JS tests, Python tests, JS↔Python parity, model smoke tests
GitHub Releases                   full checkpoints + benchmark reports per milestone
```
