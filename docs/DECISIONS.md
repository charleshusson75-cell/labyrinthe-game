# Decision log

Short, dated records of the choices that shape the project (ADR-lite). Newest at the bottom.
When a decision is revisited, add a new entry that supersedes the old one instead of rewriting history.

---

## D1 — Static web app, no build step (2026-10-21)

**Decision.** Plain HTML + CSS + native ES modules. No framework, no bundler, no npm dependencies at runtime.

**Why.** The project should be easy to read for a beginner, trivial to host for free, and runnable with
nothing more than a static file server. A 5×5 board and a few panels don't need React.

**Consequences.** UI is rendered with template strings + event delegation. Logic modules are pure and
DOM-free so they can be unit-tested with Node's built-in test runner (`node --test`) — zero dev
dependencies either.

## D2 — Online play is peer-to-peer (WebRTC via PeerJS), hosted on GitHub Pages (2026-10-21)

**Decision.** Players connect directly browser-to-browser with WebRTC data channels. Connection setup
("signaling") uses the free public PeerJS server. The site itself is static and deploys to GitHub Pages.

**Alternatives considered.**
- *Node.js + WebSocket server* (server-authoritative): most reliable connectivity, but needs a host
  (Render/Fly/Railway…), sleeps on free tiers, and is more to maintain.
- *Firebase / Supabase realtime*: needs accounts, API keys and a backend project.

**Why.** Zero infrastructure: push to GitHub, enable Pages, send the link. Matches "make it shareable
with a link" and "publishable on GitHub".

**Trade-offs / risks.**
- A small share of networks (strict NAT, some mobile carriers, corporate Wi-Fi) can't establish a
  direct WebRTC connection without a TURN relay. `src/config.js` lets you add TURN servers.
- Depends on the public PeerJS signaling server; it can be swapped for a self-hosted `peerjs-server`
  via `src/config.js`.
- The transport is abstracted (`src/net/`), so a WebSocket relay could be added later without touching
  game logic.

## D3 — Each player's browser is the sole keeper of its own maze (2026-10-21)

**Decision.** A player's walls never leave their browser during the game. When the opponent probes a
move, the *owner's* client answers "open" or "wall" — exactly like the paper game where the designer
answers.

**Why.** It makes peeking impossible: there is nothing to peek at in the opponent's browser memory
or network traffic. No server is needed to hide information.

## D4 — Commit–reveal to detect lying (2026-10-21)

**Decision.** When a player clicks *Ready*, their client publishes `SHA-256(maze + random salt)`.
At the end of the round both mazes and salts are revealed and each client checks:
the hash matches, the maze respects the rules, and **every answer given during the game matches the
revealed walls**. The end screen shows "✓ verified" or what went wrong.

**Why.** Since the owner answers moves (D3), a modified client could lie. Commit–reveal can't prevent
that, but it makes it detectable — good enough for a friendly game. SHA-256 is implemented in pure JS
(`src/game/sha256.js`) so it also works on plain-HTTP LAN addresses where WebCrypto is unavailable.

## D5 — State replication by snapshots instead of event messages (2026-10-21)

**Decision.** Apart from `move` (a request) and `chat`, each client sends a full `sync` snapshot of the
data it is authoritative for (its name, its commitment, its E/S once both are ready, the answers it gave
to the opponent's moves, its reveal, rematch/resign flags). Receiving a snapshot is an idempotent merge.

**Why.** Reconnections (page refresh, phone going to sleep, Wi-Fi drop) are the #1 source of bugs in
small online games. With snapshots, "reconnect" is just "send your snapshot again". Snapshots are tiny
(< 2 KB).

See [PROTOCOL.md](PROTOCOL.md).

## D6 — Game rules as implemented (2026-10-21)

- Square grid, default **5×5** (host can pick 4–8).
- A wall is one cell side. The outer border is always closed; the UI simply doesn't offer moves that
  leave the grid.
- Each player must place **exactly** the wall budget ("fixed total"). Default for 5×5 is **12**
  (≈ ¾ of (N−1)², the number of walls of a "perfect" maze). Host can change it.
- Entrance **E** and exit **S** (*sortie*) are any two distinct cells; the exit must be reachable from
  the entrance. Unreachable pockets are allowed.
- Both designs are secret; only E and S are revealed when both players are ready.
- On your turn you probe a side of any cell you have already reached (you start at E; you may
  "jump back" to any cell of the path you've uncovered). Open → you move and play again.
  Wall → it's noted and the turn passes.
- Already-known sides can't be probed again (no information to gain); moving along known open
  passages is free.
- First to reach the opponent's S wins immediately.
- Round 1: random starter. Following rounds: **the loser starts**.
- A player can concede.

## D7 — Seats, rooms and reconnection (2026-10-21)

**Decision.** Room = 6-char code. The host's PeerJS id is derived from the code, so the guest only
needs the link `…/#/join/<code>`. Each seat's state is saved in `localStorage`, so a refreshed or
re-opened tab resumes where it was. The host binds the first guest's random token; other people
opening the link later get "room full". A `BroadcastChannel` tab-lock prevents the same seat from being
driven by two tabs at once.

## D8 — Play vs computer reuses the network protocol (2026-10-21)

**Decision.** The bot is a second `Session` connected through an in-memory pipe, driven by a small
controller. It designs mazes (random spanning-tree mazes; on *hard* it picks the candidate that takes
its own solver the longest) and solves with an "optimistic planner": shortest path to S assuming unknown
sides are open, probe the first unknown side on it.

**Why.** One code path for all modes → the bot exercises (and tests) the real protocol.

## D9 — Bilingual UI (EN/FR), paper-notebook look (2026-10-21)

**Decision.** English and French strings in `src/i18n.js`, auto-detected from the browser and
switchable. Visual style evokes the original pen-and-paper game (graph paper, ink walls,
handwritten letters), with a chalkboard dark mode.

## D10 — License: MIT (2026-10-21)

Permissive default for a hobby project; easy to change before publishing.

## D11 — Self-healing heartbeat (2026-10-21)

**Decision.** Pings (every 4 s) carry a digest of the sender's last snapshot; if it differs from the
last snapshot actually received, the receiver asks for a `resync`. A move left unanswered for 5 s is
re-sent (the owner de-duplicates by sequence number). 14 s of silence = connection considered dead.

**Why.** Found by the reconnection tests: a message lost on a connection that *stays* open (rare
with WebRTC, but possible during network hand-overs) would otherwise freeze the game until the next
reconnect.

## D12 — Random mazes avoid E→S shortcuts (2026-10-21)

**Decision.** The generator builds a perfect maze, places E/S far apart, then knocks down walls to
reach the budget, skipping walls whose removal would shorten the E→S path (used only as a last
resort). Found in play-testing: a random maze had S right next to E with a direct opening possible.

## D13 — The host's page must stay open; reconnect as soon as it wakes up (2026-10-21)

**Context.** First real game: the guest's link showed "Waiting for the host". A probe of the
signaling server confirmed the host's room id wasn't registered: the host's page had gone away
(most likely: on a phone, switching to a messaging app to send the link suspends the page).

**Decision.** Without a server this is inherent to D2 (the game lives in the host's browser), so:
- the invite card tells the host to keep the page open and come back to it after sending the link;
- the guest's waiting message explains it, and that the host can reopen the game from "Your games";
- the PeerJS transport reconnects immediately when the page becomes visible again, when the
  network returns, and from a 10 s watchdog, instead of relying only on timers (which phones pause);
- the host keeps trying to reclaim its room id forever (the server can hold a dead registration
  for up to ~60 s); after a minute it says so but keeps trying.

**Later option.** A tiny always-on relay/lobby server would let a guest join before the host is
back, at the cost of hosting (see D2 alternatives).

## D14 — Relay (TURN) on port 443 for school / office networks (2026-10-21)

**Context.** Second real test: the guest got "can't open a direct connection" from both home Wi-Fi
and a phone hotspot, while the host was on school Wi-Fi. PeerJS already falls back to its own relay,
but only on port 3478, which such firewalls block.

**Decision.** Support a TURN provider reachable on port 443 (TLS), which looks like normal web
traffic. `CONFIG.iceServersUrl` points at a URL returning ICE servers (Metered's credentials API
format, or anything returning `[{urls, username, credential}]`); they're added to the defaults when
a game opens, and the game falls back silently if the provider is unreachable. Game traffic is tiny
(a few KB per round), so any free tier is plenty.

**Pending (user).** Pick a provider and create the account: Metered (free plan, API key usable from
the browser, simplest) or Cloudflare Realtime TURN (1,000 GB free, but credentials must be minted
server-side, e.g. by a small Worker).

**If that's not enough.** Some networks only allow web traffic through an inspecting proxy; then
WebRTC fails even with TURN, and the fallback is a WebSocket relay (e.g. a Cloudflare Worker or
Supabase Realtime) as a second transport.

## D15 — Saturation is an exploitation artifact; train and measure so it can't hide (2026-10-22)

**Context.** The user wants the bot to keep improving on big boards (8×8, 10×10), where nothing can be
brute-forced. Measured share of walls hit (`hits / W`) with today's heuristic bots:

| board, walls | random maze vs "hard" explorer | maze tuned against it | same maze vs a sloppy explorer |
|---|---|---|---|
| 5×5, 12 | 0.56 | 0.98 | 0.98 |
| 8×8, 37 | 0.63 | 0.92 | 0.99 |
| 10×10, 61 | 0.61 | 0.86 | 0.99 |

Near-100% only appears when a designer is tuned against a weak or predictable explorer. With optimal
play on both sides (exact 3×3 solve), it's about 50%. So the game has headroom at every size.

**Decision.**
- Train on **walls hit** (dense), not on win/loss; the race's tie rule isn't part of the learning signal.
- Evaluate in **duplicate format**: both bots play both seats on the same maze pairs, which removes
  maze luck and the starter's advantage.
- The wall budget W is an **input** to every network and is **randomized during training**. Default
  budgets per board size will be re-picked from self-play data so that near-equilibrium play hits
  well under W (target 50–75%).
- Tracked per board size: exploitability, near-equilibrium `hits / W`, rating against frozen
  reference bots.
- The game now allows 4×4 up to **10×10**. The bot's design effort scales with board size
  (≤ ~150 ms).

## D16 — One scalable pipeline, 5×5 first, no enumeration (2026-10-22)

**Decision.**
- No 5×5-specific method. 5×5 already has ~3×10¹² mazes, so it's a "can't brute force" case, and
  using the 8×8/10×10 pipeline from day one means everything learned transfers.
- The exact solver stays only as a correctness test on 3×3.
- Curriculum: 4×4 (fast iteration) → **5×5 (first deliverable)** → 6–8 → 10×10, fine-tuning the same
  networks.

## D17 — Represent the maze as a (2N+1)×(2N+1) lattice image; fully convolutional networks (2026-10-22)

**Decision.**
- **The lattice:** cells sit at (odd, odd), sides at (odd, even)/(even, odd), corners at (even, even),
  so walls, known passages, reached cells, E and S are all pixels. Scalars (W, walls found, board
  size, later the race state) are constant planes.
- **Actions are pixels:** a probe or a wall placement is a side pixel; placing E or S is a cell pixel.
- **Network:** a residual convolutional tower (dilated, to see a 21×21 lattice at 10×10) works for
  any board size unchanged.
- **Heads:**
  - explorer policy (masked to legal sides);
  - value (expected walls still to hit);
  - **belief head:** P(wall) for every side, trained against the true maze, which is known in
    simulation (privileged information);
  - designer policy.

## D18 — Training stack: Python + JAX in `rl/`; the JS engine stays the reference (2026-10-22)

**Decision.**
- **Simulation:** game rules re-implemented in JAX, compiled and run in parallel batches (the Pgx
  approach), so the whole training step runs as one compiled loop.
- **Hardware:** this Mac's CPU for 4×4/5×5; a free cloud GPU (Colab/Kaggle) for 8×8/10×10.
- **Parity:** a Node script exports JSON fixtures (mazes, probe sequences, answers, reached cells,
  turn order, winner) from the JS engine, and the Python tests must reproduce them exactly; CI runs
  both suites.
- **Gymnasium wrappers** for compatibility, as in Pulser-gym.
- **Shipping:** trained networks are exported to ONNX and run in the browser with onnxruntime-web,
  loaded only for the levels that need them. The site stays static.

**Alternatives.** Keeping everything in JavaScript has no serious training ecosystem. Rust plus
WebAssembly would give one engine for both, but means more engineering before any learning happens.

## D19 — Algorithm: league self-play with regularized policy gradient, plus exploiters (2026-10-22)

**Decision.**
- **Learning rule:** both roles are trained with PPO plus a magnet-style regularization (MMD /
  R-NaD), which converges toward equilibrium instead of cycling.
- **League** (from AlphaStar and PSRO): opponents are sampled from the current agents and frozen past
  snapshots, favouring the ones that are hardest to beat.
- **Exploiters:** fresh best responses are periodically trained against the frozen main agents. They
  *measure* exploitability and then join the league, which keeps progress going on big boards where
  strategies can go round in circles.
- **Explorer:** the belief head from D17 makes learning much more sample-efficient, and later
  provides particles for search at play time (search on top of a regularized self-play network
  helped in Battleships).
- **Designer:** autoregressive (E, S, then walls one by one), validity enforced by masking or
  rejection, with an entropy bonus so it keeps varying its mazes (lesson 1 of the 3×3 solve).

**Supersedes** the "PSRO with search oracles first" ordering in RL_DESIGN §3. Search and the diverse
designer are kept, as the belief head and play-time search and as the designer's entropy
regularization respectively.

## D20 — Human data only from real game logs, after self-play (2026-10-22)

Version B (adapting to one player) waits until self-play works. When it comes, human behaviour is
learned **only from real (opt-in) game logs**, never from made-up synthetic players (user decision:
otherwise not accurate).

## D21 — "Expert" level: a Bayesian explorer without learning (2026-10-22)

**Context.** First concrete step toward the self-improving bot: the no-learning explorer of the
original roadmap's P2 (now step 0 of P2). The heuristic explorer ignores the wall count and is fully
exploitable (RL_DESIGN §2, D15).

**Decision.** New level **expert** (`src/bot/expert.js`; it designs like "hard" for now).
- **Belief = particles:** mazes with exactly W walls, S reachable from E, consistent with every answer
  so far. Uniform draws come from rejection; tilted ones from Metropolis chains (swap a wall with an
  unknown open side, keep the maze valid).
- **Move = one step of policy improvement:** each legal probe is tried on each particle, then a
  greedy planner finishes the exploration; it costs each unknown side its probability of being a
  wall. The fewest expected walls hit wins, and clearly worse probes are pruned by paired comparison.
- **Budget:** 150 ms per move. Measured: 5×5 median 13 ms in the browser (max 47 ms); 10×10
  median 120 ms (first move up to ~250 ms, while it measures the prior).
- **Prior over designs (the decisive choice):** weight ∝ exp(β · E→S path length + 0.5 · cells
  reachable from E), with β ∈ {0 = uniform walls, 1.5 = winding maze}, 1 : 9 a priori. β is inferred
  during the round: log P(seen | β) is the integral over β of (mean path length of particles that fit
  what was seen − before any observation).

**Findings.**
1. **The search is near-optimal.** With a uniform prior it matches the exact Bayes-optimal explorer
   on 3×3 random walls: 0.57 vs 0.59 (3 walls), 0.73 vs 0.72 (4 walls).
2. **But the prior is what matters.** With a uniform prior it is *no better than "hard"* on designed
   mazes (11.6 of 12 walls on 5×5 "hard" mazes). Designed mazes wind: on 5×5 the E→S path is 12–15
   steps, vs 4.4 for uniform valid mazes, and every cell is reachable. A path-length prior makes the
   difference.
3. **The rollout planner must use belief costs.** Optimistic rollouts (unknown side = cost 1) throw
   the gain away (10.5 of 12). The belief-cost planner alone gets 5.0–5.4, and rollouts add about one
   wall more. Particle count barely matters past ~100.
4. **Fixed vs inferred style.** A fixed β = 1.5 is best on designed mazes (4.4 of 12) but loses on
   uniform-like ones (4.3 vs 1.9 for "hard"). We measured exploitability with an adversarial designer
   that keeps, each round, the worst of 32 candidates for that explorer (16 from the generator,
   16 uniform), out of 12 walls on 5×5:

   | explorer | candidates on average | adversarial pick |
   |---|---|---|
   | hard | 4.81 | 10.93 |
   | expert, fixed β = 1.5 | 4.33 | 7.90 |
   | **expert, inferred β (shipped)** | **3.95** | **7.60** |

   Inferring β costs about 0.7 walls on designed mazes and gains 1.4 on uniform ones, and is the
   least exploitable option tried.

**Measured** (`node scripts/benchmark.mjs 4 7 200` / `5 12 200`; walls hit when exploring each
designer's mazes):

| explorer | 4×4, 7 walls: easy / normal / hard mazes | 5×5, 12 walls: easy / normal / hard mazes |
|---|---|---|
| hard | 4.7 / 6.3 / 6.9 | 7.8 / 10.6 / 11.5 |
| **expert** | **2.5 / 1.9 / 1.9** | **5.3 / 5.4 / 5.3** |

- **Head to head:** expert beats hard in 100% of rounds on 4×4 and 99% on 5×5 (same vs normal).
- **Less luck:** in expert vs expert on 5×5 the starter wins 61% and turns are equal 15% of the time,
  vs 80% and 55% for hard vs hard.
- **3×3, best equilibrium subgame:** against the equilibrium designer, expert (as shipped) gets
  1.49 (3 walls) and 2.15 (4 walls). Hard gets 1.52 and 2.37; the equilibrium values are 1.47–1.51
  and 1.95–2.04.

**Consequences for the RL plan.**
- P2's bar moves from "hard" to "expert".
- The belief head (D17) is in effect a *learned prior over designs*. Train it against a mix of
  designers (uniform, generator, adversarial), or it inherits the weakness of a fixed prior.
- It supports D15: an explorer that models designs hits ~44% of W on 5×5 mazes tuned against "hard".

**Known limits.**
- The β evidence is noisy: MCMC burn-in bias, and the estimate is recomputed from scratch every move.
  Keeping particles between moves and updating the evidence sequentially (SMC) would fix it and save
  time.
- On 3×3 the shipped prior plays badly against uniform random walls (2.08 vs 0.72). A tiny board can't
  show whether paths wind, and the game starts at 4×4.
- It is still exploitable (7.6 of 12 under the adversarial pick above). Equilibrium play needs the
  learned, mixed strategies of P3.

## D22 — The expert in the Python evaluator: a Node bridge, not a port (2026-10-22)

**Context.** P2 measures learned explorers in `rl/`'s duplicate-format evaluator (`bench`), and its
bar is now the expert (D21), which only exists in JavaScript.

**Decision.** The bench bot `expert` runs `src/bot/expert.js` itself, in Node:
`rl/labyrinthe_rl/expert.py` sends mazes as JSON to `scripts/expert-solve.mjs`, which answers with the
walls hit on each. It designs like "hard", as in the game.
- **Fixed particle count** (200, as `scripts/benchmark.mjs` used for D21) instead of the time budget,
  so results don't depend on the machine. Each maze's seed comes from the bench's JAX key.
- **A fresh copy of expert.js per maze** (re-imported under a new URL). expert.js caches the
  unconditioned mean path length per (E, S), drawn with the caller's rng. In a shared copy, a maze's
  result depended on the mazes explored before it with the same E and S. A fresh copy costs ≈ 5%.
- **One Node process per CPU core**, each on a slice of the mazes. Thanks to the fresh copies,
  results don't depend on the split (tested).

**Alternatives considered.** A numpy port of the belief sampler and rollout policy. Per maze it
would be slower than V8, unless batched in JAX: a big job, with its own parity risk, and two copies
of the expert to keep in sync. The bridge is fast enough: about 0.16 s per 5×5 exploration on one
core.

**Measured** (`bench --deals 1000`, 5×5, 12 walls, Apple M1): expert vs hard, expert 5.29 walls
(0.44 W), hard 11.54. In the round as played the expert hits fewer walls 99.7% of the time (equal
0.2%). Expert vs expert is exactly even. This matches D21 (5.3 of 12 on hard's mazes, 99% of rounds
won). 1,000 deals of expert vs hard take 2 minutes (8 Node processes).

**Consequences.**
- `bench` needs `node` on the PATH for the expert (CI has it). Its pytest is skipped without it.
- If P2 needs the expert inside training (as a teacher or a rollout policy), a JAX port becomes worth
  it then.

## D23 — P2's learned explorer: PPO with a belief head, against a mix of fixed designers (2026-10-25)

**Context.** P2 trains the explorer against fixed designers (the designer learns in P3). D17 fixes the
network's shape (convolutional over the lattice; policy, value and belief heads) and D19 the family
(PPO, magnet-regularized, league). Left open: the framework, the network's size, the value head,
the PPO details, which designers and in what proportion, how mazes are supplied, and how progress is
measured.

**Decision** (`rl/labyrinthe_rl/network.py`, `train_explorer.py`, `learned.py`).
- **Network:** a residual convolutional tower over the (2N+1)² lattice: 32 channels × 4 blocks,
  dilations 1, 2, 4, and a global-pooling bias in every block (mean and max over the board → dense →
  added per channel, as in KataGo), so each pixel sees board-wide counts such as walls left. No
  normalization layers (small last conv per block instead). 85k parameters, the same for every size.
  - Three 1×1 heads read at the side pixels: **policy** (masked to legal probes), **belief**
    (P(wall), cross-entropy against the true maze on the unknown sides), and **hit** (P(this side is
    hit before the exit)).
  - **Value = the sum of the hit head over the unknown sides**: the expected walls still to hit. It
    is bounded by the unknown sides and scales with the board without any size-specific layer.
- **Plain JAX, optax for Adam, no Flax.** Parameters are a dict of arrays and a checkpoint is a .npz
  of named arrays plus a JSON record: what a hand-written browser forward pass (P4) needs. Flax 0.12
  would add orbax, tensorstore and friends for no gain here.
- **Size set by the CPU:** the network's gradient runs at ≈ 3,500 samples/s on the M1 whatever the
  batch size or XLA device count; 48 channels × 6 blocks would be 3× slower.
- **PPO:** 256 games × 32 probes per update, 3 epochs, minibatches of 512, clip 0.2, Adam 3·10⁻⁴,
  gradient norm clipped to 1. Reward −1 per wall hit, undiscounted (every game ends within E probes),
  GAE λ = 0.95. Loss weights: entropy 0.01, belief 1, value 0.05.
  - **Value weight 0.05, not the usual 0.5:** the value sums one term per side, so its gradient is
    large (global norm 36 vs 0.07 for the policy loss on a fresh network). At 0.5 the global clip
    shrank every step and the policy barely moved (KL ≈ 10⁻⁴ per update instead of 10⁻²).
  - **No magnet term (MMD / R-NaD) in P2.** Against a fixed designer mix the explorer solves a
    single-agent POMDP; regularizing toward a magnet matters when both sides learn and cycle, so it
    comes with P3.
- **Designers:** each new game draws its maze from a mix: uniform among valid mazes 20%, randomMaze
  15%, "normal" 15%, "hard" 30%, and an **adversary** 20%.
  - Hard has the largest share because the bench measures on hard's mazes.
  - The uniform share keeps the explorer honest on short direct paths, where a winding-maze prior
    loses: the expert hits 2.45 walls on uniform 5×5 mazes, hard 1.47.
  - The adversary aims at the learner itself, with the pick of D21's exploitability test. Every
    iteration it re-picks 128 of its 4,096 mazes, each the worst of 8 fresh randomMaze and uniform
    mazes (same budget) for the current greedy policy. It is P3's designer in miniature: an explorer
    with a fixed idea of what mazes look like gets exploited (D21).
  - Wall budgets: the default ± 2 (D15).
- **Maze supply:** designed mazes are slow (≈ 150 hard mazes/s on 5×5), so 40,000 per designer are
  made once on every core and cached; uniform and randomMaze pools (16,384) get 2,048 new mazes per
  iteration.
- **Evaluation in the loop:** every 25 iterations the greedy policy explores 512 held-out mazes per
  designer at the default budget, next to the hard explorer, a uniformly random explorer and the
  expert (256 of them) on the same mazes; plus the worst of 32 candidates over 128 rounds, a proxy
  for exploitability (`exploit.py`).
- **Bench:** the `learned` bot is the greedy policy of a checkpoint (default
  `rl/checkpoints/explorer-NxN.npz`, committed), with hard's designer, like the expert.
- **Planner prior: kept as an option, off.** `--plan true` adds to the policy logits
  −w · (expected walls on the cheapest way to the exit through each probe, planned on the belief
  head), i.e. the expert's rollout planner as a prior. It needs a good belief, and the belief head is
  weak so far (see below).
- **The minibatch loop is Python around one compiled gradient step:** XLA:CPU runs convolution
  gradients ~40× slower inside `lax.scan`.

**Measured** (Apple M1, CPU; one run per size from scratch, default config; ≈ 200–450 probes/s per
run while two runs shared the machine, ≈ 900 alone on 4×4). The 5×5 run started without the
adversary and got it at iteration 40 (`--resume --mix`).

Bench, 1,000 deals, mazes designed by the bots (the learned bot and the expert design like hard, so
these are walls hit on hard's mazes):

| board | learned (A) vs | hits A | hits B | explorer edge | score (dup) |
|---|---|---|---|---|---|
| 4×4, 7 walls (1.45 M probes, 65 min) | hard | 1.71 | 6.86 | +5.15 ± 0.04 | 100% |
| 4×4, 7 walls | expert | 1.70 | 1.88 | +0.18 ± 0.03 | 56% ± 1 |
| 5×5, 12 walls (2.06 M probes, 141 min) | hard | 4.09 | 11.54 | +7.45 ± 0.06 | 100% |
| 5×5, 12 walls | expert | 4.17 | 5.25 | +1.08 ± 0.06 | 64% ± 1 |

Expert vs hard on the same deals: 1.91 vs 6.87 (4×4), 5.30 vs 11.54 (5×5, as in D22).

Robustness: uniform mazes (bench `--mazes uniform`, 1,000 deals) and D21's adversarial pick (the
worst of 32 fresh mazes, 16 randomMaze + 16 uniform, for each explorer; `exploit`, 200 rounds):

| board | uniform: learned / expert / hard | worst of 32: learned / expert / hard |
|---|---|---|
| 4×4 | 2.53 / 2.57 / 1.08 | 6.19 / 6.07 / 6.91 |
| 5×5 | 3.20 / 2.65 / 1.58 | 9.55 / 9.49 / 11.55 |

(This setup's worst-of-32 numbers are higher than D21's 7.60 for the expert and 10.93 for hard,
which came from expert.js with its own candidates; compare within a table.)

**Findings.**
1. **It learns fast.** On 4×4 it beats hard after ~0.1 M probes (5 minutes) and reaches the expert on
   designed mazes at ~1.2 M. On 5×5 it is ahead of the expert on hard's mazes from 0.4 M probes on
   (4.87 vs 5.16 on the same mazes), and at 4.1 from 1.2 M. Uniform mazes improve slowly (4.25 →
   3.18). Neither run had plateaued on every designer when stopped.
2. **The policy learns the designers' style; the belief head carries less.** On hard 4×4 mazes, the
   belief head's cross-entropy is 0.49 nats per unknown side, vs 0.61 for the base rate (walls left
   / unknown sides). A network of the same size trained on the belief alone, on the same states,
   gets 0.47 before overfitting: the shared trunk doesn't starve it. Yet planning on these marginals
   with the expert's greedy planner hits 4.2 of 7 walls on hard mazes, against 1.65 for the policy
   head (and 1.3 vs 2.5 on uniform mazes, where it is better). Per-side marginals miss what the
   policy uses. So the planner prior (`--plan`) stays off: after 0.2 M probes it was no better than
   without (2.87 vs 2.62 walls on hard 4×4 mazes, both runs with the earlier value weight).
3. **The weakness is the expert's: what an explorer assumes about designs can be used against it.**
   On uniform mazes, which have short direct paths, hard (a straight-line planner) still does far
   better. On 5×5 the expert does better there too. The worst-of-32 pick pushes the learned
   explorer close to the budget, about as much as the expert. The adversary in the mix helped early:
   at 0.2 and 0.4 M probes, 4×4 hit 0.25 and 0.5 fewer walls on uniform mazes than a run without
   it. That run also used the larger value weight, so the two effects are mixed.
4. **The value weight.** At the usual 0.5 the policy barely moved for the first ~20 updates (KL
   10⁻⁴); at 0.05 it moves from the start. The value still explains ~50–70% of the variance of the
   returns.

**Next.**
- **P3:** a designer that learns, against the learned explorer. That is the systematic fix for
  finding 3; the adversary here is a fixed-budget stand-in.
- **A belief that plans well:** e.g. a sampler of whole mazes (particles) conditioned by the network,
  as the expert does, rather than per-side marginals (P6).
- **More compute:** neither run had plateaued when stopped. A bigger network (48 × 6) is 3× slower
  per sample on this CPU; a GPU would lift both limits.

## D24 — Train on Kaggle GPUs, develop on the Mac; no separate P2 re-run (2026-10-26)

**Context.** P2 (D23) was limited by the M1's CPU: the network's gradient runs at ≈ 3,500 samples/s,
a 48 × 6 network would be 3× slower, and neither run had plateaued when stopped (5×5: 2.06 M probes
in 141 min). The user has Kaggle notebooks with about 30 GPU hours per week (P100 or 2 × T4, sessions
up to ~12 h).

**Decision.**
- **Training runs go to Kaggle, on "GPU T4 × 2".** Not the P100: it's a Pascal GPU, and NVIDIA's next
  CUDA toolkit drops Pascal, so future JAX GPU builds may not run on it. The T4 is Turing (compute
  capability 7.5). Sessions are capped (~12 h), so training checkpoints regularly and resumes.
- **The Mac stays for development and judging:** tests, short 4×4 smoke runs, evaluation against the
  expert (Node bridge, D22), and shipping. Results come back as files: intermediate checkpoints in a
  private Kaggle dataset; small milestone checkpoints in `rl/checkpoints/` (larger ones as GitHub
  Releases in P4).
- **No standalone re-run of P2.**
  - P2 met its go/no-go on bot-made mazes (4.2 walls of 12 vs 5.3 for the expert).
  - Its two weak spots are what P3's learning designer exists to fix: uniform mazes (3.2 vs 2.7 for
    the expert) and the worst-of-32 adversarial pick (9.55 of 12).
  - Training P2 longer against the same fixed designers would polish a strategy P3 reshapes anyway.
- **The CPU limit that does matter, network size, is handled inside P3.**
  - **Calibration first** (≤ 30 GPU minutes): measure training probes/s on T4 × 2 for 32 × 4 (P2's),
    64 × 6 and 96 × 8. Keep the largest that still trains ≥ 5,000 probes/s: one 10-hour run is then
    ≥ 180 M probes, ~90× P2's 5×5 run. If the GPU isn't at least 5× faster than the Mac on this
    workload, stay on the Mac and record why.
  - **Stage 0 = P2's recipe on the GPU** with the chosen network (explorer vs the fixed designer
    mix, adversary included), until it matches or beats P2's checkpoint. This redoes P2 at almost
    no cost. P2's checkpoint stays as a frozen reference and league member.
- **Weekly budget (~30 GPU h):**
  - ≈ 1 h for calibration and stage 0;
  - two league runs of ≈ 11 h;
  - the rest for exploiters (measuring exploitability) and slack.

  Evaluation happens on the Mac between runs.
- **P3 targets** (5×5, 12 walls; guesses to steer by, not promises):
  - **Explorer:** the worst-of-32 adversarial pick (same `exploit` setup as D23) falls from 9.55
    toward ≤ 8.
  - **Designer:** the expert hits ≥ 7 of 12 walls on P3 designer mazes (5.3 on "hard" mazes today),
    and the mazes stay varied (no repeats; entropy reported).
  - **Exploitability:** a fresh exploiter trained against frozen P3 agents gains less from one
    generation to the next.
  - **Head to head:** duplicate-format scores against P2's explorer and the expert.
