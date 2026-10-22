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
