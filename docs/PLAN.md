# Implementation plan

Living checklist. ✅ done · 🟡 partial · ⬜ todo. Design rationale lives in [DECISIONS.md](DECISIONS.md).

## Milestone 1 — Core game logic (pure, tested) ✅
- ✅ `grid.js` — cells, directions, edges, coordinates (A1…E5)
- ✅ `maze.js` — validation, BFS / shortest path, random maze generator (no E→S shortcuts), settings bounds
- ✅ `sha256.js` + commitment helpers (tested against `node:crypto`)
- ✅ `match.js` — derived game view: phase, turn order, progress, winner, verification
- ✅ Unit tests (`node --test`)

## Milestone 2 — Networking ✅
- ✅ `session.js` — state, snapshot sync, moves, rematch, resign, chat, heartbeat, persistence hook
- ✅ Transports: PeerJS (online), BroadcastChannel (`?net=local`), in-memory pipe (bot/tests)
- ✅ Host seat binding, room-full rejection, reconnection, seat hand-over before the game starts
- ✅ Self-healing: digest in pings → `resync`; pending move re-sent
- ✅ Protocol tests: full game, lost messages, disconnects, reload, intruder, tampering, resign, chat

## Milestone 3 — Bot ✅
- ✅ Maze designer (easy: random · normal: best of 6 · hard: best of 30 by simulated wall hits)
- ✅ Solver (optimistic planner + noise by difficulty)
- ✅ Expert explorer: Bayesian search over particle beliefs, counts walls, infers the designer's style ([D21](DECISIONS.md))
- ✅ Controller (thinking delays, auto-rematch, banter)
- ✅ Bot-vs-bot test

## Milestone 4 — UI ✅
- ✅ Home: name, create / join / vs computer, settings, saved games, rules
- ✅ Design: wall painting (click/drag), E/S tools, random, clear, live validation, path preview
- ✅ Play: explore board (cursor, click/arrows/WASD/ZQSD/d-pad), own maze with opponent progress, banner, log
- ✅ End: reveal with undiscovered walls dashed, verification, stats, rematch, scoreboard
- ✅ Chat + quick taunts, synthesized sounds (mutable), tab-title turn alerts
- ✅ EN/FR, light (notebook) / dark (chalkboard), phone layout
- ✅ Tab lock, invite link copy/share

## Milestone 5 — Publishing ✅
- ✅ README (rules, run locally, deploy to GitHub Pages, TURN/self-hosting notes)
- ✅ PROTOCOL.md
- ✅ LICENSE, .gitignore, .nojekyll, CI workflow running tests
- ✅ Manual browser test: bot game (incl. reload + rematch), two-tab local game, two-tab PeerJS game
  (incl. guest reload mid-game), tab takeover, mobile layout, light/dark

## Testing log
- 2026-10-21 — 18 automated tests green. Browser: full bot round + rematch; `?net=local` two-tab round;
  real WebRTC two-tab round through the public PeerJS server with a guest reload mid-game — reconnected
  and verified on both sides.
- 2026-10-21 — Published on GitHub Pages; CI green. Live site: two-tab online round over WebRTC,
  guest join screen, fair-play verification ✓.
- 2026-10-21 — First real remote test (friend in France → Safari): guest stuck on "Waiting for the
  host"; host page was not registered on the signaling server. Fixed per D13; reproduced locally
  (guest first, host reopens from "Your games" → auto-connect ~11 s; host reload → back in 2 s).
- 2026-10-21 — Retry: host online, but the connection fails (guest on home Wi-Fi and on a hotspot,
  host on school Wi-Fi): the school network blocks WebRTC incl. PeerJS's relay on port 3478 → D14.
  Added `iceServersUrl` support (tested: connects with a relay list, and falls back when the URL is
  broken). Waiting on a TURN provider account.
- 2026-10-22 — Expert level: 23 automated tests green (5 new for the particle sampler and the
  explorer; the bot-vs-bot protocol test now runs the expert). Browser: level picker shows four levels,
  the expert plays its turns; 5×5 moves take 13 ms median, 47 ms max (budget 150 ms).
- 2026-10-22 — RL P1 (`rl/`): 248 pytest tests green, 205 of them parity checks against JS fixtures;
  `npm test` unchanged and green; CI gains a Python job.
- Not yet tested: two different devices / networks end-to-end, Safari iOS as host.

## Bot strength and luck (2026-10-21)

Measured with `node scripts/benchmark.mjs` (bot vs bot; a player finishes on turn *walls hit + 1*,
the starter wins ties):

| | 5×5, 12 walls (default) | 7×7, 27 walls |
|---|---|---|
| hard beats normal | 80% | 93% |
| normal beats easy | 86% | 93% |
| hard vs hard: rounds won by the starter | **74%** | 51% |
| hard vs hard: equal number of turns needed | **49%** | 18% |

- Levels are real, but they are hand-tuned settings (randomness + number of candidate mazes), not
  measured ratings.
- A wall can't be hit twice, so good designs make the explorer hit nearly every wall (11.6 / 12).
  On 5×5 between strong players, half the rounds end in equal turns and the starter takes them.
  "Loser starts next" balances this over a match, not within a round. Bigger boards fix most of it.
- Open decision: default board size, or scoring matches as pairs of rounds (each player starts once).

**Expert level (2026-10-22, [D21](DECISIONS.md)).** Walls hit when exploring each designer's mazes
(`node scripts/benchmark.mjs 4 7 200` / `5 12 200`):

| explorer | 4×4, 7 walls: easy / normal / hard | 5×5, 12 walls: easy / normal / hard |
|---|---|---|
| hard | 4.7 / 6.3 / 6.9 | 7.8 / 10.6 / 11.5 |
| expert | 2.5 / 1.9 / 1.9 | 5.3 / 5.4 / 5.3 |

- Expert beats hard in 100% (4×4) and 99% (5×5) of rounds.
- Expert vs expert on 5×5: the starter wins 61% and turns are equal 15% of the time (hard vs hard:
  80% and 55%). A stronger explorer makes rounds less of a coin toss.

## Next: self-improving bot

Full design: [RL_DESIGN.md](RL_DESIGN.md); decisions D15–D20 (2026-10-22). Ground truth:
`research/exact_solve.mjs` (exact equilibrium on 3×3). Revised phases:
- P1: JAX engine + fixtures + baselines ✅ (2026-10-22): `rl/` ([README](../rl/README.md)). Parity green
  on 203 JS fixtures. Explorer steps/s on one CPU core: 2.5 M on 5×5 (target 10⁵), 0.46 M with the
  hard planner as policy; 10×10: 0.59 M / 0.11 M. Duplicate-format evaluator in place
- P2: explorer learning — step 0 ✅ (2026-10-22): no-learning Bayesian explorer shipped as the
  "expert" level ([D21](DECISIONS.md)); the learned explorer now has to beat it
- P3: league self-play on 5×5
- P4: ship
- P5: scale to 8×8/10×10
- P6: play-time search
- P7: per-player adaptation from real logs

- 2026-10-22 — Boards up to 10×10 in the game (bot design effort scaled with size, ≤ ~150 ms).

## Ideas / later
- Spectator mode
- Optional turn timer
- Change settings between rounds (today: create a new room)
- WebSocket relay fallback for networks where WebRTC fails
- Hot-seat mode (two players, one device, screens hidden in turn)
- Move-by-move replay at the end of a round
- PWA (installable, offline bot games)
- Edit your name from inside a game
