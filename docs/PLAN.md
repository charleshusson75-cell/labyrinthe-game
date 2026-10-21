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

## Next: self-improving bot

Full design: [RL_DESIGN.md](RL_DESIGN.md). Ground truth: `research/exact_solve.mjs` (exact equilibrium
on 3×3). Phases P1–P6 there: fast simulator with shared fixtures → Bayesian explorer → PSRO
self-play → ship checkpoints as rated levels → per-player adaptation.

## Ideas / later
- Spectator mode
- Optional turn timer
- Change settings between rounds (today: create a new room)
- WebSocket relay fallback for networks where WebRTC fails
- Hot-seat mode (two players, one device, screens hidden in turn)
- Move-by-move replay at the end of a round
- PWA (installable, offline bot games)
- Edit your name from inside a game
