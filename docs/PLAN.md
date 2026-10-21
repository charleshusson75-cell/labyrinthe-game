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
- Not yet tested: two different devices / networks (needs the site deployed), Safari iOS, strict NATs.

## Ideas / later
- Spectator mode
- Optional turn timer
- Change settings between rounds (today: create a new room)
- WebSocket relay fallback for networks where WebRTC fails
- Hot-seat mode (two players, one device, screens hidden in turn)
- Move-by-move replay at the end of a round
- PWA (installable, offline bot games)
- Edit your name from inside a game
