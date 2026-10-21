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
