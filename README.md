# Labyrinthe

**The pen-and-paper maze duel, online.** Each player secretly draws a maze, then you race to find
the exit of your opponent's maze, one guess at a time. First to reach the other's **S** wins.

**▶ Play: https://charleshusson75-cell.github.io/labyrinthe-game/**

- 🧑‍🤝‍🧑 **Play a friend online.** Create a game, send the link, each plays on their own screen.
- 🤖 **Play the computer** (easy / normal / hard).
- 🔒 **No peeking, no cheating.** Your maze never leaves your browser during the game, and at the end
  both mazes are checked against a fingerprint locked in before the first move.
- 🔄 Survives page reloads, Wi-Fi drops and phones going to sleep.
- 🌍 English / Français · 🌗 light (notebook) and dark (chalkboard) themes · 📱 works on phones.
- No server, no account, no build step, no dependencies.

## How to play

1. Each player draws a maze on their own grid (5×5 by default): an exact number of **walls**
   (a wall is one side of a cell), an **entrance E** and an **exit S** (*sortie*).
2. When both are ready, you only learn where your opponent's E and S are.
3. On your turn, try to move from a cell you have reached (you start on E) to a neighbouring cell.
   - **Open?** You move and play again.
   - **Wall?** It gets drawn on your map and the turn passes.
4. You can continue from any cell you have already reached. Dead end? Jump back to a fork.
5. First to reach the other's S wins. **The loser starts the next round.**
6. At the end both mazes are revealed and verified.

Controls: click a highlighted cell, or use the arrow keys / WASD / ZQSD (or the on-screen pad on
phones). Click any cell you've reached to continue from there. Walking along passages you already
know is free. Use the chat to bluff.

> 💡 A wall you've found can't be hit twice, so with 12 walls you need at most 13 turns. Designing a
> maze is all about making your opponent bump into as many walls as possible before they find S.

## Run it locally

You need [Node.js](https://nodejs.org) 18+ (only for the dev server and tests; the game itself is
plain HTML/CSS/JS).

```bash
npm start
```

Then open http://localhost:8080. Any static file server works too (`python3 -m http.server 8080`).

Online games between two tabs on your machine work out of the box (they go through the public PeerJS
server). To test without any network, add `?net=local` to the URL: the two tabs then talk through a
`BroadcastChannel`, e.g. http://localhost:8080/?net=local#/

Run the tests (rules, protocol, reconnection, cheating detection, bot):

```bash
npm test
```

Measure the bots against each other (win rates, how often the starter wins):

```bash
node scripts/benchmark.mjs 5 12 200
```

## Publish it on GitHub Pages

1. Push this folder to a GitHub repository.
2. In the repository: **Settings → Pages → Build and deployment → Source: Deploy from a branch**,
   pick `main` and `/ (root)`, save.
3. After a minute the game is live at `https://<your-user>.github.io/<repo>/`. Share that link,
   or create a game and send the invite link it gives you.

The CI workflow (`.github/workflows/ci.yml`) runs the tests on every push and pull request.

## How online play works

```
 Alice's browser                                Bob's browser
 ┌───────────────────┐   WebRTC data channel   ┌───────────────────┐
 │ her maze (secret) │◄───────────────────────►│ his maze (secret) │
 │ answers his moves │   moves, answers, chat  │ answers her moves │
 └───────────────────┘                         └───────────────────┘
            ▲                                            ▲
            └────── PeerJS signaling (connection setup only) ──┘
```

- **Peer-to-peer.** Browsers connect directly with WebRTC, using [PeerJS](https://peerjs.com) and its
  free public server only to find each other. The game site is purely static.
- **Secret mazes.** When Bob tries a move in Alice's maze, *Alice's* browser answers "open" or
  "wall", just like on paper. Her walls are never sent until the round is over.
- **Commit–reveal.** On *Ready*, each browser publishes a SHA-256 hash of its maze plus a random salt.
  At the end, mazes and salts are revealed; each side checks the hash, the rules, and every answer it
  received. A modified client that lies gets flagged on the end screen.
- **Resilient.** Each side periodically re-sends a snapshot of what it is responsible for, so a
  dropped connection or a page reload just resumes. Game state is saved in `localStorage`.

Details: [docs/PROTOCOL.md](docs/PROTOCOL.md) · design choices: [docs/DECISIONS.md](docs/DECISIONS.md) ·
roadmap: [docs/PLAN.md](docs/PLAN.md).

### If players can't connect

Most home networks are fine. Some strict networks (corporate Wi-Fi, some mobile carriers) block
direct WebRTC connections; the game then shows a hint after a few seconds. Fixes:

- try another network (a phone hotspot usually works), or
- configure a **TURN relay on port 443** in [`src/config.js`](src/config.js): set `iceServersUrl` to
  your provider's credentials URL (e.g. Metered's free plan:
  `https://<your-app>.metered.live/api/v1/turn/credentials?apiKey=<key>`), or put servers directly in
  `peerOptions.config.iceServers`. You can also run [coturn](https://github.com/coturn/coturn).

You can also self-host the signaling server (`npx peerjs --port 9000`) and point `peerOptions` at it.

## Project structure

```
index.html, styles.css, favicon.svg   the page
src/
  main.js            routing (#/, #/room/<code>, #/join/<code>, #/bot/<code>) and header
  config.js          PeerJS / TURN settings
  i18n.js            English & French strings
  game/              pure rules, no DOM: grid, maze, match (turns, winner, verification), sha256
  net/               session (state + protocol), transports: PeerJS, BroadcastChannel, in-memory pipe
  bot/               computer player: maze designer, solver, controller
  app/               routing glue: rooms, saved games & preferences, tab lock
  ui/                screens (home, game), SVG board, sounds, DOM helpers
tests/               node --test suites
scripts/serve.mjs    zero-dependency dev server
docs/                decisions, plan, protocol
```

## License

[MIT](LICENSE)
