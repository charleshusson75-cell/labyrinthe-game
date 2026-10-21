# Network protocol (v1)

Two players, **host** and **guest**, exchange JSON messages over a reliable, ordered channel
(a WebRTC data channel via PeerJS, a `BroadcastChannel` with `?net=local`, or an in-memory pipe
for bot games and tests). Implementation: [`src/net/session.js`](../src/net/session.js).

## Principles

1. **Each side is authoritative for its own data.** Nobody can write the other's state; you can only
   *ask* (a `move`) and the other side *answers* in its own snapshot.
2. **Snapshots, not events.** Besides `move` and `chat`, everything travels in a `sync` snapshot of
   the sender's authoritative data. Receiving a snapshot is an idempotent merge, so re-sending it is
   always safe. Reconnecting = sending `hello` + `sync` again.
3. **Mazes stay secret until the round ends.** During play only E/S and individual answers cross the
   wire.

## Connection

The host registers the PeerJS id `labyrinthe-duel-v1-<room code>`; the guest dials it.
The first message on a new connection must be a `hello`. The host checks it:

| condition | result |
|---|---|
| wrong `proto` | `reject {reason: "version"}` |
| no guest bound yet, or same `token` as the bound guest | accepted (the token is bound to the room) |
| different token, previous guest offline, round 1 not started | accepted (seat handed over) |
| otherwise | `reject {reason: "full"}` |

A newly accepted connection replaces the previous one (e.g. the guest reloaded the page).

## Messages

| `t` | from | fields | meaning |
|---|---|---|---|
| `hello` | both | `proto`, `seat`, `token`, `name` | first message on a connection |
| `sync` | both | see below | snapshot of the sender's authoritative data |
| `move` | prober | `r`, `seq`, `from`, `dir` | "try side `dir` of cell `from` in your maze" |
| `chat` | both | `text` | chat line (≤ 280 chars) |
| `ping` | both | `d` | heartbeat every 4 s, carries the digest of the sender's last `sync` |
| `resync` | both | | "your `ping` digest doesn't match the last `sync` I got: send it again" |
| `reject` | host | `reason` | connection refused (`full`, `version`, `seat`) |
| `bye` | both | | leaving (tab closed / navigated away) |

Cells are integers `row * size + col` (row 0 at the top). Directions are `U`, `R`, `D`, `L`.

### `sync`

```jsonc
{
  "t": "sync", "proto": 1,
  "r": 2,                       // round number
  "name": "Alice",
  "settings": { "size": 5, "walls": 12 }, // host only
  "starter": "guest",           // host only: who moves first this round
  "ready": "9f86d0…",           // SHA-256 commitment once the sender is ready, else null
  "es": { "entrance": 3, "exit": 21 },    // once BOTH are ready (sender knows the other's commitment)
  "answers": [ { "from": 3, "dir": "D", "open": true }, … ], // answers to the OTHER player's moves
  "rematch": false,
  "resigned": false,
  "reveal": { "maze": { "walls": ["0-1", …], "entrance": 3, "exit": 21 }, "salt": "…" }, // round over only
  "d": "1x9k2"                  // digest of this snapshot
}
```

Commitment: `sha256(JSON.stringify(["labyrinthe/1", size, sortedWalls, entrance, exit, salt]))`.
Walls are edge keys `"a-b"` with `a < b`, the two adjacent cells the wall separates.

## A round

```
host                                   guest
 │  design (secret)                      │  design (secret)
 │── sync{ready: H} ────────────────────►│
 │◄──────────────────── sync{ready: G} ──│
 │── sync{es: E/S of host maze} ────────►│   (each side sends E/S once it has both commitments)
 │◄────────── sync{es: E/S of guest maze}│
 │                                       │
 │  (starter = guest)                    │
 │◄──────── move{seq:0, from, dir} ──────│
 │── sync{answers:[{…, open:true}]} ────►│   open → guest plays again
 │◄──────── move{seq:1, …} ──────────────│
 │── sync{answers:[…, {open:false}]} ───►│   wall → host's turn
 │── move{seq:0, …} ────────────────────►│
 │           …                           │
 │   someone reaches the other's S       │
 │── sync{reveal} ─────────────────────► │   both reveal and verify
 │◄───────────────────── sync{reveal} ───│
 │── sync{rematch:true} ────────────────►│
 │◄───────────────── sync{rematch:true} ─│   both → round r+1, loser starts
```

Turn order is not sent: both sides derive it from `starter` and the two answer lists (a player keeps
the turn until an answer says `open: false`).

## Robustness

- **Duplicate / late moves.** The owner only accepts `move.seq === answers.length`. A lower `seq`
  means the answer got lost: the owner re-sends its `sync`.
- **Lost messages.** A pending move is re-sent every 5 s and on reconnect. `ping` carries a digest of
  the last snapshot; a mismatch triggers `resync`.
- **Dead connections.** No message for 14 s → considered offline; the guest redials, the host waits.
- **Reloads.** State is persisted per seat in `localStorage`; on reconnect both sides re-send
  everything.
- **Round drift.** If one side is one round ahead (it saw both rematch flags first), the other side
  catches up when it receives that snapshot.
