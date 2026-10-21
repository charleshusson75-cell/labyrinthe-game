import { test } from 'node:test';
import assert from 'node:assert/strict';

import { createSession } from '../src/net/session.js';
import { createPipe } from '../src/net/pipe.js';
import { seededRng } from '../src/game/random.js';
import { chooseProbe } from '../src/bot/brain.js';
import { createBotController } from '../src/bot/controller.js';

const SETTINGS = { size: 4, walls: 3 };
// Host maze: E top-left, S bottom-right. Guest maze: mirrored.
const HOST_MAZE = { walls: ['0-1', '5-6', '10-14'], entrance: 0, exit: 15 };
const GUEST_MAZE = { walls: ['2-3', '4-8', '9-13'], entrance: 3, exit: 12 };

function memoryStorage() {
  let data = null;
  return {
    load: () => (data ? JSON.parse(data) : null),
    save: (s) => {
      data = JSON.stringify(s);
    },
  };
}

function setup({ seed = 1, delay = 0, heartbeat = false } = {}) {
  const timing = { pingEvery: 20, peerTimeout: 400, resendMoveAfter: 60 };
  const pipe = createPipe({ delay });
  const storage = { host: memoryStorage(), guest: memoryStorage() };
  const host = createSession({
    seat: 'host',
    roomCode: 'test',
    transport: pipe.host,
    storage: storage.host,
    init: { name: 'Alice', settings: SETTINGS },
    rng: seededRng(seed),
    heartbeat,
    timing,
  });
  const guest = createSession({
    seat: 'guest',
    roomCode: 'test',
    transport: pipe.guest,
    storage: storage.guest,
    init: { name: 'Bob' },
    rng: seededRng(seed + 1),
    heartbeat,
    timing,
  });
  return { pipe, host, guest, storage };
}

async function designAndReady(pipe, host, guest) {
  await pipe.settled();
  assert.ok(host.setMaze(HOST_MAZE));
  assert.ok(guest.setMaze(GUEST_MAZE));
  assert.ok(host.ready());
  await pipe.settled();
  assert.equal(guest.view().oppReady, true);
  assert.equal(host.view().phase, 'design', 'host waits for guest');
  assert.ok(guest.ready());
  await pipe.settled();
}

/** Let both sides play with the bot brain until someone wins. */
async function playOut(pipe, sessions, { rng = seededRng(7), beforeEach, afterMoves } = {}) {
  for (let i = 0; i < 400; i++) {
    await pipe.settled();
    if (sessions.every((s) => s.view().phase === 'over')) return;
    await beforeEach?.(i);
    for (const s of sessions) {
      const v = s.view();
      if (v.phase === 'play' && v.turn === s.state.seat && !s.state.round.pending) {
        const p = chooseProbe({ size: v.size, progress: v.mine, exit: s.state.round.oppES.exit, level: 'normal', rng });
        assert.equal(s.probe(p.from, p.dir), null);
      }
    }
    await afterMoves?.(i);
  }
  throw new Error('game did not finish');
}

test('a full game: design, play, verify, rematch with loser starting', async () => {
  const { pipe, host, guest } = setup();
  await pipe.settled();
  assert.equal(guest.state.settings.size, 4, 'guest received settings');
  assert.equal(host.state.names.guest, 'Bob');
  assert.equal(guest.state.names.host, 'Alice');

  await designAndReady(pipe, host, guest);
  for (const s of [host, guest]) assert.equal(s.view().phase, 'play');
  assert.deepEqual(host.state.round.oppES, { entrance: 3, exit: 12 });
  assert.equal(host.view().turn, guest.view().turn);

  // Editing is locked once ready.
  assert.equal(host.setWall('1-2', true), 'locked');

  await playOut(pipe, [host, guest]);
  const winner = host.view().winner;
  assert.ok(winner);
  assert.equal(guest.view().winner, winner);
  assert.equal(host.state.round.verification.ok, true);
  assert.equal(guest.state.round.verification.ok, true);
  assert.equal(host.state.history.length, 1);
  assert.deepEqual(host.state.history[0].moves, guest.state.history[0].moves);

  assert.ok(host.requestRematch());
  await pipe.settled();
  assert.equal(host.state.round.n, 1, 'needs both players');
  assert.ok(guest.requestRematch());
  await pipe.settled();
  for (const s of [host, guest]) {
    assert.equal(s.state.round.n, 2);
    assert.equal(s.state.round.starter, winner === 'host' ? 'guest' : 'host');
    assert.equal(s.view().phase, 'design');
    assert.equal(s.state.round.myMaze.walls.length, 3, 'previous design kept as draft');
  }
});

test('illegal and out-of-turn probes are refused', async () => {
  const { pipe, host, guest } = setup();
  await designAndReady(pipe, host, guest);
  const v = host.view();
  const [mover, waiter] = v.turn === 'host' ? [host, guest] : [guest, host];
  assert.equal(waiter.probe(waiter.state.round.oppES.entrance, 'R'), 'not-your-turn');
  const e = mover.state.round.oppES.entrance;
  assert.equal(mover.probe(e === 0 ? 1 : 0, 'D'), 'not-reached');
  assert.equal(mover.probe(e, 'U'), 'border');
  assert.equal(mover.probe(e, 'D'), null);
  assert.equal(mover.probe(e, 'L'), 'pending', 'one probe at a time');
  await pipe.settled();
  assert.equal(mover.state.round.pending, null);
  assert.equal(mover.state.round.probes[mover.state.seat].length, 1);
});

test('survives disconnections that lose messages in flight', async () => {
  const { pipe, host, guest } = setup({ seed: 3 });
  await designAndReady(pipe, host, guest);
  let cuts = 0;
  await playOut(pipe, [host, guest], {
    afterMoves: async (i) => {
      if (i % 2 === 0) {
        // The link dies right after a move was sent: the move (or its answer) is lost.
        cuts++;
        pipe.disconnect();
        await pipe.settled();
        pipe.connect();
      }
    },
  });
  assert.ok(cuts > 3);
  assert.equal(host.view().winner, guest.view().winner);
  assert.equal(host.state.round.verification?.ok, true);
});

test('heartbeat heals messages lost on a live link', async () => {
  const { pipe, host, guest } = setup({ seed: 8, heartbeat: true });
  await designAndReady(pipe, host, guest);
  const lossy = async () => {
    pipe.drop(1); // lose whatever comes next: a move or its answer
  };
  try {
    await playOut(pipe, [host, guest], {
      afterMoves: async (i) => {
        if (i % 3 === 0) await lossy();
        await new Promise((r) => setTimeout(r, 30)); // give pings a chance to run
      },
    });
  } finally {
    host.destroy();
    guest.destroy();
  }
  assert.equal(host.view().winner, guest.view().winner);
  assert.equal(guest.state.round.verification?.ok, true);
});

test('a guest can reload mid-game and resume', async () => {
  const { pipe, host, guest, storage } = setup({ seed: 5 });
  await designAndReady(pipe, host, guest);
  let current = guest;
  let reloaded = false;
  await playOut(pipe, [host, { view: () => current.view(), get state() { return current.state; }, probe: (...a) => current.probe(...a) }], {
    beforeEach: async (i) => {
      if (i === 2 && !reloaded) {
        reloaded = true;
        current.destroy();
        await pipe.settled();
        current = createSession({
          seat: 'guest',
          roomCode: 'test',
          transport: pipe.guest,
          storage: storage.guest,
          rng: seededRng(99),
          heartbeat: false,
        });
        await pipe.settled();
        pipe.connect();
      }
    },
  });
  assert.ok(reloaded);
  assert.equal(current.state.names.guest, 'Bob', 'restored from storage');
  assert.equal(host.view().winner, current.view().winner);
  assert.equal(current.state.round.verification?.ok, true);
});

test('a third person gets "room full" once the game started', async () => {
  const { pipe, host, guest } = setup();
  await designAndReady(pipe, host, guest);
  guest.destroy();
  await pipe.settled();
  const intruder = createSession({
    seat: 'guest',
    roomCode: 'test',
    transport: pipe.guest,
    init: { name: 'Eve' },
    rng: seededRng(1234),
    heartbeat: false,
  });
  pipe.connect();
  await pipe.settled();
  assert.equal(intruder.live.error?.code, 'rejected');
  assert.equal(intruder.live.error?.reason, 'full');
  assert.equal(intruder.state.settings, null);
});

test('tampering with a maze after committing is detected', async () => {
  const { pipe, host, guest } = setup({ seed: 11 });
  await designAndReady(pipe, host, guest);
  // A cheating host swaps a wall after "ready" (the UI forbids it, a hacked client wouldn't).
  host.state.round.myMaze.walls = ['0-1', '5-6', '11-15'];
  await playOut(pipe, [host, guest]);
  assert.equal(guest.state.round.verification.ok, false);
  assert.ok(guest.state.round.verification.problems.some((p) => p.code === 'hash-mismatch'));
  assert.equal(host.state.round.verification.ok, true, 'the honest side checks out');
});

test('resigning ends the round', async () => {
  const { pipe, host, guest } = setup();
  await designAndReady(pipe, host, guest);
  assert.ok(guest.resign());
  await pipe.settled();
  assert.equal(host.view().winner, 'host');
  assert.equal(guest.view().winner, 'host');
  assert.equal(host.state.round.verification.ok, true);
});

test('chat goes both ways', async () => {
  const { pipe, host, guest } = setup();
  await pipe.settled();
  assert.ok(host.sendChat('  hello   there '));
  await pipe.settled();
  assert.deepEqual(
    guest.state.chat.map((c) => [c.from, c.text]),
    [['host', 'hello there']],
  );
});

test('bot vs bot plays several rounds', async () => {
  const { pipe, host, guest } = setup({ seed: 21 });
  const bots = [
    createBotController(host, { level: 'hard', rng: seededRng(1), speed: 0 }),
    createBotController(guest, { level: 'easy', rng: seededRng(2), speed: 0 }),
  ];
  for (let i = 0; i < 2000 && host.state.history.length < 3; i++) {
    await pipe.settled();
    if (host.view().phase === 'over' && !host.state.round.rematch.host) host.requestRematch();
  }
  bots.forEach((b) => b.destroy());
  assert.ok(host.state.history.length >= 3, `played ${host.state.history.length} rounds`);
  assert.deepEqual(host.state.history.map((h) => h.winner), guest.state.history.map((h) => h.winner));
});
