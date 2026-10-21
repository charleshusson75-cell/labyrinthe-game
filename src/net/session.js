// One player's side of a game. Owns the local state, talks to the opponent through a transport,
// and exposes the actions the UI (or the bot) can take. No DOM in here.
//
// Replication model (see docs/PROTOCOL.md): each side sends a `sync` snapshot of the data it is
// authoritative for; receiving one is an idempotent merge. Moves are requests (`move`), answered by
// the maze owner through its next snapshot.

import { isCell, isDir, isEdge } from '../game/grid.js';
import {
  commitmentOf,
  deriveView,
  newRound,
  other,
  probeError,
  verifyReveal,
} from '../game/match.js';
import { canonicalMaze, emptyMaze, isOpen, normalizeSettings, randomMaze } from '../game/maze.js';
import { randomHex, secureRng } from '../game/random.js';

export const PROTO = 1;
const TIMING = { pingEvery: 4000, peerTimeout: 14000, resendMoveAfter: 5000 };
const MAX_CHAT = 150;
const MAX_TEXT = 280;
const MAX_NAME = 24;

/** Cheap string hash (FNV-1a) used to notice when the two sides' snapshots diverged. */
function digest(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(36);
}

/**
 * @param {object} o
 * @param {'host'|'guest'} o.seat
 * @param {string} o.roomCode
 * @param {object} o.transport  { start(handlers), send(msg), reset?(), destroy() }
 * @param {object} [o.storage]  { load(): state|null, save(state) }
 * @param {object} [o.init]     { name, settings (host), meta }
 */
export function createSession({
  seat,
  roomCode,
  transport,
  storage = null,
  init = {},
  rng = secureRng(),
  now = Date.now,
  heartbeat = true,
  timing = TIMING,
}) {
  const opp = other(seat);
  let state = storage?.load() ?? freshState();
  const live = {
    link: 'starting', // transport: starting | ready | error
    peerOnline: false,
    error: null, // { code, ... }
    lastSeen: 0,
  };
  const listeners = new Set();
  let lastSent = '';
  let lastSentDigest = '';
  let lastRecvDigest = '';
  let pendingSentAt = 0;
  let pingTimer = null;
  let destroyed = false;

  function freshState() {
    const s = {
      v: 1,
      roomCode,
      seat,
      token: randomHex(rng, 12),
      boundToken: null,
      settings: seat === 'host' ? normalizeSettings(init.settings) : null,
      names: { host: '', guest: '' },
      joined: false,
      round: newRound(1, seat === 'host' ? (rng() < 0.5 ? 'host' : 'guest') : null),
      history: [],
      chat: [],
      meta: init.meta ?? null,
    };
    s.names[seat] = (init.name ?? '').slice(0, MAX_NAME);
    return s;
  }

  // ---------------------------------------------------------------- plumbing

  const emit = (type, detail) => listeners.forEach((fn) => fn(type, detail));
  const event = (detail) => emit('event', detail);
  const size = () => state.settings?.size ?? 0;

  function save() {
    try {
      storage?.save(state);
    } catch {
      /* storage full or unavailable: the game still works, it just won't survive a reload */
    }
  }

  function send(msg) {
    if (live.peerOnline) transport.send(msg);
  }

  function snapshot() {
    const r = state.round;
    const view = deriveView(state);
    const snap = {
      t: 'sync',
      proto: PROTO,
      r: r.n,
      name: state.names[seat],
      ready: r.myCommit,
      answers: r.probes[opp],
      rematch: r.rematch[seat],
      resigned: r.resigned === seat,
    };
    if (seat === 'host') {
      snap.settings = state.settings;
      snap.starter = r.starter;
    }
    if (r.myCommit && r.oppCommit) snap.es = { entrance: r.myMaze.entrance, exit: r.myMaze.exit };
    if (view.phase === 'over' && r.myCommit) snap.reveal = { maze: r.myMaze, salt: r.mySalt };
    return snap;
  }

  function sendSync(force = false) {
    if (!live.peerOnline) return;
    const snap = snapshot();
    const json = JSON.stringify(snap);
    if (!force && json === lastSent) return;
    lastSent = json;
    lastSentDigest = digest(json);
    transport.send({ ...snap, d: lastSentDigest });
  }

  function sendPendingMove() {
    const p = state.round.pending;
    if (!p || !live.peerOnline) return;
    pendingSentAt = now();
    transport.send({ t: 'move', r: state.round.n, ...p });
  }

  /** Run a state change, then do the bookkeeping every change needs. */
  function mutate(fn) {
    const before = deriveView(state);
    fn();
    let view = deriveView(state);
    const r = state.round;

    if (view.phase === 'over') {
      if (!r.recorded) {
        r.recorded = true;
        state.history.push({
          n: r.n,
          winner: view.winner,
          resigned: r.resigned,
          moves: { host: r.probes.host.length, guest: r.probes.guest.length },
        });
      }
      if (r.oppReveal && !r.verification) {
        r.verification = verifyReveal({
          reveal: r.oppReveal,
          commit: r.oppCommit,
          es: r.oppES,
          answers: r.probes[seat],
          settings: state.settings,
        });
        event({ kind: 'verified', ok: r.verification.ok });
      }
      if (r.rematch.host && r.rematch.guest) {
        // Loser starts the next round. Keep my last design as a starting draft.
        state.round = newRound(r.n + 1, other(view.winner), r.myMaze);
        view = deriveView(state);
        event({ kind: 'new-round', n: state.round.n });
      }
    }

    if (before.phase !== view.phase) event({ kind: 'phase', phase: view.phase, prev: before.phase });
    if (view.phase === 'play' && view.turn === seat && (before.turn !== seat || before.phase !== 'play')) {
      event({ kind: 'your-turn' });
    }
    save();
    emit('change');
    sendSync();
  }

  function touch() {
    emit('change');
  }

  // ---------------------------------------------------------------- incoming

  function authorize(hello) {
    if (hello?.proto !== PROTO) return { ok: false, reason: 'version' };
    if (hello.seat !== 'guest' || typeof hello.token !== 'string') return { ok: false, reason: 'seat' };
    if (!state.boundToken || state.boundToken === hello.token) {
      state.boundToken = hello.token;
      save();
      return { ok: true };
    }
    // A different browser wants the seat. Allow it only before the game really started,
    // and only if the previous guest is gone (e.g. they switched device in the lobby).
    const r = state.round;
    if (!live.peerOnline && r.n === 1 && !r.oppCommit) {
      state.boundToken = hello.token;
      state.names[opp] = '';
      save();
      return { ok: true };
    }
    return { ok: false, reason: 'full' };
  }

  function onOpen() {
    const wasOnline = live.peerOnline;
    live.peerOnline = true;
    live.lastSeen = now();
    live.error = null;
    lastSent = '';
    transport.send({ t: 'hello', proto: PROTO, seat, token: state.token, name: state.names[seat] });
    sendSync(true);
    sendPendingMove();
    if (!wasOnline) event({ kind: 'opp-online' });
    touch();
  }

  function onClose() {
    if (!live.peerOnline) return;
    live.peerOnline = false;
    event({ kind: 'opp-offline' });
    touch();
  }

  function onMessage(msg) {
    if (destroyed || !msg || typeof msg !== 'object') return;
    live.lastSeen = now();
    switch (msg.t) {
      case 'hello':
        if (msg.proto !== PROTO) {
          live.error = { code: 'version' };
          touch();
        } else if (typeof msg.name === 'string' && msg.name !== state.names[opp]) {
          state.names[opp] = msg.name.slice(0, MAX_NAME);
          save();
          touch();
        }
        break;
      case 'sync':
        lastRecvDigest = msg.d;
        handleSync(msg);
        break;
      case 'ping':
        // Their last snapshot isn't the one we hold (a message got lost): ask for it again.
        if (msg.d && msg.d !== lastRecvDigest) transport.send({ t: 'resync' });
        break;
      case 'resync':
        sendSync(true);
        break;
      case 'move':
        handleMove(msg);
        break;
      case 'chat':
        if (typeof msg.text === 'string') addChat(opp, msg.text);
        break;
      case 'reject':
        live.error = { code: 'rejected', reason: msg.reason };
        live.link = 'error';
        live.peerOnline = false;
        transport.destroy();
        touch();
        break;
      case 'bye':
        onClose();
        break;
      default: // unknown: only refresh lastSeen
    }
  }

  function handleSync(msg) {
    if (msg.proto !== PROTO) {
      live.error = { code: 'version' };
      touch();
      return;
    }
    let advanced = false;
    mutate(() => {
      if (typeof msg.name === 'string') state.names[opp] = msg.name.slice(0, MAX_NAME);
      if (!state.joined) {
        state.joined = true;
        event({ kind: 'opp-joined' });
      }
      if (seat === 'guest' && msg.settings && !state.settings) state.settings = normalizeSettings(msg.settings);

      // Round alignment. Normally both sides are on the same round.
      let r = state.round;
      if (msg.r !== r.n) {
        const view = deriveView(state);
        const theyAdvanced = msg.r === r.n + 1 && view.phase === 'over' && r.rematch[seat];
        const theyWillAdvance = msg.r === r.n - 1 && msg.rematch;
        if (theyAdvanced) {
          r.rematch[opp] = true; // the advance itself happens in mutate()'s bookkeeping
          advanced = true;
          return;
        }
        if (seat === 'guest' && !theyWillAdvance && Number.isInteger(msg.r) && msg.r > 0) {
          state.round = newRound(msg.r, msg.starter ?? null, r.myMaze); // lost track: follow the host
          r = state.round;
        } else {
          return;
        }
      }
      if (seat === 'guest' && (msg.starter === 'host' || msg.starter === 'guest')) r.starter = msg.starter;

      if (typeof msg.ready === 'string' && !r.oppCommit) {
        r.oppCommit = msg.ready;
        event({ kind: 'opp-ready' });
      }
      const es = msg.es;
      if (es && !r.oppES && r.oppCommit && isCell(es.entrance, size()) && isCell(es.exit, size()) && es.entrance !== es.exit) {
        r.oppES = { entrance: es.entrance, exit: es.exit };
      }
      mergeAnswers(msg.answers);
      if (msg.rematch && !r.rematch[opp]) {
        r.rematch[opp] = true;
        event({ kind: 'rematch-request' });
      }
      if (msg.resigned && !r.resigned && deriveView(state).phase === 'play') r.resigned = opp;
      if (msg.reveal && !r.oppReveal) r.oppReveal = sanitizeReveal(msg.reveal);
    });
    // They were one round ahead: now that we caught up, merge the rest of their snapshot.
    if (advanced && msg.r === state.round.n) handleSync(msg);
  }

  function mergeAnswers(answers) {
    const r = state.round;
    if (!Array.isArray(answers) || !r.oppES) return;
    const mine = r.probes[seat];
    for (let i = mine.length; i < answers.length; i++) {
      const a = answers[i];
      if (!isCell(a?.from, size()) || !isDir(a?.dir) || typeof a.open !== 'boolean') break;
      mine.push({ from: a.from, dir: a.dir, open: a.open });
      event({ kind: 'probe', who: 'me', from: a.from, dir: a.dir, open: a.open });
    }
    if (r.pending && r.pending.seq < mine.length) r.pending = null;
  }

  function sanitizeReveal(rv) {
    const m = rv?.maze ?? {};
    return {
      maze: {
        walls: Array.isArray(m.walls) ? m.walls.filter((k) => typeof k === 'string') : [],
        entrance: m.entrance,
        exit: m.exit,
      },
      salt: String(rv?.salt ?? ''),
    };
  }

  /** The opponent probes my maze: I am the one who answers. */
  function handleMove(msg) {
    const r = state.round;
    if (msg.r !== r.n || !Number.isInteger(msg.seq)) return;
    const list = r.probes[opp];
    if (msg.seq < list.length) return sendSync(true); // they missed my answer
    if (msg.seq > list.length) return;
    const view = deriveView(state);
    if (!view.bothReady || view.winner || view.turn !== opp || probeError(view.theirs, msg.from, msg.dir, size())) {
      return sendSync(true);
    }
    mutate(() => {
      const open = isOpen(new Set(r.myMaze.walls), msg.from, msg.dir, size());
      list.push({ from: msg.from, dir: msg.dir, open });
      event({ kind: 'probe', who: 'them', from: msg.from, dir: msg.dir, open });
    });
  }

  function addChat(from, text) {
    const clean = text.replace(/\s+/g, ' ').trim().slice(0, MAX_TEXT);
    if (!clean) return null;
    const entry = { from, text: clean, ts: now() };
    state.chat.push(entry);
    if (state.chat.length > MAX_CHAT) state.chat.splice(0, state.chat.length - MAX_CHAT);
    save();
    event({ kind: 'chat', entry });
    touch();
    return entry;
  }

  // ---------------------------------------------------------------- actions

  const inDesign = () => {
    const v = deriveView(state);
    return v.phase === 'design' && !v.iAmReady;
  };

  function editMaze(fn) {
    if (!inDesign()) return false;
    mutate(() => fn(state.round.myMaze));
    return true;
  }

  const api = {
    get state() {
      return state;
    },
    get live() {
      return live;
    },
    view: () => deriveView(state),

    subscribe(fn) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },

    setName(name) {
      const clean = String(name ?? '').trim().slice(0, MAX_NAME);
      if (clean === state.names[seat]) return;
      mutate(() => {
        state.names[seat] = clean;
      });
    },

    /** Add or remove one wall. Returns 'ok' | 'budget' (no walls left) | 'locked'. */
    setWall(key, on) {
      if (!inDesign() || !isEdge(key, size())) return 'locked';
      const walls = state.round.myMaze.walls;
      const has = walls.includes(key);
      if (has === on) return 'ok';
      if (on && walls.length >= state.settings.walls) return 'budget';
      return editMaze((m) => {
        m.walls = on ? [...m.walls, key] : m.walls.filter((k) => k !== key);
      })
        ? 'ok'
        : 'locked';
    },

    toggleWall(key) {
      return api.setWall(key, !state.round.myMaze.walls.includes(key));
    },

    setEntrance(cell) {
      return editMaze((m) => {
        if (m.exit === cell) m.exit = null;
        m.entrance = m.entrance === cell ? null : cell;
      });
    },

    setExit(cell) {
      return editMaze((m) => {
        if (m.entrance === cell) m.entrance = null;
        m.exit = m.exit === cell ? null : cell;
      });
    },

    setMaze(maze) {
      return editMaze(() => {
        state.round.myMaze = canonicalMaze(maze);
      });
    },

    clearMaze() {
      return editMaze(() => {
        state.round.myMaze = emptyMaze();
      });
    },

    randomizeMaze() {
      return api.setMaze(randomMaze(state.settings, rng));
    },

    /** Lock my design and publish its commitment. */
    ready() {
      const v = deriveView(state);
      if (v.phase !== 'design' || v.iAmReady || v.issues.length) return false;
      mutate(() => {
        const r = state.round;
        r.myMaze = canonicalMaze(r.myMaze);
        r.mySalt = randomHex(rng, 16);
        r.myCommit = commitmentOf(r.myMaze, r.mySalt, size());
      });
      return true;
    },

    /** Try the `dir` side of cell `from` in the opponent's maze. Returns null or an error code. */
    probe(from, dir) {
      const v = deriveView(state);
      const r = state.round;
      if (v.phase !== 'play') return 'not-playing';
      if (v.turn !== seat) return 'not-your-turn';
      if (r.pending) return 'pending';
      const err = probeError(v.mine, from, dir, size());
      if (err) return err;
      mutate(() => {
        r.pending = { seq: r.probes[seat].length, from, dir };
      });
      sendPendingMove();
      return null;
    },

    resign() {
      if (deriveView(state).phase !== 'play') return false;
      mutate(() => {
        state.round.resigned = seat;
      });
      return true;
    },

    requestRematch() {
      if (deriveView(state).phase !== 'over') return false;
      mutate(() => {
        state.round.rematch[seat] = true;
      });
      return true;
    },

    sendChat(text) {
      if (!live.peerOnline) return false;
      const entry = addChat(seat, String(text ?? ''));
      if (entry) send({ t: 'chat', text: entry.text });
      return !!entry;
    },

    /** Politely tell the opponent we're leaving, then stop everything. */
    destroy() {
      if (destroyed) return;
      try {
        send({ t: 'bye' });
      } catch {
        /* ignore */
      }
      destroyed = true;
      clearInterval(pingTimer);
      transport.destroy();
      listeners.clear();
    },
  };

  // ---------------------------------------------------------------- start

  transport.start({
    onStatus(link, error = null) {
      if (destroyed) return;
      live.link = link;
      if (link !== 'error' || error) live.error = error;
      touch();
    },
    onOpen,
    onClose,
    onMessage,
    authorize: seat === 'host' ? authorize : undefined,
  });

  if (heartbeat) {
    pingTimer = setInterval(() => {
      if (!live.peerOnline) return;
      transport.send({ t: 'ping', d: lastSentDigest });
      if (state.round.pending && now() - pendingSentAt > timing.resendMoveAfter) sendPendingMove();
      if (now() - live.lastSeen > timing.peerTimeout) {
        onClose();
        transport.reset?.();
      }
    }, timing.pingEvery);
  }

  save();
  return api;
}

