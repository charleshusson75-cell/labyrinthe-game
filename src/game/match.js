// Match rules on top of the maze model: rounds, progress, turn order, winner, verification.
// Everything here is pure; `deriveView` turns a session state into what the UI needs.

import { edgeKey, isCell, isDir, neighbor } from './grid.js';
import { canonicalMaze, emptyMaze, isOpen, mazeIssues } from './maze.js';
import { sha256Hex } from './sha256.js';

export const other = (seat) => (seat === 'host' ? 'guest' : 'host');

/** Hash a player publishes when ready; the salt stops the opponent from brute-forcing the maze. */
export function commitmentOf(maze, salt, size) {
  const m = canonicalMaze(maze);
  return sha256Hex(JSON.stringify(['labyrinthe/1', size, m.walls, m.entrance, m.exit, salt]));
}

export function newRound(n, starter, draft) {
  return {
    n,
    starter: starter ?? null, // seat who moves first
    myMaze: draft ? canonicalMaze(draft) : emptyMaze(),
    mySalt: null,
    myCommit: null,
    oppCommit: null,
    oppES: null, // { entrance, exit } of the opponent's maze
    oppReveal: null, // { maze, salt } once the round is over
    probes: { host: [], guest: [] }, // probes[seat] = that seat's moves: { from, dir, open }
    pending: null, // my move awaiting an answer: { seq, from, dir }
    resigned: null,
    rematch: { host: false, guest: false },
    verification: null,
    recorded: false,
  };
}

/** What an explorer knows after `probes` on a maze entered at `entrance`. */
export function progressOf(probes, entrance, size) {
  const visited = new Set([entrance]);
  const open = new Set();
  const walls = new Set();
  let position = entrance;
  let hits = 0;
  for (const p of probes) {
    const to = neighbor(p.from, p.dir, size);
    if (to < 0) continue;
    const key = edgeKey(p.from, to);
    if (p.open) {
      open.add(key);
      visited.add(to);
      position = to;
    } else {
      walls.add(key);
      hits++;
      position = p.from;
    }
  }
  return { visited, open, walls, position, moves: probes.length, hits };
}

/** Why probing `dir` from `from` is not allowed (null = legal). */
export function probeError(progress, from, dir, size) {
  if (!isCell(from, size) || !isDir(dir)) return 'invalid';
  if (!progress.visited.has(from)) return 'not-reached';
  const to = neighbor(from, dir, size);
  if (to < 0) return 'border';
  const key = edgeKey(from, to);
  if (progress.open.has(key) || progress.walls.has(key)) return 'known';
  return null;
}

/**
 * Interleave both players' probes in play order: a player keeps the turn until they hit a wall.
 * Returns the ordered entries and whose turn it is now.
 */
export function timeline(probes, starter) {
  const idx = { host: 0, guest: 0 };
  const entries = [];
  let turn = starter;
  for (;;) {
    const list = probes[turn];
    if (idx[turn] >= list.length) break;
    const p = list[idx[turn]++];
    entries.push({ seat: turn, ...p });
    if (!p.open) turn = other(turn);
  }
  return { entries, turn };
}

/**
 * Check a revealed maze against its commitment, the rules, the announced E/S and every answer
 * its owner gave during the round.
 */
export function verifyReveal({ reveal, commit, es, answers, settings }) {
  const problems = [];
  try {
    const { maze, salt } = reveal;
    if (commitmentOf(maze, salt, settings.size) !== commit) problems.push({ code: 'hash-mismatch' });
    if (mazeIssues(maze, settings).length) problems.push({ code: 'rules' });
    if (es && (es.entrance !== maze.entrance || es.exit !== maze.exit)) problems.push({ code: 'es-mismatch' });
    const walls = new Set(maze.walls);
    answers.forEach((a, index) => {
      if (isOpen(walls, a.from, a.dir, settings.size) !== a.open) {
        problems.push({ code: 'lie', index, from: a.from, dir: a.dir });
      }
    });
  } catch {
    problems.push({ code: 'malformed' });
  }
  return { ok: problems.length === 0, problems };
}

/** Everything the UI (or the bot) needs to know about the current round, from one seat's view. */
export function deriveView(state) {
  const { seat, settings, round } = state;
  const opp = other(seat);
  if (!settings || !round.starter) return { phase: 'connecting', seat, opp };
  const size = settings.size;
  const iAmReady = !!round.myCommit;
  const oppReady = !!round.oppCommit;
  const bothReady = iAmReady && oppReady;
  const started = bothReady && !!round.oppES;
  // `theirs`: the opponent exploring my maze. `mine`: me exploring theirs.
  const theirs = bothReady ? progressOf(round.probes[opp], round.myMaze.entrance, size) : null;
  const mine = started ? progressOf(round.probes[seat], round.oppES.entrance, size) : null;

  let winner = null;
  if (round.resigned) winner = other(round.resigned);
  else if (mine?.visited.has(round.oppES.exit)) winner = seat;
  else if (theirs?.visited.has(round.myMaze.exit)) winner = opp;

  const { entries, turn } = timeline(round.probes, round.starter);
  const phase = winner ? 'over' : started ? 'play' : 'design';
  return {
    phase,
    seat,
    opp,
    size,
    iAmReady,
    oppReady,
    bothReady,
    mine,
    theirs,
    winner,
    turn: phase === 'play' ? turn : null,
    entries,
    issues: mazeIssues(round.myMaze, settings),
  };
}

export function scoreOf(history) {
  const score = { host: 0, guest: 0 };
  for (const h of history) if (h.winner) score[h.winner]++;
  return score;
}
