// Parity fixtures for the Python/JAX training stack in rl/ (docs/RL_DESIGN.md, phase P1).
//
//   node scripts/export-fixtures.mjs           write rl/tests/fixtures/*.json
//   node scripts/export-fixtures.mjs --check   exit 1 if the committed fixtures differ from what the
//                                              JS engine produces now (run in CI)
//
// The JS game is the reference: these files record what it does, and rl/tests must reproduce them
// exactly. Everything is seeded, so the output only changes when the JS rules or bots change.
// Per board size 4..10:
//   probes-N-KK.json   a maze, a probe sequence (legal, illegal, after the end) with every answer,
//                      newly reached cell, hit count and end flag, plus rankProbes snapshots
//   mazes-N.json       randomMaze outputs for seeds and budgets; mazeIssues on crafted mazes
//   solve-N-LEVEL.json full explorations by the easy / normal / hard bot (seeded)
//   design-N-LEVEL.json the maze each designer level picks (seeded)

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { chooseProbe, designMaze, rankProbes, simulateSolve } from '../src/bot/brain.js';
import { allEdges, DIRS, edgeKey, neighbor } from '../src/game/grid.js';
import { probeError, progressOf } from '../src/game/match.js';
import { canonicalMaze, defaultWallBudget, isOpen, maxWallBudget, mazeIssues, randomMaze } from '../src/game/maze.js';
import { pick, randInt, seededRng, shuffle } from '../src/game/random.js';

const OUT = join(dirname(fileURLToPath(import.meta.url)), '..', 'rl', 'tests', 'fixtures');
const SIZES = [4, 5, 6, 7, 8, 9, 10];
const LEVELS = ['easy', 'normal', 'hard'];
const PROBE_FIXTURES = 22; // per size
const RANK_SNAPSHOTS = 8; // per probe fixture

const dirIndex = (dir) => DIRS.indexOf(dir);

/** Budgets worth covering: few walls, default, a perfect maze's count, more than that. */
function budgets(size) {
  const perfect = (size - 1) ** 2;
  return [Math.max(1, Math.round(perfect / 3)), defaultWallBudget(size), perfect, Math.min(maxWallBudget(size), perfect + size - 1)];
}

/** Uniform among valid mazes: random walls and distinct E/S, rejected until S is reachable. */
function uniformMaze(size, budget, rng) {
  const edges = allEdges(size);
  for (;;) {
    const walls = shuffle(rng, edges).slice(0, budget);
    const entrance = randInt(rng, size * size);
    let exit = randInt(rng, size * size - 1);
    if (exit >= entrance) exit++;
    const maze = { walls, entrance, exit };
    if (!mazeIssues(maze, { size, walls: budget }).length) return canonicalMaze(maze);
  }
}

function legalProbes(progress, size) {
  const out = [];
  for (const from of progress.visited) for (const dir of DIRS) if (!probeError(progress, from, dir, size)) out.push({ from, dir });
  return out;
}

/** A probe the rules refuse: from an unreached cell, through the border, a known side, or garbage. */
function illegalProbe(progress, size, rng) {
  const cells = [...Array(size * size).keys()];
  const options = [];
  const unreached = cells.filter((c) => !progress.visited.has(c));
  if (unreached.length) options.push(() => ({ from: pick(rng, unreached), dir: pick(rng, DIRS) }));
  const visited = [...progress.visited];
  const border = visited.flatMap((from) => DIRS.filter((dir) => neighbor(from, dir, size) < 0).map((dir) => ({ from, dir })));
  if (border.length) options.push(() => pick(rng, border));
  const known = visited.flatMap((from) =>
    DIRS.filter((dir) => {
      const to = neighbor(from, dir, size);
      return to >= 0 && (progress.open.has(edgeKey(from, to)) || progress.walls.has(edgeKey(from, to)));
    }).map((dir) => ({ from, dir })),
  );
  if (known.length) options.push(() => pick(rng, known));
  options.push(() => pick(rng, [{ from: -1, dir: 'R' }, { from: size * size, dir: 'U' }, { from: pick(rng, visited), dir: 'X' }]));
  return pick(rng, options)();
}

function probeFixture(size, k) {
  const rng = seededRng(1_000_003 * size + 7919 * k);
  const budget = budgets(size)[k % 4];
  const source = k % 2 === 0 ? 'randomMaze' : 'uniform';
  const mazeSeed = 31 * size + k;
  const maze = source === 'randomMaze' ? randomMaze({ size, walls: budget }, seededRng(mazeSeed)) : uniformMaze(size, budget, seededRng(mazeSeed));
  const policy = ['random', 'hard', 'normal', 'easy', 'random'][k % 5];
  const partial = k % 7 === 6; // stop half-way, before the exit
  const walls = new Set(maze.walls);
  const accepted = [];
  const probes = [];
  const rankAt = new Set([0]);
  let progress = progressOf(accepted, maze.entrance, size);
  const cap = partial ? Math.floor(size * size * 0.6) : 10 * size * size;
  let after = 0;
  while (probes.length < cap) {
    const over = progress.visited.has(maze.exit);
    if (over && after++ >= 2) break;
    let p;
    if (over || rng() < 0.15) p = over ? { from: pick(rng, [...progress.visited]), dir: pick(rng, DIRS) } : illegalProbe(progress, size, rng);
    else if (policy === 'random') p = pick(rng, legalProbes(progress, size));
    else p = chooseProbe({ size, progress, exit: maze.exit, level: policy, rng });
    if (!over && rng() < RANK_SNAPSHOTS / (size * size)) rankAt.add(probes.length);
    const error = over ? 'over' : probeError(progress, p.from, p.dir, size);
    const entry = { from: p.from, dir: p.dir, error, open: null, reached: null };
    if (!error) {
      entry.open = isOpen(walls, p.from, p.dir, size);
      accepted.push({ from: p.from, dir: p.dir, open: entry.open });
      const before = progress.visited.size;
      progress = progressOf(accepted, maze.entrance, size);
      if (progress.visited.size > before) entry.reached = [...progress.visited][before];
    }
    Object.assign(entry, { hits: progress.hits, moves: progress.moves, done: progress.visited.has(maze.exit) });
    probes.push(entry);
  }
  // rankProbes before some of the probes (the progress replayed up to that point).
  const ranks = [...rankAt].sort((a, b) => a - b).map((at) => {
    const prior = probes.slice(0, at).filter((p) => !p.error).map((p) => ({ from: p.from, dir: p.dir, open: p.open }));
    const pr = progressOf(prior, maze.entrance, size);
    return { at, ranked: rankProbes(size, pr, maze.exit).map((r) => [r.from, dirIndex(r.dir), r.score]) };
  });
  return {
    kind: 'probes',
    size,
    budget,
    source,
    policy,
    maze,
    mazeSeed: source === 'randomMaze' ? mazeSeed : null,
    issues: mazeIssues(maze, { size, walls: budget }).map((i) => i.code),
    probes,
    ranks,
  };
}

function mazesFixture(size) {
  const random = [];
  budgets(size).concat([1, maxWallBudget(size)]).forEach((budget, i) => {
    for (const seed of [i, 100 + i]) random.push({ seed: 977 * size + seed, budget, maze: randomMaze({ size, walls: budget }, seededRng(977 * size + seed)) });
  });
  const rng = seededRng(4242 + size);
  const budget = defaultWallBudget(size);
  const base = randomMaze({ size, walls: budget }, rng);
  const otherEdges = allEdges(size).filter((e) => !base.walls.includes(e));
  // Wall S in on every side, then fill up to the budget elsewhere: S unreachable.
  const exit = base.exit;
  const around = DIRS.map((d) => neighbor(exit, d, size)).filter((n) => n >= 0).map((n) => edgeKey(exit, n));
  const rest = shuffle(rng, allEdges(size).filter((e) => !around.includes(e))).slice(0, budget - around.length);
  const crafted = [
    { name: 'valid', maze: base },
    { name: 'valid-uniform', maze: uniformMaze(size, budget, rng) },
    { name: 'one-wall-short', maze: { ...base, walls: base.walls.slice(1) } },
    { name: 'one-wall-extra', maze: { ...base, walls: [...base.walls, pick(rng, otherEdges)] } },
    { name: 'same-cell', maze: { ...base, exit: base.entrance } },
    { name: 'no-path', maze: canonicalMaze({ walls: [...around, ...rest], entrance: base.entrance, exit }) },
    { name: 'no-entrance', maze: { ...base, entrance: null } },
    { name: 'no-exit', maze: { ...base, exit: null } },
  ].map((c) => ({ ...c, issues: mazeIssues(c.maze, { size, walls: budget }).map((i) => i.code) }));
  return { kind: 'mazes', size, random, budget, crafted };
}

function solveFixture(size, level) {
  const runs = [];
  for (let i = 0; i < 4; i++) {
    const budget = budgets(size)[i];
    const mazeSeed = 555 * size + i;
    const maze = randomMaze({ size, walls: budget }, seededRng(mazeSeed));
    const seed = 7777 * size + 13 * i + LEVELS.indexOf(level);
    // simulateSolve's loop, recording the probes; checked against simulateSolve itself below.
    const rng = seededRng(seed);
    const walls = new Set(maze.walls);
    const probes = [];
    for (let guard = 0; guard < 4 * size * size; guard++) {
      const progress = progressOf(probes, maze.entrance, size);
      if (progress.visited.has(maze.exit)) break;
      const p = chooseProbe({ size, progress, exit: maze.exit, level, rng });
      probes.push({ from: p.from, dir: p.dir, open: isOpen(walls, p.from, p.dir, size) });
    }
    const { hits, moves } = progressOf(probes, maze.entrance, size);
    const ref = simulateSolve(maze, size, level, seededRng(seed));
    if (ref.hits !== hits || ref.moves !== moves) throw new Error(`solve replay drifted from simulateSolve (${size}, ${level})`);
    runs.push({ budget, mazeSeed, seed, maze, hits, moves, probes: probes.map((p) => [p.from, dirIndex(p.dir), p.open]) });
  }
  return { kind: 'solve', size, level, runs };
}

function designFixture(size, level) {
  const budget = defaultWallBudget(size);
  const seed = 31337 + 10 * size + LEVELS.indexOf(level);
  return { kind: 'design', size, level, budget, seed, maze: designMaze({ size, walls: budget }, level, seededRng(seed)) };
}

/** One top-level key per line, and one line per element of top-level arrays: readable diffs. */
function format(obj) {
  const lines = Object.entries(obj).map(([key, value]) => {
    if (Array.isArray(value) && value.length && typeof value[0] === 'object') {
      return `  ${JSON.stringify(key)}: [\n${value.map((v) => `    ${JSON.stringify(v)}`).join(',\n')}\n  ]`;
    }
    return `  ${JSON.stringify(key)}: ${JSON.stringify(value)}`;
  });
  return `{\n${lines.join(',\n')}\n}\n`;
}

function allFixtures() {
  const files = new Map();
  for (const size of SIZES) {
    for (let k = 0; k < PROBE_FIXTURES; k++) files.set(`probes-${size}-${String(k).padStart(2, '0')}.json`, probeFixture(size, k));
    files.set(`mazes-${size}.json`, mazesFixture(size));
    for (const level of LEVELS) {
      files.set(`solve-${size}-${level}.json`, solveFixture(size, level));
      files.set(`design-${size}-${level}.json`, designFixture(size, level));
    }
  }
  return new Map([...files].map(([name, obj]) => [name, format(obj)]));
}

const files = allFixtures();
if (process.argv.includes('--check')) {
  const onDisk = new Set(readdirSync(OUT).filter((f) => f.endsWith('.json')));
  const problems = [];
  for (const [name, text] of files) {
    if (!onDisk.has(name)) problems.push(`missing: ${name}`);
    else if (readFileSync(join(OUT, name), 'utf8') !== text) problems.push(`differs: ${name}`);
    onDisk.delete(name);
  }
  for (const name of onDisk) problems.push(`stale: ${name}`);
  if (problems.length) {
    console.error(`${problems.length} fixture(s) out of date; run node scripts/export-fixtures.mjs\n${problems.slice(0, 20).join('\n')}`);
    process.exit(1);
  }
  console.log(`${files.size} fixtures match the JS engine`);
} else {
  mkdirSync(OUT, { recursive: true });
  for (const name of readdirSync(OUT)) if (name.endsWith('.json') && !files.has(name)) throw new Error(`unexpected file ${name}`);
  for (const [name, text] of files) writeFileSync(join(OUT, name), text);
  console.log(`wrote ${files.size} fixtures to ${OUT}`);
}
