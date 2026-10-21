import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';

import { allEdges, cellLabel, dirBetween, edgeKey, edgeOf, isEdge, neighbor } from '../src/game/grid.js';
import {
  defaultWallBudget,
  distancesFrom,
  maxWallBudget,
  mazeIssues,
  normalizeSettings,
  randomMaze,
  shortestPath,
} from '../src/game/maze.js';
import { sha256Hex } from '../src/game/sha256.js';
import { seededRng } from '../src/game/random.js';
import { commitmentOf, progressOf, probeError, timeline, verifyReveal } from '../src/game/match.js';

const codes = (issues) => issues.map((i) => i.code);

test('grid geometry', () => {
  assert.equal(neighbor(0, 'U', 5), -1);
  assert.equal(neighbor(0, 'L', 5), -1);
  assert.equal(neighbor(0, 'R', 5), 1);
  assert.equal(neighbor(0, 'D', 5), 5);
  assert.equal(neighbor(4, 'R', 5), -1, 'no wrap-around');
  assert.equal(edgeKey(6, 1), '1-6');
  assert.equal(edgeOf(24, 'D', 5), null);
  assert.equal(dirBetween(7, 12, 5), 'D');
  assert.equal(dirBetween(4, 5, 5), null, 'end of row is not adjacent to next row start');
  assert.equal(allEdges(5).length, 40);
  assert.ok(allEdges(5).every((k) => isEdge(k, 5)));
  assert.ok(!isEdge('4-5', 5));
  assert.ok(!isEdge('5-4', 5));
  assert.equal(cellLabel(0, 5), 'A1');
  assert.equal(cellLabel(13, 5), 'D3');
});

test('settings bounds', () => {
  assert.deepEqual(normalizeSettings({}), { size: 5, walls: 12 });
  assert.equal(defaultWallBudget(5), 12);
  assert.equal(normalizeSettings({ size: 99 }).size, 8);
  assert.equal(normalizeSettings({ size: 5, walls: 999 }).walls, maxWallBudget(5));
});

test('maze validation', () => {
  const settings = { size: 4, walls: 2 };
  assert.deepEqual(codes(mazeIssues({ walls: [], entrance: null, exit: null }, settings)), [
    'walls-left',
    'no-entrance',
    'no-exit',
  ]);
  assert.deepEqual(codes(mazeIssues({ walls: ['0-1', '0-4'], entrance: 0, exit: 0 }, settings)), ['same-cell']);
  // Cell 0 boxed in by its two internal sides.
  assert.deepEqual(codes(mazeIssues({ walls: ['0-1', '0-4'], entrance: 0, exit: 15 }, settings)), ['no-path']);
  assert.deepEqual(codes(mazeIssues({ walls: ['0-1', '1-5'], entrance: 0, exit: 15 }, settings)), []);
  assert.deepEqual(codes(mazeIssues({ walls: ['0-1', '0-1'], entrance: 0, exit: 15 }, settings)), ['bad-walls']);
  assert.deepEqual(codes(mazeIssues({ walls: ['3-4', '0-1'], entrance: 0, exit: 15 }, settings)), ['bad-walls']);
});

test('shortest path and distances', () => {
  const walls = new Set(['0-1']);
  assert.deepEqual(shortestPath(3, walls, 0, 1), [0, 3, 4, 1]);
  assert.equal(distancesFrom(3, walls, 0)[1], 3);
  assert.equal(shortestPath(3, new Set(['0-1', '0-3']), 0, 8), null);
});

test('random mazes are always valid', () => {
  const rng = seededRng(42);
  for (let size = 4; size <= 8; size++) {
    for (const walls of [1, defaultWallBudget(size), (size - 1) ** 2, maxWallBudget(size)]) {
      for (let i = 0; i < 20; i++) {
        const maze = randomMaze({ size, walls }, rng);
        assert.deepEqual(mazeIssues(maze, { size, walls }), [], `size ${size}, ${walls} walls`);
      }
    }
  }
});

test('sha256 matches node:crypto', () => {
  for (const s of ['', 'abc', 'é🧱 labyrinthe', 'x'.repeat(55), 'y'.repeat(56), 'z'.repeat(64), 'w'.repeat(1000)]) {
    assert.equal(sha256Hex(s), createHash('sha256').update(s).digest('hex'));
  }
});

test('progress, legality and timeline', () => {
  const size = 3;
  const probes = [
    { from: 0, dir: 'R', open: true },
    { from: 1, dir: 'R', open: false },
  ];
  const p = progressOf(probes, 0, size);
  assert.deepEqual([...p.visited].sort(), [0, 1]);
  assert.equal(p.hits, 1);
  assert.equal(p.position, 1);
  assert.equal(probeError(p, 1, 'R', size), 'known');
  assert.equal(probeError(p, 0, 'R', size), 'known');
  assert.equal(probeError(p, 2, 'L', size), 'not-reached');
  assert.equal(probeError(p, 0, 'U', size), 'border');
  assert.equal(probeError(p, 1, 'D', size), null);

  const { entries, turn } = timeline(
    { host: [{ open: true }, { open: false }, { open: false }], guest: [{ open: false }] },
    'host',
  );
  assert.deepEqual(
    entries.map((e) => e.seat),
    ['host', 'host', 'guest', 'host'],
  );
  assert.equal(turn, 'guest');
});

test('verifyReveal catches lies and swapped mazes', () => {
  const settings = { size: 4, walls: 2 };
  const maze = { walls: ['0-1', '1-5'], entrance: 0, exit: 15 };
  const salt = 'abc';
  const commit = commitmentOf(maze, salt, 4);
  const honest = [{ from: 0, dir: 'R', open: false }, { from: 0, dir: 'D', open: true }];
  const es = { entrance: 0, exit: 15 };
  assert.deepEqual(verifyReveal({ reveal: { maze, salt }, commit, es, answers: honest, settings }), { ok: true, problems: [] });

  const lie = [{ from: 0, dir: 'D', open: false }];
  assert.deepEqual(codes(verifyReveal({ reveal: { maze, salt }, commit, es, answers: lie, settings }).problems), ['lie']);

  const swapped = { ...maze, walls: ['0-4', '1-5'] };
  assert.deepEqual(codes(verifyReveal({ reveal: { maze: swapped, salt }, commit, es, answers: [], settings }).problems), [
    'hash-mismatch',
  ]);
  assert.deepEqual(codes(verifyReveal({ reveal: null, commit, es, answers: [], settings }).problems), ['malformed']);
});
