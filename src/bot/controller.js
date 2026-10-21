// Drives a Session as the computer player: designs, gets ready, explores, accepts rematches, banters.

import { chooseProbe, designMaze } from './brain.js';

/**
 * @param session  the bot's own Session (guest seat)
 * @param o.level  'easy' | 'normal' | 'hard' | 'expert'
 * @param o.line   (kind) => string | null — localized banter, kinds: hit | bonk | win | lose | hello
 */
export function createBotController(session, { level = 'normal', rng = Math.random, line = () => null, speed = 1 }) {
  let timer = null;
  let greeted = session.state.chat.length > 0; // resumed game: we already said hi

  const wait = (ms) => Math.round(ms * speed);
  const say = (kind, chance = 1) => {
    if (rng() >= chance) return;
    const text = line(kind);
    if (text) setTimeout(() => session.sendChat(text), wait(400 + rng() * 500));
  };

  function nextAction() {
    const v = session.view();
    const s = session.state;
    if (v.phase === 'design' && !v.iAmReady) {
      return { delay: 1200 + rng() * 1200, run: () => session.setMaze(designMaze(s.settings, level, rng)) && session.ready() };
    }
    if (v.phase === 'play' && v.turn === s.seat && !s.round.pending) {
      return {
        delay: 550 + rng() * 650,
        run: () => {
          const { entrance, exit } = s.round.oppES;
          const p = chooseProbe({ size: v.size, progress: v.mine, entrance, exit, wallCount: s.settings.walls, level, rng });
          if (p) session.probe(p.from, p.dir);
        },
      };
    }
    if (v.phase === 'over' && s.round.rematch[v.opp] && !s.round.rematch[s.seat]) {
      return { delay: 900, run: () => session.requestRematch() };
    }
    return null;
  }

  function schedule() {
    if (timer) return;
    const first = nextAction();
    if (!first) return;
    timer = setTimeout(() => {
      timer = null;
      nextAction()?.run(); // re-evaluate: the situation may have changed while "thinking"
      schedule();
    }, wait(first.delay));
  }

  const unsubscribe = session.subscribe((type, e) => {
    if (type === 'change') schedule();
    if (type !== 'event') return;
    if (e.kind === 'opp-online' && !greeted) {
      greeted = true;
      say('hello', 0.8);
    }
    if (e.kind === 'probe' && e.who === 'them' && !e.open) say('bonk', 0.3);
    if (e.kind === 'probe' && e.who === 'me' && !e.open) say('hit', 0.15);
    if (e.kind === 'phase' && e.phase === 'over') say(session.view().winner === session.state.seat ? 'win' : 'lose', 0.9);
  });
  schedule();

  return {
    destroy() {
      clearTimeout(timer);
      unsubscribe();
    },
  };
}
