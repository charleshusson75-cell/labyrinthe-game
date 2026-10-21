// Turns a route (online host / online guest / bot game) into a running Session with the right
// transport and storage.

import { createBotController } from '../bot/controller.js';
import { normalizeSettings } from '../game/maze.js';
import { pick, randomCode, secureRng } from '../game/random.js';
import { tt } from '../i18n.js';
import { createLocalTransport } from '../net/local.js';
import { createPeerTransport } from '../net/peer.js';
import { createPipe } from '../net/pipe.js';
import { createSession } from '../net/session.js';
import { gameStorage, prefs } from './prefs.js';

export const CODE_RE = /^[a-z0-9]{4,12}$/;

// Settings chosen on the home screen, waiting for the game route to pick them up.
const pending = new Map();

export function newOnlineGame({ name, settings }) {
  const code = randomCode(secureRng());
  pending.set(`online:${code}`, { name, settings: normalizeSettings(settings) });
  return code;
}

export function newBotGame({ name, settings, level }) {
  const code = randomCode(secureRng());
  pending.set(`bot:${code}`, { name, settings: normalizeSettings(settings), meta: { level } });
  return code;
}

/** `?net=local` → play between two tabs of the same browser, no network (handy for development). */
const useLocalNet = () => new URLSearchParams(location.search).get('net') === 'local';

export function inviteUrl(code) {
  const url = new URL(location.href);
  url.hash = `#/join/${code}`;
  return url.toString();
}

/** Extract a game code from whatever the user pasted (code or full link). */
export function parseCode(input) {
  const s = String(input ?? '').trim().toLowerCase();
  const fromLink = s.match(/#\/(?:join|room)\/([a-z0-9]+)/);
  const code = fromLink ? fromLink[1] : s.replace(/[^a-z0-9]/g, '');
  return CODE_RE.test(code) ? code : null;
}

/**
 * @returns {{ redirect: string } | { session, kind, seat, botLevel?, inviteUrl?, destroy() }}
 */
export function openGame(kind, code, seat) {
  return kind === 'bot' ? openBotGame(code) : openOnlineGame(code, seat);
}

function openOnlineGame(code, seat) {
  const storage = gameStorage('online', code, seat);
  const init = pending.get(`online:${code}`);
  pending.delete(`online:${code}`);
  // Someone opened the host's own URL without being the host: they are the guest.
  if (seat === 'host' && !init && !storage.exists()) return { redirect: `#/join/${code}` };

  const transport = useLocalNet()
    ? createLocalTransport({ role: seat, roomCode: code })
    : createPeerTransport({ role: seat, roomCode: code });
  const session = createSession({
    seat,
    roomCode: code,
    transport,
    storage,
    init: init ?? { name: prefs.get('name', '') },
  });
  return { session, kind: 'online', seat, inviteUrl: inviteUrl(code), destroy: () => session.destroy() };
}

function openBotGame(code) {
  const humanStore = gameStorage('bot', code, 'host');
  const botStore = gameStorage('bot', code, 'guest');
  const init = pending.get(`bot:${code}`);
  pending.delete(`bot:${code}`);
  if (!init && !humanStore.exists()) return { redirect: '#/' };

  const level = init?.meta?.level ?? humanStore.load()?.meta?.level ?? 'normal';
  const pipe = createPipe();
  const session = createSession({ seat: 'host', roomCode: code, transport: pipe.host, storage: humanStore, init });
  const bot = createSession({ seat: 'guest', roomCode: code, transport: pipe.guest, storage: botStore, init: { name: '' } });
  const controller = createBotController(bot, {
    level,
    line: (kind) => pick(Math.random, tt(`bot.${kind}`)),
  });
  return {
    session,
    kind: 'bot',
    seat: 'host',
    botLevel: level,
    destroy() {
      controller.destroy();
      bot.destroy();
      session.destroy();
    },
  };
}
