// Entry point: header controls and hash routing.
//   #/              home
//   #/room/<code>   online game, host seat
//   #/join/<code>   online game, guest seat
//   #/bot/<code>    game against the computer

import { gameStorage, prefs } from './app/prefs.js';
import { CODE_RE, openGame } from './app/rooms.js';
import { heldElsewhere, hold, takeOver } from './app/tablock.js';
import { getLang, setLang, t, tt } from './i18n.js';
import { $, esc } from './ui/dom.js';
import { mountGame } from './ui/game.js';
import { mountHome } from './ui/home.js';
import { toggleMute } from './ui/sound.js';

const app = document.getElementById('app');
let screen = null; // what's mounted in #app: { destroy() }
let remount = null; // how to rebuild it (after a language switch)
let game = null; // the open game, if any
let routeId = 0;

function mount(fn) {
  screen?.destroy?.();
  remount = fn;
  screen = fn() ?? null;
  window.scrollTo(0, 0);
}

function closeGame() {
  screen?.destroy?.();
  screen = null;
  game?.release?.();
  game?.destroy();
  game = null;
}

function notice({ title, text, button, onClick }) {
  mount(() => {
    app.innerHTML = `<div class="notice card"><h2>${title}</h2><p>${text}</p>
      ${button ? `<button class="btn btn-primary" data-go>${button}</button>` : ''}
      <a class="btn btn-ghost" href="#/">${t('over.menu')}</a></div>`;
    $('[data-go]', app)?.addEventListener('click', onClick);
  });
}

function startGame(kind, code, seat) {
  const opened = openGame(kind, code, seat);
  if (opened.redirect) return location.replace(opened.redirect);
  game = opened;
  const lockKey = `${kind}:${code}:${seat}`;
  game.release = hold(lockKey, () => {
    closeGame();
    notice({ title: t('lock.title'), text: t('lock.away'), button: t('lock.here'), onClick: () => claim(kind, code, seat) });
  });
  mount(() => mountGame(app, game));
}

function claim(kind, code, seat) {
  takeOver(`${kind}:${code}:${seat}`);
  // Give the other tab a moment to close its connection (and free the room id).
  setTimeout(() => startGame(kind, code, seat), 400);
}

function joinPrompt(code) {
  mount(() => {
    app.innerHTML = `<form class="notice card" data-join>
      <h2>${t('join.title')}</h2><p>${t('join.desc')}</p>
      <label class="field">${t('home.name')}
        <input name="name" maxlength="24" required autocomplete="nickname" placeholder="${t('home.namePlaceholder')}" value="${esc(prefs.get('name', ''))}">
      </label>
      <button class="btn btn-primary">${t('join.go')}</button></form>`;
    const form = $('[data-join]', app);
    form.elements.name.focus();
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      prefs.set('name', form.elements.name.value.trim().slice(0, 24));
      route(true);
    });
  });
}

async function route(skipPrompt = false) {
  const id = ++routeId;
  closeGame();
  const [name, rawCode] = location.hash.replace(/^#\/?/, '').split('/');
  const code = rawCode?.toLowerCase();
  const kind = name === 'bot' ? 'bot' : 'online';
  const seat = name === 'join' ? 'guest' : 'host';

  if (!['room', 'join', 'bot'].includes(name) || !CODE_RE.test(code ?? '')) {
    if (location.hash && location.hash !== '#/') history.replaceState(null, '', '#/');
    return mount(() => mountHome(app));
  }
  if (seat === 'guest' && !skipPrompt && !gameStorage('online', code, 'guest').exists() && !prefs.get('name')) {
    return joinPrompt(code);
  }
  mount(() => {
    app.innerHTML = '<div class="connecting"><div class="spinner"></div></div>';
  });
  if (await heldElsewhere(`${kind}:${code}:${seat}`)) {
    if (id !== routeId) return;
    return notice({ title: t('lock.title'), text: t('lock.desc'), button: t('lock.here'), onClick: () => claim(kind, code, seat) });
  }
  if (id === routeId) startGame(kind, code, seat);
}

// ------------------------------------------------------------------ header

function renderHeader() {
  document.documentElement.lang = getLang();
  const muted = prefs.get('muted', false);
  $('[data-nav=sound]').textContent = muted ? '🔇' : '🔊';
  $('[data-nav=sound]').title = tt('nav.sound');
  $('[data-nav=lang]').textContent = tt('nav.lang');
}

$('[data-nav=sound]').addEventListener('click', () => {
  toggleMute();
  renderHeader();
});
$('[data-nav=lang]').addEventListener('click', () => {
  setLang(getLang() === 'fr' ? 'en' : 'fr');
  renderHeader();
  if (remount) mount(remount);
});

// Debug handle for the console: __labyrinthe.game.session.state, .view(), .probe(…)
globalThis.__labyrinthe = {
  get game() {
    return game;
  },
};

window.addEventListener('hashchange', () => route());
renderHeader();
route();
