// Home screen: name, create an online game, play the bot, join with a code, resume saved games, rules.

import { forgetGame, prefs, savedGames } from '../app/prefs.js';
import { newBotGame, newOnlineGame, parseCode } from '../app/rooms.js';
import { LEVELS } from '../bot/brain.js';
import { other, scoreOf } from '../game/match.js';
import { MAX_SIZE, MIN_SIZE, defaultWallBudget, maxWallBudget, randomMaze, shortestPath } from '../game/maze.js';
import { seededRng } from '../game/random.js';
import { t, tt } from '../i18n.js';
import { renderBoard } from './board.js';
import { $, esc, on, toast } from './dom.js';

function heroBoard() {
  const maze = randomMaze({ size: 5, walls: 16 }, seededRng(11));
  return renderBoard({
    size: 5,
    variant: 'hero',
    walls: maze.walls,
    entrance: maze.entrance,
    exit: maze.exit,
    solution: shortestPath(5, new Set(maze.walls), maze.entrance, maze.exit),
    label: 'Labyrinthe',
  });
}

function settingsFields(prefix, settings) {
  const sizes = [];
  for (let s = MIN_SIZE; s <= MAX_SIZE; s++) {
    sizes.push(`<option value="${s}" ${s === settings.size ? 'selected' : ''}>${s} × ${s}</option>`);
  }
  return `<div class="field-row">
    <label class="field">${t('home.size')}<select name="size">${sizes.join('')}</select></label>
    <label class="field">${t('home.walls')}
      <input name="walls" type="number" inputmode="numeric" min="1" max="${maxWallBudget(settings.size)}" value="${settings.walls}">
      <small data-walls-hint>${t('home.wallsHint', { max: maxWallBudget(settings.size) })}</small>
    </label>
  </div>`;
}

function savedList() {
  const games = savedGames().slice(0, 8);
  if (!games.length) return '';
  const items = games.map(({ kind, code, seat, state }) => {
    const opp = other(seat);
    const oppName =
      kind === 'bot'
        ? `${tt('computer')} · ${tt(`level.${state.meta?.level ?? 'normal'}`)}`
        : state.names?.[opp] || (state.joined ? tt('opponent') : '');
    const score = scoreOf(state.history ?? []);
    const href = kind === 'bot' ? `#/bot/${code}` : seat === 'host' ? `#/room/${code}` : `#/join/${code}`;
    const title = oppName ? t('home.saved.vs', { name: oppName }) : t('home.saved.waiting');
    const meta = [
      `${score[seat]} – ${score[opp]}`,
      t('home.saved.round', { n: state.round?.n ?? 1 }),
      kind === 'online' ? `<code>${esc(code)}</code>` : '',
    ].filter(Boolean).join(' · ');
    return `<li class="saved-item">
      <a href="${href}" class="saved-link"><span class="saved-title">${title}</span><span class="saved-meta">${meta}</span></a>
      <button class="btn btn-ghost btn-small" data-forget="${kind}:${code}" aria-label="${t('home.saved.forget')}">✕</button>
    </li>`;
  });
  return `<h2>${t('home.saved.title')}</h2><ul class="saved-list">${items.join('')}</ul>`;
}

export function mountHome(root) {
  const settings = prefs.get('settings', { size: 5, walls: defaultWallBudget(5) });
  const level = prefs.get('botLevel', 'normal');

  root.innerHTML = `
    <div class="home">
      <section class="hero">
        <div class="hero-board">${heroBoard()}</div>
        <div class="hero-text">
          <h1>Labyrinthe</h1>
          <p class="tagline">${t('tagline')}</p>
          <label class="field name-field">${t('home.name')}
            <input name="name" maxlength="24" autocomplete="nickname" placeholder="${t('home.namePlaceholder')}" value="${esc(prefs.get('name', ''))}">
          </label>
        </div>
      </section>

      <div class="cards">
        <form class="card card--main" data-form="friend">
          <h2>${t('home.friend.title')}</h2>
          <p>${t('home.friend.desc')}</p>
          ${settingsFields('friend', settings)}
          <button class="btn btn-primary">${t('home.create')}</button>
        </form>

        <form class="card" data-form="bot">
          <h2>${t('home.bot.title')}</h2>
          <p>${t('home.bot.desc')}</p>
          <fieldset class="seg seg--levels">
            <legend>${t('home.level')}</legend>
            ${LEVELS.map((l) => `<label><input type="radio" name="level" value="${l}" ${l === level ? 'checked' : ''}><span>${t(`level.${l}`)}</span></label>`).join('')}
          </fieldset>
          ${settingsFields('bot', settings)}
          <button class="btn">${t('home.start')}</button>
        </form>

        <form class="card" data-form="join">
          <h2>${t('home.join.title')}</h2>
          <p>${t('home.join.desc')}</p>
          <label class="field"><span class="sr-only">${t('home.join.placeholder')}</span>
            <input name="code" placeholder="${t('home.join.placeholder')}" autocomplete="off" autocapitalize="off" spellcheck="false">
          </label>
          <button class="btn">${t('home.join.button')}</button>
        </form>
      </div>

      <section class="saved" data-saved>${savedList()}</section>

      <details class="rules card" ${prefs.get('seenRules') ? '' : 'open'}>
        <summary><h2>${t('rules.title')}</h2></summary>
        <ol>${t('rules').map((r) => `<li>${r}</li>`).join('')}</ol>
      </details>
    </div>`;

  const nameInput = $('input[name=name]', root);
  const saveName = () => prefs.set('name', nameInput.value.trim().slice(0, 24));
  const readSettings = (form) => ({ size: Number(form.elements.size.value), walls: Number(form.elements.walls.value) });

  const offs = [
    on(root, 'change', 'select[name=size]', (e, select) => {
      const form = select.form;
      const size = Number(select.value);
      const walls = form.elements.walls;
      walls.max = maxWallBudget(size);
      walls.value = defaultWallBudget(size);
      $('[data-walls-hint]', form).textContent = tt('home.wallsHint', { max: maxWallBudget(size) });
    }),
    on(root, 'input', 'input[name=name]', saveName),
    on(root, 'submit', 'form', (e, form) => {
      e.preventDefault();
      saveName();
      prefs.set('seenRules', true);
      const name = prefs.get('name', '');
      if (form.dataset.form === 'friend') {
        const s = readSettings(form);
        prefs.set('settings', s);
        location.hash = `#/room/${newOnlineGame({ name, settings: s })}`;
      } else if (form.dataset.form === 'bot') {
        const s = readSettings(form);
        const lvl = form.elements.level.value;
        prefs.set('settings', s);
        prefs.set('botLevel', lvl);
        location.hash = `#/bot/${newBotGame({ name, settings: s, level: lvl })}`;
      } else {
        const code = parseCode(form.elements.code.value);
        if (!code) return toast(tt('home.join.invalid'), { kind: 'warn' });
        location.hash = `#/join/${code}`;
      }
    }),
    on(root, 'click', '[data-forget]', (e, b) => {
      const [kind, code] = b.dataset.forget.split(':');
      forgetGame(kind, code);
      $('[data-saved]', root).innerHTML = savedList();
    }),
  ];

  return { destroy: () => offs.forEach((off) => off()) };
}
