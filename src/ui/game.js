// The game screen: top bar, status banner, phase panel (design / play / over), and a side column
// with your own maze, the move log and the chat.

import { ARROW, DIRS, cellLabel, edgeKey, neighbor } from '../game/grid.js';
import { other, probeError, scoreOf } from '../game/match.js';
import { shortestPath } from '../game/maze.js';
import { t, tt } from '../i18n.js';
import { cellAtPoint, edgeAtPoint, previewPath, renderBoard } from './board.js';
import { $, copyText, esc, on, setHTML, toast } from './dom.js';
import { sound } from './sound.js';

const KEY_DIRS = {
  ArrowUp: 'U', ArrowDown: 'D', ArrowLeft: 'L', ArrowRight: 'R',
  w: 'U', s: 'D', a: 'L', d: 'R', z: 'U', q: 'L', // AZERTY too
};

/**
 * @param root     element to render into
 * @param o.session
 * @param o.kind   'online' | 'bot'
 * @param o.inviteUrl  link for the guest (online host only)
 */
export function mountGame(root, { session, kind, inviteUrl, botLevel }) {
  const ui = {
    phase: null,
    tool: 'walls',
    showPath: false,
    cursor: null,
    cursorKey: '',
    drag: null,
    fresh: null, // { who, key, open, at }
    chatCount: -1,
    logCount: -1,
  };
  const baseTitle = document.title;
  const cleanups = [];
  let destroyed = false;
  const s = () => session.state;
  const seat = () => s().seat;
  const n = () => s().settings?.size ?? 5;

  const names = () => {
    const st = s();
    const them = kind === 'bot' ? `${t('computer')} · ${t(`level.${botLevel}`)}` : st.names[other(st.seat)];
    return { me: st.names[st.seat] || t('you'), them: them || t('opponent') };
  };
  const theirName = () => names().them;

  // ---------------------------------------------------------------- skeleton

  root.innerHTML = `
    <div class="game">
      <div class="g-top">
        <div class="g-score">
          <span class="g-player is-me"><span class="g-swatch"></span><span data-name-me></span></span>
          <span class="g-points" data-score></span>
          <span class="g-player is-them"><span class="g-dot" data-dot></span><span data-name-them></span></span>
        </div>
        ${kind === 'online' ? `<div class="g-room"><span>${t('game.room')}</span> <code>${esc(s().roomCode)}</code>
          <button class="btn btn-small" data-act="copy-link">${t('game.copyLink')}</button></div>` : ''}
      </div>
      <div class="g-banner" data-banner role="status" aria-live="polite"></div>
      <div class="g-invite" data-invite hidden></div>
      <div class="g-layout">
        <section class="g-main" data-main></section>
        <aside class="g-side">
          <div class="panel g-mine" data-side-board hidden></div>
          <div class="panel g-log" data-log hidden></div>
          <div class="panel g-chat">
            <h3>${t('chat.title')}</h3>
            <ol class="chat-list" data-chat></ol>
            <div class="chat-quick">${tt('chat.quick').map((q) => `<button class="chip" data-quick="${esc(q)}">${esc(q)}</button>`).join('')}</div>
            <form class="chat-form" data-chat-form autocomplete="off">
              <input name="text" maxlength="280" aria-label="${t('chat.title')}">
              <button class="btn btn-small">${t('chat.send')}</button>
            </form>
          </div>
        </aside>
      </div>
    </div>`;

  const el = {
    nameMe: $('[data-name-me]', root),
    nameThem: $('[data-name-them]', root),
    score: $('[data-score]', root),
    dot: $('[data-dot]', root),
    banner: $('[data-banner]', root),
    invite: $('[data-invite]', root),
    main: $('[data-main]', root),
    sideBoard: $('[data-side-board]', root),
    log: $('[data-log]', root),
    chat: $('[data-chat]', root),
    chatForm: $('[data-chat-form]', root),
  };

  // ---------------------------------------------------------------- banner

  function connectionProblem() {
    const { live } = session;
    const code = live.error?.code;
    if (code === 'version') return { tone: 'warn', text: t('status.version') };
    if (live.link === 'error') {
      return { tone: 'warn', text: t({ 'room-busy': 'status.roomBusy', 'peerjs-load': 'status.peerjsLoad', browser: 'status.browser' }[code] ?? 'status.network') };
    }
    if (live.peerOnline) return null;
    if (code === 'p2p-failed') return { tone: 'warn', text: t('status.p2pFailed') };
    if (code === 'room-busy') return { tone: 'warn', text: t('status.roomBusy') };
    if (code === 'network' || code === 'reclaiming') return { tone: 'wait', text: t(`status.${code}`) };
    if (seat() === 'guest') {
      if (code === 'host-offline') return { tone: 'wait', text: t('status.hostOffline', { name: s().names.host || t('status.theHost') }) };
      return { tone: 'wait', text: t(code === 'slow' ? 'status.slow' : 'status.connecting') };
    }
    if (kind === 'online' && live.link !== 'ready') return { tone: 'wait', text: t('status.opening') };
    if (s().joined) return { tone: 'warn', text: t('status.oppOffline', { name: theirName() }) };
    return null;
  }

  function phaseBanner(v) {
    const name = theirName();
    if (v.phase === 'connecting') return { tone: 'wait', text: t('status.connecting') };
    if (v.phase === 'design') {
      if (!s().joined) return { tone: 'wait', text: t('banner.waitJoin') };
      if (v.iAmReady) return { tone: 'wait', text: t('banner.meReady', { name }) };
      if (v.oppReady) return { tone: 'turn', text: t('banner.oppReady', { name }) };
      return { tone: 'info', text: t('banner.drawing', { name }) };
    }
    if (v.phase === 'play') {
      if (v.turn !== v.seat) return { tone: 'wait', text: t('banner.theirTurn', { name }) };
      if (s().round.pending) return { tone: 'wait', text: t('banner.pending') };
      return { tone: 'turn', text: t('banner.myTurn') };
    }
    return v.winner === v.seat ? { tone: 'win', text: t('banner.won') } : { tone: 'lose', text: t('banner.lost', { name }) };
  }

  // ---------------------------------------------------------------- phases

  function mountPhase(phase) {
    ui.phase = phase;
    el.main.className = `g-main phase--${phase}`;
    if (phase === 'connecting') {
      el.main.innerHTML = `<div class="connecting"><div class="spinner"></div></div>`;
    } else if (phase === 'design') {
      el.main.innerHTML = `
        <div class="phase-head">
          <h2>${t('design.title')}</h2>
          <p>${t('design.help', { walls: s().settings.walls })}</p>
        </div>
        <div class="toolbar" data-toolbar>
          <div class="seg" role="radiogroup" aria-label="${t('design.tools')}">
            <button role="radio" data-tool="walls"><span class="tool-wall"></span>${t('tool.walls')}</button>
            <button role="radio" data-tool="entrance"><b class="letter-e">E</b>${t('tool.entrance')}</button>
            <button role="radio" data-tool="exit"><b class="letter-s">S</b>${t('tool.exit')}</button>
          </div>
          <div class="toolbar-actions">
            <button class="btn btn-small" data-act="random">🎲 ${t('design.random')}</button>
            <button class="btn btn-small" data-act="clear">${t('design.clear')}</button>
          </div>
        </div>
        <div class="board-wrap is-editable" data-board="design"></div>
        <div class="design-foot">
          <div class="design-status" data-status></div>
          <label class="check"><input type="checkbox" data-act="show-path" ${ui.showPath ? 'checked' : ''}> ${t('design.showPath')}</label>
        </div>
        <div class="design-cta" data-cta></div>`;
      bindDesignBoard($('[data-board=design]', el.main));
    } else if (phase === 'play') {
      el.main.innerHTML = `
        <div class="phase-head">
          <h2 data-title></h2>
          <p class="hint">${t('play.hint')}</p>
        </div>
        <div class="board-wrap is-explore" data-board="explore"></div>
        <div class="play-controls">
          <div class="dpad" aria-label="${t('play.dpad')}">
            ${DIRS.map((d) => `<button class="dpad-${d}" data-dir="${d}" aria-label="${t(`dir.${d}`)}">${ARROW[d]}</button>`).join('')}
          </div>
          <div class="play-stats" data-stats></div>
          <button class="btn btn-ghost btn-small" data-act="resign">${t('play.resign')}</button>
        </div>`;
      $('[data-board=explore]', el.main).addEventListener('click', (e) => {
        const svg = $('svg', e.currentTarget);
        const cell = svg && cellAtPoint(svg, e, n());
        if (cell != null) clickExplore(cell);
      });
    } else {
      el.main.innerHTML = `
        <div class="result" data-result></div>
        <div class="verify" data-verify></div>
        <div class="phase-head"><h2 data-reveal-title></h2><p class="hint">${t('over.legend')}</p></div>
        <div class="board-wrap" data-board="reveal"></div>
        <table class="over-stats" data-stats></table>
        <div class="over-actions">
          <button class="btn btn-primary btn-big" data-act="rematch"></button>
          <a class="btn" href="#/">${t('over.menu')}</a>
        </div>`;
    }
  }

  // ---------------------------------------------------------------- design

  function bindDesignBoard(wrap) {
    const locked = () => session.view().iAmReady;
    const svgOf = () => $('svg', wrap);
    const preview = (key) => {
      const p = $('.b-preview', wrap);
      if (!p) return;
      p.setAttribute('d', previewPath(key, n()));
      p.classList.toggle('is-erase', !!key && s().round.myMaze.walls.includes(key));
    };
    const applyWall = (key, onOff) => {
      const res = session.setWall(key, onOff);
      if (res === 'budget' && !ui.drag?.warned) {
        if (ui.drag) ui.drag.warned = true;
        toast(tt('design.budget'), { kind: 'warn' });
      }
    };
    wrap.addEventListener('pointerdown', (e) => {
      const svg = svgOf();
      if (!svg || locked() || e.button > 0) return;
      if (ui.tool === 'walls') {
        const key = edgeAtPoint(svg, e, n());
        if (!key) return;
        e.preventDefault();
        ui.drag = { on: !s().round.myMaze.walls.includes(key), last: key, warned: false };
        applyWall(key, ui.drag.on);
        wrap.setPointerCapture?.(e.pointerId);
      } else {
        const cell = cellAtPoint(svg, e, n());
        if (cell == null) return;
        if (ui.tool === 'entrance') session.setEntrance(cell);
        else session.setExit(cell);
        // Natural flow: after placing E, offer S; after S, back to walls if some are left.
        const m = s().round.myMaze;
        if (ui.tool === 'entrance' && m.entrance != null && m.exit == null) ui.tool = 'exit';
        else if (ui.tool === 'exit' && m.exit != null && m.entrance == null) ui.tool = 'entrance';
        update();
      }
    });
    wrap.addEventListener('pointermove', (e) => {
      const svg = svgOf();
      if (!svg || ui.tool !== 'walls' || locked()) return;
      const key = edgeAtPoint(svg, e, n());
      if (ui.drag) {
        if (key && key !== ui.drag.last) {
          ui.drag.last = key;
          applyWall(key, ui.drag.on);
        }
      } else preview(key);
    });
    const end = () => {
      ui.drag = null;
    };
    wrap.addEventListener('pointerup', end);
    wrap.addEventListener('pointercancel', end);
    wrap.addEventListener('pointerleave', () => !ui.drag && preview(null));
  }

  function issueText(issue) {
    return t(`issue.${issue.code}`, { count: issue.count });
  }

  function updateDesign(v) {
    const r = s().round;
    const m = r.myMaze;
    const settings = s().settings;
    const solution = ui.showPath && m.entrance != null && m.exit != null
      ? shortestPath(settings.size, new Set(m.walls), m.entrance, m.exit)
      : null;
    const wrap = $('[data-board=design]', el.main);
    wrap.classList.toggle('is-locked', v.iAmReady);
    wrap.dataset.tool = ui.tool;
    setHTML(wrap, renderBoard({
      size: settings.size,
      variant: 'design',
      walls: m.walls,
      entrance: m.entrance,
      exit: m.exit,
      solution,
      label: t('design.title'),
    }));

    const toolbar = $('[data-toolbar]', el.main);
    toolbar.classList.toggle('is-hidden', v.iAmReady);
    for (const b of toolbar.querySelectorAll('[data-tool]')) {
      const active = b.dataset.tool === ui.tool;
      b.classList.toggle('is-active', active);
      b.setAttribute('aria-checked', String(active));
    }

    const used = m.walls.length;
    const issue = v.issues[0];
    const pct = Math.min(100, (used / settings.walls) * 100);
    setHTML($('[data-status]', el.main), `
      <span class="counter ${used === settings.walls ? 'is-full' : ''}">
        <span class="counter-bar"><span style="width:${pct}%"></span></span>
        ${t('design.counter', { used, total: settings.walls })}
      </span>
      <span class="issue ${issue ? '' : 'is-ok'}">${issue ? issueText(issue) : t('design.ok')}</span>`);

    const cta = $('[data-cta]', el.main);
    if (v.iAmReady) {
      setHTML(cta, `<div class="locked">${t('design.locked')}</div>`);
    } else {
      setHTML(cta, `<button class="btn btn-primary btn-big" data-act="ready" ${v.issues.length ? 'disabled' : ''}>${t('design.ready')}</button>
        <p class="hint">${t(`design.hint.${ui.tool}`)}</p>`);
    }
  }

  // ---------------------------------------------------------------- play

  function cursorOf(v) {
    const key = `${s().round.n}:${v.mine.moves}`;
    if (ui.cursorKey !== key || !v.mine.visited.has(ui.cursor)) {
      ui.cursorKey = key;
      ui.cursor = v.mine.position;
    }
    return ui.cursor;
  }

  const myMove = (v) => v.phase === 'play' && v.turn === v.seat && !s().round.pending;

  function freshFor(who) {
    return ui.fresh && ui.fresh.who === who && Date.now() - ui.fresh.at < 1500 ? ui.fresh : null;
  }

  function tryProbe(from, dir) {
    const err = session.probe(from, dir);
    if (err === 'not-your-turn') toast(tt('toast.notYourTurn'));
    else if (err === 'known') toast(tt('toast.known'));
  }

  function moveDir(dir) {
    const v = session.view();
    if (v.phase !== 'play') return;
    const from = cursorOf(v);
    const to = neighbor(from, dir, n());
    if (to < 0) return;
    const key = edgeKey(from, to);
    if (v.mine.open.has(key)) {
      // Walking along passages you already know is free.
      ui.cursor = to;
      update();
    } else if (v.mine.walls.has(key)) {
      toast(tt('toast.known'));
    } else {
      tryProbe(from, dir);
    }
  }

  function clickExplore(cell) {
    const v = session.view();
    if (v.phase !== 'play') return;
    if (v.mine.visited.has(cell)) {
      ui.cursor = cell;
      return update();
    }
    const cursor = cursorOf(v);
    const options = DIRS.map((d) => ({ from: neighbor(cell, d, n()), dir: { U: 'D', D: 'U', L: 'R', R: 'L' }[d] }))
      .filter((o) => o.from >= 0 && !probeError(v.mine, o.from, o.dir, n()));
    if (!options.length) return toast(tt('toast.notReached'));
    const pick = options.find((o) => o.from === cursor) ?? (options.length === 1 ? options[0] : null);
    if (!pick) return toast(tt('toast.pickStart'));
    ui.cursor = pick.from;
    tryProbe(pick.from, pick.dir);
  }

  function updatePlay(v) {
    const r = s().round;
    const cursor = cursorOf(v);
    const targets = myMove(v)
      ? DIRS.filter((d) => !probeError(v.mine, cursor, d, n())).map((d) => ({ cell: cursor, dir: d }))
      : [];
    setHTML($('[data-title]', el.main), t('play.title', { name: theirName() }));
    const wrap = $('[data-board=explore]', el.main);
    wrap.classList.toggle('is-pending', !!r.pending);
    wrap.classList.toggle('is-waiting', !myMove(v));
    setHTML(wrap, renderBoard({
      size: n(),
      variant: 'explore',
      who: 'me',
      visited: v.mine.visited,
      passages: [...v.mine.open],
      walls: [...v.mine.walls],
      entrance: r.oppES.entrance,
      exit: r.oppES.exit,
      token: cursor,
      targets,
      fresh: freshFor('me'),
      label: t('play.title', { name: theirName() }),
    }));
    for (const b of el.main.querySelectorAll('[data-dir]')) {
      const to = neighbor(cursor, b.dataset.dir, n());
      const known = to >= 0 && v.mine.open.has(edgeKey(cursor, to));
      b.disabled = !(known || targets.some((x) => x.dir === b.dataset.dir));
    }
    setHTML($('[data-stats]', el.main), t('play.stats', { moves: v.mine.moves, hits: v.mine.hits }));
  }

  // ---------------------------------------------------------------- over

  function updateOver(v) {
    const r = s().round;
    const name = theirName();
    const won = v.winner === v.seat;
    const resultKey = r.resigned ? (won ? 'over.winResign' : 'over.loseResign') : won ? 'over.win' : 'over.lose';
    setHTML($('[data-result]', el.main), `<div class="result-text ${won ? 'is-win' : 'is-lose'}">${t(resultKey, { name })}</div>`);

    const verify = $('[data-verify]', el.main);
    const ver = r.verification;
    if (!ver) {
      verify.className = 'verify is-wait';
      setHTML(verify, t('verify.wait', { name }));
    } else if (ver.ok) {
      verify.className = 'verify is-ok';
      setHTML(verify, t('verify.ok', { name }));
    } else {
      verify.className = 'verify is-bad';
      const codes = [...new Set(ver.problems.map((p) => p.code))];
      setHTML(verify, `${t('verify.bad', { name })}<ul>${codes.map((c) => `<li>${t(`verify.${c}`)}</li>`).join('')}</ul>`);
    }

    setHTML($('[data-reveal-title]', el.main), t('over.revealed', { name }));
    const rev = r.oppReveal?.maze;
    const mine = v.mine;
    if (mine && r.oppES) {
      setHTML($('[data-board=reveal]', el.main), renderBoard({
        size: n(),
        variant: 'reveal',
        who: 'me',
        visited: mine.visited,
        passages: [...mine.open],
        walls: rev ? rev.walls.filter((k) => mine.walls.has(k)) : [...mine.walls],
        ghostWalls: rev ? rev.walls.filter((k) => !mine.walls.has(k)) : [],
        entrance: r.oppES.entrance,
        exit: r.oppES.exit,
        token: mine.position,
        label: t('over.revealed', { name }),
      }));
    }

    const row = (label, p) => `<tr><th>${label}</th><td>${p?.moves ?? 0}</td><td>${p?.hits ?? 0}</td></tr>`;
    setHTML($('[data-stats]', el.main), `
      <thead><tr><th></th><th>${t('over.moves')}</th><th>${t('over.hits')}</th></tr></thead>
      <tbody>${row(esc(names().me), mine)}${row(esc(name), v.theirs)}</tbody>`);

    const btn = $('[data-act=rematch]', el.main);
    const meAsked = r.rematch[v.seat];
    const theyAsked = r.rematch[v.opp];
    btn.disabled = meAsked;
    setHTML(btn, meAsked ? t('over.rematchWaiting', { name }) : theyAsked ? t('over.rematchAccept', { name }) : t('over.rematch'));
    btn.classList.toggle('is-pulsing', theyAsked && !meAsked);
  }

  // ---------------------------------------------------------------- side column

  function updateSide(v) {
    const playing = v.phase === 'play' || v.phase === 'over';
    el.sideBoard.hidden = !playing || !v.theirs;
    el.log.hidden = !playing;
    if (!playing) return;
    const r = s().round;
    const name = theirName();
    if (v.theirs) {
      setHTML(el.sideBoard, `<h3>${t('side.yourMaze', { name })}</h3>
        <div class="board-wrap is-small">${renderBoard({
          size: n(),
          variant: 'defend',
          who: 'them',
          walls: r.myMaze.walls,
          visited: v.theirs.visited,
          passages: [...v.theirs.open],
          hits: [...v.theirs.walls],
          entrance: r.myMaze.entrance,
          exit: r.myMaze.exit,
          token: v.theirs.position,
          fresh: freshFor('them'),
          label: t('side.yourMaze', { name }),
        })}</div>
        <p class="hint">${t('play.stats', { moves: v.theirs.moves, hits: v.theirs.hits })}</p>`);
    }
    if (ui.logCount === v.entries.length && el.log.firstChild) return;
    ui.logCount = v.entries.length;
    const exitOf = (entrySeat) => (entrySeat === v.seat ? r.oppES?.exit : r.myMaze.exit);
    let turn = 0;
    let prev = null;
    const items = v.entries.map((e) => {
      if (e.seat !== prev) turn++;
      prev = e.seat;
      const who = e.seat === v.seat ? 'me' : 'them';
      const to = neighbor(e.from, e.dir, n());
      const res = !e.open ? `🧱 ${t('log.wall')}` : to === exitOf(e.seat) ? `🏁 ${t('log.exit')}` : '✓';
      return `<li class="log-item is-${who} ${e.open ? '' : 'is-wall'}" data-turn="${turn % 2}">
        <span class="log-who">${esc(who === 'me' ? names().me : name)}</span>
        <span class="log-move">${cellLabel(e.from, n())} ${ARROW[e.dir]}</span>
        <span class="log-res">${res}</span></li>`;
    });
    setHTML(el.log, `<h3>${t('log.title')}</h3><ol class="log-list">${items.join('') || `<li class="log-empty">${t('log.empty')}</li>`}</ol>`);
    const list = $('.log-list', el.log);
    list.scrollTop = list.scrollHeight;
  }

  function updateChat() {
    const chat = s().chat;
    const input = el.chatForm.elements.text;
    input.disabled = !session.live.peerOnline;
    input.placeholder = session.live.peerOnline ? tt('chat.placeholder') : tt('chat.offline');
    for (const b of root.querySelectorAll('[data-quick]')) b.disabled = !session.live.peerOnline;
    if (ui.chatCount === chat.length) return;
    ui.chatCount = chat.length;
    setHTML(el.chat, chat.map((c) => `<li class="chat-msg is-${c.from === seat() ? 'me' : 'them'}">${esc(c.text)}</li>`).join('')
      || `<li class="chat-empty">${t('chat.empty')}</li>`);
    el.chat.scrollTop = el.chat.scrollHeight;
  }

  // ---------------------------------------------------------------- main update

  function update() {
    if (destroyed) return;
    const v = session.view();
    const st = s();
    if (session.live.error?.code === 'rejected') return showRejected(session.live.error.reason);

    const nm = names();
    el.nameMe.textContent = nm.me;
    el.nameThem.textContent = nm.them;
    const score = scoreOf(st.history);
    el.score.textContent = `${score[st.seat]} – ${score[other(st.seat)]}`;
    el.dot.className = `g-dot ${session.live.peerOnline ? 'is-online' : st.joined || seat() === 'guest' ? 'is-offline' : 'is-waiting'}`;

    const banner = connectionProblem() ?? phaseBanner(v);
    el.banner.className = `g-banner is-${banner.tone}`;
    setHTML(el.banner, banner.text);

    const showInvite = kind === 'online' && seat() === 'host' && !st.joined;
    el.invite.hidden = !showInvite;
    if (showInvite && !el.invite.firstChild) {
      el.invite.innerHTML = `
        <div><strong>${t('invite.title')}</strong> ${t('invite.desc')}</div>
        <div class="invite-row">
          <input readonly value="${esc(inviteUrl)}" aria-label="${t('invite.title')}" data-invite-url>
          <button class="btn btn-primary btn-small" data-act="copy-link">${t('game.copyLink')}</button>
          ${navigator.share ? `<button class="btn btn-small" data-act="share">${t('game.share')}</button>` : ''}
        </div>
        <p class="hint">${t('invite.meanwhile')}</p>`;
    }

    if (v.phase !== ui.phase) mountPhase(v.phase);
    if (v.phase === 'design') updateDesign(v);
    else if (v.phase === 'play') updatePlay(v);
    else if (v.phase === 'over') updateOver(v);
    updateSide(v);
    updateChat();
  }

  function showRejected(reason) {
    root.innerHTML = `<div class="notice card">
      <h2>${t('rejected.title')}</h2>
      <p>${t(reason === 'full' ? 'rejected.full' : reason === 'version' ? 'status.version' : 'rejected.other')}</p>
      <a class="btn btn-primary" href="#/">${t('over.menu')}</a></div>`;
  }

  // ---------------------------------------------------------------- events

  const alertTitle = (key) => {
    if (document.hidden) document.title = `● ${tt(key)} — ${baseTitle}`;
  };
  const onVisible = () => {
    if (!document.hidden) document.title = baseTitle;
  };
  document.addEventListener('visibilitychange', onVisible);
  cleanups.push(() => document.removeEventListener('visibilitychange', onVisible));

  cleanups.push(session.subscribe((type, e) => {
    if (type === 'change') return update();
    const name = theirName();
    switch (e.kind) {
      case 'probe': {
        const key = edgeKey(e.from, neighbor(e.from, e.dir, n()));
        ui.fresh = { who: e.who, key, open: e.open, at: Date.now() };
        if (e.open) sound.open();
        else sound.wall();
        if (e.who === 'me' && !e.open) toast(tt('toast.wallHit', { name }), { kind: 'warn', ms: 1800 });
        setTimeout(() => !destroyed && update(), 1600); // let the "fresh" highlight fade
        break;
      }
      case 'your-turn':
        sound.turn();
        alertTitle('title.yourTurn');
        break;
      case 'phase':
        if (e.phase === 'over') {
          if (session.view().winner === seat()) sound.win();
          else sound.lose();
        }
        if (e.phase === 'play') toast(session.view().turn === seat() ? tt('toast.youStart') : tt('toast.theyStart', { name }));
        break;
      case 'opp-joined':
        if (kind === 'online') toast(tt('toast.oppJoined', { name }), { kind: 'good' });
        sound.turn();
        break;
      case 'opp-offline':
        if (kind === 'online' && s().joined) toast(tt('toast.oppLeft', { name }), { kind: 'warn' });
        break;
      case 'opp-ready':
        toast(tt('toast.oppReady', { name }));
        break;
      case 'rematch-request':
        toast(tt('toast.rematch', { name }), { kind: 'good' });
        break;
      case 'new-round':
        toast(tt(s().round.starter === seat() ? 'toast.newRoundYou' : 'toast.newRoundThem', { n: e.n, name }), { kind: 'good', ms: 3500 });
        break;
      case 'verified':
        if (!e.ok) toast(tt('verify.bad', { name }), { kind: 'warn', ms: 5000 });
        break;
      case 'chat':
        if (e.entry.from !== seat()) {
          sound.chat();
          alertTitle('title.chat');
        }
        break;
      default:
    }
  }));

  cleanups.push(on(root, 'click', '[data-act]', async (e, b) => {
    const act = b.dataset.act;
    if (act === 'copy-link') {
      if (await copyText(inviteUrl)) toast(tt('game.copied'), { kind: 'good' });
    } else if (act === 'share') {
      navigator.share?.({ title: 'Labyrinthe', text: tt('invite.shareText'), url: inviteUrl }).catch(() => {});
    } else if (act === 'random') {
      session.randomizeMaze();
    } else if (act === 'clear') {
      session.clearMaze();
      ui.tool = 'walls';
      update();
    } else if (act === 'ready') {
      session.ready();
    } else if (act === 'resign') {
      if (confirm(tt('play.resignConfirm'))) session.resign();
    } else if (act === 'rematch') {
      session.requestRematch();
    }
  }));
  cleanups.push(on(root, 'change', '[data-act=show-path]', (e, b) => {
    ui.showPath = b.checked;
    update();
  }));
  cleanups.push(on(root, 'click', '[data-tool]', (e, b) => {
    ui.tool = b.dataset.tool;
    update();
  }));
  cleanups.push(on(root, 'click', '[data-dir]', (e, b) => moveDir(b.dataset.dir)));
  cleanups.push(on(root, 'click', '[data-quick]', (e, b) => session.sendChat(b.dataset.quick)));
  cleanups.push(on(root, 'focusin', '[data-invite-url]', (e, input) => input.select()));
  el.chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const input = el.chatForm.elements.text;
    if (session.sendChat(input.value)) input.value = '';
  });

  const onKey = (e) => {
    if (e.target.closest?.('input, textarea, select') || e.metaKey || e.ctrlKey || e.altKey) return;
    const v = session.view();
    if (v.phase === 'play') {
      const dir = KEY_DIRS[e.key] ?? KEY_DIRS[e.key.toLowerCase()];
      if (dir) {
        e.preventDefault();
        moveDir(dir);
      }
    } else if (v.phase === 'design' && !v.iAmReady) {
      const tool = { w: 'walls', e: 'entrance', s: 'exit' }[e.key.toLowerCase()];
      if (tool) {
        ui.tool = tool;
        update();
      }
    }
  };
  document.addEventListener('keydown', onKey);
  cleanups.push(() => document.removeEventListener('keydown', onKey));

  update();

  return {
    destroy() {
      destroyed = true;
      cleanups.forEach((fn) => fn());
      document.title = baseTitle;
    },
  };
}
