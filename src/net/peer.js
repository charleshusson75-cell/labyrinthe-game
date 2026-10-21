// Online transport: WebRTC data channel via PeerJS. The host registers a peer id derived from the
// room code; the guest dials it. Both sides retry on their own after network hiccups.

import { CONFIG } from '../config.js';
import { createHostGate } from './gate.js';

let peerLib = null;

function loadPeerJS() {
  if (globalThis.Peer) return Promise.resolve(globalThis.Peer);
  peerLib ??= new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = CONFIG.peerjsUrl;
    s.integrity = CONFIG.peerjsIntegrity;
    s.crossOrigin = 'anonymous';
    s.onload = () => resolve(globalThis.Peer);
    s.onerror = () => {
      peerLib = null;
      s.remove();
      reject(new Error('Could not load PeerJS'));
    };
    document.head.append(s);
  });
  return peerLib;
}

export const hostPeerId = (roomCode) => CONFIG.peerIdPrefix + roomCode;

// PeerJS's own defaults, spelled out because passing `config` replaces them.
const DEFAULT_ICE = [
  { urls: 'stun:stun.l.google.com:19302' },
  { urls: ['turn:eu-0.turn.peerjs.com:3478', 'turn:us-0.turn.peerjs.com:3478'], username: 'peerjs', credential: 'peerjsp' },
];

/** Peer options, plus relay servers from `CONFIG.iceServersUrl` when configured (best effort). */
async function peerOptions() {
  const base = CONFIG.peerOptions;
  if (!CONFIG.iceServersUrl || base.config?.iceServers) return base;
  try {
    const res = await fetch(CONFIG.iceServersUrl, { signal: AbortSignal.timeout(5000) });
    const data = await res.json();
    const extra = [].concat(Array.isArray(data) ? data : data.iceServers).filter((s) => s?.urls);
    if (!extra.length) return base;
    return { ...base, config: { ...base.config, iceServers: [...DEFAULT_ICE, ...extra] } };
  } catch {
    return base; // relay provider down or misconfigured: direct connections still work
  }
}

const DIAL_TIMEOUT = 15000;
const WATCHDOG_EVERY = 10000;

export function createPeerTransport({ role, roomCode }) {
  let handlers = null;
  let peer = null;
  let options = CONFIG.peerOptions;
  let destroyed = false;
  let restart = null; // builds a fresh Peer for this role
  let restartTimer = null;
  let onWake = null; // role-specific extra work when the page wakes up
  const timers = new Set();

  const later = (fn, ms) => {
    const id = setTimeout(() => {
      timers.delete(id);
      if (!destroyed) fn();
    }, ms);
    timers.add(id);
    return id;
  };
  const cancel = (id) => {
    clearTimeout(id);
    timers.delete(id);
  };
  const scheduleRestart = (ms) => {
    cancel(restartTimer);
    restartTimer = later(() => {
      restartTimer = null;
      restart();
    }, ms);
  };

  /** Shared Peer lifecycle: create, keep the signaling link alive, report status. */
  function makePeer(id, { onReady, onError }) {
    handlers.onStatus('starting');
    const p = id ? new globalThis.Peer(id, options) : new globalThis.Peer(options);
    let opened = false;
    peer = p;
    p.on('open', () => {
      opened = true;
      onReady(p);
    });
    p.on('disconnected', () => {
      // Lost the signaling server (network drop, phone put the page to sleep…).
      // Open data connections keep working meanwhile.
      if (destroyed || p.destroyed) return;
      handlers.onStatus('starting', { code: 'network' });
      later(() => !p.destroyed && p.disconnected && p.reconnect(), 2000);
    });
    p.on('error', (err) => {
      if (destroyed) return;
      if (onError(err, p)) return;
      if (err.type === 'browser-incompatible') return handlers.onStatus('error', { code: 'browser' });
      // Once registered, network errors are followed by 'disconnected' (handled above).
      // Before that, start over with a fresh Peer.
      if (!opened && !p.destroyed) {
        p.destroy();
        handlers.onStatus('starting', { code: 'network' });
        scheduleRestart(3000);
      }
    });
    return p;
  }

  /**
   * Called when the page comes back to the foreground, the network returns, or by the watchdog:
   * phones suspend background pages and kill their connections, so don't wait for timers.
   */
  function wake() {
    if (destroyed || !restart) return;
    if (!peer || peer.destroyed) {
      if (!restartTimer) restart();
    } else if (peer.disconnected) {
      peer.reconnect();
    }
    onWake?.();
  }
  const onVisible = () => document.visibilityState === 'visible' && wake();

  // ---------------------------------------------------------------- host

  function startHost() {
    const gate = createHostGate(handlers);
    let idAttempts = 0;
    api.send = (msg) => gate.send(msg);
    api.reset = () => gate.drop();

    const onReady = (p) => {
      idAttempts = 0;
      handlers.onStatus('ready');
      p.removeAllListeners('connection');
      p.on('connection', (conn) => {
        const wrapped = {
          send: (m) => conn.open && conn.send(m),
          close: () => conn.close(),
        };
        conn.on('data', (d) => gate.onData(wrapped, d));
        conn.on('close', () => gate.onClose(wrapped));
        conn.on('error', () => gate.onClose(wrapped));
      });
    };
    const onError = (err, p) => {
      if (err.type !== 'unavailable-id') return false;
      // Our id is still held by the server: our previous connection hasn't timed out yet (page
      // reloaded or woke up), or the room is open on another device. Keep trying; after a minute,
      // say so.
      p.destroy();
      idAttempts++;
      handlers.onStatus('starting', { code: idAttempts > 20 ? 'room-busy' : 'reclaiming' });
      scheduleRestart(idAttempts > 20 ? 5000 : 3000);
      return true;
    };
    restart = () => makePeer(hostPeerId(roomCode), { onReady, onError });
    restart();
  }

  // ---------------------------------------------------------------- guest

  function startGuest() {
    let conn = null;
    let wasOpen = false;
    let failures = 0;
    let dialTimer = null;
    let openTimer = null;

    const dropConn = () => {
      const c = conn;
      conn = null;
      cancel(openTimer);
      if (c) c.close();
      if (wasOpen) {
        wasOpen = false;
        handlers.onClose();
      }
    };
    const scheduleDial = (ms) => {
      cancel(dialTimer);
      dialTimer = later(dial, ms);
    };

    function dial() {
      if (!peer || peer.destroyed || peer.disconnected) return scheduleDial(2000);
      const c = peer.connect(hostPeerId(roomCode), { reliable: true, serialization: 'json' });
      conn = c;
      openTimer = later(() => {
        if (conn !== c || c.open) return;
        failures++;
        handlers.onStatus('ready', { code: failures >= 2 ? 'p2p-failed' : 'slow' });
        dropConn();
        scheduleDial(1000);
      }, DIAL_TIMEOUT);
      c.on('open', () => {
        if (conn !== c) return;
        cancel(openTimer);
        failures = 0;
        wasOpen = true;
        handlers.onStatus('ready');
        handlers.onOpen();
      });
      c.on('data', (d) => conn === c && handlers.onMessage(d));
      const lost = () => {
        if (conn !== c) return;
        dropConn();
        scheduleDial(1500);
      };
      c.on('close', lost);
      c.on('error', lost);
    }

    api.send = (msg) => conn?.open && conn.send(msg);
    api.reset = () => {
      dropConn();
      scheduleDial(500);
    };
    onWake = () => {
      if (!conn) scheduleDial(0); // not connected nor dialing: knock right away
    };

    restart = () =>
      makePeer(null, {
        onReady: () => {
          handlers.onStatus('ready');
          if (!conn) scheduleDial(0);
        },
        onError: (err) => {
          if (err.type !== 'peer-unavailable') return false;
          // Host not online (yet / anymore). Keep knocking.
          conn = null;
          cancel(openTimer);
          handlers.onStatus('ready', { code: 'host-offline' });
          scheduleDial(4000);
          return true;
        },
      });
    restart();
  }

  const api = {
    start(h) {
      handlers = h;
      Promise.all([loadPeerJS(), peerOptions()])
        .then(([, opts]) => {
          if (destroyed) return;
          options = opts;
          if (role === 'host') startHost();
          else startGuest();
          document.addEventListener('visibilitychange', onVisible);
          window.addEventListener('online', wake);
          window.addEventListener('pageshow', wake);
          const watchdog = setInterval(() => document.visibilityState === 'visible' && wake(), WATCHDOG_EVERY);
          timers.add(watchdog);
        })
        .catch(() => handlers.onStatus('error', { code: 'peerjs-load' }));
    },
    send() {},
    reset() {},
    destroy() {
      destroyed = true;
      timers.forEach((id) => {
        clearTimeout(id);
        clearInterval(id);
      });
      timers.clear();
      document.removeEventListener('visibilitychange', onVisible);
      window.removeEventListener('online', wake);
      window.removeEventListener('pageshow', wake);
      peer?.destroy();
    },
  };
  return api;
}
