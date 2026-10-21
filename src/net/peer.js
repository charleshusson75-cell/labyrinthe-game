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

const DIAL_TIMEOUT = 15000;

export function createPeerTransport({ role, roomCode }) {
  let handlers = null;
  let peer = null;
  let destroyed = false;
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

  /** Shared Peer lifecycle: (re)create, keep the signaling link alive, report status. */
  function makePeer(id, { onReady, onError }) {
    handlers.onStatus('starting');
    const p = id ? new globalThis.Peer(id, CONFIG.peerOptions) : new globalThis.Peer(CONFIG.peerOptions);
    let opened = false;
    peer = p;
    p.on('open', () => {
      opened = true;
      onReady(p);
    });
    p.on('disconnected', () => {
      // Lost the signaling server; open data connections keep working meanwhile.
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
        later(() => makePeer(id, { onReady, onError }), 3000);
      }
    });
    return p;
  }

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
      // Our id is still held by the server (we just reloaded) or by another tab/device.
      p.destroy();
      if (++idAttempts > 15) handlers.onStatus('error', { code: 'room-busy' });
      else {
        handlers.onStatus('starting', { code: 'reclaiming' });
        later(() => makePeer(hostPeerId(roomCode), { onReady, onError }), 2000);
      }
      return true;
    };
    makePeer(hostPeerId(roomCode), { onReady, onError });
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
  }

  const api = {
    start(h) {
      handlers = h;
      loadPeerJS()
        .then(() => {
          if (destroyed) return;
          if (role === 'host') startHost();
          else startGuest();
        })
        .catch(() => handlers.onStatus('error', { code: 'peerjs-load' }));
    },
    send() {},
    reset() {},
    destroy() {
      destroyed = true;
      timers.forEach(clearTimeout);
      timers.clear();
      peer?.destroy();
    },
  };
  return api;
}
