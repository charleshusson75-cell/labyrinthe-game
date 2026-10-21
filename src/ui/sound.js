// Small synthesized sound effects (no audio files). Muting is remembered.

import { prefs } from '../app/prefs.js';

let ctx = null;

function audio() {
  if (prefs.get('muted', false)) return null;
  try {
    ctx ??= new (globalThis.AudioContext || globalThis.webkitAudioContext)();
    if (ctx.state === 'suspended') ctx.resume();
    return ctx;
  } catch {
    return null;
  }
}

function tone(freq, { at = 0, dur = 0.15, type = 'sine', gain = 0.18, slide = 0 } = {}) {
  const a = audio();
  if (!a) return;
  const t = a.currentTime + at;
  const osc = a.createOscillator();
  const g = a.createGain();
  osc.type = type;
  osc.frequency.setValueAtTime(freq, t);
  if (slide) osc.frequency.exponentialRampToValueAtTime(freq * slide, t + dur);
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(gain, t + 0.01);
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  osc.connect(g).connect(a.destination);
  osc.start(t);
  osc.stop(t + dur + 0.02);
}

export const sound = {
  open: () => tone(660, { dur: 0.09, type: 'triangle', gain: 0.12, slide: 1.25 }),
  wall: () => {
    tone(120, { dur: 0.22, type: 'square', gain: 0.12, slide: 0.6 });
    tone(70, { dur: 0.25, gain: 0.2 });
  },
  turn: () => {
    tone(880, { dur: 0.12, gain: 0.1 });
    tone(1320, { at: 0.1, dur: 0.18, gain: 0.08 });
  },
  chat: () => tone(1046, { dur: 0.07, type: 'triangle', gain: 0.07 }),
  win: () => [523, 659, 784, 1046].forEach((f, i) => tone(f, { at: i * 0.11, dur: 0.22, type: 'triangle', gain: 0.14 })),
  lose: () => [392, 330, 262].forEach((f, i) => tone(f, { at: i * 0.15, dur: 0.25, type: 'sine', gain: 0.13 })),
};

export function toggleMute() {
  const muted = !prefs.get('muted', false);
  prefs.set('muted', muted);
  return muted;
}
