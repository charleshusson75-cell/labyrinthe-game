// Tiny DOM helpers. All dynamic text goes through `esc` before reaching innerHTML.

const ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
export const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ESC[c]);

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

/** Delegated listener: `on(root, 'click', '[data-act]', (e, el) => …)`. */
export function on(root, type, selector, fn, opts) {
  const handler = (e) => {
    const el = e.target.closest?.(selector);
    if (el && root.contains(el)) fn(e, el);
  };
  root.addEventListener(type, handler, opts);
  return () => root.removeEventListener(type, handler, opts);
}

export function setHTML(el, html) {
  if (el && el.innerHTML !== html) el.innerHTML = html;
}

let toastRoot = null;

export function toast(message, { kind = 'info', ms = 2600 } = {}) {
  toastRoot ??= document.getElementById('toasts');
  if (!toastRoot) return;
  const el = document.createElement('div');
  el.className = `toast toast--${kind}`;
  el.textContent = message;
  toastRoot.append(el);
  requestAnimationFrame(() => el.classList.add('is-in'));
  setTimeout(() => {
    el.classList.remove('is-in');
    setTimeout(() => el.remove(), 300);
  }, ms);
}

export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const ta = document.createElement('textarea');
    ta.value = text;
    document.body.append(ta);
    ta.select();
    const ok = document.execCommand('copy');
    ta.remove();
    return ok;
  }
}
