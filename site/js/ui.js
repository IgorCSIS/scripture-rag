// Small UI toolkit: safe element builder, toasts, drawers with focus trap, verified CDN imports.
// All text is inserted with text nodes (textContent), never as HTML.

export const $ = (id) => document.getElementById(id);

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === false || v == null) continue;
    if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : String(v));
  }
  for (const c of children.flat()) if (c != null && c !== false) node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return node;
}

const SVG_NS = "http://www.w3.org/2000/svg";
export function icon(path, size = 24) {
  const s = document.createElementNS(SVG_NS, "svg");
  s.setAttribute("viewBox", `0 0 ${size} ${size}`);
  s.setAttribute("aria-hidden", "true");
  const p = document.createElementNS(SVG_NS, "path");
  p.setAttribute("d", path);
  s.append(p);
  return s;
}
export const ICONS = {
  quote: "M5 7h5v5H7.5c0 1.6 1 2.6 2.5 3v2c-3 0-5-2.2-5-5zm9 0h5v5h-2.5c0 1.6 1 2.6 2.5 3v2c-3 0-5-2.2-5-5z",
  search: "M10 3a7 7 0 0 1 5.6 11.2l5 5-1.4 1.4-5-5A7 7 0 1 1 10 3zm0 2a5 5 0 1 0 0 10 5 5 0 0 0 0-10z",
  warn: "M12 2 1 21h22zm-1 7h2v6h-2zm0 8h2v2h-2z",
  book: "M5 3h11a3 3 0 0 1 3 3v15H7a2 2 0 0 1-2-2zm2 14v2h10v-2z",
  trash: "M9 3h6l1 2h4v2H4V5h4zm-3 6h12l-1 12H7z",
};

export const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

export function toast(msg, kind = "info", ms = 3600) {
  const t = el("div", { class: `toast ${kind}` }, msg);
  $("toasts").append(t);
  setTimeout(() => t.remove(), ms);
}

/** Make a file name safe to display: no paths, control or bidi characters, bounded length. */
export function cleanFileName(name) {
  const base = String(name).split(/[\\/]/).pop();
  const s = base.replace(/[\u0000-\u001f\u007f\u200b-\u200f\u202a-\u202e\u2066-\u2069]/g, "").replace(/\s+/g, " ").trim();
  return (s.length > 120 ? s.slice(0, 117) + "..." : s) || "untitled";
}

export function formatBytes(n) {
  return n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(0)} KB` : `${(n / 1048576).toFixed(1)} MB`;
}

// Drawers: one open at a time, scrim, focus trap, Esc closes, focus returns to opener.
let openDrawer = null, opener = null;
const FOCUSABLE = 'a[href],button:not([disabled]),input:not([disabled]):not([tabindex="-1"]),select,textarea,[tabindex]:not([tabindex="-1"])';

export function showDrawer(node, from = document.activeElement, { modal = true } = {}) {
  if (openDrawer && openDrawer !== node) hideDrawer(false);
  opener = from;
  node.hidden = false;
  node.classList.add("open");
  if (modal) $("scrim").hidden = false;
  openDrawer = node;
  document.querySelectorAll(`[aria-controls="${node.id}"]`).forEach((b) => b.setAttribute("aria-expanded", "true"));
  const first = node.querySelector("input,[data-autofocus]") || node.querySelector(FOCUSABLE);
  requestAnimationFrame(() => first?.focus());
}

export function hideDrawer(restore = true) {
  const node = openDrawer;
  if (!node) return false;
  node.classList.remove("open");
  if (node.getAttribute("role") === "dialog") node.hidden = true;
  $("scrim").hidden = true;
  document.querySelectorAll(`[aria-controls="${node.id}"]`).forEach((b) => b.setAttribute("aria-expanded", "false"));
  openDrawer = null;
  if (restore && opener?.isConnected) opener.focus();
  return true;
}

export const drawerIsOpen = () => openDrawer;

export function trapFocus(e) {
  if (!openDrawer || e.key !== "Tab") return;
  const items = [...openDrawer.querySelectorAll(FOCUSABLE)].filter((n) => n.offsetParent !== null);
  if (!items.length) return;
  const [first, last] = [items[0], items.at(-1)];
  if (e.shiftKey && document.activeElement === first) { last.focus(); e.preventDefault(); }
  else if (!e.shiftKey && document.activeElement === last) { first.focus(); e.preventDefault(); }
}

async function sha384(buf) {
  const d = new Uint8Array(await crypto.subtle.digest("SHA-384", buf));
  let s = "";
  for (const b of d) s += String.fromCharCode(b);
  return "sha384-" + btoa(s);
}

/** Fetch a pinned CDN file, check its Subresource Integrity hash, and return a blob URL. */
export async function verifiedBlobUrl(url, integrity, type = "text/javascript") {
  const res = await fetch(url, { integrity, credentials: "omit", referrerPolicy: "no-referrer" });
  if (!res.ok) throw new Error(`Could not download ${url}`);
  const buf = await res.arrayBuffer();
  if ((await sha384(buf)) !== integrity) throw new Error("Integrity check failed for a CDN script");
  return URL.createObjectURL(new Blob([buf], { type }));
}

export async function verifiedImport(url, integrity) {
  return import(await verifiedBlobUrl(url, integrity));
}
