// IndexedDB persistence. Everything stays in this browser; nothing is sent to a server.
// Stores: books (key bookId), chunks (key chunkId, index bookId), history (auto key, index ts), meta (key k).

const DB_NAME = "scripture-rag";
const DB_VERSION = 1;

function req(r) {
  return new Promise((resolve, reject) => { r.onsuccess = () => resolve(r.result); r.onerror = () => reject(r.error); });
}

export async function openDb() {
  const r = indexedDB.open(DB_NAME, DB_VERSION);
  r.onupgradeneeded = () => {
    const db = r.result;
    db.createObjectStore("books", { keyPath: "bookId" });
    db.createObjectStore("chunks", { keyPath: "chunkId" }).createIndex("bookId", "bookId");
    db.createObjectStore("history", { keyPath: "id", autoIncrement: true }).createIndex("ts", "ts");
    db.createObjectStore("meta", { keyPath: "k" });
  };
  return req(r);
}

function store(db, name, mode = "readonly") {
  return db.transaction(name, mode).objectStore(name);
}

export const getAll = (db, name) => req(store(db, name).getAll());
export const put = (db, name, value) => req(store(db, name, "readwrite").put(value));
export const del = (db, name, key) => req(store(db, name, "readwrite").delete(key));

/** Save a book and its chunks in one transaction. O(c) writes. */
export function saveBook(db, book, chunks) {
  const tx = db.transaction(["books", "chunks"], "readwrite");
  tx.objectStore("books").put(book);
  const cs = tx.objectStore("chunks");
  for (const c of chunks) cs.put(c);
  return new Promise((resolve, reject) => { tx.oncomplete = resolve; tx.onerror = () => reject(tx.error); });
}

export async function deleteBook(db, bookId) {
  const tx = db.transaction(["books", "chunks"], "readwrite");
  tx.objectStore("books").delete(bookId);
  const idx = tx.objectStore("chunks").index("bookId");
  const keys = await req(idx.getAllKeys(bookId));
  for (const k of keys) tx.objectStore("chunks").delete(k);
  return new Promise((resolve) => { tx.oncomplete = resolve; });
}

export async function clearAll(db) {
  for (const name of ["books", "chunks", "history", "meta"]) await req(store(db, name, "readwrite").clear());
}

/** Export everything as JSON; vectors become plain arrays. */
export async function exportJson(db) {
  const chunks = (await getAll(db, "chunks")).map((c) => ({ ...c, vec: Array.from(c.vec) }));
  return JSON.stringify({ format: "scripture-rag/1", books: await getAll(db, "books"), chunks,
    history: await getAll(db, "history"), meta: await getAll(db, "meta") });
}

/** Import a previous export, validating shape and types before writing. */
export async function importJson(db, text) {
  const data = JSON.parse(text);
  if (data.format !== "scripture-rag/1" || !Array.isArray(data.books) || !Array.isArray(data.chunks)) {
    throw new Error("Not a Scripture RAG export");
  }
  for (const b of data.books) if (typeof b.bookId === "string") await put(db, "books", b);
  const tx = db.transaction("chunks", "readwrite");
  for (const c of data.chunks) {
    if (typeof c.chunkId !== "string" || typeof c.text !== "string" || !Array.isArray(c.vec)) continue;
    tx.objectStore("chunks").put({ ...c, vec: Float32Array.from(c.vec.map(Number)) });
  }
  await new Promise((resolve) => { tx.oncomplete = resolve; });
  for (const h of data.history || []) { const { id, ...rest } = h; await put(db, "history", rest); }
  for (const m of data.meta || []) await put(db, "meta", m);
}
