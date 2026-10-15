// Headless Chrome smoke test of the full browser flow.
// Usage: node scripts/smoke_browser.mjs [chromePath]  (needs playwright-core on NODE_PATH and a server on :8765)
import { chromium } from "playwright-core";
import { writeFileSync } from "node:fs";

const exe = process.argv[2] || "/usr/bin/google-chrome";
const url = process.env.SMOKE_URL || "http://127.0.0.1:8765/";
const browser = await chromium.launch({ executablePath: exe, headless: true });
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
page.on("console", (m) => { if (m.type() === "error" && !/archive\.org|favicon/.test(m.text())) errors.push(m.text()); });
const step = (name, ok) => { console.log(`${ok ? "PASS" : "FAIL"} ${name}`); if (!ok) process.exitCode = 1; };

await page.addInitScript(() => localStorage.setItem("srag-onboarded", "1"));
await page.goto(url);
await page.waitForFunction(() => globalThis.ScriptureRAG?.state.db);
// The tiny sample needs small windows to give the theme map a few points (default is 200 words).
await page.evaluate(() => { document.getElementById("chunkSize").value = "30"; });
await page.click("#sample");
await page.waitForFunction(() => globalThis.ScriptureRAG?.state.books.length >= 1, null, { timeout: 15000 });
writeFileSync("/tmp/notes_on_psalms.txt", "Notes on the Psalms. A commentary. The shepherd in this psalm is read as the Lord who guides and feeds his people, a figure of care.");
await page.setInputFiles("#files", "/tmp/notes_on_psalms.txt");
await page.waitForFunction(() => globalThis.ScriptureRAG.state.books.length >= 2, null, { timeout: 15000 });
step("library has 2 books", true);
step("role selector on each book", (await page.locator("#library .role-select").count()) === 2);
const nb = await page.evaluate(async () => { const s = globalThis.ScriptureRAG; const b = s.state.books.find((x) => /notes/.test(x.bookId)); await s.setRole(b.bookId, "commentary"); return s.roleOf(b.bookId); });
step("role set to commentary", nb === "commentary");

await page.fill("#ask", "Who is my shepherd?");
await page.click("#askBtn");
await page.waitForSelector("#answer .chip", { timeout: 15000 });
step("answer has citation chips with role tags", (await page.locator("#answer .chip .role-tag").count()) > 0);

await page.selectOption("#libRole", "commentary");
await page.click("#askBtn");
await page.waitForTimeout(800);
const chipsC = await page.locator("#answer .chip .role-tag").allTextContents();
step("role filter scopes citations to commentary", chipsC.every((t) => t === "commentary"));
await page.selectOption("#libRole", "");

await page.fill("#ask", "What is the capital of France?");
await page.click("#askBtn");
await page.waitForSelector("#answer .refusal", { timeout: 15000 });
step("unsupported question abstains", true);

await page.fill("#ask", "Quem é o meu pastor? O Senhor não me deixa faltar nada?");
await page.click("#askBtn");
await page.waitForFunction(() => document.querySelector("#answer")?.textContent.includes("Portuguese questions are not supported yet"), null, { timeout: 15000 });
step("Portuguese question gets the unsupported-language message", (await page.locator("#answer .chip").count()) === 0);

await page.fill("#ask", "Who is my shepherd?");
await page.click("#tab-compare");
await page.selectOption("#compareBy", "role");
await page.click("#compareGo");
await page.waitForSelector(".compare-col", { timeout: 15000 });
step("compare mode shows side by side columns", (await page.locator(".compare-col").count()) === 2);
const shots = process.env.SMOKE_SHOTS;
if (shots) await page.evaluate(() => { document.getElementById("toasts").hidden = true; });
if (shots) await page.locator("#panel-compare").screenshot({ path: `${shots}/compare-tab.png` });

await page.click("#tab-themes");
await page.waitForSelector("#themeMap .pt", { timeout: 15000 });
step("theme map renders", (await page.locator("#themeMap .pt").count()) > 0);
step("outlier box present", (await page.locator("#outlierBox").count()) === 1);

const sb = page.locator("details:has(#searchBooks)");
if (await sb.count()) {
  if (!(await sb.evaluate((d) => d.open))) await sb.locator("summary").click();
  step("books to search list renders", (await page.locator("#searchBooks input, #searchBooks label").count()) > 0);
  if (shots) await sb.screenshot({ path: `${shots}/books-to-search.png` });
}
await page.click("#manifestBox summary");
step("manifest table has 2 rows", (await page.locator("#manifestTable tbody tr").count()) === 2);
const [dl] = await Promise.all([page.waitForEvent("download"), page.click("#manifestCsv")]);
step("manifest CSV export downloads", /manifest\.csv$/.test(dl.suggestedFilename()));

await page.click("#tab-archive");
await page.waitForSelector("#iaQuery", { timeout: 10000 });
const filt = await page.evaluate(() => ["iaLang", "iaTrad", "iaSort", "iaCreator", "iaColl", "iaFrom", "iaTo", "iaPdf", "iaPd", "iaGroup", "iaRelated"].every((id) => document.getElementById(id)));
step("archive filters, grouping and related controls mounted", filt);
if (shots) await page.locator("#panel-archive").screenshot({ path: `${shots}/ia-panel.png` });
await page.screenshot({ path: process.env.SMOKE_SHOT || "/tmp/smoke.png", fullPage: false });
step(`no page errors (${errors.join(" | ")})`, errors.length === 0);
await browser.close();
