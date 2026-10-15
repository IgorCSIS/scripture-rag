// Recapture every image in docs/screenshots from a running copy of the site.
// Usage: SHOT_URL=http://127.0.0.1:8765/ node tools/screenshots/capture.mjs [outDir] [chromePath]
// Needs playwright-core on NODE_PATH.
import { chromium } from "playwright-core";

const out = process.argv[2] || "docs/screenshots";
const exe = process.argv[3] || "/usr/bin/google-chrome";
const url = process.env.SHOT_URL || "http://127.0.0.1:8765/";
const browser = await chromium.launch({ executablePath: exe, headless: true });

const VIEWS = {
  desktop: { viewport: { width: 1440, height: 900 } },
  mobile: { viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true },
};

async function open(view, theme, { onboarded = true, sample = true } = {}) {
  const ctx = await browser.newContext(VIEWS[view]);
  const page = await ctx.newPage();
  await page.addInitScript(([t, ob]) => {
    localStorage.setItem("srag-theme", t);
    if (ob) localStorage.setItem("srag-onboarded", "1");
  }, [theme, onboarded]);
  await page.goto(url);
  await page.waitForFunction(() => globalThis.ScriptureRAG?.state.db);
  if (sample) {
    await page.evaluate(() => { document.getElementById("chunkSize").value = "30"; });
    if (view === "mobile") await page.evaluate(() => document.getElementById("sample").click());
    else await page.click("#sample");
    await page.waitForFunction(() => globalThis.ScriptureRAG?.state.books.length >= 1, null, { timeout: 20000 });
  }
  return { ctx, page };
}

const quiet = (page) => page.evaluate(() => { document.getElementById("toasts").hidden = true; });
const toAsk = (page) => page.evaluate(() => { const r = document.querySelector("#ask").getBoundingClientRect(); window.scrollBy(0, r.top - 90); });

async function ask(page, q, sel) {
  await page.fill("#ask", q);
  await page.click("#askBtn");
  await page.waitForSelector(sel, { timeout: 20000 });
  await page.waitForTimeout(600);
}

for (const view of ["desktop", "mobile"]) {
  for (const theme of ["light", "dark"]) {
    const p = (name) => `${out}/${view}-${theme}-${name}.png`;
    // Onboarding, first visit.
    let { ctx, page } = await open(view, theme, { onboarded: false, sample: false });
    await page.waitForSelector("#onboard[open], #onboard:not([hidden])", { timeout: 5000 }).catch(() => {});
    await page.waitForTimeout(400);
    await page.screenshot({ path: p("onboarding") });
    await ctx.close();
    // Empty state.
    ({ ctx, page } = await open(view, theme, { sample: false }));
    await page.waitForTimeout(300);
    await page.screenshot({ path: p("empty") });
    await ctx.close();
    // Sample loaded: answer, refusal, drawer, history, themes, library, confirm clear.
    ({ ctx, page } = await open(view, theme));
    await quiet(page);
    await ask(page, "Who is my shepherd?", "#answer .chip");
    await page.waitForTimeout(view === "desktop" ? 4000 : 1500);
    await toAsk(page); await page.waitForTimeout(300);
    await page.screenshot({ path: p("answer") });
    await page.locator("#answer .chip").first().click();
    await page.waitForTimeout(500);
    await page.screenshot({ path: p("passage-drawer") });
    await page.keyboard.press("Escape"); await page.waitForTimeout(300);
    await ask(page, "What is the capital of France?", "#answer .refusal");
    await toAsk(page); await page.waitForTimeout(300);
    await page.screenshot({ path: p("refusal") });
    await page.click("#historyToggle"); await page.waitForTimeout(500);
    await page.screenshot({ path: p("history") });
    await page.keyboard.press("Escape"); await page.waitForTimeout(300);
    await page.click("#tab-themes");
    await page.waitForSelector("#themeMap .pt", { timeout: 20000 });
    await page.locator("#panel-themes").scrollIntoViewIfNeeded();
    await page.evaluate(() => window.scrollBy(0, -80)); await page.waitForTimeout(400);
    await page.screenshot({ path: p("themes") });
    if (view === "desktop" && theme === "light") {
      await page.locator("#panel-themes").screenshot({ path: `${out}/theme_map.png` });
      await page.click("#tab-ask"); await page.waitForTimeout(300);
      await page.locator("#app").screenshot({ path: `${out}/ask.png` });
    }
    if (view === "mobile") {
      await page.click("#libraryToggle"); await page.waitForTimeout(500);
      await page.screenshot({ path: p("library") });
    }
    if (view === "mobile") await page.evaluate(() => document.getElementById("clear").click());
    else { await page.locator("#clear").scrollIntoViewIfNeeded(); await page.click("#clear"); }
    await page.waitForSelector("#confirmDlg[open]", { timeout: 5000 });
    await page.waitForTimeout(300);
    await page.screenshot({ path: p("confirm-clear") });
    await ctx.close();
    console.log(`captured ${view} ${theme}`);
  }
}
await browser.close();
