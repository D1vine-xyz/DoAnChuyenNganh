import { pathToFileURL } from "url";
import path from "path";
import fs from "fs/promises";
import { existsSync } from "fs";

const appData = process.env.APPDATA || "C:\\Users\\ACER\\AppData\\Roaming";
const localAppData = process.env.LOCALAPPDATA || "C:\\Users\\ACER\\AppData\\Local";

const prPath = path.join(appData, "npm", "node_modules", "notebooklm-mcp", "node_modules", "patchright", "index.mjs");
const prUrl = pathToFileURL(prPath).href;
const { chromium } = await import(prUrl);

const dataDir = path.join(localAppData, "notebooklm-mcp", "Data");
const chromeProfileDir = path.join(dataDir, "chrome_profile");
const browserStateDir = path.join(dataDir, "browser_state");
const stateFilePath = path.join(browserStateDir, "state.json");
const sessionFilePath = path.join(browserStateDir, "session.json");

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

console.log("====================================================");
console.log("  NOTEBOOKLM MCP - SMART LOGIN & STATE SAVER");
console.log("====================================================");
console.log("Dang khoi dong trinh duyet voi ho so hien tai...");

try {
  await fs.mkdir(browserStateDir, { recursive: true });

  const context = await chromium.launchPersistentContext(chromeProfileDir, {
    headless: false,
    viewport: { width: 1920, height: 1080 },
    locale: "en-US",
    args: [
      "--disable-blink-features=AutomationControlled",
      "--disable-dev-shm-usage",
      "--no-first-run",
      "--no-default-browser-check",
    ],
  });

  const pages = context.pages();
  const page = pages.length > 0 ? pages[0] : await context.newPage();

  console.log("Dang mo trang dang nhap NotebookLM...");
  const loginUrl = "https://notebook.google.com/login?continue=https://notebooklm.google.com/notebook/dc714098-32b3-4287-8ccb-6ec5f242adad";
  await page.goto(loginUrl, { timeout: 60000 });

  console.log("====================================================");
  console.log("👉 VUI LONG DANG NHAP GOOGLE TREN CUA SO TRINH DUYET");
  console.log("👉 Khi da vao NotebookLM, ban co the:");
  console.log("   1. BAM NUT XANH LA 'BAM VAO DAY DE LUU' o giua tren cung trinh duyet");
  console.log("   2. HOAC GO PHIM ENTER TAI TERMINAL NAY DE LUU NGAY");
  console.log("====================================================");

  const maxAttempts = 300; // 5 minutes
  let saved = false;
  let forceSave = false;

  try {
    process.stdin.resume();
    process.stdin.on("data", () => {
      console.log("\n⚡ Nhan tin hieu ENTER! Dang luu phien ngay lap tuc...");
      forceSave = true;
    });
  } catch (e) {}

  for (let i = 0; i < maxAttempts; i++) {
    const allPages = context.pages();
    const cookies = await context.cookies().catch(() => []);
    const hasSid = cookies.some(c => c.name === "SID" || c.name === "__Secure-1PSID" || c.name === "__Secure-OSID");

    let activeNotebookPage = null;
    let anyNotebookUrl = "";

    for (const p of allPages) {
      try {
        const u = p.url();
        if ((u.includes("notebooklm.google.com") || u.includes("notebook.google.com")) && !u.includes("accounts.google.com") && !u.includes("/trynow")) {
          activeNotebookPage = p;
          anyNotebookUrl = u;
          break;
        }
      } catch (e) {}
    }

    // Inject banner into pages so user can click to save anytime
    let clickedBanner = false;
    for (const p of allPages) {
      try {
        const clicked = await p.evaluate(() => window.__MCP_FORCE_SAVE__).catch(() => false);
        if (clicked) {
          clickedBanner = true;
        }

        await p.evaluate(() => {
          if (document.getElementById("mcp-save-banner")) return;
          const div = document.createElement("div");
          div.id = "mcp-save-banner";
          div.innerHTML = "👉 <b>BAM VAO DAY DE LUU PHIEN NOTEBOOKLM</b> 👈";
          div.style.cssText = "position:fixed;top:15px;left:50%;transform:translateX(-50%);z-index:9999999;background:#16a34a;color:#fff;padding:12px 24px;font-size:16px;font-weight:bold;border-radius:30px;cursor:pointer;box-shadow:0 6px 20px rgba(0,0,0,0.4);border:2px solid #fff;user-select:none;";
          div.onclick = () => {
            div.innerText = "⏳ DANG LUU COOKIES...";
            div.style.background = "#eab308";
            window.__MCP_FORCE_SAVE__ = true;
          };
          document.body.appendChild(div);
        }).catch(() => {});
      } catch (err) {}
    }

    // If any page is on trynow, click Sign in button automatically
    for (const p of allPages) {
      try {
        if (p.url().includes("/trynow")) {
          const signInBtn = await p.$('a[href*="/login"]');
          if (signInBtn) {
            console.log("Tu dong bam nut Dang nhap tren trang Try Now...");
            await signInBtn.click();
          }
        }
      } catch (e) {}
    }

    const isNotebookLM = forceSave || clickedBanner || (hasSid && activeNotebookPage !== null) || (activeNotebookPage && activeNotebookPage.url().includes("/notebook/"));

    if (isNotebookLM) {
      const targetPage = activeNotebookPage || (allPages.length > 0 ? allPages[0] : page);
      const title = await targetPage.title().catch(() => "");
      console.log(`\n🎉 Da xac thuc thanh cong! (URL: ${anyNotebookUrl || (targetPage ? targetPage.url() : "")}, Title: ${title})`);
      console.log("Dang luu trang thai xac thuc va cookies...");

      await sleep(3000);
      await context.storageState({ path: stateFilePath });

      try {
        if (targetPage) {
          const sessionStorageData = await targetPage.evaluate(() => {
            const storage = {};
            for (let k = 0; k < sessionStorage.length; k++) {
              const key = sessionStorage.key(k);
              if (key) storage[key] = sessionStorage.getItem(key) || "";
            }
            return JSON.stringify(storage);
          });
          await fs.writeFile(sessionFilePath, sessionStorageData, { encoding: "utf-8" });
        }
      } catch (err) { }

      // Update file timestamp
      try {
        const now = new Date();
        await fs.utimes(stateFilePath, now, now);
      } catch (err) { }

      console.log("====================================================");
      console.log("✅ LUU THANH CONG: state.json va session.json!");
      console.log(`Cookies luu duoc: ${cookies.length}`);
      console.log("====================================================");
      saved = true;
      break;
    }

    if (i % 5 === 0 && i > 0) {
      const currentUrl = page ? page.url() : "";
      if (currentUrl.includes("accounts.google.com")) {
        console.log(`[${i}s] Dang cho ban nhap Email/Password Google... (Hoac bam Enter de luu ngay)`);
      } else {
        console.log(`[${i}s] Dang cho... (URL: ${currentUrl.slice(0, 50)}) (Hoac bam Enter de luu ngay)`);
      }
    }
    await sleep(1000);
  }

  await context.close().catch(() => {});
  if (saved) {
    console.log("Trinh duyet da dong an toan. Ban co the quay lai IDE!");
  } else {
    console.log("Het thoi gian cho dang nhap.");
  }
} catch (error) {
  console.error("Loi:", error);
}

process.exit(0);
