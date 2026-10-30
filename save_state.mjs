import { pathToFileURL } from "url";
import path from "path";
import fs from "fs/promises";

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
console.log("Dang khoi dong trinh duyet...");

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

  console.log("Dang mo trang NotebookLM...");
  const loginUrl = "https://notebooklm.google.com/notebook/dc714098-32b3-4287-8ccb-6ec5f242adad";
  await page.goto(loginUrl, { timeout: 60000 }).catch(() => {});

  console.log("====================================================");
  console.log("👉 VUI LONG DANG NHAP GOOGLE TREN CUA SO TRINH DUYET");
  console.log("👉 Khi dang nhap thanh cong, trinh duyet se TU DONG LUU VA DONG!");
  console.log("   (Hoac ban co the go ENTER tai day bat cu luc nao de luu)");
  console.log("====================================================");

  let forceSave = false;
  try {
    process.stdin.resume();
    process.stdin.on("data", () => {
      console.log("\n⚡ Nhan tin hieu ENTER! Dang luu phien ngay...");
      forceSave = true;
    });
  } catch (e) {}

  const maxAttempts = 300; // 5 minutes
  let saved = false;

  for (let i = 0; i < maxAttempts; i++) {
    const allPages = context.pages();
    const cookies = await context.cookies().catch(() => []);
    const hasSid = cookies.some(c => 
      c.name === "SID" || 
      c.name === "__Secure-1PSID" || 
      c.name === "__Secure-3PSID" || 
      c.name === "__Secure-OSID" ||
      c.name === "HSID" ||
      c.name === "SSID"
    );

    let activeNotebookPage = null;
    let anyNotebookUrl = "";

    for (const p of allPages) {
      try {
        const u = p.url();
        const t = (await p.title().catch(() => "")).toLowerCase();
        const isNb = u.includes("notebooklm.google") || 
                     u.includes("notebook.google") || 
                     u.includes("/notebook/") ||
                     t.includes("notebooklm");
        const isNotAuth = !u.includes("accounts.google.com") && !u.includes("/trynow");

        if (isNb && isNotAuth) {
          activeNotebookPage = p;
          anyNotebookUrl = u;
          break;
        } else if (hasSid && isNotAuth && !u.startsWith("chrome://") && !u.startsWith("about:")) {
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
          div.innerHTML = "👉 <b>BAM VAO DAY DE LUU PHIEN VA DONG TRINH DUYET</b> 👈";
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

    const isSuccess = forceSave || clickedBanner || (hasSid && activeNotebookPage !== null);

    if (isSuccess) {
      const targetPage = activeNotebookPage || (allPages.length > 0 ? allPages[0] : page);
      const title = await targetPage.title().catch(() => "");
      console.log(`\n🎉 Da phat hien dang nhap thanh cong! (URL: ${anyNotebookUrl || targetPage.url()}, Title: ${title})`);
      console.log("Dang luu trang thai xac thuc va cookies vao state.json...");

      await sleep(2000);
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
      } catch (err) {}

      // Update file timestamp
      try {
        const now = new Date();
        await fs.utimes(stateFilePath, now, now);
      } catch (err) {}

      console.log("====================================================");
      console.log("✅ LUU THANH CONG: state.json va session.json!");
      console.log(`Tong so cookies da luu: ${cookies.length}`);
      console.log("Trinh duyet se tu dong dong ngay bay gio...");
      console.log("====================================================");
      saved = true;
      break;
    }

    if (i % 5 === 0 && i > 0) {
      const currentUrl = page ? page.url() : "";
      if (currentUrl.includes("accounts.google.com")) {
        console.log(`[${i}s] Dang cho dang nhap Google... (URL: accounts.google.com)`);
      } else {
        console.log(`[${i}s] Dang cho... (URL: ${currentUrl.slice(0, 50)}) (Cookies: ${cookies.length}, hasSid: ${hasSid})`);
      }
    }
    await sleep(1000);
  }

  await sleep(1500);
  await context.close().catch(() => {});
  if (saved) {
    console.log("Trinh duyet da dong thanh cong. San sang su dung MCP!");
  } else {
    console.log("Het thoi gian cho dang nhap.");
  }
} catch (error) {
  console.error("Loi:", error);
}

process.exit(0);
