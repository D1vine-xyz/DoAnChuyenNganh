import { pathToFileURL } from "url";
const prUrl = pathToFileURL("C:/Users/ACER/AppData/Roaming/npm/node_modules/notebooklm-mcp/node_modules/patchright/index.mjs").href;
const { chromium } = await import(prUrl);
import path from "path";
import fs from "fs/promises";
import { existsSync } from "fs";

const dataDir = "C:\\Users\\ACER\\AppData\\Local\\notebooklm-mcp\\Data";
const chromeProfileDir = path.join(dataDir, "chrome_profile");
const browserStateDir = path.join(dataDir, "browser_state");
const stateFilePath = path.join(browserStateDir, "state.json");
const sessionFilePath = path.join(browserStateDir, "session.json");

console.log("====================================================");
console.log("  NOTEBOOKLM MCP - SMART LOGIN & STATE SAVER");
console.log("====================================================");
console.log("Đang khởi động trình duyệt với hồ sơ hiện tại...");

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

  console.log("Đang mở Google / NotebookLM...");
  await page.goto("https://notebooklm.google.com/", { timeout: 60000 });

  console.log("Đang kiểm tra trạng thái đăng nhập...");
  const maxAttempts = 300; // 5 minutes
  let saved = false;

  for (let i = 0; i < maxAttempts; i++) {
    const url = page.url();
    if (url.includes("notebooklm.google.com") || url.includes("notebook.google.com")) {
      // Check if page loaded
      const title = await page.title();
      console.log(`Đã phát hiện NotebookLM! (URL: ${url}, Title: ${title})`);
      console.log("Đang lưu trạng thái xác thực và cookies...");

      await page.waitForTimeout(3000);
      await context.storageState({ path: stateFilePath });

      try {
        const sessionStorageData = await page.evaluate(() => {
          const storage = {};
          for (let k = 0; k < sessionStorage.length; k++) {
            const key = sessionStorage.key(k);
            if (key) storage[key] = sessionStorage.getItem(key) || "";
          }
          return JSON.stringify(storage);
        });
        await fs.writeFile(sessionFilePath, sessionStorageData, { encoding: "utf-8" });
      } catch (err) { }

      console.log("====================================================");
      console.log("✅ LƯU THÀNH CÔNG: state.json và session.json!");
      console.log(`File: ${stateFilePath}`);
      console.log("====================================================");
      saved = true;
      break;
    }

    if (i % 10 === 0 && i > 0) {
      console.log(`Đang chờ đăng nhập... (${i}s elapsed, URL: ${url.slice(0, 60)})`);
    }
    await page.waitForTimeout(1000);
  }

  await context.close();
  if (saved) {
    console.log("Trình duyệt đã đóng an toàn. Bạn có thể quay lại IDE!");
  } else {
    console.log("Hết thời gian chờ đăng nhập.");
  }
} catch (error) {
  console.error("Lỗi:", error);
}

process.exit(0);
