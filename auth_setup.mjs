import { pathToFileURL } from 'url';
import path from 'path';

process.env.NOTEBOOKLM_BROWSER_CHANNEL = "chromium";

const appData = process.env.APPDATA || "C:\\Users\\ACER\\AppData\\Roaming";
const authManagerPath = path.join(appData, "npm", "node_modules", "notebooklm-mcp", "dist", "auth", "auth-manager.js");
const modUrl = pathToFileURL(authManagerPath).href;
const { AuthManager } = await import(modUrl);

const auth = new AuthManager();

console.log("====================================================");
console.log("  NOTEBOOKLM MCP - GOOGLE AUTHENTICATION");
console.log("====================================================");
console.log("Trình duyệt đăng nhập sẽ mở ngay bây giờ.");
console.log("Vui lòng đăng nhập tài khoản Google của bạn trên cửa sổ mở ra.");
console.log("Khi đăng nhập thành công và vào NotebookLM, cửa sổ sẽ tự động đóng và lưu phiên đăng nhập.\n");

try {
  const success = await auth.performSetup((msg, current, total) => {
    console.log(`[Tiến trình ${current}/${total}] ${msg}`);
  }, true);

  if (success) {
    console.log("\n====================================================");
    console.log("✅ ĐĂNG NHẬP THÀNH CÔNG VÀ ĐÃ LƯU COOKIES/SESSION!");
    console.log("====================================================");
  } else {
    console.log("\n❌ Đăng nhập chưa hoàn tất hoặc bị hủy.");
  }
} catch (err) {
  console.error("Lỗi xác thực:", err);
}

console.log("\nNhấn phím bất kỳ để thoát...");
if (process.stdin.isTTY) {
  process.stdin.setRawMode(true);
  process.stdin.resume();
  process.stdin.on('data', () => process.exit(0));
} else {
  process.exit(0);
}
