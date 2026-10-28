import { pathToFileURL } from "url";
import path from "path";

process.env.BROWSER_CHANNEL = "chromium";
process.env.NOTEBOOKLM_BROWSER_CHANNEL = "chromium";

const appData = process.env.APPDATA || "C:\\Users\\ACER\\AppData\\Roaming";
const basePath = path.join(appData, "npm", "node_modules", "notebooklm-mcp", "dist");

const { AuthManager } = await import(pathToFileURL(path.join(basePath, "auth", "auth-manager.js")).href);
const { SessionManager } = await import(pathToFileURL(path.join(basePath, "session", "session-manager.js")).href);
const { NotebookLibrary } = await import(pathToFileURL(path.join(basePath, "library", "notebook-library.js")).href);
const { ToolHandlers } = await import(pathToFileURL(path.join(basePath, "tools", "index.js")).href);

console.log("====================================================");
console.log("  TRUY VẤN LỊCH TRÌNH TỪ NOTEBOOKLM");
console.log("====================================================");
console.log("Đang khởi tạo kết nối với NotebookLM...");

const authManager = new AuthManager();
const sessionManager = new SessionManager(authManager);
const library = new NotebookLibrary();
const toolHandlers = new ToolHandlers(sessionManager, authManager, library);

try {
  console.log("Đang gửi câu hỏi về lịch trình làm việc đến NotebookLM...");
  const result = await toolHandlers.handleAskQuestion({
    notebook_id: "nghi-n-c-u-ph-t-hi-n-v-n-b-n-l",
    question: "Hãy đọc và tóm tắt chi tiết toàn bộ thông tin về lịch trình làm việc, các mốc thời gian (milestones), các giai đoạn thực hiện đề tài và nhiệm vụ hoặc sản phẩm cụ thể được đề cập trong các tài liệu nguồn của notebook này.",
    source_format: "footnotes"
  }, (msg) => {
    console.log(`[Tiến trình] ${msg}`);
  });

  console.log("\n====================================================");
  console.log("KẾT QUẢ TỪ NOTEBOOKLM:");
  console.log("====================================================");
  if (result.answer) {
    console.log(result.answer);
  } else {
    console.log(JSON.stringify(result, null, 2));
  }
} catch (error) {
  console.error("Lỗi truy vấn:", error);
} finally {
  await toolHandlers.cleanup();
  process.exit(0);
}
