/**
 * AI Text Detector - Background Service Worker (Manifest v3)
 * Xử lý Context Menu chuột phải và điều phối thông điệp.
 */

chrome.runtime.onInstalled.addListener(() => {
  // Tạo menu chuột phải khi người dùng bôi đen văn bản
  chrome.contextMenus.create({
    id: "ai_detect_selection",
    title: "🔍 Quét văn bản AI (AI Text Detector)",
    contexts: ["selection"],
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "ai_detect_selection" && info.selectionText) {
    if (tab && tab.id) {
      chrome.tabs.sendMessage(tab.id, {
        action: "ANALYZE_SELECTION",
        text: info.selectionText,
      });
    }
  }
});
