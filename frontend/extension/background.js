/**
 * VeriText AI PRO - Background Service Worker (Manifest v3)
 * Xử lý Context Menu chuột phải (bôi đen văn bản & quét toàn trang LMS) và điều phối thông điệp.
 */

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.removeAll(() => {
    // 1. Menu chuột phải khi bôi đen văn bản
    chrome.contextMenus.create({
      id: "ai_detect_selection",
      title: "🔍 Quét đoạn văn bản bôi đen (VeriText AI)",
      contexts: ["selection"],
    });

    // 2. Menu chuột phải quét toàn trang LMS / Google Classroom / Blog
    chrome.contextMenus.create({
      id: "ai_detect_full_page",
      title: "🌐 Quét toàn trang web hiện tại (VeriText AI)",
      contexts: ["page"],
    });
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (!tab || !tab.id) return;

  if (info.menuItemId === "ai_detect_selection" && info.selectionText) {
    chrome.tabs.sendMessage(
      tab.id,
      {
        action: "ANALYZE_SELECTION",
        text: info.selectionText,
        pageTitle: tab.title || "",
      },
      () => {
        void chrome.runtime.lastError;
      }
    );
  } else if (info.menuItemId === "ai_detect_full_page") {
    chrome.tabs.sendMessage(
      tab.id,
      {
        action: "ANALYZE_FULL_PAGE",
        pageTitle: tab.title || "",
      },
      () => {
        void chrome.runtime.lastError;
      }
    );
  }
});
