/**
 * VeriText AI PRO - Content Script (Manifest v3)
 * - Bắt sự kiện bôi đen (selection) trên mọi trang web -> Floating Button & Tooltip Modal
 * - Hỗ trợ Quét Toàn Trang Web (LMS / Google Classroom / Blog) từ Context Menu chuột phải
 * - Tự động đính kèm JWT Token từ chrome.storage.local để đồng bộ lịch sử SQL Server
 * - Hiển thị cảnh báo Anti-Bypass, Phân loại 4 cấp độ & Bằng chứng XAI rút gọn
 */

const API_BASE_URL = "http://localhost:8000";

let floatingBtn = null;
let tooltipModal = null;
let lastSelectedText = "";
let lastSelectionRange = null;

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = String(str ?? "");
  return div.innerHTML;
}

function getExtensionSettings() {
  return new Promise((resolve) => {
    if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
      chrome.storage.local.get(
        ["auth_token", "auth_user", "threshold", "filter_quotes", "filter_references"],
        (res) => resolve(res || {})
      );
    } else {
      resolve({});
    }
  });
}

function extractMainPageText() {
  const MAX_CHARS = 12000;
  const candidateSelectors = [
    ".submissionstatustable",
    ".assignsubmission_onlinetext",
    "#region-main",
    ".kix-appview-editor",
    "article",
    "main",
    "[role='main']",
    ".post-content",
    ".entry-content",
    ".article-content",
  ];

  let rootEl = null;
  for (const sel of candidateSelectors) {
    const el = document.querySelector(sel);
    if (el && (el.innerText || "").trim().length >= 80) {
      rootEl = el;
      break;
    }
  }
  if (!rootEl) rootEl = document.body;

  const clone = rootEl.cloneNode(true);
  [
    "script",
    "style",
    "noscript",
    "svg",
    "iframe",
    "nav",
    "header",
    "footer",
    "aside",
    "#ai-detect-floating-btn",
    "#ai-detect-tooltip-modal",
  ].forEach((sel) => {
    clone.querySelectorAll(sel).forEach((node) => node.remove());
  });

  const rawText = (clone.innerText || clone.textContent || "")
    .replace(/\r\n/g, "\n")
    .replace(/[ \t]{2,}/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();

  return rawText.length > MAX_CHARS ? rawText.slice(0, MAX_CHARS) : rawText;
}

// ==============================================================================
// 1. TẠO VÀ QUẢN LÝ CÁC PHẦN TỬ GIAO DIỆN NỔI (FLOATING UI)
// ==============================================================================

function getOrCreateFloatingBtn() {
  if (!floatingBtn) {
    floatingBtn = document.createElement("div");
    floatingBtn.id = "ai-detect-floating-btn";
    floatingBtn.className = "ai-detect-floating-btn hidden";
    floatingBtn.innerHTML = `<span>🤖 Quét AI</span>`;
    document.body.appendChild(floatingBtn);

    floatingBtn.addEventListener("mousedown", (e) => {
      e.preventDefault(); // Ngăn mất vùng bôi đen
      e.stopPropagation();
      triggerDetection(lastSelectedText, {
        mode: "selection",
        pageTitle: document.title || window.location.hostname,
      });
    });
  }
  return floatingBtn;
}

function getOrCreateTooltipModal() {
  if (!tooltipModal) {
    tooltipModal = document.createElement("div");
    tooltipModal.id = "ai-detect-tooltip-modal";
    tooltipModal.className = "ai-detect-tooltip-modal hidden";
    document.body.appendChild(tooltipModal);
  }
  return tooltipModal;
}

// ==============================================================================
// 2. BẮT SỰ KIỆN CHUỘT KHI BÔI ĐEN VĂN BẢN
// ==============================================================================

document.addEventListener("mouseup", (e) => {
  if (tooltipModal && tooltipModal.contains(e.target)) return;
  if (floatingBtn && floatingBtn.contains(e.target)) return;

  const selection = window.getSelection();
  const selectedText = selection ? selection.toString().trim() : "";

  if (selectedText && selectedText.length >= 15) {
    lastSelectedText = selectedText;
    if (selection.rangeCount > 0) {
      lastSelectionRange = selection.getRangeAt(0).cloneRange();
    }

    const rect = selection.getRangeAt(0).getBoundingClientRect();
    const btn = getOrCreateFloatingBtn();
    btn.style.top = `${window.scrollY + rect.bottom + 6}px`;
    btn.style.left = `${window.scrollX + rect.right - 70}px`;
    btn.classList.remove("hidden");
  } else {
    if (floatingBtn) floatingBtn.classList.add("hidden");
    if (tooltipModal && !tooltipModal.contains(e.target)) {
      tooltipModal.classList.add("hidden");
    }
  }
});

// Lắng nghe từ Context Menu chuột phải (background.js)
chrome.runtime.onMessage.addListener((request) => {
  if (request.action === "ANALYZE_SELECTION" && request.text) {
    lastSelectedText = request.text;
    triggerDetection(request.text, {
      mode: "selection",
      pageTitle: request.pageTitle || document.title || window.location.hostname,
    });
  } else if (request.action === "ANALYZE_FULL_PAGE") {
    lastSelectionRange = null;
    const pageText = extractMainPageText();
    if (!pageText || pageText.length < 20) {
      showErrorModal("Trang web hiện tại không có đủ văn bản để phân tích (tối thiểu 20 ký tự).");
      return;
    }
    triggerDetection(pageText, {
      mode: "page",
      pageTitle: request.pageTitle || document.title || window.location.hostname,
    });
  }
});

// ==============================================================================
// 3. GỌI API FASTAPI (CÓ JWT SYNC) VÀ HIỂN THỊ KẾT QUẢ ANTI-BYPASS + XAI
// ==============================================================================

function showErrorModal(message) {
  const modal = getOrCreateTooltipModal();
  positionModalNearSelection(true);
  modal.innerHTML = `
    <div class="ai-detect-modal-header">
      <strong>⚠️ Lỗi Phân Tích</strong>
      <button class="ai-detect-close-btn" id="ai-detect-close">&times;</button>
    </div>
    <div class="ai-detect-error-body">
      <p>${escapeHtml(message)}</p>
      <small>Kiểm tra Backend FastAPI tại: <code>http://localhost:8000</code></small>
    </div>
  `;
  modal.classList.remove("hidden");
  document.getElementById("ai-detect-close").addEventListener("click", () => {
    modal.classList.add("hidden");
  });
}

async function triggerDetection(text, options = {}) {
  if (floatingBtn) floatingBtn.classList.add("hidden");

  const isFullPage = options.mode === "page";
  const pageTitle = (options.pageTitle || document.title || "Web").trim();
  const modal = getOrCreateTooltipModal();
  positionModalNearSelection(isFullPage);

  modal.innerHTML = `
    <div class="ai-detect-modal-header">
      <div class="ai-detect-brand">
        <span class="ai-detect-icon">🛡️</span>
        <strong>VeriText AI PRO ${isFullPage ? "• Quét Toàn Trang" : ""}</strong>
      </div>
      <button class="ai-detect-close-btn" id="ai-detect-close">&times;</button>
    </div>
    <div class="ai-detect-loading">
      <div class="ai-detect-spinner"></div>
      <span>Đang phân tích văn bản qua PhoBERT / Ensemble + Anti-Bypass + XAI...</span>
    </div>
  `;
  modal.classList.remove("hidden");

  document.getElementById("ai-detect-close").addEventListener("click", () => {
    modal.classList.add("hidden");
  });

  try {
    const settings = await getExtensionSettings();
    const threshold = typeof settings.threshold === "number" ? settings.threshold : 0.5;
    const headers = { "Content-Type": "application/json" };
    if (settings.auth_token) {
      headers["Authorization"] = `Bearer ${settings.auth_token}`;
    }

    const scanTitle = isFullPage
      ? `[LMS/Web] ${pageTitle}`.slice(0, 120)
      : `[Bôi đen] ${pageTitle}`.slice(0, 120);

    const res = await fetch(`${API_BASE_URL}/api/v1/detect/sentences`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        text: text,
        threshold: threshold,
        title: scanTitle,
        filter_quotes: Boolean(settings.filter_quotes),
        filter_references: Boolean(settings.filter_references),
      }),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(
        errData.detail || "Không thể kết nối FastAPI server. Hãy chắc chắn 'start_server.bat' đang chạy!"
      );
    }

    const data = await res.json();
    renderModalResults(data, settings.auth_user || null, Boolean(settings.auth_token));
  } catch (err) {
    showErrorModal(err.message || "Không thể phân tích văn bản.");
  }
}

function positionModalNearSelection(centerTop = false) {
  const modal = getOrCreateTooltipModal();
  if (!centerTop && lastSelectionRange) {
    const rect = lastSelectionRange.getBoundingClientRect();
    modal.style.top = `${window.scrollY + rect.bottom + 8}px`;
    modal.style.left = `${Math.max(12, Math.min(window.innerWidth - 400, window.scrollX + rect.left))}px`;
  } else {
    modal.style.top = `${window.scrollY + 24}px`;
    modal.style.left = `${Math.max(12, window.scrollX + window.innerWidth - 415)}px`;
  }
}

function renderModalResults(data, authUser, isSynced) {
  const modal = getOrCreateTooltipModal();
  const isAI = data.overall.label === "AI-generated";
  const prob = (data.overall.ai_probability * 100).toFixed(1);
  const colorClass = isAI ? "verdict-ai" : "verdict-human";
  const badgeText = isAI ? "🤖 DO AI SINH (ChatGPT/LLM)" : "👤 DO NGƯỜI VIẾT (HUMAN)";

  // Anti-Bypass Warning Banner
  const bypass = data.bypass_report;
  const bypassHtml =
    bypass && bypass.has_bypass_warning
      ? `<div class="ai-detect-bypass-alert">
           <strong>🚨 CẢNH BÁO ANTI-BYPASS:</strong>
           <span>${escapeHtml(bypass.summary)}</span>
         </div>`
      : "";

  // 4-Tier Authorship Badge
  const tier = data.authorship_tier;
  const tierHtml =
    tier && tier.label_vi
      ? `<div class="ai-detect-tier ${escapeHtml(tier.badge_class || "")}" title="${escapeHtml(tier.reason || "")}">
           ${escapeHtml(tier.label_vi)}
         </div>`
      : "";

  // Compact XAI Strip
  const xai = data.xai_metrics;
  let xaiHtml = "";
  if (xai && typeof xai.burstiness_score === "number") {
    const clichesHtml =
      xai.ai_cliches && xai.ai_cliches.length > 0
        ? `<div class="ai-detect-cliches">⚠️ Văn mẫu AI (${escapeHtml(xai.ai_cliche_count)}): ${xai.ai_cliches
            .slice(0, 3)
            .map((c) => `"${escapeHtml(c.phrase)}"`)
            .join(", ")}</div>`
        : "";

    xaiHtml = `
      <div class="ai-detect-xai-box">
        <div class="ai-detect-xai-grid">
          <div class="ai-detect-xai-cell"><span>Burstiness</span><b>${escapeHtml(xai.burstiness_score)}/100</b></div>
          <div class="ai-detect-xai-cell"><span>Từ vựng</span><b>${escapeHtml(xai.lexical_diversity)}%</b></div>
          <div class="ai-detect-xai-cell"><span>Perplexity</span><b>${escapeHtml(xai.perplexity_score)}/100</b></div>
        </div>
        ${clichesHtml}
      </div>
    `;
  }

  // Sync status chip
  const syncHtml =
    isSynced && authUser
      ? `<div class="ai-detect-sync-chip synced">✓ Đã đồng bộ SQL Server (#${escapeHtml(data.scan_id || "")}) • ${escapeHtml(authUser.full_name || authUser.email)}</div>`
      : `<div class="ai-detect-sync-chip guest">ℹ️ Chế độ khách • Đăng nhập ở Popup Extension để lưu lịch sử</div>`;

  let sentencesHtml = "";
  if (data.sentences && data.sentences.length > 0) {
    sentencesHtml = data.sentences
      .map((s) => {
        return `
          <div class="ai-detect-sentence-row ${escapeHtml(s.highlight_color)}">
            <span class="ai-detect-tag">[${escapeHtml(s.ai_percentage)} AI]</span>
            <span>${escapeHtml(s.sentence)}</span>
          </div>
        `;
      })
      .join("");
  }

  const langTag = data.overall.language === "vi" ? "🇻🇳 Tiếng Việt (PhoBERT)" : "🇬🇧 Tiếng Anh (Ensemble)";

  modal.innerHTML = `
    <div class="ai-detect-modal-header">
      <div class="ai-detect-brand">
        <span class="ai-detect-icon">🛡️</span>
        <strong>VeriText AI • ${langTag}</strong>
      </div>
      <button class="ai-detect-close-btn" id="ai-detect-close">&times;</button>
    </div>

    <div class="ai-detect-modal-body">
      ${bypassHtml}

      <!-- Verdict Bar -->
      <div class="ai-detect-verdict-bar ${colorClass}">
        <span class="verdict-title">${badgeText}</span>
        <span class="verdict-prob">${prob}% AI</span>
      </div>

      ${tierHtml}

      <div class="ai-detect-meta">
        <span>Độ tin cậy: <b>${escapeHtml(data.overall.confidence)}</b></span>
        <span>Thời gian: <b>${escapeHtml(data.total_processing_time_ms)} ms</b></span>
      </div>

      ${xaiHtml}

      <!-- Hotspots -->
      <div class="ai-detect-hotspots-box">
        <div class="hotspots-header">
          <strong>📑 Bóc tách chi tiết từng câu:</strong>
          <small>Đỏ: AI • Vàng: Nghi vấn • Xanh: Người</small>
        </div>
        <div class="ai-detect-sentences-scroll">
          ${sentencesHtml}
        </div>
      </div>

      ${syncHtml}
    </div>
  `;

  document.getElementById("ai-detect-close").addEventListener("click", () => {
    modal.classList.add("hidden");
  });
}
