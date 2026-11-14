/**
 * AI Text Detector - Content Script (Manifest v3)
 * Bắt sự kiện bôi đen (selection) trên mọi trang web, hiển thị Floating Button và Tooltip Modal
 */

const API_BASE_URL = "http://localhost:8000";

let floatingBtn = null;
let tooltipModal = null;
let lastSelectedText = "";
let lastSelectionRange = null;

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
      triggerDetection(lastSelectedText);
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
  // Nếu nhấp vào modal thì không ẩn
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
    // Chỉ ẩn modal nếu click ra ngoài hoàn toàn
    if (tooltipModal && !tooltipModal.contains(e.target)) {
      tooltipModal.classList.add("hidden");
    }
  }
});

// Lắng nghe từ context menu chuột phải (background.js)
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "ANALYZE_SELECTION" && request.text) {
    lastSelectedText = request.text;
    triggerDetection(request.text);
  }
});

// ==============================================================================
// 3. GỌI API FASTAPI VÀ HIỂN THỊ KẾT QUẢ
// ==============================================================================

async function triggerDetection(text) {
  if (floatingBtn) floatingBtn.classList.add("hidden");

  const modal = getOrCreateTooltipModal();
  positionModalNearSelection();

  modal.innerHTML = `
    <div class="ai-detect-modal-header">
      <div class="ai-detect-brand">
        <span class="ai-detect-icon">🛡️</span>
        <strong>AI Text Detector (Ensemble PLMs)</strong>
      </div>
      <button class="ai-detect-close-btn" id="ai-detect-close">&times;</button>
    </div>
    <div class="ai-detect-loading">
      <div class="ai-detect-spinner"></div>
      <span>Đang phân tích xác suất qua BERT, RoBERTa, Longformer...</span>
    </div>
  `;
  modal.classList.remove("hidden");

  document.getElementById("ai-detect-close").addEventListener("click", () => {
    modal.classList.add("hidden");
  });

  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/detect/sentences`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text, threshold: 0.5 }),
    });

    if (!res.ok) {
      throw new Error("Không thể kết nối FastAPI server. Hãy chắc chắn 'start_server.bat' đang chạy!");
    }

    const data = await res.json();
    renderModalResults(data);
  } catch (err) {
    modal.innerHTML = `
      <div class="ai-detect-modal-header">
        <strong>⚠️ Lỗi Phân Tích</strong>
        <button class="ai-detect-close-btn" id="ai-detect-close">&times;</button>
      </div>
      <div class="ai-detect-error-body">
        <p>${err.message}</p>
        <small>Kiểm tra Backend FastAPI tại: <code>http://localhost:8000</code></small>
      </div>
    `;
    document.getElementById("ai-detect-close").addEventListener("click", () => {
      modal.classList.add("hidden");
    });
  }
}

function positionModalNearSelection() {
  const modal = getOrCreateTooltipModal();
  if (lastSelectionRange) {
    const rect = lastSelectionRange.getBoundingClientRect();
    modal.style.top = `${window.scrollY + rect.bottom + 8}px`;
    modal.style.left = `${Math.max(10, window.scrollX + rect.left)}px`;
  } else {
    modal.style.top = `${window.scrollY + 50}px`;
    modal.style.left = "50px";
  }
}

function renderModalResults(data) {
  const modal = getOrCreateTooltipModal();
  const isAI = data.overall.label === "AI-generated";
  const prob = (data.overall.ai_probability * 100).toFixed(1);
  const colorClass = isAI ? "verdict-ai" : "verdict-human";
  const badgeText = isAI ? "🤖 DO AI SINH (ChatGPT/LLM)" : "👤 DO NGƯỜI VIẾT (HUMAN)";

  let sentencesHtml = "";
  if (data.sentences && data.sentences.length > 0) {
    sentencesHtml = data.sentences.map((s) => {
      return `
        <div class="ai-detect-sentence-row ${s.highlight_color}">
          <span class="ai-detect-tag">[${s.ai_percentage} AI]</span>
          <span>${s.sentence}</span>
        </div>
      `;
    }).join("");
  }

  modal.innerHTML = `
    <div class="ai-detect-modal-header">
      <div class="ai-detect-brand">
        <span class="ai-detect-icon">🛡️</span>
        <strong>Kết Quả Phân Tích AI</strong>
      </div>
      <button class="ai-detect-close-btn" id="ai-detect-close">&times;</button>
    </div>

    <div class="ai-detect-modal-body">
      <!-- Verdict Bar -->
      <div class="ai-detect-verdict-bar ${colorClass}">
        <span class="verdict-title">${badgeText}</span>
        <span class="verdict-prob">${prob}% AI</span>
      </div>

      <div class="ai-detect-meta">
        <span>Độ tin cậy: <b>${data.overall.confidence}</b></span>
        <span>Thời gian: <b>${data.total_processing_time_ms} ms</b></span>
      </div>

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
    </div>
  `;

  document.getElementById("ai-detect-close").addEventListener("click", () => {
    modal.classList.add("hidden");
  });
}
