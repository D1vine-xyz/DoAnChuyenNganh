/**
 * VeriText AI - Web Dashboard Single Page Application Logic
 * Đồ án Chuyên ngành Kỹ thuật Phần mềm (Chủ đề 6)
 */

const API_BASE = "";

// Global State
let currentScanId = null;
let currentUploadedFile = null;

// DOM Elements - Navigation
const navTabs = document.querySelectorAll(".nav-tab");
const tabPanes = document.querySelectorAll(".tab-pane");
const serverStatusEl = document.getElementById("server-status");
const serverStatusTextEl = document.getElementById("server-status-text");

// DOM Elements - Inspector
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const fileInfoBar = document.getElementById("file-info");
const fileNameEl = document.getElementById("file-name");
const fileSizeEl = document.getElementById("file-size");
const btnRemoveFile = document.getElementById("btn-remove-file");

const textInput = document.getElementById("text-input");
const statWords = document.getElementById("stat-words");
const statChars = document.getElementById("stat-chars");
const statSentences = document.getElementById("stat-sentences");
const thresholdRange = document.getElementById("threshold-range");
const thresholdVal = document.getElementById("threshold-val");

const btnSubmit = document.getElementById("btn-submit");
const spinner = document.getElementById("spinner");
const btnUpload = document.getElementById("btn-upload");
const btnPaste = document.getElementById("btn-paste");
const btnClear = document.getElementById("btn-clear");
const btnSampleAi = document.getElementById("btn-sample-ai");
const btnSampleHuman = document.getElementById("btn-sample-human");
const btnExportPdf = document.getElementById("btn-export-pdf");
const resultActions = document.getElementById("result-actions");

// DOM Elements - Results
const resultPlaceholder = document.getElementById("result-placeholder");
const resultContent = document.getElementById("result-content");
const verdictBanner = document.getElementById("verdict-banner");
const verdictIcon = document.getElementById("verdict-icon");
const verdictTitle = document.getElementById("verdict-title");
const verdictSubtitle = document.getElementById("verdict-subtitle");
const verdictPercent = document.getElementById("verdict-percent");

const metricHumanProb = document.getElementById("metric-human-prob");
const metricConfidence = document.getElementById("metric-confidence");
const metricLatency = document.getElementById("metric-latency");
const metricCoverage = document.getElementById("metric-coverage");

const barLongformer = document.getElementById("bar-longformer");
const probLongformer = document.getElementById("prob-longformer");
const barBert = document.getElementById("bar-bert");
const probBert = document.getElementById("prob-bert");
const barRoberta = document.getElementById("bar-roberta");
const probRoberta = document.getElementById("prob-roberta");
const sentencesContainer = document.getElementById("sentences-container");

// DOM Elements - History
const kpiTotalScans = document.getElementById("kpi-total-scans");
const kpiAiRate = document.getElementById("kpi-ai-rate");
const kpiTotalWords = document.getElementById("kpi-total-words");
const kpiAvgLatency = document.getElementById("kpi-avg-latency");
const historyTableBody = document.getElementById("history-table-body");
const btnRefreshHistory = document.getElementById("btn-refresh-history");


// ==============================================================================
// 1. KHỞI TẠO & ĐIỀU HƯỚNG TABS
// ==============================================================================

document.addEventListener("DOMContentLoaded", () => {
  // Theme Toggle Sync
  const btnThemeToggle = document.getElementById("btn-theme-toggle");
  const themeToggleIcon = document.getElementById("theme-toggle-icon");

  function updateThemeIcon(theme) {
    if (themeToggleIcon) {
      themeToggleIcon.textContent = theme === "dark" ? "☀️" : "🌙";
    }
  }

  const initialTheme = document.documentElement.getAttribute("data-theme") || "light";
  updateThemeIcon(initialTheme);

  if (btnThemeToggle) {
    btnThemeToggle.addEventListener("click", () => {
      const activeTheme = document.documentElement.getAttribute("data-theme") || "light";
      const nextTheme = activeTheme === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", nextTheme);
      try {
        localStorage.setItem("veritext_theme", nextTheme);
      } catch (e) {}
      updateThemeIcon(nextTheme);
    });
  }

  checkServerHealth();
  updateTextCounts();

  // Tab switching
  navTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const targetId = tab.dataset.tab;
      navTabs.forEach((t) => t.classList.remove("active"));
      tabPanes.forEach((p) => p.classList.remove("active"));

      tab.classList.add("active");
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add("active");

      if (targetId === "tab-history") {
        loadHistoryAndStats();
      }
    });
  });
});

async function checkServerHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      const data = await res.json();
      serverStatusEl.className = "status-indicator " + (data.ready ? "status-online" : "status-offline");
      serverStatusTextEl.textContent = data.ready ? `FastAPI: Sẵn sàng (${data.device})` : "Đang nạp mô hình...";
    } else {
      serverStatusEl.className = "status-indicator status-offline";
      serverStatusTextEl.textContent = "FastAPI: Lỗi phản hồi";
    }
  } catch (err) {
    serverStatusEl.className = "status-indicator status-offline";
    serverStatusTextEl.textContent = "FastAPI: Chưa kết nối";
  }
}


// ==============================================================================
// 2. XỬ LÝ NHẬP LIỆU & FILE DRAG & DROP
// ==============================================================================

textInput.addEventListener("input", updateTextCounts);

function updateTextCounts() {
  const text = textInput.value.trim();
  const words = text ? text.split(/\s+/).length : 0;
  const chars = text.length;
  const sents = text ? (text.match(/[^.!?]+[.!?]+/g) || [text]).length : 0;

  statWords.innerHTML = `<b>${words}</b> từ`;
  statChars.innerHTML = `<b>${chars}</b> ký tự`;
  statSentences.innerHTML = `<b>${sents}</b> câu`;
}

thresholdRange.addEventListener("input", () => {
  thresholdVal.textContent = `${thresholdRange.value}%`;
});

// File upload button
if (btnUpload) {
  btnUpload.addEventListener("click", () => {
    fileInput.click();
  });
}

// Drag and drop events on input card
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
    handleFileSelected(e.dataTransfer.files[0]);
  }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files && fileInput.files.length > 0) {
    handleFileSelected(fileInput.files[0]);
  }
});

function handleFileSelected(file) {
  const ext = file.name.split(".").pop().toLowerCase();
  if (!["txt", "docx", "doc", "pdf"].includes(ext)) {
    alert("Chỉ hỗ trợ tệp tin định dạng: .PDF, .DOCX, .DOC, hoặc .TXT");
    return;
  }
  currentUploadedFile = file;
  fileNameEl.textContent = file.name;
  fileSizeEl.textContent = `(${(file.size / 1024).toFixed(1)} KB)`;
  fileInfoBar.classList.remove("hidden");
}

btnRemoveFile.addEventListener("click", (e) => {
  e.stopPropagation();
  currentUploadedFile = null;
  fileInput.value = "";
  fileInfoBar.classList.add("hidden");
});

// Action buttons
btnPaste.addEventListener("click", async () => {
  try {
    const text = await navigator.clipboard.readText();
    textInput.value = text;
    updateTextCounts();
  } catch (err) {
    alert("Không thể đọc Clipboard. Vui lòng nhấn Ctrl + V để dán trực tiếp.");
  }
});

btnClear.addEventListener("click", () => {
  textInput.value = "";
  updateTextCounts();
  currentUploadedFile = null;
  fileInput.value = "";
  fileInfoBar.classList.add("hidden");
  resultContent.classList.add("hidden");
  resultActions.classList.add("hidden");
});

btnSampleAi.addEventListener("click", () => {
  textInput.value = (
    "Artificial intelligence models have demonstrated exceptional capabilities across various natural language processing domains. "
    + "These sophisticated architectures leverage deep contextual self-attention layers to produce coherent, syntactically fluent discourse. "
    + "However, the widespread dissemination of synthetic content introduces non-trivial risks regarding academic dishonesty and journalistic trust. "
    + "Therefore, developing resilient multi-model ensemble detection frameworks is of paramount importance to safeguard digital information fidelity."
  );
  updateTextCounts();
});

btnSampleHuman.addEventListener("click", () => {
  textInput.value = (
    "Last Sunday morning, my roommate and I decided to take a spontaneous road trip out to the coast. "
    + "We packed a couple of sandwiches, some cold drinks, and hit the highway before the morning traffic started piling up. "
    + "The weather was completely overcast with a cool breeze coming off the ocean, but the seafood shack by the pier was fantastic. "
    + "It was honestly one of the most relaxing weekends we have had all semester."
  );
  updateTextCounts();
});


// ==============================================================================
// 3. THỰC HIỆN THẨM ĐỊNH (SUBMIT ANALYSIS)
// ==============================================================================

btnSubmit.addEventListener("click", async () => {
  const threshold = parseFloat(thresholdRange.value) / 100;

  // Trường hợp 1: Có file đính kèm
  if (currentUploadedFile) {
    setLoading(true);
    const formData = new FormData();
    formData.append("file", currentUploadedFile);
    formData.append("threshold", threshold);

    try {
      const res = await fetch(`${API_BASE}/api/v1/detect/upload`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Lỗi xử lý file từ server.");
      }

      const data = await res.json();
      currentScanId = data.scan_id;
      renderResults(data);
    } catch (err) {
      alert("Lỗi thẩm định tệp: " + err.message);
    } finally {
      setLoading(false);
    }
    return;
  }

  // Trường hợp 2: Nhập văn bản trực tiếp
  const rawText = textInput.value.trim();
  if (rawText.length < 10) {
    alert("Vui lòng nhập hoặc dán ít nhất 10 ký tự tiếng Anh để hệ thống phân tích.");
    return;
  }

  setLoading(true);
  try {
    const res = await fetch(`${API_BASE}/api/v1/detect/web`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: rawText,
        threshold: threshold,
        title: rawText.slice(0, 35) + "...",
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Lỗi suy luận từ máy chủ.");
    }

    const data = await res.json();
    currentScanId = data.scan_id;
    renderResults(data);
  } catch (err) {
    alert("Lỗi thẩm định văn bản: " + err.message);
  } finally {
    setLoading(false);
  }
});

function setLoading(isLoading) {
  btnSubmit.disabled = isLoading;
  const btnText = btnSubmit.querySelector(".btn-text");
  if (isLoading) {
    btnText.textContent = "Đang kiểm tra qua 3 PLMs...";
    spinner.classList.remove("hidden");
  } else {
    btnText.textContent = "Kiểm tra nội dung AI";
    spinner.classList.add("hidden");
  }
}


// ==============================================================================
// 4. HIỂN THỊ KẾT QUẢ THẨM ĐỊNH
// ==============================================================================

function renderResults(data) {
  if (resultPlaceholder) resultPlaceholder.classList.add("hidden");
  resultContent.classList.remove("hidden");
  resultActions.classList.remove("hidden");

  const overall = data.overall || data;
  const isAI = overall.label === "AI-generated" || overall.ai_probability >= 0.5;
  const aiPct = (overall.ai_probability * 100).toFixed(1);
  const humanPct = (overall.human_probability * 100).toFixed(1);

  // 1. Verdict Banner
  verdictBanner.className = "verdict-header " + (isAI ? "verdict-ai" : "verdict-human");
  verdictIcon.textContent = isAI ? "🤖" : "👤";
  verdictTitle.textContent = isAI ? "VĂN BẢN DO AI SINH (AI-GENERATED)" : "VĂN BẢN DO CON NGƯỜI VIẾT (HUMAN-WRITTEN)";
  verdictSubtitle.textContent = isAI
    ? "Phát hiện dấu hiệu cú pháp và thống kê đặc trưng của Mô hình Ngôn ngữ Lớn (LLM)"
    : "Văn phong tự nhiên, mức độ biến thiên từ vựng cao, không phát hiện dấu vết máy sinh";
  verdictPercent.textContent = `${aiPct}%`;

  // 2. Metrics Box
  metricHumanProb.textContent = `${humanPct}%`;
  metricConfidence.textContent = overall.confidence || "Rất cao";
  metricLatency.textContent = `${data.total_processing_time_ms || overall.inference_time_ms || 120} ms`;
  if (overall.text_stats) {
    metricCoverage.textContent = overall.text_stats.context_coverage || "1024 Tokens";
  }

  // 3. Model Breakdown
  if (overall.model_breakdown && overall.model_breakdown.individual_models) {
    const m = overall.model_breakdown.individual_models;
    const pLong = (m.longformer.probability * 100).toFixed(1);
    const pBert = (m.bert.probability * 100).toFixed(1);
    const pRob = (m.roberta.probability * 100).toFixed(1);

    probLongformer.textContent = `${pLong}%`;
    barLongformer.style.width = `${pLong}%`;

    probBert.textContent = `${pBert}%`;
    barBert.style.width = `${pBert}%`;

    probRoberta.textContent = `${pRob}%`;
    barRoberta.style.width = `${pRob}%`;
  }

  // 4. Sentences Hotspots
  renderSentences(data.sentences || []);

  // Smooth scroll into view
  resultContent.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderSentences(sentences) {
  sentencesContainer.innerHTML = "";
  if (!sentences || sentences.length === 0) {
    sentencesContainer.innerHTML = "<p style='color: var(--text-muted); font-size: 12px; padding: 10px;'>Không có phân tích cấp câu.</p>";
    return;
  }

  sentences.forEach((s) => {
    const item = document.createElement("div");
    item.className = `sentence-item ${s.highlight_color}`;

    const scoreTag = document.createElement("span");
    scoreTag.className = "sentence-score-tag";
    scoreTag.textContent = `[${s.ai_percentage} AI]`;

    const text = document.createElement("span");
    text.textContent = ` ${s.sentence}`;

    item.appendChild(scoreTag);
    item.appendChild(text);
    sentencesContainer.appendChild(item);
  });
}

// Export PDF Button
btnExportPdf.addEventListener("click", () => {
  if (!currentScanId) {
    alert("Chưa có lượt quét nào để xuất báo cáo PDF.");
    return;
  }
  window.open(`${API_BASE}/api/v1/report/pdf/${currentScanId}`, "_blank");
});


// ==============================================================================
// 5. QUẢN LÝ LỊCH SỬ QUÉT & KPIS (TAB 2)
// ==============================================================================

async function loadHistoryAndStats() {
  try {
    // 1. Thống kê KPI
    const statsRes = await fetch(`${API_BASE}/api/v1/stats`);
    if (statsRes.ok) {
      const stats = await statsRes.json();
      kpiTotalScans.textContent = stats.total_scans.toLocaleString();
      kpiAiRate.textContent = `${stats.ai_percentage}%`;
      kpiTotalWords.textContent = stats.total_words_scanned.toLocaleString();
      kpiAvgLatency.textContent = `${stats.avg_inference_time_ms} ms`;
    }

    // 2. Bảng dữ liệu
    const histRes = await fetch(`${API_BASE}/api/v1/history?limit=30`);
    if (histRes.ok) {
      const data = await histRes.json();
      renderHistoryTable(data.history || []);
    }
  } catch (err) {
    console.error("Lỗi nạp lịch sử:", err);
  }
}

btnRefreshHistory.addEventListener("click", loadHistoryAndStats);

function renderHistoryTable(items) {
  historyTableBody.innerHTML = "";
  if (!items || items.length === 0) {
    historyTableBody.innerHTML = `
      <tr>
        <td colspan="8" class="text-center" style="padding: 30px; color: var(--text-muted);">
          Chưa có lượt thẩm định nào được lưu. Hãy thực hiện kiểm tra ở Tab Thẩm Định!
        </td>
      </tr>
    `;
    return;
  }

  items.forEach((item) => {
    const tr = document.createElement("tr");
    const isAi = item.verdict === "AI-generated" || item.ai_probability >= 0.5;
    const badgeClass = isAi ? "badge-ai-table" : "badge-human-table";
    const badgeText = isAi ? "🤖 AI-Generated" : "👤 Human";

    tr.innerHTML = `
      <td><b>#${item.id}</b></td>
      <td style="color: var(--text-secondary); font-size: 12px;">${item.timestamp}</td>
      <td><b>${escapeHtml(item.title || item.filename || "Văn bản")}</b></td>
      <td>${item.word_count} từ</td>
      <td><span class="data-badge ${badgeClass}">${badgeText}</span></td>
      <td><b style="font-family: var(--font-mono);">${(item.ai_probability * 100).toFixed(1)}%</b></td>
      <td style="color: var(--text-secondary);">${item.inference_time_ms} ms</td>
      <td style="text-align: right;">
        <button class="table-btn" onclick="downloadPdf(${item.id})">📥 PDF</button>
        <button class="table-btn" onclick="viewHistoryItem(${item.id})">👁️ Xem</button>
        <button class="table-btn text-danger" onclick="deleteHistoryItem(${item.id})">🗑️</button>
      </td>
    `;
    historyTableBody.appendChild(tr);
  });
}

window.downloadPdf = function(id) {
  window.open(`${API_BASE}/api/v1/report/pdf/${id}`, "_blank");
};

window.viewHistoryItem = async function(id) {
  try {
    const res = await fetch(`${API_BASE}/api/v1/history/${id}`);
    if (res.ok) {
      const data = await res.json();
      currentScanId = id;
      textInput.value = data.full_text || "";
      updateTextCounts();

      // Render lại kết quả trên tab inspect
      renderResults({
        scan_id: id,
        overall: {
          label: data.verdict,
          ai_probability: data.ai_probability,
          human_probability: 1.0 - data.ai_probability,
          confidence: data.confidence,
          inference_time_ms: data.inference_time_ms,
          model_breakdown: data.details ? data.details.model_breakdown : null,
          text_stats: { word_count: data.word_count, char_count: data.char_count, context_coverage: "1024 Tokens" },
        },
        sentences: data.details ? data.details.sentences : [],
        total_processing_time_ms: data.inference_time_ms,
      });

      // Switch to Inspect Tab
      document.querySelector('[data-tab="tab-inspect"]').click();
    }
  } catch (err) {
    alert("Không thể tải chi tiết: " + err.message);
  }
};

window.deleteHistoryItem = async function(id) {
  if (!confirm(`Bạn có chắc chắn muốn xóa lượt quét #${id} không?`)) return;
  try {
    const res = await fetch(`${API_BASE}/api/v1/history/${id}`, { method: "DELETE" });
    if (res.ok) {
      loadHistoryAndStats();
    }
  } catch (err) {
    alert("Lỗi xóa mục: " + err.message);
  }
};

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}
