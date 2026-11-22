/**
 * VeriText AI - Web Dashboard Single Page Application Logic
 * Đồ án Chuyên ngành Kỹ thuật Phần mềm (Chủ đề 6)
 */

const API_BASE = window.location.protocol === "file:" ? "http://localhost:8000" : "";

// Global State
let currentScanId = null;
let currentUploadedFile = null;
let currentUser = null;

function getAuthHeaders() {
  const token = localStorage.getItem("veritext_token");
  return token ? { "Authorization": `Bearer ${token}` } : {};
}

// DOM Elements - Auth & Roles
const btnOpenAuth = document.getElementById("btn-open-auth");
const userProfileBox = document.getElementById("user-profile-box");
const userRoleBadge = document.getElementById("user-role-badge");
const userDisplayName = document.getElementById("user-display-name");
const btnLogout = document.getElementById("btn-logout");

const authModal = document.getElementById("auth-modal");
const btnCloseAuthModal = document.getElementById("btn-close-auth-modal");
const tabBtnLogin = document.getElementById("tab-btn-login");
const tabBtnRegister = document.getElementById("tab-btn-register");
const formLogin = document.getElementById("form-login");
const formRegister = document.getElementById("form-register");
const authAlert = document.getElementById("auth-alert");

const btnQuickTeacher = document.getElementById("btn-quick-teacher");
const btnQuickStudent = document.getElementById("btn-quick-student");
const labelRoleStudent = document.getElementById("label-role-student");
const labelRoleTeacher = document.getElementById("label-role-teacher");

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
const btnSampleAiVi = document.getElementById("btn-sample-ai-vi");
const btnSampleHumanVi = document.getElementById("btn-sample-human-vi");
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
const verdictLangBadge = document.getElementById("verdict-lang-badge");

const metricHumanProb = document.getElementById("metric-human-prob");
const metricConfidence = document.getElementById("metric-confidence");
const metricLatency = document.getElementById("metric-latency");
const metricCoverage = document.getElementById("metric-coverage");

const modelsBoxTitle = document.getElementById("models-box-title");
const nameModel1 = document.getElementById("name-model-1");
const weightModel1 = document.getElementById("weight-model-1");
const barLongformer = document.getElementById("bar-longformer");
const probLongformer = document.getElementById("prob-longformer");

const nameModel2 = document.getElementById("name-model-2");
const weightModel2 = document.getElementById("weight-model-2");
const barBert = document.getElementById("bar-bert");
const probBert = document.getElementById("prob-bert");

const nameModel3 = document.getElementById("name-model-3");
const weightModel3 = document.getElementById("weight-model-3");
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
  // Theme Toggle Sync with Silky Circular View Transition
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
    btnThemeToggle.addEventListener("click", (e) => {
      const activeTheme = document.documentElement.getAttribute("data-theme") || "light";
      const nextTheme = activeTheme === "dark" ? "light" : "dark";

      // 1. Icon flip spring animation
      if (themeToggleIcon) {
        themeToggleIcon.style.transform = "rotate(180deg) scale(0.2)";
        setTimeout(() => {
          updateThemeIcon(nextTheme);
          themeToggleIcon.style.transform = "rotate(360deg) scale(1)";
        }, 140);
        setTimeout(() => {
          themeToggleIcon.style.transform = "";
        }, 320);
      }

      const applyTheme = () => {
        document.documentElement.setAttribute("data-theme", nextTheme);
        try {
          localStorage.setItem("veritext_theme", nextTheme);
        } catch (err) {}
      };

      // 2. Silky Circular Ripple Transition (View Transitions API)
      if (document.startViewTransition) {
        const rect = btnThemeToggle.getBoundingClientRect();
        const x = e.clientX || rect.left + rect.width / 2;
        const y = e.clientY || rect.top + rect.height / 2;
        const endRadius = Math.hypot(
          Math.max(x, window.innerWidth - x),
          Math.max(y, window.innerHeight - y)
        );

        const transition = document.startViewTransition(() => {
          applyTheme();
        });

        transition.ready.then(() => {
          document.documentElement.animate(
            {
              clipPath: [
                `circle(0px at ${x}px ${y}px)`,
                `circle(${endRadius}px at ${x}px ${y}px)`,
              ],
            },
            {
              duration: 460,
              easing: "cubic-bezier(0.4, 0, 0.2, 1)",
              pseudoElement: "::view-transition-new(root)",
            }
          );
        });
      } else {
        // Fallback for browsers without View Transition API
        applyTheme();
      }
    });
  }

  checkServerHealth();
  updateTextCounts();
  initAuth();

  // Tự động kiểm tra định kỳ trạng thái máy chủ
  let healthInterval = setInterval(async () => {
    const isOnline = await checkServerHealth();
    if (isOnline) {
      clearInterval(healthInterval);
      setInterval(checkServerHealth, 20000);
    }
  }, 3500);

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
      serverStatusEl.className = "status-pill " + (data.ready ? "status-online" : "status-offline");
      serverStatusTextEl.textContent = data.ready ? `FastAPI: Sẵn sàng (${data.device})` : "Đang nạp mô hình...";
      return data.ready;
    } else {
      serverStatusEl.className = "status-pill status-offline";
      serverStatusTextEl.textContent = "FastAPI: Lỗi phản hồi";
      return false;
    }
  } catch (err) {
    serverStatusEl.className = "status-pill status-offline";
    serverStatusTextEl.textContent = "FastAPI: Chưa kết nối";
    return false;
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

fileInput.addEventListener("click", () => {
  fileInput.value = null;
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

  // Nếu là file .txt thì nạp xem trước ngay vào textarea
  if (ext === "txt") {
    const reader = new FileReader();
    reader.onload = (e) => {
      textInput.value = e.target.result;
      updateTextCounts();
    };
    reader.readAsText(file);
  } else {
    // Với tài liệu .docx, .doc, .pdf
    textInput.placeholder = `[Đã chọn tệp: ${file.name} (${(file.size / 1024).toFixed(1)} KB)]\nNhấn "Kiểm tra nội dung AI" để máy chủ tự động bóc tách và phân tích toàn bộ văn bản.`;
  }
}

btnRemoveFile.addEventListener("click", (e) => {
  e.stopPropagation();
  currentUploadedFile = null;
  fileInput.value = "";
  fileInfoBar.classList.add("hidden");
  textInput.placeholder = "Nhập văn bản vào đây hoặc tải lên tệp để kiểm tra Nội dung AI.";
  textInput.value = "";
  updateTextCounts();
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
  textInput.placeholder = "Nhập văn bản vào đây hoặc tải lên tệp để kiểm tra Nội dung AI.";
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

if (btnSampleAiVi) {
  btnSampleAiVi.addEventListener("click", () => {
    textInput.value = (
      "Trí tuệ nhân tạo (AI) ngày nay đang đóng vai trò then chốt trong sự phát triển của nền công nghệ toàn cầu. "
      + "Các mô hình xử lý ngôn ngữ tự nhiên hiện đại có khả năng tiếp nhận và xử lý lượng thông tin khổng lồ một cách ấn tượng. "
      + "Tuy nhiên, sự xuất hiện ngày càng phổ biến của các nội dung do máy sinh đòi hỏi các giải pháp kiểm định nghiêm ngặt và đáng tin cậy. "
      + "Chính vì vậy, việc ứng dụng các mô hình học sâu như PhoBERT vào bài toán phát hiện văn bản AI là một hướng đi vô cùng cấp thiết."
    );
    updateTextCounts();
  });
}

if (btnSampleHumanVi) {
  btnSampleHumanVi.addEventListener("click", () => {
    textInput.value = (
      "Chè cam tào phớ giải nhiệt mùa hè là một món giải khát rất dễ làm, vừa giúp thanh lọc cơ thể lại không bị béo ngậy. "
      + "Cam tươi gọt sạch vỏ, bóc lấy phần tép và giữ lại một chút vỏ thái chỉ để tạo mùi thơm dịu nhẹ. "
      + "Nước cam hòa cùng đường phèn đun sôi liu riu trên bếp, thêm chút bột năng khuấy đều đến khi sánh mịn thì tắt bếp. "
      + "Múc từng lát tào phớ mềm mượt ra bát, chan nước cam lên trên và thưởng thức cùng vài viên đá mát lạnh."
    );
    updateTextCounts();
  });
}


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
        headers: getAuthHeaders(),
        body: formData,
      });

      if (!res.ok) {
        let errDetail = "Lỗi xử lý file từ server.";
        try {
          const err = await res.json();
          errDetail = err.detail || errDetail;
        } catch (e) {
          errDetail = `HTTP ${res.status}: ${res.statusText}`;
        }
        throw new Error(errDetail);
      }

      const data = await res.json();
      currentScanId = data.scan_id;

      // Đưa toàn bộ nội dung đã bóc tách vào ô văn bản để người dùng theo dõi
      if (data.extracted_text) {
        textInput.value = data.extracted_text;
        updateTextCounts();
      }

      renderResults(data);
    } catch (err) {
      console.error("Lỗi upload:", err);
      if (err.message.includes("Failed to fetch") || err.message.includes("NetworkError")) {
        alert("Không thể kết nối đến máy chủ FastAPI (http://localhost:8000).\nVui lòng chạy 'start_server.bat' để khởi động server trước khi quét!");
      } else {
        alert("Lỗi thẩm định tệp: " + err.message);
      }
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
      headers: { "Content-Type": "application/json", ...getAuthHeaders() },
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

  // 1. Language Badge & Verdict Banner
  const isVi = overall.language === "vi";
  if (verdictLangBadge) {
    if (isVi) {
      verdictLangBadge.textContent = "🇻🇳 Tiếng Việt (PhoBERT v2)";
      verdictLangBadge.style.background = "rgba(16, 185, 129, 0.15)";
      verdictLangBadge.style.color = "#059669";
      verdictLangBadge.style.borderColor = "rgba(16, 185, 129, 0.35)";
    } else {
      verdictLangBadge.textContent = "🇬🇧 Tiếng Anh (Ensemble 3 PLMs)";
      verdictLangBadge.style.background = "rgba(37, 99, 235, 0.12)";
      verdictLangBadge.style.color = "#2563eb";
      verdictLangBadge.style.borderColor = "rgba(37, 99, 235, 0.25)";
    }
  }

  verdictBanner.className = "verdict-header " + (isAI ? "verdict-ai" : "verdict-human");
  verdictIcon.textContent = isAI ? "🤖" : "👤";
  verdictTitle.textContent = isAI ? "VĂN BẢN DO AI SINH (AI-GENERATED)" : "VĂN BẢN DO CON NGƯỜI VIẾT (HUMAN-WRITTEN)";
  verdictSubtitle.textContent = isAI
    ? (isVi ? "Phát hiện dấu hiệu văn phong và phân bố token đặc trưng của ChatGPT / LLMs tiếng Việt" : "Phát hiện dấu hiệu cú pháp và thống kê đặc trưng của Mô hình Ngôn ngữ Lớn (LLM)")
    : (isVi ? "Văn phong tự nhiên, mức độ biến thiên từ vựng phong phú, không phát hiện dấu hiệu máy sinh" : "Văn phong tự nhiên, mức độ biến thiên từ vựng cao, không phát hiện dấu vết máy sinh");
  verdictPercent.textContent = `${aiPct}%`;

  // 2. Metrics Box
  metricHumanProb.textContent = `${humanPct}%`;
  metricConfidence.textContent = overall.confidence || "Rất cao";
  metricLatency.textContent = `${data.total_processing_time_ms || overall.inference_time_ms || 120} ms`;
  if (overall.text_stats) {
    metricCoverage.textContent = overall.text_stats.context_coverage || (isVi ? "256 Tokens (Full)" : "1024 Tokens");
  }

  // 3. Model Breakdown
  if (modelsBoxTitle) {
    modelsBoxTitle.textContent = isVi
      ? "Đánh giá từ Mô hình Chuyên biệt Tiếng Việt (PhoBERT base v2 & PyVi):"
      : "Đóng góp từ 3 Mô hình Nền tảng (Ensemble PLMs):";
  }

  if (isVi) {
    if (nameModel1) nameModel1.textContent = "PhoBERT v2";
    if (weightModel1) weightModel1.textContent = "Chuyên biệt (100%)";
    probLongformer.textContent = `${aiPct}%`;
    barLongformer.style.width = `${aiPct}%`;

    if (nameModel2) nameModel2.textContent = "PyVi Segmenter";
    if (weightModel2) weightModel2.textContent = "Tách từ ghép";
    probBert.textContent = "Hoàn tất";
    barBert.style.width = "100%";

    if (nameModel3) nameModel3.textContent = "Language Router";
    if (weightModel3) weightModel3.textContent = "Định tuyến tự động";
    probRoberta.textContent = "Tiếng Việt 🇻🇳";
    barRoberta.style.width = "100%";
  } else if (overall.model_breakdown && overall.model_breakdown.individual_models) {
    if (nameModel1) nameModel1.textContent = "Longformer";
    if (weightModel1) weightModel1.textContent = "Trụ cột (48.7%)";
    if (nameModel2) nameModel2.textContent = "BERT";
    if (weightModel2) weightModel2.textContent = "Trọng số 34.0%";
    if (nameModel3) nameModel3.textContent = "RoBERTa";
    if (weightModel3) weightModel3.textContent = "Trọng số 17.4%";

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
    const statsRes = await fetch(`${API_BASE}/api/v1/stats`, {
      headers: getAuthHeaders(),
    });
    if (statsRes.ok) {
      const stats = await statsRes.json();
      kpiTotalScans.textContent = stats.total_scans.toLocaleString();
      kpiAiRate.textContent = `${stats.ai_percentage}%`;
      kpiTotalWords.textContent = stats.total_words_scanned.toLocaleString();
      kpiAvgLatency.textContent = `${stats.avg_inference_time_ms} ms`;
    }

    // 2. Bảng dữ liệu
    const histRes = await fetch(`${API_BASE}/api/v1/history?limit=30`, {
      headers: getAuthHeaders(),
    });
    if (histRes.ok) {
      const data = await histRes.json();
      renderHistoryTable(data.history || [], data.current_user);
    }
  } catch (err) {
    console.error("Lỗi nạp lịch sử:", err);
  }
}

btnRefreshHistory.addEventListener("click", loadHistoryAndStats);

function renderHistoryTable(items, user) {
  historyTableBody.innerHTML = "";
  if (!items || items.length === 0) {
    const isStudent = user && user.role === "student";
    const msg = isStudent
      ? "Bạn chưa có lượt thẩm định nào. Hãy nhập văn bản ở tab 'Thẩm Định Văn Bản' để kiểm tra!"
      : "Chưa có lượt thẩm định nào được lưu. Hãy thực hiện kiểm tra ở Tab Thẩm Định!";
    historyTableBody.innerHTML = `
      <tr>
        <td colspan="8" class="text-center" style="padding: 30px; color: var(--text-muted);">
          ${msg}
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
      <td>
        <b>${escapeHtml(item.title || item.filename || "Văn bản")}</b>
        <div style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">
          👤 Người nộp: <b>${escapeHtml(item.author_name || "Khách")}</b>
        </div>
      </td>
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
    const res = await fetch(`${API_BASE}/api/v1/history/${id}`, {
      headers: getAuthHeaders(),
    });
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
    const res = await fetch(`${API_BASE}/api/v1/history/${id}`, {
      method: "DELETE",
      headers: getAuthHeaders(),
    });
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

// ==============================================================================
// 6. XÁC THỰC NGƯỜI DÙNG & PHÂN QUYỀN (JWT AUTHENTICATION & RBAC)
// ==============================================================================

function initAuth() {
  const savedUser = localStorage.getItem("veritext_user");
  const savedToken = localStorage.getItem("veritext_token");
  if (savedUser && savedToken) {
    try {
      currentUser = JSON.parse(savedUser);
      updateAuthUI(currentUser);
      checkUserToken();
    } catch (e) {
      handleLogout();
    }
  }

  if (btnOpenAuth) {
    btnOpenAuth.addEventListener("click", () => showAuthModal("login"));
  }
  if (btnCloseAuthModal) {
    btnCloseAuthModal.addEventListener("click", hideAuthModal);
  }
  if (authModal) {
    authModal.addEventListener("click", (e) => {
      if (e.target === authModal) hideAuthModal();
    });
  }

  if (tabBtnLogin && tabBtnRegister) {
    tabBtnLogin.addEventListener("click", () => switchAuthTab("login"));
    tabBtnRegister.addEventListener("click", () => switchAuthTab("register"));
  }

  const roleRadios = document.querySelectorAll('input[name="reg-role"]');
  roleRadios.forEach((r) => {
    r.addEventListener("change", () => {
      if (labelRoleStudent && labelRoleTeacher) {
        labelRoleStudent.classList.toggle("active", r.value === "student");
        labelRoleTeacher.classList.toggle("active", r.value === "teacher");
      }
    });
  });

  if (btnQuickTeacher) {
    btnQuickTeacher.addEventListener("click", async () => {
      document.getElementById("login-email").value = "giangvien@veritext.ai";
      document.getElementById("login-password").value = "123456";
      await handleLogin("giangvien@veritext.ai", "123456");
    });
  }
  if (btnQuickStudent) {
    btnQuickStudent.addEventListener("click", async () => {
      document.getElementById("login-email").value = "sinhvien@veritext.ai";
      document.getElementById("login-password").value = "123456";
      await handleLogin("sinhvien@veritext.ai", "123456");
    });
  }

  if (formLogin) {
    formLogin.addEventListener("submit", async (e) => {
      e.preventDefault();
      const email = document.getElementById("login-email").value.trim();
      const password = document.getElementById("login-password").value;
      await handleLogin(email, password);
    });
  }

  if (formRegister) {
    formRegister.addEventListener("submit", async (e) => {
      e.preventDefault();
      const fullName = document.getElementById("reg-fullname").value.trim();
      const email = document.getElementById("reg-email").value.trim();
      const password = document.getElementById("reg-password").value;
      const role = document.querySelector('input[name="reg-role"]:checked')?.value || "student";
      await handleRegister(fullName, email, password, role);
    });
  }

  if (btnLogout) {
    btnLogout.addEventListener("click", handleLogout);
  }
}

async function checkUserToken() {
  try {
    const res = await fetch(`${API_BASE}/api/v1/auth/me`, {
      headers: getAuthHeaders(),
    });
    if (res.ok) {
      const data = await res.json();
      currentUser = data.user;
      localStorage.setItem("veritext_user", JSON.stringify(currentUser));
      updateAuthUI(currentUser);
    } else {
      handleLogout();
    }
  } catch (e) {}
}

function showAuthModal(tab = "login") {
  if (!authModal) return;
  authModal.classList.remove("hidden");
  switchAuthTab(tab);
  clearAuthAlert();
}

function hideAuthModal() {
  if (!authModal) return;
  authModal.classList.add("hidden");
  clearAuthAlert();
}

function switchAuthTab(tab) {
  if (tab === "login") {
    tabBtnLogin.classList.add("active");
    tabBtnRegister.classList.remove("active");
    formLogin.classList.remove("hidden");
    formRegister.classList.add("hidden");
  } else {
    tabBtnLogin.classList.remove("active");
    tabBtnRegister.classList.add("active");
    formLogin.classList.add("hidden");
    formRegister.classList.remove("hidden");
  }
  clearAuthAlert();
}

function showAuthAlert(msg, isSuccess = false) {
  if (!authAlert) return;
  authAlert.textContent = msg;
  authAlert.className = "auth-alert " + (isSuccess ? "auth-alert-success" : "auth-alert-error");
  authAlert.classList.remove("hidden");
}

function clearAuthAlert() {
  if (!authAlert) return;
  authAlert.classList.add("hidden");
  authAlert.textContent = "";
}

async function handleLogin(email, password) {
  const submitBtn = document.getElementById("btn-submit-login");
  if (submitBtn) submitBtn.disabled = true;
  clearAuthAlert();

  try {
    const res = await fetch(`${API_BASE}/api/v1/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Đăng nhập thất bại.");
    }

    localStorage.setItem("veritext_token", data.access_token);
    localStorage.setItem("veritext_user", JSON.stringify(data.user));
    currentUser = data.user;
    updateAuthUI(currentUser);

    showAuthAlert("Đăng nhập thành công! Đang chuyển hướng...", true);
    setTimeout(() => {
      hideAuthModal();
      loadHistoryAndStats();
    }, 600);
  } catch (err) {
    showAuthAlert(err.message || "Lỗi đăng nhập.");
  } finally {
    if (submitBtn) submitBtn.disabled = false;
  }
}

async function handleRegister(fullName, email, password, role) {
  const submitBtn = document.getElementById("btn-submit-register");
  if (submitBtn) submitBtn.disabled = true;
  clearAuthAlert();

  try {
    const res = await fetch(`${API_BASE}/api/v1/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ full_name: fullName, email, password, role }),
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Đăng ký thất bại.");
    }

    localStorage.setItem("veritext_token", data.access_token);
    localStorage.setItem("veritext_user", JSON.stringify(data.user));
    currentUser = data.user;
    updateAuthUI(currentUser);

    showAuthAlert("Đăng ký tài khoản thành công!", true);
    setTimeout(() => {
      hideAuthModal();
      loadHistoryAndStats();
    }, 600);
  } catch (err) {
    showAuthAlert(err.message || "Lỗi đăng ký.");
  } finally {
    if (submitBtn) submitBtn.disabled = false;
  }
}

function handleLogout() {
  localStorage.removeItem("veritext_token");
  localStorage.removeItem("veritext_user");
  currentUser = null;
  updateAuthUI(null);
  loadHistoryAndStats();
}

function updateAuthUI(user) {
  if (user) {
    if (btnOpenAuth) btnOpenAuth.classList.add("hidden");
    if (userProfileBox) userProfileBox.classList.remove("hidden");
    const isTeacher = user.role === "teacher";
    if (userRoleBadge) {
      userRoleBadge.textContent = isTeacher ? "👨‍🏫" : "🎓";
      userRoleBadge.title = isTeacher ? "Giảng viên" : "Sinh viên";
    }
    if (userDisplayName) {
      userDisplayName.textContent = user.full_name || user.email;
      userDisplayName.title = `${user.full_name} (${isTeacher ? "Giảng viên" : "Sinh viên"})`;
    }
  } else {
    if (btnOpenAuth) btnOpenAuth.classList.remove("hidden");
    if (userProfileBox) userProfileBox.classList.add("hidden");
  }
}
