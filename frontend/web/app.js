/**
 * VeriText AI - Web Dashboard Single Page Application Logic
 * Đồ án Chuyên ngành Kỹ thuật Phần mềm (Chủ đề 6)
 */

const API_BASE = window.location.protocol === "file:" ? "http://localhost:8000" : "";

// Global State
let currentScanId = null;
let currentAuditCode = null;
let currentUploadedFile = null;
let currentUser = null;
let lastRenderedResult = null;
let progressStepTimers = [];

// ==============================================================================
// 0. HỆ THỐNG THÔNG BÁO TOAST CHUẨN SAAS (THAY THẾ HOÀN TOÀN ALERT TRÌNH DUYỆT)
// ==============================================================================

const TOAST_ICONS = {
  success: `<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`,
  error: `<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>`,
  warning: `<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`,
  info: `<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>`,
};

function showToast(message, type = "info", duration = 3800) {
  const container = document.getElementById("toast-container");
  if (!container) return;

  const toast = document.createElement("div");
  const safeType = TOAST_ICONS[type] ? type : "info";
  toast.className = `toast-item toast-${safeType}`;
  toast.innerHTML = `
    ${TOAST_ICONS[safeType]}
    <div class="toast-msg">${escapeHtml(String(message || ""))}</div>
    <button type="button" class="toast-close" aria-label="Đóng">&times;</button>
  `;

  const removeToast = () => {
    if (!toast.parentNode) return;
    toast.classList.add("toast-leaving");
    setTimeout(() => toast.remove(), 200);
  };

  toast.querySelector(".toast-close")?.addEventListener("click", removeToast);
  container.appendChild(toast);

  if (duration > 0) {
    setTimeout(removeToast, duration);
  }
}

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
const btnSampleBypass = document.getElementById("btn-sample-bypass");
const btnExportPdf = document.getElementById("btn-export-pdf");
const resultActions = document.getElementById("result-actions");

// DOM Elements - Academic Presets & Smart Filters
const presetBtns = document.querySelectorAll(".preset-pill-btn[data-preset]");
const chkFilterQuotes = document.getElementById("chk-filter-quotes");
const chkFilterReferences = document.getElementById("chk-filter-references");
let currentStrictnessPreset = "balanced";

// DOM Elements - Forensics, Anti-Bypass, Authorship Tier & XAI
const bypassWarningBanner = document.getElementById("bypass-warning-banner");
const bypassRiskBadge = document.getElementById("bypass-risk-badge");
const bypassBannerSummary = document.getElementById("bypass-banner-summary");
const bypassSamplesRow = document.getElementById("bypass-samples-row");

const filterInfoBanner = document.getElementById("filter-info-banner");
const filterInfoText = document.getElementById("filter-info-text");

const authorshipTierBox = document.getElementById("authorship-tier-box");
const authorshipTierBadge = document.getElementById("authorship-tier-badge");
const authorshipTierReason = document.getElementById("authorship-tier-reason");

const xaiEvidenceBox = document.getElementById("xai-evidence-box");
const xaiBurstinessVal = document.getElementById("xai-burstiness-val");
const xaiBurstinessBar = document.getElementById("xai-burstiness-bar");
const xaiBurstinessLabel = document.getElementById("xai-burstiness-label");
const xaiPerplexityVal = document.getElementById("xai-perplexity-val");
const xaiPerplexityBar = document.getElementById("xai-perplexity-bar");
const xaiPerplexityLabel = document.getElementById("xai-perplexity-label");
const xaiMattrVal = document.getElementById("xai-mattr-val");
const xaiMattrBar = document.getElementById("xai-mattr-bar");
const xaiMattrLabel = document.getElementById("xai-mattr-label");
const xaiClichesWrap = document.getElementById("xai-cliches-wrap");
const xaiClichesRow = document.getElementById("xai-cliches-row");
const xaiBulletsList = document.getElementById("xai-bullets-list");

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
const annotatedDocContainer = document.getElementById("annotated-doc-container");
const sentencesContainer = document.getElementById("sentences-container");
const btnViewAnnotated = document.getElementById("btn-view-annotated");
const btnViewList = document.getElementById("btn-view-list");
const btnCopyAuditCode = document.getElementById("btn-copy-audit-code");
const resultAuditCodeEl = document.getElementById("result-audit-code");
const btnCopySummary = document.getElementById("btn-copy-summary");
const verdictRingProgress = document.getElementById("verdict-ring-progress");
const sentenceDistWrap = document.getElementById("sentence-dist-wrap");
const distSegRed = document.getElementById("dist-seg-red");
const distSegYellow = document.getElementById("dist-seg-yellow");
const distSegGreen = document.getElementById("dist-seg-green");
const distSummaryText = document.getElementById("dist-summary-text");
const sentenceTooltip = document.getElementById("sentence-tooltip");
const btnSampleMenu = document.getElementById("btn-sample-menu");
const sampleDropdownMenu = document.getElementById("sample-dropdown-menu");
const analysisProgressBar = document.getElementById("analysis-progress-bar");
const progStep1 = document.getElementById("prog-step-1");
const progStep2 = document.getElementById("prog-step-2");
const progStep3 = document.getElementById("prog-step-3");

// DOM Elements - History & Filters
const kpiTotalScans = document.getElementById("kpi-total-scans");
const kpiAiRate = document.getElementById("kpi-ai-rate");
const kpiTotalWords = document.getElementById("kpi-total-words");
const kpiAvgLatency = document.getElementById("kpi-avg-latency");
const historyTableBody = document.getElementById("history-table-body");
const btnRefreshHistory = document.getElementById("btn-refresh-history");
const histSearchInput = document.getElementById("hist-search-input");
const histFilterLang = document.getElementById("hist-filter-lang");
const histFilterVerdict = document.getElementById("hist-filter-verdict");
const histFilterBypass = document.getElementById("hist-filter-bypass");

// DOM Elements - Verify Certificate Modal
const btnOpenVerify = document.getElementById("btn-open-verify");
const verifyModal = document.getElementById("verify-modal");
const btnCloseVerifyModal = document.getElementById("btn-close-verify-modal");
const formVerifyCert = document.getElementById("form-verify-cert");
const verifyCodeInput = document.getElementById("verify-code-input");
const btnSubmitVerify = document.getElementById("btn-submit-verify");
const verifyResultBox = document.getElementById("verify-result-box");


// ==============================================================================
// 1. KHỞI TẠO & ĐIỀU HƯỚNG TABS
// ==============================================================================

document.addEventListener("DOMContentLoaded", () => {
  // Theme Toggle Sync with Butter-Smooth Bidirectional Circular View Transition
  const btnThemeToggle = document.getElementById("btn-theme-toggle");
  let isThemeAnimating = false;

  function syncThemeButtonA11y(theme) {
    if (!btnThemeToggle) return;
    const isDark = theme === "dark";
    btnThemeToggle.setAttribute("aria-pressed", String(isDark));
    btnThemeToggle.title = isDark ? "Chuyển sang Chế độ Sáng" : "Chuyển sang Chế độ Tối";
  }

  const initialTheme = document.documentElement.getAttribute("data-theme") || "light";
  syncThemeButtonA11y(initialTheme);

  if (btnThemeToggle) {
    btnThemeToggle.addEventListener("click", () => {
      if (isThemeAnimating) return;

      const root = document.documentElement;
      const activeTheme = root.getAttribute("data-theme") || "light";
      const nextTheme = activeTheme === "dark" ? "light" : "dark";
      const toDark = nextTheme === "dark";

      const applyTheme = () => {
        root.setAttribute("data-theme", nextTheme);
        syncThemeButtonA11y(nextTheme);
        try {
          localStorage.setItem("veritext_theme", nextTheme);
        } catch (err) {}
      };

      const prefersReducedMotion =
        window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

      // GPU-composited Bidirectional Circular Reveal (không xung đột CSS transition)
      if (document.startViewTransition && !prefersReducedMotion) {
        isThemeAnimating = true;
        const rect = btnThemeToggle.getBoundingClientRect();
        const x = rect.left + rect.width / 2;
        const y = rect.top + rect.height / 2;
        const endRadius = Math.hypot(
          Math.max(x, window.innerWidth - x),
          Math.max(y, window.innerHeight - y)
        );

        const dirClass = toDark ? "theme-to-dark" : "theme-to-light";
        root.classList.add("theme-transitioning", dirClass);

        const transition = document.startViewTransition(() => {
          applyTheme();
        });

        transition.ready
          .then(() => {
            const clipExpand = [
              `circle(0px at ${x}px ${y}px)`,
              `circle(${endRadius}px at ${x}px ${y}px)`,
            ];
            root.animate(
              {
                clipPath: toDark ? clipExpand : [...clipExpand].reverse(),
              },
              {
                duration: 560,
                easing: "cubic-bezier(0.22, 1, 0.36, 1)",
                pseudoElement: toDark ? "::view-transition-new(root)" : "::view-transition-old(root)",
              }
            );
          })
          .catch(() => {});

        transition.finished.finally(() => {
          root.classList.remove("theme-transitioning", "theme-to-dark", "theme-to-light");
          isThemeAnimating = false;
        });
      } else {
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
      } else if (targetId === "tab-classroom") {
        syncClassroomAccessUI();
      }
    });
  });

  // Click logo -> quay về Tab Thẩm định chính
  const brandHome = document.getElementById("brand-home");
  if (brandHome) {
    brandHome.style.cursor = "pointer";
    brandHome.addEventListener("click", () => {
      const inspectTab = document.querySelector('[data-tab="tab-inspect"]');
      if (inspectTab) inspectTab.click();
    });
  }

  initClassroomBatchUI();
  initHotspotViewTabs();
  initHistoryFilters();
  initVerifyCertificateModal();
  initSampleDropdownAndShortcuts();

  // Đóng nhanh các hộp thoại Modal & Dropdown bằng phím Escape
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    authModal?.classList.add("hidden");
    document.getElementById("profile-modal")?.classList.add("hidden");
    verifyModal?.classList.add("hidden");
    closeSampleDropdown();
  });
});

function closeSampleDropdown() {
  if (sampleDropdownMenu) sampleDropdownMenu.classList.add("hidden");
  if (btnSampleMenu) {
    btnSampleMenu.setAttribute("aria-expanded", "false");
    btnSampleMenu.closest(".sample-dropdown-wrap")?.classList.remove("open");
  }
}

function initSampleDropdownAndShortcuts() {
  if (btnSampleMenu && sampleDropdownMenu) {
    btnSampleMenu.addEventListener("click", (e) => {
      e.stopPropagation();
      const willOpen = sampleDropdownMenu.classList.contains("hidden");
      if (willOpen) {
        sampleDropdownMenu.classList.remove("hidden");
        btnSampleMenu.setAttribute("aria-expanded", "true");
        btnSampleMenu.closest(".sample-dropdown-wrap")?.classList.add("open");
      } else {
        closeSampleDropdown();
      }
    });

    document.addEventListener("click", (e) => {
      if (!sampleDropdownMenu.contains(e.target) && !btnSampleMenu.contains(e.target)) {
        closeSampleDropdown();
      }
    });
  }

  // Phím tắt Ctrl + Enter (hoặc Cmd + Enter trên macOS) để quét nhanh ngay trong ô nhập
  if (textInput) {
    textInput.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        if (!btnSubmit.disabled) btnSubmit.click();
      }
    });
  }
}

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
  const val = parseInt(thresholdRange.value, 10);
  thresholdVal.textContent = `${val}%`;
  currentStrictnessPreset = null;
  presetBtns.forEach((b) => {
    const match = parseInt(b.dataset.val, 10) === val;
    b.classList.toggle("active", match);
    if (match) currentStrictnessPreset = b.dataset.preset;
  });
});

presetBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    const val = btn.dataset.val;
    currentStrictnessPreset = btn.dataset.preset;
    thresholdRange.value = val;
    thresholdVal.textContent = `${val}%`;
    presetBtns.forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
  });
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
    showToast("Chỉ hỗ trợ tệp tin định dạng: .PDF, .DOCX, .DOC, hoặc .TXT", "warning");
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

function clearSelectedUploadFile() {
  currentUploadedFile = null;
  if (fileInput) fileInput.value = "";
  if (fileInfoBar) fileInfoBar.classList.add("hidden");
  if (textInput) textInput.placeholder = "Nhập văn bản vào đây hoặc tải lên tệp để kiểm tra Nội dung AI.";
}

btnRemoveFile.addEventListener("click", (e) => {
  e.stopPropagation();
  clearSelectedUploadFile();
  textInput.value = "";
  updateTextCounts();
});

// Action buttons
btnPaste.addEventListener("click", async () => {
  try {
    const text = await navigator.clipboard.readText();
    clearSelectedUploadFile();
    textInput.value = text;
    updateTextCounts();
    showToast("Đã dán văn bản từ Clipboard.", "info", 2000);
  } catch (err) {
    showToast("Không thể đọc Clipboard. Vui lòng nhấn Ctrl + V để dán trực tiếp.", "warning");
  }
});

btnClear.addEventListener("click", () => {
  textInput.value = "";
  updateTextCounts();
  currentScanId = null;
  currentAuditCode = null;
  lastRenderedResult = null;
  clearSelectedUploadFile();
  resultContent.classList.add("hidden");
  resultActions.classList.add("hidden");
  if (btnCopyAuditCode) btnCopyAuditCode.classList.add("hidden");
});

function applySampleText(sampleContent, sampleLabel) {
  clearSelectedUploadFile();
  closeSampleDropdown();
  textInput.value = sampleContent;
  updateTextCounts();
  showToast(`Đã nạp ${sampleLabel} — nhấn Ctrl + Enter để kiểm tra!`, "info", 2500);
}

btnSampleAi.addEventListener("click", () => {
  applySampleText(
    "Artificial intelligence models have demonstrated exceptional capabilities across various natural language processing domains. "
    + "These sophisticated architectures leverage deep contextual self-attention layers to produce coherent, syntactically fluent discourse. "
    + "However, the widespread dissemination of synthetic content introduces non-trivial risks regarding academic dishonesty and journalistic trust. "
    + "Therefore, developing resilient multi-model ensemble detection frameworks is of paramount importance to safeguard digital information fidelity.",
    "Mẫu AI Tiếng Anh"
  );
});

btnSampleHuman.addEventListener("click", () => {
  applySampleText(
    "Animal cognition encompasses the mental capacities of non-human animals including insect cognition. "
    + "The study of animal conditioning and learning used in this field was developed from comparative psychology. "
    + "It has also been strongly influenced by research in ethology, behavioral ecology, and evolutionary psychology; the alternative name cognitive ethology is sometimes used. "
    + "Many behaviors associated with the term animal intelligence are also subsumed within animal cognition.",
    "Mẫu Người Tiếng Anh"
  );
});

if (btnSampleAiVi) {
  btnSampleAiVi.addEventListener("click", () => {
    applySampleText(
      "Trí tuệ nhân tạo (AI) ngày nay đang đóng vai trò then chốt trong sự phát triển của nền công nghệ toàn cầu. "
      + "Các mô hình xử lý ngôn ngữ tự nhiên hiện đại có khả năng tiếp nhận và xử lý lượng thông tin khổng lồ một cách ấn tượng. "
      + "Tuy nhiên, sự xuất hiện ngày càng phổ biến của các nội dung do máy sinh đòi hỏi các giải pháp kiểm định nghiêm ngặt và đáng tin cậy. "
      + "Chính vì vậy, việc ứng dụng các mô hình học sâu như PhoBERT vào bài toán phát hiện văn bản AI là một hướng đi vô cùng cấp thiết.",
      "Mẫu AI Tiếng Việt"
    );
  });
}

if (btnSampleHumanVi) {
  btnSampleHumanVi.addEventListener("click", () => {
    applySampleText(
      "Tôi cũng nhiều lúc rơi vào tình trạng cuồng chân. "
      + "Đi du lịch không được, về quê không xong, gặp gỡ, thăm nom ai đó cũng phải hỏi kỹ càng bởi chỗ nào cũng bị cảnh báo. "
      + "Bố mẹ, anh chị, hàng xóm của tôi đều là F2, F3, con phố tôi ở cũng có người F0 rồi. "
      + "Đành bó chân ở nhà viết bài, đọc sách, nghiên cứu tài liệu, soạn bài giảng, dạy con học và tập thể dục qua ngày.",
      "Mẫu Người Tiếng Việt"
    );
  });
}

if (btnSampleBypass) {
  btnSampleBypass.addEventListener("click", () => {
    // Mẫu AI bị cài ký tự tàng hình Zero-Width (\u200b) + chữ Cyrillic giả Latin (\u0430, \u0435, \u043e) + mục Tài liệu tham khảo
    applySampleText(
      "Trí\u200b tuệ\u200b nhân\u200b t\u0430o (AI) đóng v\u0430i trò then chốt tr\u043eng bối c\u0430nh chuyển đổi số hiện n\u0430y. "
      + "Không thể phủ nhận rằng các mô hình ngôn ngữ lớn m\u0430ng lại nhiều lợi ích vượt trội ch\u043e giáo dục đại học. "
      + "Bên cạnh đó, việc ứng dụng công\u200b nghệ\u200b này cũng đặt r\u0430 nhiều thách thức đối với tính trung thực học thuật. "
      + "Tóm lại, xây dựng hệ thống kiểm định minh bạch là một hướng đi vô cùng cấp thiết.\n\n"
      + "Tài liệu tham khảo:\n"
      + "[1] Vaswani et al. (2017). Attention Is All You Need. NeurIPS.\n"
      + "[2] Nguyen & Nguyen (2020). PhoBERT: Pre-trained language models for Vietnamese.",
      "Mẫu Lách Luật Anti-Bypass"
    );
  });
}


// ==============================================================================
// 3. THỰC HIỆN THẨM ĐỊNH (SUBMIT ANALYSIS)
// ==============================================================================

btnSubmit.addEventListener("click", async () => {
  const threshold = parseFloat(thresholdRange.value) / 100;
  const filterQuotes = chkFilterQuotes ? chkFilterQuotes.checked : false;
  const filterReferences = chkFilterReferences ? chkFilterReferences.checked : false;

  // Trường hợp 1: Có file đính kèm
  if (currentUploadedFile) {
    setLoading(true);
    const formData = new FormData();
    formData.append("file", currentUploadedFile);
    formData.append("threshold", threshold);
    if (currentStrictnessPreset) {
      formData.append("strictness_preset", currentStrictnessPreset);
    }
    formData.append("filter_quotes", filterQuotes);
    formData.append("filter_references", filterReferences);

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
        showToast("Không thể kết nối máy chủ FastAPI (http://localhost:8000). Vui lòng chạy 'start_server.bat'!", "error", 5000);
      } else {
        showToast("Lỗi thẩm định tệp: " + err.message, "error", 4800);
      }
    } finally {
      setLoading(false);
    }
    return;
  }

  // Trường hợp 2: Nhập văn bản trực tiếp
  const rawText = textInput.value.trim();
  if (rawText.length < 10) {
    showToast("Vui lòng nhập hoặc dán ít nhất 10 ký tự (Anh hoặc Việt) để hệ thống phân tích.", "warning");
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
        strictness_preset: currentStrictnessPreset,
        filter_quotes: filterQuotes,
        filter_references: filterReferences,
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
    showToast("Lỗi thẩm định văn bản: " + err.message, "error", 4800);
  } finally {
    setLoading(false);
  }
});

function setLoading(isLoading) {
  btnSubmit.disabled = isLoading;
  const btnText = btnSubmit.querySelector(".btn-text");
  const kbdHint = btnSubmit.querySelector(".kbd-hint");

  progressStepTimers.forEach((t) => clearTimeout(t));
  progressStepTimers = [];

  if (isLoading) {
    btnText.textContent = "Đang phân tích...";
    if (kbdHint) kbdHint.classList.add("hidden");
    spinner.classList.remove("hidden");

    if (analysisProgressBar) {
      analysisProgressBar.classList.remove("hidden");
      [progStep1, progStep2, progStep3].forEach((el) => el?.classList.remove("active", "done"));
      progStep1?.classList.add("active");

      progressStepTimers.push(setTimeout(() => {
        progStep1?.classList.remove("active");
        progStep1?.classList.add("done");
        progStep2?.classList.add("active");
      }, 260));

      progressStepTimers.push(setTimeout(() => {
        progStep2?.classList.remove("active");
        progStep2?.classList.add("done");
        progStep3?.classList.add("active");
      }, 1050));
    }
  } else {
    btnText.textContent = "Kiểm tra nội dung AI";
    if (kbdHint) kbdHint.classList.remove("hidden");
    spinner.classList.add("hidden");
    if (analysisProgressBar) {
      analysisProgressBar.classList.add("hidden");
      [progStep1, progStep2, progStep3].forEach((el) => el?.classList.remove("active", "done"));
    }
  }
}


// ==============================================================================
// 4. HIỂN THỊ KẾT QUẢ THẨM ĐỊNH
// ==============================================================================

function renderResults(data) {
  lastRenderedResult = data;
  if (resultPlaceholder) resultPlaceholder.classList.add("hidden");
  resultContent.classList.remove("hidden");
  resultActions.classList.remove("hidden");

  const overall = data.overall || data;
  const aiProb = Math.max(0, Math.min(1, Number(overall.ai_probability) || 0));
  const isAI = overall.label ? overall.label === "AI-generated" : aiProb >= 0.5;
  const aiPct = (aiProb * 100).toFixed(1);
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

  // Animated Circular SVG Ring Gauge (r=36 -> circumference = 226.195)
  if (verdictRingProgress) {
    const CIRC = 2 * Math.PI * 36;
    const ringColor = aiProb >= 0.70 ? "#EF4444" : aiProb >= 0.40 ? "#F59E0B" : "#10B981";
    verdictRingProgress.style.stroke = ringColor;
    verdictRingProgress.style.strokeDasharray = `${CIRC}`;
    verdictRingProgress.style.strokeDashoffset = `${CIRC}`;
    requestAnimationFrame(() => {
      verdictRingProgress.style.strokeDashoffset = `${CIRC * (1 - aiProb)}`;
    });
  }

  // 2. Metrics Box
  metricHumanProb.textContent = `${humanPct}%`;
  metricConfidence.textContent = overall.confidence || "Rất cao";
  metricLatency.textContent = `${data.total_processing_time_ms || overall.inference_time_ms || 120} ms`;
  if (overall.text_stats) {
    metricCoverage.textContent = overall.text_stats.context_coverage || (isVi ? "256 Tokens (Full)" : "1024 Tokens");
  }

  // 2b. Anti-Bypass Alert, Academic Filter Notice, 4-Tier Authorship & XAI Evidence
  renderForensicsAndXai(data);

  // 2c. Cryptographic Anti-Forgery Audit Code Pill
  currentAuditCode = data.audit_code || null;
  if (btnCopyAuditCode && resultAuditCodeEl) {
    if (currentAuditCode) {
      resultAuditCodeEl.textContent = currentAuditCode;
      btnCopyAuditCode.classList.remove("hidden");
    } else {
      btnCopyAuditCode.classList.add("hidden");
    }
  }

  // 3. Model Breakdown
  if (modelsBoxTitle) {
    modelsBoxTitle.textContent = isVi
      ? "Đánh giá từ Mô hình Chuyên biệt Tiếng Việt (PhoBERT v3.1 - 15K & PyVi):"
      : "Đóng góp từ 3 Mô hình Nền tảng (Ensemble Stacking + Temp Scaling):";
  }

  if (isVi) {
    const chunkInfo = overall.model_breakdown && overall.model_breakdown.chunk_analysis;
    if (chunkInfo && chunkInfo.chunk_count > 1) {
      const meanPct = (chunkInfo.mean_probability * 100).toFixed(1);
      const peakPct = (chunkInfo.peak_probability * 100).toFixed(1);

      if (nameModel1) nameModel1.textContent = "PhoBERT v2 (Hợp nhất)";
      if (weightModel1) weightModel1.textContent = `Sliding-Window (${chunkInfo.chunk_count} đoạn)`;
      probLongformer.textContent = `${aiPct}%`;
      barLongformer.style.width = `${aiPct}%`;

      if (nameModel2) nameModel2.textContent = "Trung bình các đoạn";
      if (weightModel2) weightModel2.textContent = "Trọng số độ dài (65%)";
      probBert.textContent = `${meanPct}%`;
      barBert.style.width = `${meanPct}%`;

      if (nameModel3) nameModel3.textContent = "Đỉnh cục bộ (Hotspot)";
      if (weightModel3) weightModel3.textContent = "Bắt trộn AI (35%)";
      probRoberta.textContent = `${peakPct}%`;
      barRoberta.style.width = `${peakPct}%`;
    } else {
      if (nameModel1) nameModel1.textContent = "PhoBERT v2";
      if (weightModel1) weightModel1.textContent = "Chuyên biệt 15K (100%)";
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
    }
  } else if (overall.model_breakdown && overall.model_breakdown.individual_models) {
    const m = overall.model_breakdown.individual_models;
    const wLong = (m.longformer && m.longformer.weight_pct) || "15.9%";
    const wBert = (m.bert && m.bert.weight_pct) || "46.6%";
    const wRob = (m.roberta && m.roberta.weight_pct) || "37.5%";

    if (nameModel1) nameModel1.textContent = "Longformer";
    if (weightModel1) weightModel1.textContent = `1024 Tokens (${wLong})`;
    if (nameModel2) nameModel2.textContent = "BERT";
    if (weightModel2) weightModel2.textContent = `Trụ cột (${wBert})`;
    if (nameModel3) nameModel3.textContent = "RoBERTa";
    if (weightModel3) weightModel3.textContent = `Trọng số ${wRob}`;

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

  // 4. Turnitin-Style Annotated Essay & Sentence List Hotspots
  renderAnnotatedDocument(data.sentences || []);
  renderSentences(data.sentences || []);

  // Smooth scroll into view
  resultContent.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderForensicsAndXai(data) {
  // 1. Anti-Bypass Warning Banner
  const bypass = data.bypass_report;
  if (bypassWarningBanner) {
    if (bypass && bypass.has_bypass_warning) {
      bypassWarningBanner.classList.remove("hidden");
      if (bypassRiskBadge) bypassRiskBadge.textContent = `RỦI RO ${bypass.risk_level || "HIGH"}`;
      if (bypassBannerSummary) bypassBannerSummary.textContent = bypass.summary || "";
      if (bypassSamplesRow) {
        bypassSamplesRow.innerHTML = "";
        (bypass.flagged_samples || []).forEach((s) => {
          const chip = document.createElement("span");
          chip.className = "bypass-sample-chip";
          chip.textContent = s;
          bypassSamplesRow.appendChild(chip);
        });
      }
    } else {
      bypassWarningBanner.classList.add("hidden");
    }
  }

  // 2. Academic Filter Notice
  const flt = data.filter_report;
  if (filterInfoBanner) {
    if (flt && flt.filter_applied && flt.filter_summary) {
      filterInfoBanner.classList.remove("hidden");
      if (filterInfoText) filterInfoText.textContent = flt.filter_summary;
    } else {
      filterInfoBanner.classList.add("hidden");
    }
  }

  // 3. 4-Tier Authorship Box (hỗ trợ cả object chi tiết lẫn mã chuỗi từ SQL Server)
  const TIER_META_MAP = {
    AI_PURE: { label_vi: "🤖 AI sinh toàn phần (Pure AI)", badge_class: "tier-ai-pure" },
    HYBRID_MIXED: { label_vi: "🧩 Văn bản lai: Trộn Người + AI (Hybrid)", badge_class: "tier-hybrid" },
    AI_ASSISTED: { label_vi: "✍️ Nghi vấn AI biên tập / Paraphrase", badge_class: "tier-assisted" },
    HUMAN_PURE: { label_vi: "👤 Tự viết hoàn toàn (Human-Written)", badge_class: "tier-human" },
  };
  let tier = data.authorship_tier;
  if (typeof tier === "string" && TIER_META_MAP[tier]) {
    tier = { code: tier, ...TIER_META_MAP[tier], reason: "" };
  }
  if (authorshipTierBox) {
    if (tier && typeof tier === "object" && tier.label_vi) {
      authorshipTierBox.classList.remove("hidden");
      if (authorshipTierBadge) {
        authorshipTierBadge.textContent = tier.label_vi;
        authorshipTierBadge.className = `authorship-tier-badge ${tier.badge_class || "tier-assisted"}`;
      }
      if (authorshipTierReason) authorshipTierReason.textContent = tier.reason || "";
    } else {
      authorshipTierBox.classList.add("hidden");
    }
  }

  // 4. Explainable AI (XAI) Evidence Panel
  const xai = data.xai_metrics;
  if (xaiEvidenceBox) {
    if (xai && typeof xai === "object" && xai.burstiness_score !== undefined) {
      xaiEvidenceBox.classList.remove("hidden");
      const setBar = (el, val) => {
        if (!el) return;
        const pct = Math.max(4, Math.min(100, Number(val) || 0));
        el.style.width = `${pct}%`;
        el.className = "xai-bar-fill " + (pct < 38 ? "bar-low" : pct < 62 ? "bar-mid" : "bar-high");
      };

      if (xaiBurstinessVal) xaiBurstinessVal.textContent = `${xai.burstiness_score}/100`;
      setBar(xaiBurstinessBar, xai.burstiness_score);
      if (xaiBurstinessLabel) xaiBurstinessLabel.textContent = xai.burstiness_label || "";

      if (xaiPerplexityVal) xaiPerplexityVal.textContent = `${xai.perplexity_score}/100`;
      setBar(xaiPerplexityBar, xai.perplexity_score);
      if (xaiPerplexityLabel) xaiPerplexityLabel.textContent = xai.perplexity_label || "";

      if (xaiMattrVal) xaiMattrVal.textContent = `${xai.lexical_diversity}%`;
      setBar(xaiMattrBar, xai.lexical_diversity);
      if (xaiMattrLabel) {
        const st = xai.sentence_stats || {};
        xaiMattrLabel.textContent = `TB ${st.avg_length || 0} từ/câu (±${st.std_length || 0} từ)`;
      }

      if (xaiClichesWrap && xaiClichesRow) {
        const cliches = xai.ai_cliches || [];
        if (cliches.length > 0) {
          xaiClichesWrap.classList.remove("hidden");
          xaiClichesRow.innerHTML = "";
          cliches.forEach((c) => {
            const chip = document.createElement("span");
            chip.className = "xai-cliche-chip";
            chip.textContent = `"${c.phrase}" ×${c.count}`;
            xaiClichesRow.appendChild(chip);
          });
        } else {
          xaiClichesWrap.classList.add("hidden");
        }
      }

      if (xaiBulletsList) {
        xaiBulletsList.innerHTML = "";
        (xai.explanation_bullets || []).forEach((b) => {
          const li = document.createElement("li");
          li.textContent = b;
          xaiBulletsList.appendChild(li);
        });
      }
    } else {
      xaiEvidenceBox.classList.add("hidden");
    }
  }
}

function initHotspotViewTabs() {
  if (btnViewAnnotated && btnViewList) {
    btnViewAnnotated.addEventListener("click", () => {
      btnViewAnnotated.classList.add("active");
      btnViewList.classList.remove("active");
      annotatedDocContainer?.classList.remove("hidden");
      sentencesContainer?.classList.add("hidden");
    });

    btnViewList.addEventListener("click", () => {
      btnViewList.classList.add("active");
      btnViewAnnotated.classList.remove("active");
      annotatedDocContainer?.classList.add("hidden");
      sentencesContainer?.classList.remove("hidden");
    });
  }

  if (btnCopyAuditCode) {
    btnCopyAuditCode.addEventListener("click", () => {
      if (!currentAuditCode) return;
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(currentAuditCode).catch(() => {});
      }
      showToast(`Đã sao chép mã chứng thực ${currentAuditCode}`, "success", 2600);
      window.openVerifyWithCode(currentAuditCode);
    });
  }

  if (btnCopySummary) {
    btnCopySummary.addEventListener("click", async () => {
      if (!lastRenderedResult) {
        showToast("Chưa có kết quả thẩm định để sao chép.", "warning");
        return;
      }
      const d = lastRenderedResult;
      const ov = d.overall || d;
      const aiPct = ((Number(ov.ai_probability) || 0) * 100).toFixed(1);
      const isAi = ov.label === "AI-generated" || Number(ov.ai_probability) >= 0.5;
      const langTxt = ov.language === "vi" ? "Tiếng Việt (PhoBERT v2)" : "Tiếng Anh (Ensemble 3 PLMs)";
      const tierTxt = typeof d.authorship_tier === "object" && d.authorship_tier?.label_vi
        ? d.authorship_tier.label_vi
        : (d.authorship_tier || "--");
      const bypassTxt = d.bypass_report?.has_bypass_warning
        ? `CÓ CẢNH BÁO (${d.bypass_report.summary || "Phát hiện ký tự bất thường"})`
        : "Không phát hiện";
      const xaiLine = d.xai_metrics?.burstiness_score !== undefined
        ? `• XAI Ngôn ngữ học: Burstiness ${d.xai_metrics.burstiness_score}/100 | Perplexity ${d.xai_metrics.perplexity_score}/100 | MATTR ${d.xai_metrics.lexical_diversity}%`
        : "";
      const summaryLines = [
        "[KẾT QUẢ THẨM ĐỊNH VĂN BẢN — VERITEXT AI]",
        `• Kết luận: ${isAi ? "Văn bản do AI sinh (AI-Generated)" : "Văn bản do Con người viết (Human-Written)"} (${aiPct}% AI)`,
        `• Phân loại 4 mức: ${tierTxt}`,
        `• Tuyến mô hình: ${langTxt} | Độ tin cậy: ${ov.confidence || "--"}`,
        xaiLine,
        `• Kiểm tra Anti-Bypass: ${bypassTxt}`,
        d.audit_code ? `• Mã chứng thực chống giả mạo: ${d.audit_code}` : "",
      ].filter(Boolean).join("\n");

      try {
        await navigator.clipboard.writeText(summaryLines);
        showToast("Đã sao chép tóm tắt nhận xét vào Clipboard!", "success", 2800);
      } catch (e) {
        showToast("Trình duyệt không cho phép truy cập Clipboard tự động.", "warning");
      }
    });
  }
}

function showSentenceTooltip(spanEl, event) {
  if (!sentenceTooltip) return;
  const idx = spanEl.dataset.sIdx || "1";
  const words = spanEl.dataset.sWords || "";
  const pct = spanEl.dataset.sPct || "0%";
  const desc = spanEl.dataset.sDesc || "";
  const color = spanEl.dataset.sColor || "green";

  sentenceTooltip.innerHTML = `
    <div class="sentence-tooltip-head">
      <span class="sentence-tooltip-idx">Câu #${escapeHtml(idx)}${words ? ` (${escapeHtml(words)} từ)` : ""}</span>
      <span class="sentence-tooltip-pct tooltip-pct-${escapeHtml(color)}">${escapeHtml(pct)} AI</span>
    </div>
    <div class="sentence-tooltip-desc">${escapeHtml(desc)}</div>
  `;
  sentenceTooltip.classList.remove("hidden");

  const pad = 12;
  const tipRect = sentenceTooltip.getBoundingClientRect();
  let left = event.clientX - tipRect.width / 2;
  let top = event.clientY - tipRect.height - 14;
  if (top < pad) top = event.clientY + 18;
  left = Math.max(pad, Math.min(window.innerWidth - tipRect.width - pad, left));

  sentenceTooltip.style.left = `${left}px`;
  sentenceTooltip.style.top = `${top}px`;
}

function hideSentenceTooltip() {
  if (sentenceTooltip) sentenceTooltip.classList.add("hidden");
}

function renderSentenceDistribution(sentences) {
  if (!sentenceDistWrap) return;
  if (!sentences || sentences.length === 0) {
    sentenceDistWrap.classList.add("hidden");
    return;
  }

  const total = sentences.length;
  const red = sentences.filter((s) => s.highlight_color === "red").length;
  const yellow = sentences.filter((s) => s.highlight_color === "yellow").length;
  const green = total - red - yellow;

  const redPct = (red / total) * 100;
  const yellowPct = (yellow / total) * 100;
  const greenPct = (green / total) * 100;

  if (distSegRed) distSegRed.style.width = `${redPct.toFixed(1)}%`;
  if (distSegYellow) distSegYellow.style.width = `${yellowPct.toFixed(1)}%`;
  if (distSegGreen) distSegGreen.style.width = `${greenPct.toFixed(1)}%`;
  if (distSummaryText) {
    distSummaryText.innerHTML = `
      <span>Phân bố <b>${total} câu</b>: <b>${red}</b> AI cao (${redPct.toFixed(0)}%) • <b>${yellow}</b> nghi vấn (${yellowPct.toFixed(0)}%) • <b>${green}</b> tự nhiên (${greenPct.toFixed(0)}%)</span>
      <span style="color: var(--text-muted); font-size: 11.5px;">Rê chuột vào từng câu để xem điểm chi tiết</span>
    `;
  }
  sentenceDistWrap.classList.remove("hidden");
}

function renderAnnotatedDocument(sentences) {
  renderSentenceDistribution(sentences);
  if (!annotatedDocContainer) return;
  annotatedDocContainer.innerHTML = "";
  hideSentenceTooltip();

  if (!sentences || sentences.length === 0) {
    const rawPreview = (textInput && textInput.value.trim()) || "";
    if (rawPreview) {
      const p = document.createElement("p");
      p.style.margin = "0";
      p.textContent = rawPreview;
      annotatedDocContainer.appendChild(p);
    } else {
      annotatedDocContainer.innerHTML = "<span style='color: var(--text-muted); font-family: var(--font-sans); font-size: 13px;'>Không có dữ liệu tách câu chi tiết cho bản ghi này.</span>";
    }
    return;
  }

  sentences.forEach((s, idx) => {
    const span = document.createElement("span");
    const color = ["red", "yellow", "green"].includes(s.highlight_color) ? s.highlight_color : "green";
    span.className = `hl-sent ${color}`;
    span.textContent = s.sentence;
    span.dataset.sIdx = String(s.sentence_index || (idx + 1));
    span.dataset.sWords = s.word_count ? String(s.word_count) : "";
    span.dataset.sPct = String(s.ai_percentage || "0%");
    span.dataset.sDesc = String(s.assessment || "");
    span.dataset.sColor = color;

    span.addEventListener("mouseenter", (e) => showSentenceTooltip(span, e));
    span.addEventListener("mousemove", (e) => showSentenceTooltip(span, e));
    span.addEventListener("mouseleave", hideSentenceTooltip);

    annotatedDocContainer.appendChild(span);

    if (idx < sentences.length - 1) {
      annotatedDocContainer.appendChild(document.createTextNode(" "));
    }
  });
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

async function openAuthenticatedPdf(scanId) {
  try {
    const res = await fetch(`${API_BASE}/api/v1/report/pdf/${scanId}`, {
      headers: getAuthHeaders(),
    });
    if (!res.ok) {
      let errDetail = "Không thể xuất báo cáo PDF.";
      try {
        const err = await res.json();
        errDetail = err.detail || errDetail;
      } catch (e) {}
      throw new Error(errDetail);
    }
    const blob = await res.blob();
    const blobUrl = URL.createObjectURL(blob);
    const popup = window.open(blobUrl, "_blank");
    if (!popup) {
      // Fallback tải trực tiếp nếu trình duyệt chặn cửa sổ bật lên (Popup Blocker)
      const a = document.createElement("a");
      a.href = blobUrl;
      a.download = `AI_Audit_Report_${String(scanId).padStart(6, "0")}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    }
    setTimeout(() => URL.revokeObjectURL(blobUrl), 60000);
  } catch (err) {
    showToast("Lỗi xuất báo cáo PDF: " + err.message, "error");
  }
}

// Export PDF Button
btnExportPdf.addEventListener("click", () => {
  if (!currentScanId) {
    showToast("Chưa có lượt quét nào để xuất báo cáo PDF.", "warning");
    return;
  }
  openAuthenticatedPdf(currentScanId);
});


// ==============================================================================
// 5. QUẢN LÝ LỊCH SỬ QUÉT, TÌM KIẾM ĐA TIÊU CHÍ & KPIS (TAB 2)
// ==============================================================================

let historyFilterTimer = null;
let historyAbortCtrl = null;

function initHistoryFilters() {
  if (histSearchInput) {
    histSearchInput.addEventListener("input", () => {
      clearTimeout(historyFilterTimer);
      historyFilterTimer = setTimeout(() => {
        loadHistoryAndStats();
      }, 280);
    });
  }
  [histFilterLang, histFilterVerdict, histFilterBypass].forEach((el) => {
    if (el) {
      el.addEventListener("change", () => loadHistoryAndStats());
    }
  });
}

function hasActiveHistoryFilter() {
  return Boolean(
    (histSearchInput && histSearchInput.value.trim()) ||
    (histFilterLang && histFilterLang.value) ||
    (histFilterVerdict && histFilterVerdict.value) ||
    (histFilterBypass && histFilterBypass.checked)
  );
}

async function loadHistoryAndStats() {
  if (historyAbortCtrl) {
    historyAbortCtrl.abort();
  }
  historyAbortCtrl = new AbortController();
  const { signal } = historyAbortCtrl;

  try {
    // 1. Thống kê KPI & 2. Bảng dữ liệu có lọc đa tiêu chí (chạy song song, tự hủy request cũ tránh race condition)
    const params = new URLSearchParams({ limit: "50" });
    const qVal = histSearchInput ? histSearchInput.value.trim() : "";
    const langVal = histFilterLang ? histFilterLang.value : "";
    const verdictVal = histFilterVerdict ? histFilterVerdict.value : "";
    const bypassVal = histFilterBypass ? histFilterBypass.checked : false;

    if (qVal) params.set("q", qVal);
    if (langVal) params.set("language", langVal);
    if (verdictVal) params.set("verdict", verdictVal);
    if (bypassVal) params.set("bypass_only", "true");

    const [statsRes, histRes] = await Promise.all([
      fetch(`${API_BASE}/api/v1/stats`, { headers: getAuthHeaders(), signal }),
      fetch(`${API_BASE}/api/v1/history?${params.toString()}`, { headers: getAuthHeaders(), signal }),
    ]);

    if (statsRes.ok) {
      const stats = await statsRes.json();
      kpiTotalScans.textContent = stats.total_scans.toLocaleString();
      kpiAiRate.textContent = `${stats.ai_percentage}%`;
      kpiTotalWords.textContent = stats.total_words_scanned.toLocaleString();
      kpiAvgLatency.textContent = `${stats.avg_inference_time_ms} ms`;
    }

    if (histRes.ok) {
      const data = await histRes.json();
      renderHistoryTable(data.history || [], data.current_user);
    }
  } catch (err) {
    if (err && err.name === "AbortError") return;
    console.error("Lỗi nạp lịch sử:", err);
  }
}

btnRefreshHistory.addEventListener("click", loadHistoryAndStats);

function renderHistoryTable(items, user) {
  historyTableBody.innerHTML = "";
  if (!items || items.length === 0) {
    const isFiltered = hasActiveHistoryFilter();
    const isStudent = user && user.role === "student";
    const msg = isFiltered
      ? "Không tìm thấy lượt thẩm định nào khớp với bộ lọc hiện tại."
      : (isStudent
        ? "Bạn chưa có lượt thẩm định nào. Hãy nhập văn bản ở tab 'Thẩm Định Văn Bản' để kiểm tra!"
        : "Chưa có lượt thẩm định nào được lưu. Hãy thực hiện kiểm tra ở Tab Thẩm Định!");
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
    const isAi = item.verdict ? item.verdict === "AI-generated" : item.ai_probability >= 0.5;
    const badgeClass = isAi ? "badge-ai-table" : "badge-human-table";
    const badgeText = isAi ? "🤖 AI-Generated" : "👤 Human";

    const langBadge = item.language === "vi" ? "🇻🇳 VI" : "🇬🇧 EN";
    const bypassTag = item.has_bypass_warning ? `<span class="badge-bypass-table" title="Phát hiện ký tự ẩn / Cyrillic giả mạo">⚠️ Bypass</span>` : "";
    const auditTag = item.audit_code
      ? `<span class="audit-mini-tag" title="Nhấp để kiểm chứng chống giả mạo mã ${escapeHtml(item.audit_code)}" onclick="openVerifyWithCode('${escapeHtml(item.audit_code)}')">🔐 ${escapeHtml(item.audit_code)}</span>`
      : "";

    tr.innerHTML = `
      <td><b>#${item.id}</b></td>
      <td style="color: var(--text-sub); font-size: 12px;">${item.timestamp}</td>
      <td>
        <b>${escapeHtml(item.title || item.filename || "Văn bản")}</b>${bypassTag}
        <div style="font-size: 11px; color: var(--text-muted); margin-top: 3px;">
          ${langBadge} • 👤 Người nộp: <b>${escapeHtml(item.author_name || "Khách")}</b> ${auditTag}
        </div>
      </td>
      <td>${item.word_count} từ</td>
      <td><span class="data-badge ${badgeClass}">${badgeText}</span></td>
      <td><b style="font-family: var(--font-mono);">${(item.ai_probability * 100).toFixed(1)}%</b></td>
      <td style="color: var(--text-sub);">${item.inference_time_ms} ms</td>
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
  openAuthenticatedPdf(id);
};

window.viewHistoryItem = async function(id) {
  try {
    const res = await fetch(`${API_BASE}/api/v1/history/${id}`, {
      headers: getAuthHeaders(),
    });
    if (res.ok) {
      const data = await res.json();
      currentScanId = id;
      textInput.value = data.full_text || data.text_snippet || "";
      updateTextCounts();

      const resolvedLabel = data.label || data.verdict;
      const isVi = data.language === "vi";
      const savedCoverage = (data.details && data.details.text_stats && data.details.text_stats.context_coverage)
        || (isVi ? "256 Tokens (PhoBERT)" : "1024 Tokens");

      // Render lại kết quả trên tab inspect (bao gồm Audit Code, XAI & Anti-Bypass nếu có)
      renderResults({
        scan_id: id,
        audit_code: data.audit_code || null,
        overall: {
          label: resolvedLabel,
          language: data.language || "en",
          ai_probability: data.ai_probability,
          human_probability: data.human_probability ?? (1.0 - data.ai_probability),
          confidence: data.confidence,
          inference_time_ms: data.inference_time_ms,
          model_breakdown: data.details ? data.details.model_breakdown : null,
          text_stats: {
            word_count: data.word_count,
            char_count: data.char_count,
            context_coverage: savedCoverage,
          },
        },
        authorship_tier: data.authorship_tier || (data.details ? data.details.authorship_tier : null),
        bypass_report: data.bypass_report || (data.details ? data.details.bypass_report : null),
        xai_metrics: data.xai_metrics || (data.details ? data.details.xai_metrics : null),
        sentences: data.details ? data.details.sentences : [],
        total_processing_time_ms: data.inference_time_ms,
      });

      // Switch to Inspect Tab
      document.querySelector('[data-tab="tab-inspect"]').click();
    } else {
      const err = await res.json().catch(() => ({}));
      showToast(err.detail || "Không thể tải chi tiết lượt quét này.", "error");
    }
  } catch (err) {
    showToast("Không thể tải chi tiết: " + err.message, "error");
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
      showToast(`Đã xóa lượt quét #${id}.`, "success", 2500);
      loadHistoryAndStats();
    }
  } catch (err) {
    showToast("Lỗi xóa mục: " + err.message, "error");
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
      const confirmPassword = document.getElementById("reg-password-confirm")?.value || "";
      const role = document.querySelector('input[name="reg-role"]:checked')?.value || "student";
      if (password !== confirmPassword) {
        showAuthAlert("Mật khẩu xác nhận không khớp! Vui lòng nhập lại.");
        return;
      }
      await handleRegister(fullName, email, password, confirmPassword, role);
    });
  }

  // Toggle ẩn/hiện mật khẩu cho tất cả các ô password
  document.querySelectorAll(".btn-toggle-pwd").forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetId = btn.dataset.target;
      const input = document.getElementById(targetId);
      if (!input) return;
      const isHidden = input.type === "password";
      input.type = isHidden ? "text" : "password";
      btn.textContent = isHidden ? "🙈" : "👁️";
    });
  });

  // Thanh đo độ mạnh mật khẩu trực tiếp khi đăng ký
  const regPwdInput = document.getElementById("reg-password");
  const regPwdBar = document.getElementById("reg-pwd-strength-bar");
  const regPwdText = document.getElementById("reg-pwd-strength-text");
  if (regPwdInput && regPwdBar && regPwdText) {
    regPwdInput.addEventListener("input", () => {
      const v = regPwdInput.value;
      let score = 0;
      if (v.length >= 6) score += 1;
      if (v.length >= 10) score += 1;
      if (/[A-Z]/.test(v) && /[a-z]/.test(v)) score += 1;
      if (/\d/.test(v)) score += 1;
      if (/[^A-Za-z0-9]/.test(v)) score += 1;

      if (!v) {
        regPwdBar.style.width = "0%";
        regPwdText.textContent = "Độ mạnh mật khẩu";
      } else if (score <= 2) {
        regPwdBar.style.width = "33%";
        regPwdBar.style.backgroundColor = "#EF4444";
        regPwdText.textContent = "Yếu";
      } else if (score <= 3) {
        regPwdBar.style.width = "66%";
        regPwdBar.style.backgroundColor = "#F59E0B";
        regPwdText.textContent = "Trung bình";
      } else {
        regPwdBar.style.width = "100%";
        regPwdBar.style.backgroundColor = "#10B981";
        regPwdText.textContent = "Mạnh ✓";
      }
    });
  }

  if (btnLogout) {
    btnLogout.addEventListener("click", handleLogout);
  }

  // Khởi tạo Google Sign-In (OAuth 2.0 / GIS) & Profile Modal
  initGoogleSignIn();
  initProfileModal();
}

async function initGoogleSignIn() {
  const gisContainer = document.getElementById("google-gis-button");
  const fallbackBtn = document.getElementById("btn-google-fallback");
  const setupHint = document.getElementById("google-setup-hint");

  if (fallbackBtn && setupHint) {
    fallbackBtn.addEventListener("click", () => {
      setupHint.classList.toggle("hidden");
    });
  }

  try {
    const res = await fetch(`${API_BASE}/api/v1/auth/config`);
    if (!res.ok) return;
    const cfg = await res.json();
    if (!cfg.google_enabled || !cfg.google_client_id) return;

    const mountGoogleButton = () => {
      if (!window.google || !window.google.accounts || !window.google.accounts.id) return false;
      window.google.accounts.id.initialize({
        client_id: cfg.google_client_id,
        callback: handleGoogleCredentialResponse,
      });
      if (gisContainer) {
        gisContainer.innerHTML = "";
        window.google.accounts.id.renderButton(gisContainer, {
          theme: document.documentElement.getAttribute("data-theme") === "dark" ? "filled_black" : "outline",
          size: "large",
          width: 360,
          text: "continue_with",
          locale: "vi",
        });
        gisContainer.classList.remove("hidden");
      }
      if (fallbackBtn) fallbackBtn.classList.add("hidden");
      if (setupHint) setupHint.classList.add("hidden");
      return true;
    };

    if (!mountGoogleButton()) {
      let tries = 0;
      const poll = setInterval(() => {
        tries += 1;
        if (mountGoogleButton() || tries >= 20) clearInterval(poll);
      }, 300);
    }
  } catch (e) {}
}

async function handleGoogleCredentialResponse(response) {
  if (!response || !response.credential) {
    showAuthAlert("Không nhận được mã xác thực từ Google.");
    return;
  }
  clearAuthAlert();
  const role = document.querySelector('input[name="reg-role"]:checked')?.value || "student";
  try {
    const res = await fetch(`${API_BASE}/api/v1/auth/google`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ credential: response.credential, role }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Đăng nhập Google thất bại.");

    localStorage.setItem("veritext_token", data.access_token);
    localStorage.setItem("veritext_user", JSON.stringify(data.user));
    currentUser = data.user;
    updateAuthUI(currentUser);
    showAuthAlert("Đăng nhập bằng Google thành công!", true);
    setTimeout(() => {
      hideAuthModal();
      loadHistoryAndStats();
    }, 550);
  } catch (err) {
    showAuthAlert(err.message || "Lỗi đăng nhập Google.");
  }
}

function initProfileModal() {
  const btnOpenProfile = document.getElementById("btn-open-profile");
  const profileModal = document.getElementById("profile-modal");
  const btnCloseProfile = document.getElementById("btn-close-profile-modal");
  const tabProfInfo = document.getElementById("tab-btn-prof-info");
  const tabProfPwd = document.getElementById("tab-btn-prof-pwd");
  const formProfInfo = document.getElementById("form-profile-info");
  const formProfPwd = document.getElementById("form-profile-pwd");
  const profAlert = document.getElementById("profile-alert");

  function showProfAlert(msg, isSuccess = false) {
    if (!profAlert) return;
    profAlert.textContent = msg;
    profAlert.className = "auth-alert " + (isSuccess ? "auth-alert-success" : "auth-alert-error");
    profAlert.classList.remove("hidden");
  }

  function clearProfAlert() {
    if (profAlert) profAlert.classList.add("hidden");
  }

  function switchProfTab(tab) {
    clearProfAlert();
    if (tab === "info") {
      tabProfInfo?.classList.add("active");
      tabProfPwd?.classList.remove("active");
      formProfInfo?.classList.remove("hidden");
      formProfPwd?.classList.add("hidden");
    } else {
      tabProfInfo?.classList.remove("active");
      tabProfPwd?.classList.add("active");
      formProfInfo?.classList.add("hidden");
      formProfPwd?.classList.remove("hidden");
    }
  }

  function openProfileModal() {
    if (!currentUser || !profileModal) return;
    clearProfAlert();
    switchProfTab("info");

    const avatarBox = document.getElementById("profile-avatar-box");
    const nameEl = document.getElementById("profile-summary-name");
    const emailEl = document.getElementById("profile-summary-email");
    const badgesEl = document.getElementById("profile-summary-badges");
    const tsEl = document.getElementById("profile-timestamps");
    const oldPwdGroup = document.getElementById("group-old-password");
    const oldPwdInput = document.getElementById("prof-old-pwd");

    if (avatarBox) {
      if (currentUser.avatar_url) {
        avatarBox.innerHTML = `<img src="${escapeHtml(currentUser.avatar_url)}" alt="Avatar" referrerpolicy="no-referrer">`;
      } else {
        const initials = (currentUser.full_name || currentUser.email || "U").trim().charAt(0).toUpperCase();
        avatarBox.textContent = initials;
      }
    }
    if (nameEl) nameEl.textContent = currentUser.full_name || currentUser.email;
    if (emailEl) emailEl.textContent = currentUser.email || "";
    if (badgesEl) {
      const roleLabel = currentUser.role === "teacher" ? "👨‍🏫 Giảng viên" : "🎓 Sinh viên";
      const providerLabel = currentUser.is_google_linked || currentUser.auth_provider === "google" ? "🔵 Google OAuth 2.0" : "🔑 Email / Mật khẩu";
      badgesEl.innerHTML = `<span class="data-badge badge-human-table">${roleLabel}</span> <span class="data-badge" style="background: var(--bg-card); border: 1px solid var(--border-light);">${providerLabel}</span>`;
    }
    if (tsEl) {
      tsEl.textContent = `Ngày tạo: ${currentUser.created_at || "--"} • Đăng nhập gần nhất: ${currentUser.last_login || "Vừa xong"}`;
    }

    const inputFullname = document.getElementById("prof-fullname");
    const selectRole = document.getElementById("prof-role");
    if (inputFullname) inputFullname.value = currentUser.full_name || "";
    if (selectRole) selectRole.value = currentUser.role || "student";

    const isPureGoogle = currentUser.auth_provider === "google";
    if (oldPwdGroup) oldPwdGroup.classList.toggle("hidden", isPureGoogle);
    if (oldPwdInput) oldPwdInput.required = !isPureGoogle;

    profileModal.classList.remove("hidden");
  }

  function closeProfileModal() {
    if (profileModal) profileModal.classList.add("hidden");
  }

  if (btnOpenProfile) btnOpenProfile.addEventListener("click", openProfileModal);
  if (btnCloseProfile) btnCloseProfile.addEventListener("click", closeProfileModal);
  if (profileModal) {
    profileModal.addEventListener("click", (e) => {
      if (e.target === profileModal) closeProfileModal();
    });
  }
  if (tabProfInfo) tabProfInfo.addEventListener("click", () => switchProfTab("info"));
  if (tabProfPwd) tabProfPwd.addEventListener("click", () => switchProfTab("pwd"));

  if (formProfInfo) {
    formProfInfo.addEventListener("submit", async (e) => {
      e.preventDefault();
      const fullName = document.getElementById("prof-fullname").value.trim();
      const role = document.getElementById("prof-role").value;
      clearProfAlert();
      try {
        const res = await fetch(`${API_BASE}/api/v1/auth/profile`, {
          method: "PUT",
          headers: { "Content-Type": "application/json", ...getAuthHeaders() },
          body: JSON.stringify({ full_name: fullName, role }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Không thể cập nhật hồ sơ.");
        localStorage.setItem("veritext_token", data.access_token);
        localStorage.setItem("veritext_user", JSON.stringify(data.user));
        currentUser = data.user;
        updateAuthUI(currentUser);
        showProfAlert("Đã cập nhật hồ sơ thành công!", true);
        loadHistoryAndStats();
      } catch (err) {
        showProfAlert(err.message);
      }
    });
  }

  if (formProfPwd) {
    formProfPwd.addEventListener("submit", async (e) => {
      e.preventDefault();
      const oldPwd = document.getElementById("prof-old-pwd")?.value || "";
      const newPwd = document.getElementById("prof-new-pwd").value;
      const confirmPwd = document.getElementById("prof-new-pwd-confirm").value;
      clearProfAlert();
      if (newPwd !== confirmPwd) {
        showProfAlert("Mật khẩu mới xác nhận không khớp!");
        return;
      }
      try {
        const res = await fetch(`${API_BASE}/api/v1/auth/change-password`, {
          method: "POST",
          headers: { "Content-Type": "application/json", ...getAuthHeaders() },
          body: JSON.stringify({ old_password: oldPwd || null, new_password: newPwd }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Đổi mật khẩu thất bại.");
        formProfPwd.reset();
        if (currentUser && currentUser.auth_provider === "google") {
          currentUser.auth_provider = "hybrid";
          localStorage.setItem("veritext_user", JSON.stringify(currentUser));
          const oldPwdGroup = document.getElementById("group-old-password");
          const oldPwdInput = document.getElementById("prof-old-pwd");
          if (oldPwdGroup) oldPwdGroup.classList.remove("hidden");
          if (oldPwdInput) oldPwdInput.required = true;
        }
        showProfAlert("Đổi mật khẩu thành công!", true);
      } catch (err) {
        showProfAlert(err.message);
      }
    });
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

async function handleRegister(fullName, email, password, confirmPassword, role) {
  const submitBtn = document.getElementById("btn-submit-register");
  if (submitBtn) submitBtn.disabled = true;
  clearAuthAlert();

  try {
    const res = await fetch(`${API_BASE}/api/v1/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        full_name: fullName,
        email,
        password,
        confirm_password: confirmPassword,
        role,
      }),
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
  const avatarImg = document.getElementById("user-avatar-img");
  if (user) {
    if (btnOpenAuth) btnOpenAuth.classList.add("hidden");
    if (userProfileBox) userProfileBox.classList.remove("hidden");
    const isTeacher = user.role === "teacher";
    if (avatarImg) {
      if (user.avatar_url) {
        avatarImg.src = user.avatar_url;
        avatarImg.classList.remove("hidden");
      } else {
        avatarImg.classList.add("hidden");
      }
    }
    if (userRoleBadge) {
      userRoleBadge.textContent = isTeacher ? "👨‍🏫" : "🎓";
      userRoleBadge.title = isTeacher ? "Giảng viên" : "Sinh viên";
    }
    if (userDisplayName) {
      userDisplayName.textContent = user.full_name || user.email;
      userDisplayName.title = `${user.full_name} (${isTeacher ? "Giảng viên" : "Sinh viên"}) • Nhấn để xem hồ sơ`;
    }
  } else {
    if (btnOpenAuth) btnOpenAuth.classList.remove("hidden");
    if (userProfileBox) userProfileBox.classList.add("hidden");
    if (avatarImg) avatarImg.classList.add("hidden");
  }
  syncClassroomAccessUI();
}


// ==============================================================================
// 7. CHẤM BÀI LỚP HỌC (.ZIP / MULTI-FILE) & XUẤT BẢNG ĐIỂM EXCEL/CSV
// ==============================================================================

let selectedBatchFiles = [];
let activeBatchDetail = null;
let activeBatchRiskFilter = "all";

const classroomTeacherGate = document.getElementById("classroom-teacher-gate");
const classroomWorkspace = document.getElementById("classroom-workspace");
const btnGateTeacherLogin = document.getElementById("btn-gate-teacher-login");
const btnGateOpenModal = document.getElementById("btn-gate-open-modal");

const batchNameInput = document.getElementById("batch-name-input");
const batchPresetSelect = document.getElementById("batch-preset-select");
const batchFilterQuotes = document.getElementById("batch-filter-quotes");
const batchFilterReferences = document.getElementById("batch-filter-references");
const batchDropZone = document.getElementById("batch-drop-zone");
const batchFileInput = document.getElementById("batch-file-input");
const batchSelectedWrap = document.getElementById("batch-selected-wrap");
const batchSelectedCount = document.getElementById("batch-selected-count");
const batchSelectedChips = document.getElementById("batch-selected-chips");
const btnClearBatchFiles = document.getElementById("btn-clear-batch-files");
const btnStartBatchScan = document.getElementById("btn-start-batch-scan");
const batchSpinner = document.getElementById("batch-spinner");

const batchResultPanel = document.getElementById("batch-result-panel");
const batchResultTitle = document.getElementById("batch-result-title");
const batchResultMeta = document.getElementById("batch-result-meta");
const btnExportXlsx = document.getElementById("btn-export-xlsx");
const btnExportCsv = document.getElementById("btn-export-csv");
const bkpiTotal = document.getElementById("bkpi-total");
const bkpiHigh = document.getElementById("bkpi-high");
const bkpiMod = document.getElementById("bkpi-mod");
const bkpiSafe = document.getElementById("bkpi-safe");
const bkpiBypass = document.getElementById("bkpi-bypass");
const batchSkippedBox = document.getElementById("batch-skipped-box");
const batchSkippedText = document.getElementById("batch-skipped-text");
const batchSubmissionsTbody = document.getElementById("batch-submissions-tbody");
const batchRiskFilters = document.querySelectorAll(".batch-risk-filter");
const savedBatchesList = document.getElementById("saved-batches-list");
const btnRefreshBatches = document.getElementById("btn-refresh-batches");

const TIER_SHORT_LABELS = {
  AI_PURE: "🤖 AI toàn phần",
  HYBRID_MIXED: "🧩 Văn bản lai",
  AI_ASSISTED: "✍️ AI biên tập",
  HUMAN_PURE: "👤 Tự viết",
};

function isTeacherRole(user) {
  return Boolean(user && (user.role === "teacher" || user.role === "admin"));
}

function syncClassroomAccessUI() {
  const isTeacher = isTeacherRole(currentUser);
  if (classroomTeacherGate) classroomTeacherGate.classList.toggle("hidden", isTeacher);
  if (classroomWorkspace) classroomWorkspace.classList.toggle("hidden", !isTeacher);
  if (isTeacher) {
    loadSavedBatches();
  } else {
    activeBatchDetail = null;
    if (batchResultPanel) batchResultPanel.classList.add("hidden");
  }
}

function initClassroomBatchUI() {
  if (btnGateTeacherLogin) {
    btnGateTeacherLogin.addEventListener("click", async () => {
      await handleLogin("giangvien@veritext.ai", "123456");
      syncClassroomAccessUI();
    });
  }
  if (btnGateOpenModal) {
    btnGateOpenModal.addEventListener("click", () => showAuthModal("login"));
  }

  if (batchDropZone && batchFileInput) {
    batchDropZone.addEventListener("click", () => batchFileInput.click());
    batchDropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      batchDropZone.classList.add("dragover");
    });
    batchDropZone.addEventListener("dragleave", () => {
      batchDropZone.classList.remove("dragover");
    });
    batchDropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      batchDropZone.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        addBatchFiles(Array.from(e.dataTransfer.files));
      }
    });
    batchFileInput.addEventListener("change", () => {
      if (batchFileInput.files && batchFileInput.files.length > 0) {
        addBatchFiles(Array.from(batchFileInput.files));
        batchFileInput.value = "";
      }
    });
  }

  if (btnClearBatchFiles) {
    btnClearBatchFiles.addEventListener("click", () => {
      selectedBatchFiles = [];
      renderSelectedBatchFiles();
    });
  }

  if (btnStartBatchScan) {
    btnStartBatchScan.addEventListener("click", runClassroomBatchScan);
  }

  if (btnExportXlsx) {
    btnExportXlsx.addEventListener("click", () => {
      if (activeBatchDetail) downloadBatchExport(activeBatchDetail.id, "xlsx");
    });
  }
  if (btnExportCsv) {
    btnExportCsv.addEventListener("click", () => {
      if (activeBatchDetail) downloadBatchExport(activeBatchDetail.id, "csv");
    });
  }

  batchRiskFilters.forEach((btn) => {
    btn.addEventListener("click", () => {
      activeBatchRiskFilter = btn.dataset.risk || "all";
      batchRiskFilters.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      if (activeBatchDetail) renderBatchSubmissionsTable(activeBatchDetail.submissions || []);
    });
  });

  if (btnRefreshBatches) {
    btnRefreshBatches.addEventListener("click", loadSavedBatches);
  }
}

function addBatchFiles(fileList) {
  const allowed = ["zip", "docx", "doc", "pdf", "txt"];
  fileList.forEach((f) => {
    const ext = (f.name.split(".").pop() || "").toLowerCase();
    if (!allowed.includes(ext)) return;
    if (!selectedBatchFiles.some((x) => x.name === f.name && x.size === f.size)) {
      selectedBatchFiles.push(f);
    }
  });
  renderSelectedBatchFiles();
}

function renderSelectedBatchFiles() {
  if (!batchSelectedWrap || !batchSelectedChips) return;
  if (selectedBatchFiles.length === 0) {
    batchSelectedWrap.classList.add("hidden");
    batchSelectedChips.innerHTML = "";
    return;
  }
  batchSelectedWrap.classList.remove("hidden");
  if (batchSelectedCount) {
    batchSelectedCount.textContent = `Đã chọn ${selectedBatchFiles.length} tệp sẵn sàng chấm`;
  }
  batchSelectedChips.innerHTML = "";
  selectedBatchFiles.forEach((f) => {
    const chip = document.createElement("span");
    chip.className = "batch-file-chip";
    const icon = f.name.toLowerCase().endsWith(".zip") ? "📦" : "📄";
    chip.textContent = `${icon} ${f.name} (${(f.size / 1024).toFixed(1)} KB)`;
    batchSelectedChips.appendChild(chip);
  });
}

async function runClassroomBatchScan() {
  if (!isTeacherRole(currentUser)) {
    showToast("Vui lòng đăng nhập tài khoản Giảng viên để chấm bài lớp học.", "warning");
    return;
  }
  if (selectedBatchFiles.length === 0) {
    showToast("Vui lòng kéo thả tệp .ZIP hoặc chọn ít nhất 1 tệp bài nộp của sinh viên!", "warning");
    return;
  }

  const preset = batchPresetSelect ? batchPresetSelect.value : "balanced";
  const presetThresholds = { strict: 0.3, balanced: 0.5, lenient: 0.7 };
  const formData = new FormData();
  formData.append("batch_name", (batchNameInput && batchNameInput.value.trim()) || "Đợt chấm bài lớp học");
  formData.append("threshold", presetThresholds[preset] || 0.5);
  formData.append("strictness_preset", preset);
  formData.append("filter_quotes", batchFilterQuotes ? batchFilterQuotes.checked : false);
  formData.append("filter_references", batchFilterReferences ? batchFilterReferences.checked : true);
  selectedBatchFiles.forEach((file) => formData.append("files", file));

  btnStartBatchScan.disabled = true;
  const txtEl = btnStartBatchScan.querySelector(".btn-text");
  if (txtEl) txtEl.textContent = "Đang giải nén & chấm toàn lớp...";
  if (batchSpinner) batchSpinner.classList.remove("hidden");

  try {
    const res = await fetch(`${API_BASE}/api/v1/classroom/batch-scan`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: formData,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.detail || `Lỗi HTTP ${res.status}`);
    }

    selectedBatchFiles = [];
    renderSelectedBatchFiles();
    renderBatchDetail(data.batch, data.skipped_files || []);
    loadSavedBatches();
    showToast(`Hoàn tất chấm lớp: ${data.batch.total_files} bài nộp!`, "success", 3600);
  } catch (err) {
    showToast("Lỗi chấm bài lớp học: " + err.message, "error", 5000);
  } finally {
    btnStartBatchScan.disabled = false;
    if (txtEl) txtEl.textContent = "🚀 Bắt Đầu Chấm Cả Lớp";
    if (batchSpinner) batchSpinner.classList.add("hidden");
  }
}

function renderBatchDetail(batch, skippedFiles = []) {
  if (!batch || !batchResultPanel) return;
  activeBatchDetail = batch;
  activeBatchRiskFilter = "all";
  batchRiskFilters.forEach((b) => b.classList.toggle("active", b.dataset.risk === "all"));

  batchResultPanel.classList.remove("hidden");
  if (batchResultTitle) batchResultTitle.textContent = `📊 ${batch.batch_name}`;
  if (batchResultMeta) {
    batchResultMeta.textContent = `Mã đợt #${batch.id} • Giảng viên: ${batch.teacher_name} • ${batch.created_at} • AI trung bình lớp: ${(batch.avg_ai_probability * 100).toFixed(1)}%`;
  }

  if (bkpiTotal) bkpiTotal.textContent = batch.total_files;
  if (bkpiHigh) bkpiHigh.textContent = batch.high_risk_count;
  if (bkpiMod) bkpiMod.textContent = batch.moderate_risk_count;
  if (bkpiSafe) bkpiSafe.textContent = batch.safe_count;
  if (bkpiBypass) bkpiBypass.textContent = batch.bypass_count || 0;

  if (batchSkippedBox && batchSkippedText) {
    if (skippedFiles && skippedFiles.length > 0) {
      batchSkippedBox.classList.remove("hidden");
      batchSkippedText.textContent = `Đã bỏ qua ${skippedFiles.length} tệp không hợp lệ: ` +
        skippedFiles.slice(0, 4).map((s) => `${s.filename} (${s.reason})`).join(" • ");
    } else {
      batchSkippedBox.classList.add("hidden");
    }
  }

  renderBatchSubmissionsTable(batch.submissions || []);
  batchResultPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderBatchSubmissionsTable(submissions) {
  if (!batchSubmissionsTbody) return;
  const filtered = submissions.filter((s) => {
    if (activeBatchRiskFilter === "all") return true;
    if (activeBatchRiskFilter === "bypass") return !!s.has_bypass_warning;
    return s.risk_level === activeBatchRiskFilter;
  });

  batchSubmissionsTbody.innerHTML = "";
  if (filtered.length === 0) {
    batchSubmissionsTbody.innerHTML = `<tr><td colspan="8" class="text-center" style="padding: 20px; color: var(--text-muted);">Không có bài nộp nào thuộc nhóm lọc này.</td></tr>`;
    return;
  }

  filtered.forEach((s, idx) => {
    const tr = document.createElement("tr");
    const riskClass = s.risk_level === "high" ? "risk-high" : s.risk_level === "moderate" ? "risk-moderate" : "risk-safe";
    const riskText = s.risk_level === "high" ? "🔴 Nguy cơ cao" : s.risk_level === "moderate" ? "🟡 Cần xem xét" : "🟢 An toàn";
    const langTag = s.language === "vi" ? "🇻🇳 VI" : "🇬🇧 EN";
    const bypassBadge = s.has_bypass_warning ? `<span class="badge-bypass-table">⚠️ Bypass</span>` : "";
    const tierText = TIER_SHORT_LABELS[s.authorship_tier] || s.verdict;
    const burstStr = s.burstiness_score !== null && s.burstiness_score !== undefined ? `${s.burstiness_score}/100` : "--";

    tr.innerHTML = `
      <td><b>${idx + 1}</b></td>
      <td>
        <b>${escapeHtml(s.student_label || s.title)}</b>${bypassBadge}
        <div style="font-size: 11px; color: var(--text-muted);">${s.word_count} từ • ${escapeHtml(s.title)}</div>
      </td>
      <td>${langTag}</td>
      <td><b style="font-family: var(--font-mono);">${(s.ai_probability * 100).toFixed(1)}%</b></td>
      <td><span class="risk-pill ${riskClass}">${riskText}</span></td>
      <td style="font-size: 12px;">${escapeHtml(tierText)}</td>
      <td style="font-family: var(--font-mono); font-size: 12px;">${burstStr}</td>
      <td style="text-align: right;">
        <button class="table-btn" onclick="downloadPdf(${s.scan_id})">📥 PDF</button>
        <button class="table-btn" onclick="viewHistoryItem(${s.scan_id})">👁️ Xem</button>
      </td>
    `;
    batchSubmissionsTbody.appendChild(tr);
  });
}

async function loadSavedBatches() {
  if (!savedBatchesList || !isTeacherRole(currentUser)) return;
  try {
    const res = await fetch(`${API_BASE}/api/v1/classroom/batches`, {
      headers: getAuthHeaders(),
    });
    if (!res.ok) return;
    const data = await res.json();
    const batches = data.batches || [];

    if (batches.length === 0) {
      savedBatchesList.innerHTML = `<p style="color: var(--text-muted); font-size: 13px; text-align: center; padding: 18px;">Chưa có đợt chấm bài lớp học nào. Hãy kéo thả tệp .ZIP ở trên để bắt đầu!</p>`;
      return;
    }

    savedBatchesList.innerHTML = "";
    batches.forEach((b) => {
      const row = document.createElement("div");
      row.className = "saved-batch-row";
      row.innerHTML = `
        <div class="saved-batch-info">
          <strong>#${b.id} • ${escapeHtml(b.batch_name)}</strong>
          <div class="saved-batch-stats">
            🕒 ${b.created_at} • 📄 ${b.total_files} bài •
            🔴 ${b.high_risk_count} cao • 🟡 ${b.moderate_risk_count} nghi vấn • 🟢 ${b.safe_count} an toàn
            ${b.bypass_count ? ` • ⚠️ ${b.bypass_count} Bypass` : ""}
          </div>
        </div>
        <div class="saved-batch-actions">
          <button class="table-btn" onclick="openClassroomBatch(${b.id})">👁️ Mở bảng điểm</button>
          <button class="table-btn" onclick="downloadBatchExport(${b.id}, 'xlsx')">📊 Excel</button>
          <button class="table-btn text-danger" onclick="deleteClassroomBatch(${b.id})">🗑️</button>
        </div>
      `;
      savedBatchesList.appendChild(row);
    });
  } catch (err) {
    console.error("Lỗi nạp đợt chấm lớp học:", err);
  }
}

window.openClassroomBatch = async function(batchId) {
  try {
    const res = await fetch(`${API_BASE}/api/v1/classroom/batches/${batchId}`, {
      headers: getAuthHeaders(),
    });
    if (!res.ok) throw new Error("Không thể tải chi tiết đợt chấm.");
    const detail = await res.json();
    renderBatchDetail(detail, []);
  } catch (err) {
    showToast(err.message, "error");
  }
};

window.deleteClassroomBatch = async function(batchId) {
  if (!confirm(`Bạn có chắc muốn xóa đợt chấm lớp học #${batchId}?`)) return;
  try {
    const res = await fetch(`${API_BASE}/api/v1/classroom/batches/${batchId}`, {
      method: "DELETE",
      headers: getAuthHeaders(),
    });
    if (res.ok) {
      if (activeBatchDetail && activeBatchDetail.id === batchId && batchResultPanel) {
        batchResultPanel.classList.add("hidden");
        activeBatchDetail = null;
      }
      showToast(`Đã xóa đợt chấm #${batchId}.`, "success", 2500);
      loadSavedBatches();
    }
  } catch (err) {
    showToast("Lỗi xóa đợt chấm: " + err.message, "error");
  }
};

window.downloadBatchExport = async function(batchId, fmt = "xlsx") {
  try {
    const res = await fetch(`${API_BASE}/api/v1/classroom/batches/${batchId}/export?format=${fmt}`, {
      headers: getAuthHeaders(),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Không thể xuất bảng điểm.");
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `Bang_Diem_Kiem_Dinh_Lop_${String(batchId).padStart(4, "0")}.${fmt}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
    showToast(`Đã tải xuống bảng điểm .${fmt.toUpperCase()} thành công!`, "success", 2800);
  } catch (err) {
    showToast("Lỗi xuất bảng điểm: " + err.message, "error");
  }
};


// ==============================================================================
// 8. CỔNG XÁC THỰC CHỐNG GIẢ MẠO BIÊN BẢN PDF (HMAC-SHA256 + QR CODE)
// ==============================================================================

function initVerifyCertificateModal() {
  if (!verifyModal) return;

  const closeVerify = () => {
    verifyModal.classList.add("hidden");
  };

  if (btnOpenVerify) {
    btnOpenVerify.addEventListener("click", () => {
      verifyModal.classList.remove("hidden");
      if (verifyCodeInput && currentAuditCode && !verifyCodeInput.value.trim()) {
        verifyCodeInput.value = currentAuditCode;
      }
      verifyCodeInput?.focus();
    });
  }

  if (btnCloseVerifyModal) {
    btnCloseVerifyModal.addEventListener("click", closeVerify);
  }

  verifyModal.addEventListener("click", (e) => {
    if (e.target === verifyModal) closeVerify();
  });

  if (formVerifyCert) {
    formVerifyCert.addEventListener("submit", async (e) => {
      e.preventDefault();
      const rawCode = verifyCodeInput ? verifyCodeInput.value.trim() : "";
      if (!rawCode) return;
      await verifyAuditCode(rawCode);
    });
  }

  // Hỗ trợ tự động mở & xác thực khi người dùng quét mã QR trên biên bản PDF (?verify=VT-...)
  try {
    const verifyParam = new URLSearchParams(window.location.search).get("verify");
    if (verifyParam && verifyParam.trim()) {
      window.openVerifyWithCode(verifyParam.trim());
    }
  } catch (e) {}
}

window.openVerifyWithCode = function(code) {
  if (!verifyModal) return;
  verifyModal.classList.remove("hidden");
  if (verifyCodeInput) {
    verifyCodeInput.value = (code || "").trim().toUpperCase();
  }
  if (code && code.trim()) {
    verifyAuditCode(code.trim());
  }
};

async function verifyAuditCode(rawCode) {
  if (!verifyResultBox) return;
  const cleanCode = rawCode.trim().toUpperCase();
  if (btnSubmitVerify) btnSubmitVerify.disabled = true;
  verifyResultBox.className = "verify-result-box";
  verifyResultBox.innerHTML = "<span>⏳ Đang đối chiếu chữ ký mật mã HMAC-SHA256 với SQL Server VeriTextAI...</span>";
  verifyResultBox.classList.remove("hidden");

  try {
    const res = await fetch(`${API_BASE}/api/v1/verify/${encodeURIComponent(cleanCode)}`);
    const data = await res.json();

    if (res.ok && data.valid && data.certificate) {
      const c = data.certificate;
      const aiPct = ((Number(c.ai_probability) || 0) * 100).toFixed(2);
      const isAi = c.verdict === "AI-generated" || Number(c.ai_probability) >= 0.5;
      const verdictStr = isAi ? "🤖 VĂN BẢN DO AI SINH" : "👤 DO CON NGƯỜI VIẾT";
      const langStr = c.language === "vi" ? "🇻🇳 Tiếng Việt (PhoBERT v2)" : "🇬🇧 Tiếng Anh (Ensemble PLMs)";
      const tierStr = TIER_SHORT_LABELS[c.authorship_tier] || c.authorship_tier || "--";
      const bypassStr = c.has_bypass_warning ? "🚨 Có phát hiện tiểu xảo" : "✅ Sạch (Không phát hiện)";

      verifyResultBox.className = "verify-result-box verify-valid";
      verifyResultBox.innerHTML = `
        <div style="font-weight: 700; font-size: 13.5px; margin-bottom: 4px;">
          ✅ BIÊN BẢN KIỂM ĐỊNH THẬT — CHỮ KÝ SỐ HỢP LỆ 100%
        </div>
        <div style="font-size: 12px; opacity: 0.92;">${escapeHtml(data.message || "")}</div>
        <div class="verify-cert-grid">
          <div><span>Mã chứng thực</span><strong style="font-family: var(--font-mono);">${escapeHtml(data.audit_code)}</strong></div>
          <div><span>Thời điểm lưu trữ</span><strong>${escapeHtml(c.timestamp || "--")}</strong></div>
          <div><span>Kết luận gốc</span><strong>${verdictStr} (${aiPct}% AI)</strong></div>
          <div><span>Phân loại 4 mức</span><strong>${escapeHtml(tierStr)}</strong></div>
          <div><span>Ngôn ngữ &amp; Độ dài</span><strong>${langStr} • ${Number(c.word_count || 0).toLocaleString()} từ</strong></div>
          <div><span>Kiểm tra Anti-Bypass</span><strong>${bypassStr}</strong></div>
          <div><span>Tài liệu / Tiêu đề</span><strong>${escapeHtml(c.title || c.filename || "Văn bản trực tiếp")}</strong></div>
          <div><span>Tài khoản thẩm định</span><strong>${escapeHtml(c.author_name || "Khách vãng lai")}</strong></div>
        </div>
      `;
    } else {
      verifyResultBox.className = "verify-result-box verify-invalid";
      verifyResultBox.innerHTML = `
        <div style="font-weight: 700; font-size: 13.5px; margin-bottom: 4px;">
          🚨 KHÔNG HỢP LỆ / CẢNH BÁO NGHI VẤN CHỈNH SỬA GIẢ MẠO
        </div>
        <div>${escapeHtml(data.message || "Không thể xác thực mã kiểm định này.")}</div>
      `;
    }
  } catch (err) {
    verifyResultBox.className = "verify-result-box verify-invalid";
    verifyResultBox.innerHTML = `<strong>⚠️ Lỗi kết nối máy chủ:</strong> ${escapeHtml(err.message || "Không thể kiểm tra.")}`;
  } finally {
    if (btnSubmitVerify) btnSubmitVerify.disabled = false;
  }
}
