/**
 * VeriText AI PRO - Chrome Extension Popup Logic (Manifest v3)
 * - Đồng bộ tài khoản JWT + SQL Server qua chrome.storage.local
 * - Quét Toàn Trang Web Hiện Tại (LMS / Google Classroom / Blog)
 * - Hiển thị Anti-Bypass, Phân loại 4 cấp độ & Bằng chứng XAI rút gọn
 */

const API_BASE_URL = "http://localhost:8000";

// Core DOM Elements
const textInput = document.getElementById("text-input");
const wordCountEl = document.getElementById("word-count");
const thresholdSlider = document.getElementById("threshold-slider");
const thresholdValEl = document.getElementById("threshold-val");
const chkFilterQuotes = document.getElementById("chk-filter-quotes");
const chkFilterReferences = document.getElementById("chk-filter-references");
const btnPaste = document.getElementById("btn-paste");
const btnClear = document.getElementById("btn-clear");
const btnAnalyze = document.getElementById("btn-analyze");
const btnScanPage = document.getElementById("btn-scan-page");
const pageSourceBanner = document.getElementById("page-source-banner");
const pageSourceText = document.getElementById("page-source-text");
const btnClearSource = document.getElementById("btn-clear-source");
const spinner = document.getElementById("spinner");
const apiStatusEl = document.getElementById("api-status");
const statusTextEl = document.getElementById("status-text");
const errorBox = document.getElementById("error-box");

// Auth DOM Elements
const authGuestView = document.getElementById("auth-guest-view");
const authUserView = document.getElementById("auth-user-view");
const btnToggleLogin = document.getElementById("btn-toggle-login");
const btnCloseLogin = document.getElementById("btn-close-login");
const loginDrawer = document.getElementById("login-drawer");
const loginEmail = document.getElementById("login-email");
const loginPassword = document.getElementById("login-password");
const loginError = document.getElementById("login-error");
const btnLoginSubmit = document.getElementById("btn-login-submit");
const btnDemoTeacher = document.getElementById("btn-demo-teacher");
const linkOpenWebAuth = document.getElementById("link-open-web-auth");
const authAvatar = document.getElementById("auth-avatar");
const authRoleIcon = document.getElementById("auth-role-icon");
const authUserName = document.getElementById("auth-user-name");
const authUserSub = document.getElementById("auth-user-sub");
const btnOpenDashboard = document.getElementById("btn-open-dashboard");
const btnLogout = document.getElementById("btn-logout");

// Results DOM Elements
const resultsCard = document.getElementById("results-card");
const bypassBanner = document.getElementById("bypass-banner");
const bypassSummary = document.getElementById("bypass-summary");
const verdictBadge = document.getElementById("verdict-badge");
const verdictIcon = document.getElementById("verdict-icon");
const verdictTitle = document.getElementById("verdict-title");
const tierBadge = document.getElementById("tier-badge");
const inferenceTimeEl = document.getElementById("inference-time");
const aiPercentageText = document.getElementById("ai-percentage-text");
const gaugeBarFill = document.getElementById("gauge-bar-fill");
const humanPercentageText = document.getElementById("human-percentage-text");
const confidenceText = document.getElementById("confidence-text");
const xaiStrip = document.getElementById("xai-strip");
const xaiBurstiness = document.getElementById("xai-burstiness");
const xaiMattr = document.getElementById("xai-mattr");
const xaiPerplexity = document.getElementById("xai-perplexity");
const xaiCliches = document.getElementById("xai-cliches");
const scoreLongformer = document.getElementById("score-longformer");
const scoreBert = document.getElementById("score-bert");
const scoreRoberta = document.getElementById("score-roberta");
const syncResultBar = document.getElementById("sync-result-bar");
const syncResultText = document.getElementById("sync-result-text");
const linkViewHistory = document.getElementById("link-view-history");
const btnToggleSentences = document.getElementById("btn-toggle-sentences");
const sentencesList = document.getElementById("sentences-list");

// State
let currentSentenceData = [];
let currentAuthToken = null;
let currentAuthUser = null;
let activePageTitle = null;

// ==============================================================================
// 1. KHỞI TẠO, NẠP CÀI ĐẶT & ĐỒNG BỘ PHIÊN ĐĂNG NHẬP
// ==============================================================================

function storageGet(keys) {
  return new Promise((resolve) => {
    if (chrome.storage && chrome.storage.local) {
      chrome.storage.local.get(keys, (res) => resolve(res || {}));
    } else {
      resolve({});
    }
  });
}

function storageSet(obj) {
  return new Promise((resolve) => {
    if (chrome.storage && chrome.storage.local) {
      chrome.storage.local.set(obj, () => resolve());
    } else {
      resolve();
    }
  });
}

function storageRemove(keys) {
  return new Promise((resolve) => {
    if (chrome.storage && chrome.storage.local) {
      chrome.storage.local.remove(keys, () => resolve());
    } else {
      resolve();
    }
  });
}

document.addEventListener("DOMContentLoaded", async () => {
  const saved = await storageGet([
    "threshold",
    "filter_quotes",
    "filter_references",
    "auth_token",
    "auth_user",
  ]);

  if (typeof saved.threshold === "number") {
    thresholdSlider.value = Math.round(saved.threshold * 100);
    thresholdValEl.textContent = `${thresholdSlider.value}%`;
  }
  if (typeof saved.filter_quotes === "boolean") {
    chkFilterQuotes.checked = saved.filter_quotes;
  }
  if (typeof saved.filter_references === "boolean") {
    chkFilterReferences.checked = saved.filter_references;
  }

  currentAuthToken = saved.auth_token || null;
  currentAuthUser = saved.auth_user || null;
  renderAuthUI();

  updateCounts();
  const isOnline = await checkApiHealth();
  if (isOnline && currentAuthToken) {
    await verifyStoredSession();
  }

  // Nhấp vào badge trạng thái API để kết nối lại nhanh mà không cần đóng/mở popup
  apiStatusEl.style.cursor = "pointer";
  apiStatusEl.title = "Nhấp để kiểm tra lại kết nối Backend FastAPI";
  apiStatusEl.addEventListener("click", async () => {
    statusTextEl.textContent = "Đang kiểm tra...";
    const ok = await checkApiHealth();
    if (ok && currentAuthToken) {
      await verifyStoredSession();
    }
  });
});

async function checkApiHealth() {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2500);

    const res = await fetch(`${API_BASE_URL}/health`, {
      method: "GET",
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (res.ok) {
      const data = await res.json();
      if (data.ready) {
        setApiStatus(true, `API Sẵn sàng (${data.device})`);
        return true;
      }
      setApiStatus(false, "Đang nạp mô hình...");
      return false;
    }
    setApiStatus(false, "Lỗi phản hồi API");
    return false;
  } catch (err) {
    setApiStatus(false, "Backend chưa chạy");
    return false;
  }
}

function setApiStatus(isOnline, text) {
  apiStatusEl.className = "status-badge " + (isOnline ? "status-online" : "status-offline");
  statusTextEl.textContent = text;
  if (!isOnline) {
    showError("Không thể kết nối Backend FastAPI! Vui lòng chạy 'start_server.bat' trước.");
  } else {
    hideError();
  }
}

// ==============================================================================
// 2. QUẢN LÝ ĐĂNG NHẬP JWT & ĐỒNG BỘ TÀI KHOẢN
// ==============================================================================

async function verifyStoredSession() {
  if (!currentAuthToken) return;
  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/auth/me`, {
      method: "GET",
      headers: { Authorization: `Bearer ${currentAuthToken}` },
    });
    if (res.ok) {
      const data = await res.json();
      if (data.user) {
        currentAuthUser = data.user;
        await storageSet({ auth_user: currentAuthUser });
        renderAuthUI();
      }
    } else if (res.status === 401) {
      // Token hết hạn -> dọn sạch phiên
      await clearAuthSession();
    }
  } catch (_) {
    // Bỏ qua nếu mất mạng tạm thời
  }
}

function renderAuthUI() {
  if (currentAuthToken && currentAuthUser) {
    authGuestView.classList.add("hidden");
    loginDrawer.classList.add("hidden");
    authUserView.classList.remove("hidden");

    const isTeacher = currentAuthUser.role === "teacher" || currentAuthUser.role === "admin";
    authUserName.textContent = currentAuthUser.full_name || currentAuthUser.email;
    authUserSub.textContent = isTeacher
      ? "👨‍🏫 Giảng viên • Đồng bộ SQL Server"
      : "🎓 Sinh viên • Đồng bộ SQL Server";

    if (currentAuthUser.avatar_url) {
      authAvatar.src = currentAuthUser.avatar_url;
      authAvatar.classList.remove("hidden");
      authRoleIcon.classList.add("hidden");
    } else {
      authAvatar.classList.add("hidden");
      authRoleIcon.classList.remove("hidden");
      authRoleIcon.textContent = isTeacher ? "👨‍🏫" : "🎓";
    }
  } else {
    authUserView.classList.add("hidden");
    authGuestView.classList.remove("hidden");
  }
}

async function clearAuthSession() {
  currentAuthToken = null;
  currentAuthUser = null;
  await storageRemove(["auth_token", "auth_user"]);
  renderAuthUI();
}

btnToggleLogin.addEventListener("click", () => {
  loginDrawer.classList.toggle("hidden");
  loginError.classList.add("hidden");
  if (!loginDrawer.classList.contains("hidden")) {
    loginEmail.focus();
  }
});

btnCloseLogin.addEventListener("click", () => {
  loginDrawer.classList.add("hidden");
});

btnDemoTeacher.addEventListener("click", () => {
  loginEmail.value = "giangvien@veritext.ai";
  loginPassword.value = "123456";
  loginError.classList.add("hidden");
});

loginDrawer.addEventListener("submit", async (e) => {
  e.preventDefault();
  loginError.classList.add("hidden");
  btnLoginSubmit.disabled = true;
  btnLoginSubmit.textContent = "Đang đăng nhập...";

  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        email: loginEmail.value.trim(),
        password: loginPassword.value,
      }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.detail || "Email hoặc mật khẩu không chính xác!");
    }

    currentAuthToken = data.access_token;
    currentAuthUser = data.user;
    await storageSet({
      auth_token: currentAuthToken,
      auth_user: currentAuthUser,
    });
    loginPassword.value = "";
    renderAuthUI();
  } catch (err) {
    loginError.textContent = err.message || "Không thể kết nối máy chủ đăng nhập.";
    loginError.classList.remove("hidden");
  } finally {
    btnLoginSubmit.disabled = false;
    btnLoginSubmit.textContent = "Đăng nhập";
  }
});

btnLogout.addEventListener("click", async () => {
  await clearAuthSession();
  syncResultBar.classList.add("hidden");
});

function openWebDashboard(e) {
  if (e) e.preventDefault();
  if (chrome.tabs && chrome.tabs.create) {
    chrome.tabs.create({ url: API_BASE_URL });
  } else {
    window.open(API_BASE_URL, "_blank");
  }
}

btnOpenDashboard.addEventListener("click", openWebDashboard);
linkOpenWebAuth.addEventListener("click", openWebDashboard);
linkViewHistory.addEventListener("click", openWebDashboard);

// ==============================================================================
// 3. QUÉT TOÀN TRANG WEB HIỆN TẠI (LMS / GOOGLE CLASSROOM / BLOG)
// ==============================================================================

/**
 * Hàm chạy trực tiếp trong context của tab đang mở để trích xuất nội dung bài viết chính,
 * ưu tiên các khung bài nộp LMS / article / main và loại bỏ thanh điều hướng, menu, script.
 */
function extractMainPageTextFromDOM() {
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
  if (!rootEl) {
    rootEl = document.body;
  }

  const SKIP_TAGS = new Set([
    "SCRIPT", "STYLE", "NOSCRIPT", "SVG", "IFRAME",
    "NAV", "HEADER", "FOOTER", "ASIDE", "BUTTON", "SELECT",
  ]);
  const chunks = [];
  let lastBlock = null;

  const walker = document.createTreeWalker(rootEl, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      if (!node.nodeValue || !node.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
      let cur = node.parentElement;
      while (cur && cur !== rootEl.parentElement) {
        if (SKIP_TAGS.has(cur.tagName)) return NodeFilter.FILTER_REJECT;
        if (cur.id === "ai-detect-floating-btn" || cur.id === "ai-detect-tooltip-modal") {
          return NodeFilter.FILTER_REJECT;
        }
        if (cur.hidden || cur.getAttribute("aria-hidden") === "true") {
          return NodeFilter.FILTER_REJECT;
        }
        try {
          const st = window.getComputedStyle(cur);
          if (st.display === "none" || st.visibility === "hidden") return NodeFilter.FILTER_REJECT;
        } catch (_) {}
        cur = cur.parentElement;
      }
      return NodeFilter.FILTER_ACCEPT;
    },
  });

  let n;
  while ((n = walker.nextNode())) {
    const parentBlock = n.parentElement
      ? n.parentElement.closest("p,div,li,h1,h2,h3,h4,h5,h6,tr,section,article,blockquote,pre")
      : null;
    if (lastBlock && parentBlock !== lastBlock) {
      chunks.push("\n");
    }
    chunks.push(n.nodeValue.replace(/\s+/g, " "));
    lastBlock = parentBlock;
  }

  const rawText = chunks
    .join(" ")
    .replace(/[ \t]*\n[ \t]*/g, "\n")
    .replace(/[ \t]{2,}/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();

  const trimmedText = rawText.length > MAX_CHARS ? rawText.slice(0, MAX_CHARS) : rawText;
  const words = trimmedText ? trimmedText.split(/\s+/).length : 0;

  return {
    title: (document.title || window.location.hostname || "Trang web").trim(),
    url: window.location.href || "",
    text: trimmedText,
    wordCount: words,
    truncated: rawText.length > MAX_CHARS,
  };
}

btnScanPage.addEventListener("click", async () => {
  hideError();
  if (!chrome.tabs || !chrome.scripting) {
    showError("Trình duyệt không hỗ trợ trích xuất trang tự động trong ngữ cảnh này.");
    return;
  }

  btnScanPage.disabled = true;
  const origLabel = btnScanPage.innerHTML;
  btnScanPage.innerHTML = "<span>⏳ Đang trích xuất nội dung trang web...</span>";

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab || !tab.id) {
      throw new Error("Không tìm thấy tab trình duyệt đang mở.");
    }
    if (tab.url && /^(chrome|edge|about|chrome-extension):/i.test(tab.url)) {
      throw new Error("Không thể quét trang hệ thống trình duyệt (chrome://). Hãy mở một trang web, bài viết hoặc LMS!");
    }

    const injected = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: extractMainPageTextFromDOM,
    });

    const payload = injected && injected[0] ? injected[0].result : null;
    if (!payload || !payload.text || payload.text.length < 20) {
      throw new Error("Trang hiện tại không có đủ văn bản để phân tích (tối thiểu 20 ký tự).");
    }

    textInput.value = payload.text;
    activePageTitle = `[LMS/Web] ${payload.title}`.slice(0, 120);
    pageSourceText.textContent = `🌐 Đã lấy ${payload.wordCount} từ từ: ${payload.title.slice(0, 42)}${payload.title.length > 42 ? "..." : ""}${payload.truncated ? " (đã rút gọn)" : ""}`;
    pageSourceBanner.classList.remove("hidden");
    updateCounts();

    // Tự động kích hoạt phân tích ngay sau khi lấy văn bản trang
    await runAnalysis();
  } catch (err) {
    showError(err.message || "Không thể trích xuất nội dung từ trang hiện tại.");
  } finally {
    btnScanPage.disabled = false;
    btnScanPage.innerHTML = origLabel;
  }
});

btnClearSource.addEventListener("click", () => {
  activePageTitle = null;
  pageSourceBanner.classList.add("hidden");
});

// ==============================================================================
// 4. SỰ KIỆN TƯƠNG TÁC INPUT & BỘ LỌC HỌC THUẬT
// ==============================================================================

textInput.addEventListener("input", () => {
  updateCounts();
  if (!textInput.value.trim()) {
    activePageTitle = null;
    pageSourceBanner.classList.add("hidden");
  }
});

function updateCounts() {
  const text = textInput.value.trim();
  const words = text ? text.split(/\s+/).length : 0;
  const chars = text.length;
  wordCountEl.textContent = `${words} từ | ${chars} ký tự`;
}

thresholdSlider.addEventListener("input", () => {
  const val = thresholdSlider.value;
  thresholdValEl.textContent = `${val}%`;
  storageSet({ threshold: val / 100 });
});

chkFilterQuotes.addEventListener("change", () => {
  storageSet({ filter_quotes: chkFilterQuotes.checked });
});

chkFilterReferences.addEventListener("change", () => {
  storageSet({ filter_references: chkFilterReferences.checked });
});

btnPaste.addEventListener("click", async () => {
  try {
    const text = await navigator.clipboard.readText();
    textInput.value = text;
    activePageTitle = null;
    pageSourceBanner.classList.add("hidden");
    updateCounts();
  } catch (e) {
    showError("Không thể đọc từ Clipboard. Hãy dùng phím Ctrl + V để dán.");
  }
});

btnClear.addEventListener("click", () => {
  textInput.value = "";
  activePageTitle = null;
  pageSourceBanner.classList.add("hidden");
  updateCounts();
  resultsCard.classList.add("hidden");
  hideError();
});

// ==============================================================================
// 5. XỬ LÝ PHÂN TÍCH VĂN BẢN (ENSEMBLE + ANTI-BYPASS + XAI + JWT SYNC)
// ==============================================================================

btnAnalyze.addEventListener("click", runAnalysis);

async function runAnalysis() {
  if (btnAnalyze.disabled) return;
  const text = textInput.value.trim();
  if (text.length < 10) {
    showError("Vui lòng nhập ít nhất 10 ký tự (tiếng Anh hoặc tiếng Việt) để mô hình phân tích chính xác.");
    return;
  }

  hideError();
  setLoading(true);

  const threshold = parseFloat(thresholdSlider.value) / 100;
  const headers = { "Content-Type": "application/json" };
  if (currentAuthToken) {
    headers["Authorization"] = `Bearer ${currentAuthToken}`;
  }

  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/detect/sentences`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        text: text,
        threshold: threshold,
        title: activePageTitle || `[Extension] ${text.slice(0, 42)}...`,
        filter_quotes: chkFilterQuotes.checked,
        filter_references: chkFilterReferences.checked,
      }),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || "Lỗi xử lý API");
    }

    const data = await res.json();
    const overallData = data.overall || data;
    if (data.total_processing_time_ms && !overallData.inference_time_ms) {
      overallData.inference_time_ms = data.total_processing_time_ms;
    }
    currentSentenceData = data.sentences || [];

    renderResults(data, overallData, currentSentenceData);
  } catch (err) {
    showError(err.message || "Không thể phân tích. Kiểm tra lại kết nối server!");
  } finally {
    setLoading(false);
  }
}

function renderResults(fullResponse, data, sentences) {
  resultsCard.classList.remove("hidden");

  const isAI = data.label === "AI-generated";
  const aiPct = (data.ai_probability * 100).toFixed(1);
  const humanPct = (data.human_probability * 100).toFixed(1);

  // 1. Cảnh báo Anti-Bypass (nếu phát hiện ký tự tàng hình / Cyrillic giả Latin)
  const bypass = fullResponse.bypass_report;
  if (bypass && bypass.has_bypass_warning) {
    bypassSummary.textContent = bypass.summary || "Phát hiện ký tự bất thường cố tình qua mặt bộ lọc AI.";
    bypassBanner.classList.remove("hidden");
  } else {
    bypassBanner.classList.add("hidden");
  }

  // 2. Badge kết luận chính
  verdictBadge.className = "verdict-badge " + (isAI ? "badge-ai" : "badge-human");
  verdictIcon.textContent = isAI ? "🤖" : "👤";
  verdictTitle.textContent = isAI ? "DO AI SINH (ChatGPT/LLM)" : "DO NGƯỜI VIẾT (HUMAN)";

  // 3. Phân loại 4 cấp độ thực tế
  const tier = fullResponse.authorship_tier;
  if (tier && tier.label_vi) {
    tierBadge.className = `tier-badge ${tier.badge_class || ""}`;
    tierBadge.textContent = tier.label_vi;
    tierBadge.title = tier.reason || "";
    tierBadge.classList.remove("hidden");
  } else {
    tierBadge.classList.add("hidden");
  }

  // 4. Thời gian & Gauge
  inferenceTimeEl.textContent = `${data.inference_time_ms} ms`;
  aiPercentageText.textContent = `${aiPct}%`;
  aiPercentageText.style.color = isAI ? "#ef4444" : "#10b981";
  gaugeBarFill.style.width = `${aiPct}%`;

  humanPercentageText.textContent = `Người viết: ${humanPct}%`;
  confidenceText.textContent = `Độ tin cậy: ${data.confidence}`;

  // 5. Bằng chứng Ngôn ngữ học XAI rút gọn
  const xai = fullResponse.xai_metrics;
  if (xai && typeof xai.burstiness_score === "number") {
    xaiBurstiness.textContent = `${xai.burstiness_score}/100`;
    xaiMattr.textContent = `${xai.lexical_diversity}%`;
    xaiPerplexity.textContent = `${xai.perplexity_score}/100`;

    if (xai.ai_cliches && xai.ai_cliches.length > 0) {
      const phrases = xai.ai_cliches.slice(0, 4).map((c) => `"${c.phrase}"`).join(", ");
      xaiCliches.textContent = `⚠️ Văn mẫu AI (${xai.ai_cliche_count}): ${phrases}`;
      xaiCliches.classList.remove("hidden");
    } else {
      xaiCliches.classList.add("hidden");
    }
    xaiStrip.classList.remove("hidden");
  } else {
    xaiStrip.classList.add("hidden");
  }

  // 6. Breakdown các mô hình (hỗ trợ hiển thị Sliding-Window đa đoạn cho bài luận dài Tiếng Việt)
  if (data.model_breakdown && data.model_breakdown.individual_models) {
    const m = data.model_breakdown.individual_models;
    const chunkInfo = data.model_breakdown.chunk_analysis;
    if (data.language === "vi") {
      if (chunkInfo && chunkInfo.chunk_count > 1) {
        scoreLongformer.textContent = `${aiPct}% (${chunkInfo.chunk_count} đoạn)`;
        scoreBert.textContent = `TB ${(chunkInfo.mean_probability * 100).toFixed(1)}%`;
        scoreRoberta.textContent = `Đỉnh ${(chunkInfo.peak_probability * 100).toFixed(1)}%`;
      } else {
        scoreLongformer.textContent = `${aiPct}% (PhoBERT)`;
        scoreBert.textContent = "100% (PyVi)";
        scoreRoberta.textContent = "Tiếng Việt 🇻🇳";
      }
    } else {
      scoreLongformer.textContent = `${(m.longformer.probability * 100).toFixed(1)}% AI`;
      scoreBert.textContent = `${(m.bert.probability * 100).toFixed(1)}% AI`;
      scoreRoberta.textContent = `${(m.roberta.probability * 100).toFixed(1)}% AI`;
    }
  }

  // 7. Trạng thái đồng bộ SQL Server & mã chứng thực (kiểm chứng thực tế từ cờ saved_by_user)
  if (fullResponse.scan_id) {
    const auditSuffix = fullResponse.audit_code ? ` • ${fullResponse.audit_code}` : "";
    if (fullResponse.saved_by_user) {
      const ownerName = fullResponse.saved_user_name || (currentAuthUser && currentAuthUser.full_name) || "Tài khoản";
      syncResultText.textContent = `✓ Đã lưu #${fullResponse.scan_id} (${ownerName})${auditSuffix}`;
    } else if (currentAuthToken) {
      // Token hết hạn -> dọn phiên và báo rõ cho người dùng
      clearAuthSession();
      syncResultText.textContent = `⚠️ Hết hạn phiên • Quét khách (#${fullResponse.scan_id})${auditSuffix}`;
    } else {
      syncResultText.textContent = `✓ Đã quét (#${fullResponse.scan_id})${auditSuffix}`;
    }
    syncResultBar.classList.remove("hidden");
  } else {
    syncResultBar.classList.add("hidden");
  }

  // 8. Cấp câu
  renderSentencesList(sentences);
}

function renderSentencesList(sentences) {
  sentencesList.innerHTML = "";
  if (!sentences || sentences.length === 0) {
    sentencesList.innerHTML = "<div class='sentence-item'>Không có dữ liệu cấp câu.</div>";
    return;
  }

  sentences.forEach((s) => {
    const item = document.createElement("div");
    item.className = `sentence-item ${s.highlight_color}`;

    const scoreTag = document.createElement("span");
    scoreTag.className = "sentence-score";
    scoreTag.textContent = `[${s.ai_percentage} AI]`;

    const textSpan = document.createElement("span");
    textSpan.textContent = ` ${s.sentence}`;

    item.appendChild(scoreTag);
    item.appendChild(textSpan);
    sentencesList.appendChild(item);
  });
}

btnToggleSentences.addEventListener("click", () => {
  sentencesList.classList.toggle("hidden");
  if (sentencesList.classList.contains("hidden")) {
    btnToggleSentences.textContent = "📑 Xem chi tiết từng câu (AI Hotspots)";
  } else {
    btnToggleSentences.textContent = "Ẩn chi tiết từng câu";
  }
});

// ==============================================================================
// 6. TIỆN ÍCH TRỢ GIÚP
// ==============================================================================

function setLoading(isLoading) {
  btnAnalyze.disabled = isLoading;
  btnScanPage.disabled = isLoading;
  const btnText = btnAnalyze.querySelector(".btn-text");
  if (isLoading) {
    btnText.textContent = "Đang phân tích...";
    spinner.classList.remove("hidden");
  } else {
    btnText.textContent = "🔍 Phân tích Văn bản (Ensemble + XAI)";
    spinner.classList.add("hidden");
  }
}

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.classList.remove("hidden");
}

function hideError() {
  errorBox.classList.add("hidden");
}
