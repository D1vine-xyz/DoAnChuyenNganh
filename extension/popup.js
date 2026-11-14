/**
 * AI Text Detector - Chrome Extension Popup Logic (Manifest v3)
 * Kết nối trực tiếp với FastAPI Backend (http://localhost:8000)
 */

const API_BASE_URL = "http://localhost:8000";

// DOM Elements
const textInput = document.getElementById("text-input");
const wordCountEl = document.getElementById("word-count");
const thresholdSlider = document.getElementById("threshold-slider");
const thresholdValEl = document.getElementById("threshold-val");
const btnPaste = document.getElementById("btn-paste");
const btnClear = document.getElementById("btn-clear");
const btnAnalyze = document.getElementById("btn-analyze");
const spinner = document.getElementById("spinner");
const apiStatusEl = document.getElementById("api-status");
const statusTextEl = document.getElementById("status-text");
const errorBox = document.getElementById("error-box");

// Results DOM
const resultsCard = document.getElementById("results-card");
const verdictBadge = document.getElementById("verdict-badge");
const verdictIcon = document.getElementById("verdict-icon");
const verdictTitle = document.getElementById("verdict-title");
const inferenceTimeEl = document.getElementById("inference-time");
const aiPercentageText = document.getElementById("ai-percentage-text");
const gaugeBarFill = document.getElementById("gauge-bar-fill");
const humanPercentageText = document.getElementById("human-percentage-text");
const confidenceText = document.getElementById("confidence-text");
const scoreLongformer = document.getElementById("score-longformer");
const scoreBert = document.getElementById("score-bert");
const scoreRoberta = document.getElementById("score-roberta");

const btnToggleSentences = document.getElementById("btn-toggle-sentences");
const sentencesList = document.getElementById("sentences-list");

let currentSentenceData = [];

// ==============================================================================
// 1. KHỞI TẠO & KIỂM TRA SỨC KHỎE API (HEALTH CHECK)
// ==============================================================================

document.addEventListener("DOMContentLoaded", async () => {
  // Nạp ngưỡng đã lưu từ bộ nhớ extension
  if (chrome.storage && chrome.storage.local) {
    chrome.storage.local.get(["threshold"], (res) => {
      if (res.threshold) {
        thresholdSlider.value = res.threshold * 100;
        thresholdValEl.textContent = `${thresholdSlider.value}%`;
      }
    });
  }

  // Cập nhật đếm từ ban đầu
  updateCounts();
  // Kiểm tra kết nối API
  await checkApiHealth();
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
        setApiStatus(true, `FastAPI Sẵn sàng (${data.device})`);
      } else {
        setApiStatus(false, "Đang khởi tạo mô hình...");
      }
    } else {
      setApiStatus(false, "Lỗi phản hồi API");
    }
  } catch (err) {
    setApiStatus(false, "FastAPI Chưa chạy");
  }
}

function setApiStatus(isOnline, text) {
  apiStatusEl.className = "status-badge " + (isOnline ? "status-online" : "status-offline");
  statusTextEl.textContent = text;
  if (!isOnline) {
    showError("Không thể kết nối Backend FastAPI! Vui lòng nhấp đúp file 'start_server.bat' để khởi chạy server.");
  } else {
    hideError();
  }
}

// ==============================================================================
// 2. SỰ KIỆN TƯƠNG TÁC INPUT
// ==============================================================================

textInput.addEventListener("input", updateCounts);

function updateCounts() {
  const text = textInput.value.trim();
  const words = text ? text.split(/\s+/).length : 0;
  const chars = text.length;
  wordCountEl.textContent = `${words} từ | ${chars} ký tự`;
}

thresholdSlider.addEventListener("input", () => {
  const val = thresholdSlider.value;
  thresholdValEl.textContent = `${val}%`;
  if (chrome.storage && chrome.storage.local) {
    chrome.storage.local.set({ threshold: val / 100 });
  }
});

btnPaste.addEventListener("click", async () => {
  try {
    const text = await navigator.clipboard.readText();
    textInput.value = text;
    updateCounts();
  } catch (e) {
    showError("Không thể đọc từ Clipboard. Hãy dùng phím Ctrl + V để dán.");
  }
});

btnClear.addEventListener("click", () => {
  textInput.value = "";
  updateCounts();
  resultsCard.classList.add("hidden");
  hideError();
});

// ==============================================================================
// 3. XỬ LÝ PHÂN TÍCH VĂN BẢN (ENSEMBLE DETECT)
// ==============================================================================

btnAnalyze.addEventListener("click", async () => {
  const text = textInput.value.trim();
  if (text.length < 10) {
    showError("Vui lòng nhập ít nhất 10 ký tự tiếng Anh để mô hình phân tích chính xác.");
    return;
  }

  hideError();
  setLoading(true);

  const threshold = parseFloat(thresholdSlider.value) / 100;

  try {
    // 1. Gửi request phân tích tổng thể
    const detectRes = await fetch(`${API_BASE_URL}/api/v1/detect`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: text,
        threshold: threshold,
        detailed: true,
      }),
    });

    if (!detectRes.ok) {
      const errData = await detectRes.json();
      throw new Error(errData.detail || "Lỗi xử lý API");
    }

    const detectData = await detectRes.json();

    // 2. Gửi request phân tích từng câu (AI Hotspots)
    const sentenceRes = await fetch(`${API_BASE_URL}/api/v1/detect/sentences`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: text,
        threshold: threshold,
      }),
    });

    if (sentenceRes.ok) {
      const sData = await sentenceRes.json();
      currentSentenceData = sData.sentences || [];
    } else {
      currentSentenceData = [];
    }

    renderResults(detectData, currentSentenceData);
  } catch (err) {
    showError(err.message || "Không thể phân tích. Kiểm tra lại kết nối server!");
  } finally {
    setLoading(false);
  }
});

function renderResults(data, sentences) {
  resultsCard.classList.remove("hidden");

  const isAI = data.label === "AI-generated";
  const aiPct = (data.ai_probability * 100).toFixed(1);
  const humanPct = (data.human_probability * 100).toFixed(1);

  // 1. Badge kết luận
  verdictBadge.className = "verdict-badge " + (isAI ? "badge-ai" : "badge-human");
  verdictIcon.textContent = isAI ? "🤖" : "👤";
  verdictTitle.textContent = isAI ? "DO AI SINH (ChatGPT/LLM)" : "DO NGƯỜI VIẾT (HUMAN)";

  // 2. Thời gian
  inferenceTimeEl.textContent = `${data.inference_time_ms} ms`;

  // 3. Gauge
  aiPercentageText.textContent = `${aiPct}%`;
  aiPercentageText.style.color = isAI ? "#ef4444" : "#10b981";
  gaugeBarFill.style.width = `${aiPct}%`;

  humanPercentageText.textContent = `Người viết: ${humanPct}%`;
  confidenceText.textContent = `Độ tin cậy: ${data.confidence}`;

  // 4. Breakdown các mô hình
  if (data.model_breakdown && data.model_breakdown.individual_models) {
    const m = data.model_breakdown.individual_models;
    scoreLongformer.textContent = `${(m.longformer.probability * 100).toFixed(1)}% AI`;
    scoreBert.textContent = `${(m.bert.probability * 100).toFixed(1)}% AI`;
    scoreRoberta.textContent = `${(m.roberta.probability * 100).toFixed(1)}% AI`;
  }

  // 5. Cấp câu
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
// 4. TIỆN ÍCH TRỢ GIÚP
// ==============================================================================

function setLoading(isLoading) {
  btnAnalyze.disabled = isLoading;
  const btnText = btnAnalyze.querySelector(".btn-text");
  if (isLoading) {
    btnText.textContent = "Đang phân tích...";
    spinner.classList.remove("hidden");
  } else {
    btnText.textContent = "🔍 Phân tích Văn bản (Ensemble)";
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
