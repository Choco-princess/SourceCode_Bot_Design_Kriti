/* ================================================================
   Rover Dashboard — script.js
   Capture + Detect workflow with live ESP32-CAM feed (HUD UI)
   ================================================================ */

const btnCapture    = document.getElementById("btn-capture");
const btnStart      = document.getElementById("btn-start");
const btnStop       = document.getElementById("btn-stop");
const timerEl       = document.getElementById("timer");
const statusBadge   = document.getElementById("status-badge");
const resultsEl     = document.getElementById("results-container");
const captureInline = document.getElementById("capture-inline");
const capturedImg   = document.getElementById("captured-image");
const logEl         = document.getElementById("detection-log");
const loadingOverlay = document.getElementById("loading-overlay");

let _timerInterval = null;
let _runStartTime  = 0;
let _isRunning     = false;

// ── Capture & Detect ────────────────────────────────────────────
async function capturePhoto() {
    btnCapture.disabled = true;
    btnCapture.textContent = "ANALYZING_OPTICAL_DATA...";
    loadingOverlay.style.display = "flex";

    try {
        const resp = await fetch("/api/capture", { method: "POST" });
        const data = await resp.json();

        if (data.error) {
            showError(data.error);
            return;
        }

        // Show captured image thumbnail inline
        if (data.capture_image) {
            capturedImg.src = "data:image/jpeg;base64," + data.capture_image;
            captureInline.style.display = "flex";
        }

        // Display results
        displayResults(data);

        // Update log
        fetchLog();

    } catch (err) {
        showError("UPLINK_FAILURE: " + err.message);
    } finally {
        btnCapture.disabled = false;
        btnCapture.textContent = "INITIATE CAPTURE & DETECT";
        loadingOverlay.style.display = "none";
    }
}

// ── Display Detection Results ───────────────────────────────────
function displayResults(data) {
    let html = "";

    // 1) Classifier — Top 3
    const cls = data.classifier || [];
    if (cls.length > 0) {
        html += `<div class="result-section">`;
        html += `<h3>NEURAL_NET_CLASS (TOP ${cls.length})</h3>`;
        const ranks = ["[PRM]", "[SEC]", "[TER]"]; // Primary / Secondary / Tertiary
        cls.forEach((d, i) => {
            const conf = d.confidence ? `${(d.confidence * 100).toFixed(1)}%` : "";
            const rank = ranks[i] || `[#${i+1}]`;
            html += `<div class="result-item result-classifier">
                ${rank} <b>${d.category}</b>: ${d.content}
                ${conf ? `<span class="conf-badge">${conf}</span>` : ""}
            </div>`;
        });
        html += `</div>`;
    }

    // 2) QR Codes
    const qr = data.qr || [];
    if (qr.length > 0) {
        html += `<div class="result-section">`;
        html += `<h3>DATA_MATRIX_SCAN</h3>`;
        qr.forEach(d => {
            html += `<div class="result-item result-qr">
                <b>${d.category}</b>: ${d.content}
            </div>`;
        });
        html += `</div>`;
    }

    // 3) Face Recognition
    const face = data.face || [];
    if (face.length > 0) {
        html += `<div class="result-section">`;
        html += `<h3>BIOMETRIC_MATCH</h3>`;
        face.forEach(d => {
            html += `<div class="result-item result-face">
                <b>${d.category}</b>: ${d.content}
            </div>`;
        });
        html += `</div>`;
    }

    // 4) Number Plate
    const plate = data.plate || [];
    if (plate.length > 0) {
        html += `<div class="result-section">`;
        html += `<h3>ALPHANUMERIC_PLATE_ID</h3>`;
        plate.forEach(d => {
            html += `<div class="result-item result-plate">
                <b>${d.category}</b>: ${d.content}
            </div>`;
        });
        html += `</div>`;
    }

    // Nothing detected?
    if (!html) {
        html = `<p class="results-placeholder">NO_TARGETS_ACQUIRED</p>`;
    }

    resultsEl.innerHTML = html;
}

function showError(msg) {
    resultsEl.innerHTML = `<p class="results-error">[SYSTEM_ERR] ${msg}</p>`;
}

// ── Run Controls ────────────────────────────────────────────────
function startRun() {
    fetch("/api/start", { method: "POST" })
        .then(r => r.json())
        .then(() => {
            _isRunning = true;
            _runStartTime = Date.now();
            btnStart.disabled = true;
            btnStop.disabled = false;
            statusBadge.textContent = "SYS_ACTIVE";
            statusBadge.className = "badge badge-running";
            logEl.innerHTML = "";

            // Start timer updates
            if (_timerInterval) clearInterval(_timerInterval);
            _timerInterval = setInterval(updateTimer, 100);
        });
}

function stopRun() {
    fetch("/api/stop", { method: "POST" })
        .then(r => r.json())
        .then(() => {
            _isRunning = false;
            btnStart.disabled = false;
            btnStop.disabled = true;
            statusBadge.textContent = "SYS_HALTED";
            statusBadge.className = "badge badge-stopped";

            if (_timerInterval) {
                clearInterval(_timerInterval);
                _timerInterval = null;
            }
        });
}

function updateTimer() {
    if (!_isRunning) return;
    const elapsed = (Date.now() - _runStartTime) / 1000;
    const mins = Math.floor(elapsed / 60);
    const secs = Math.floor(elapsed % 60);
    timerEl.textContent = `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;

    // Auto-stop at 5 minutes
    if (elapsed >= 300) {
        stopRun();
    }
}

// ── Fetch detection log from server ─────────────────────────────
async function fetchLog() {
    try {
        const resp = await fetch("/api/status");
        const data = await resp.json();
        const log = data.log || [];

        if (log.length === 0) {
            logEl.innerHTML = `<li class="log-empty">NO_DATA_AVAILABLE</li>`;
        } else {
            logEl.innerHTML = log.map(entry =>
                `<li>${entry}</li>`
            ).join("");
        }

        // Sync run state
        if (data.running && !_isRunning) {
            _isRunning = true;
            _runStartTime = Date.now() - (data.elapsed * 1000);
            btnStart.disabled = true;
            btnStop.disabled = false;
            statusBadge.textContent = "SYS_ACTIVE";
            statusBadge.className = "badge badge-running";
            if (!_timerInterval) {
                _timerInterval = setInterval(updateTimer, 100);
            }
        } else if (!data.running && _isRunning) {
            _isRunning = false;
            btnStart.disabled = false;
            btnStop.disabled = true;
            statusBadge.textContent = "SYS_HALTED";
            statusBadge.className = "badge badge-stopped";
            if (_timerInterval) {
                clearInterval(_timerInterval);
                _timerInterval = null;
            }
        }
    } catch (e) {
        // Ignore — just a log update
    }
}

// ── Poll status periodically ────────────────────────────────────
setInterval(fetchLog, 3000);
fetchLog();

// ── Lightbox (expand captured image) ────────────────────────────
function openLightbox() {
    const src = document.getElementById("captured-image").src;
    if (!src) return;
    const lb = document.getElementById("lightbox");
    document.getElementById("lightbox-img").src = src;
    lb.style.display = "flex";
}
function closeLightbox() {
    document.getElementById("lightbox").style.display = "none";
}
// Close on Escape key
document.addEventListener("keydown", e => {
    if (e.key === "Escape") closeLightbox();
});

// ── Live Feed ──────────────────────────────────────────────────────────
// Uses native MJPEG stream via <img src="/video_feed">.
// Browser renders multipart/x-mixed-replace natively — zero JS overhead.
// No polling loop needed.