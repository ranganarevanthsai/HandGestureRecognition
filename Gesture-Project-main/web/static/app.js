const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const capture = document.getElementById("capture");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const autoPredict = document.getElementById("autoPredict");
const fps = document.getElementById("fps");
const fpsVal = document.getElementById("fpsVal");

const symbolEl = document.getElementById("symbol");
const statusEl = document.getElementById("status");

let stream = null;
let timer = null;
let inFlight = false;

function setStatus(text) {
  statusEl.textContent = text;
}

function setSymbol(text) {
  symbolEl.textContent = text && text.trim().length ? text : "—";
}

function resizeCanvases() {
  const w = video.videoWidth || 640;
  const h = video.videoHeight || 480;
  overlay.width = w;
  overlay.height = h;
  capture.width = w;
  capture.height = h;
}

function drawOverlay() {
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  ctx.strokeStyle = "rgba(255,255,255,0.2)";
  ctx.lineWidth = 2;
  const pad = 24;
  ctx.strokeRect(pad, pad, overlay.width - pad * 2, overlay.height - pad * 2);
}

async function startCamera() {
  if (stream) return;
  setStatus("Requesting camera permission…");

  stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();

  resizeCanvases();
  drawOverlay();

  stopBtn.disabled = false;
  setStatus("Camera running.");

  if (autoPredict.checked) {
    startLoop();
  }
}

function stopCamera() {
  stopLoop();
  if (stream) {
    for (const t of stream.getTracks()) t.stop();
  }
  stream = null;
  video.srcObject = null;
  stopBtn.disabled = true;
  setStatus("Camera stopped.");
  setSymbol("");
}

function startLoop() {
  stopLoop();
  const rate = Number(fps.value || 5);
  timer = setInterval(() => {
    predictOnce().catch((e) => setStatus(`Predict error: ${e.message || e}`));
  }, Math.max(100, Math.floor(1000 / rate)));
}

function stopLoop() {
  if (timer) clearInterval(timer);
  timer = null;
}

async function predictOnce() {
  if (!stream) return;
  if (inFlight) return;
  inFlight = true;

  try {
    const ctx = capture.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(video, 0, 0, capture.width, capture.height);
    const imageDataUrl = capture.toDataURL("image/jpeg", 0.75);

    setStatus("Predicting…");
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ imageDataUrl }),
    });
    const json = await res.json();
    if (!json.ok) {
      setStatus(json.error || "Prediction failed.");
      setSymbol("");
      return;
    }

    if (!json.hasHand) {
      setStatus("No hand detected.");
      setSymbol("");
      return;
    }

    setSymbol(json.symbol);
    setStatus("OK");
  } finally {
    inFlight = false;
  }
}

startBtn.addEventListener("click", () => startCamera().catch((e) => setStatus(`Camera error: ${e.message || e}`)));
stopBtn.addEventListener("click", () => stopCamera());

autoPredict.addEventListener("change", () => {
  if (!stream) return;
  if (autoPredict.checked) startLoop();
  else stopLoop();
});

fps.addEventListener("input", () => {
  fpsVal.textContent = `${fps.value} fps`;
  if (stream && autoPredict.checked) startLoop();
});

window.addEventListener("resize", () => {
  if (!stream) return;
  resizeCanvases();
  drawOverlay();
});

