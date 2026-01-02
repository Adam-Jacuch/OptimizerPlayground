// ------------------------
// Canvas + data handling
// ------------------------
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const points = []; // {x, y, label} in normalized [-1,1] coords
let currentLabel = 0;
let brush = 6;
let showBackground = true;

let toolMode = "draw"; // "draw" | "erase"

let stepCount = 0;

function setSteps(n) {
  stepCount = n | 0;
  const el = document.getElementById("stepsValue");
  if (el) el.textContent = String(stepCount);
}

// View mode
let viewMode = "grid"; // "grid" | "graph"

// Training runs (loss history)
let runs = []; // { id, name, points: [{step, loss}], createdAt }
let activeRunId = null;
let runCounter = 1;

function newRun() {
  const id = String(Date.now()) + "_" + Math.random().toString(16).slice(2);
  const r = {
    id,
    name: `Run ${runCounter++}`,
    points: [],
    createdAt: Date.now(),
  };
  runs.unshift(r);
  activeRunId = id;
  renderRunsList();
  redraw(); // if in graph mode, update
  return r;
}

function getActiveRun() {
  return runs.find(r => r.id === activeRunId) || null;
}

function deleteRun(id) {
  runs = runs.filter(r => r.id !== id);
  if (activeRunId === id) {
    activeRunId = runs.length ? runs[0].id : null;
  }
  renderRunsList();
  redraw();
}

function canvasToXY(e) {
  const rect = canvas.getBoundingClientRect();
  const px = (e.clientX - rect.left) * (canvas.width / rect.width);
  const py = (e.clientY - rect.top)  * (canvas.height / rect.height);

  // map pixel -> [-1, 1]
  const x = (px / canvas.width) * 2 - 1;
  const y = -((py / canvas.height) * 2 - 1);
  return { x, y, px, py };
}

function drawGrid() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function drawPoints() {
  for (const p of points) {
    const px = ((p.x + 1) / 2) * canvas.width;
    const py = ((-p.y + 1) / 2) * canvas.height;

    ctx.beginPath();
    ctx.arc(px, py, brush, 0, Math.PI * 2);
    ctx.fillStyle = p.label === 0 ? "#4cc9f0" : "#ff4d6d";
    ctx.fill();
    ctx.strokeStyle = "rgba(0,0,0,0.35)";
    ctx.stroke();
  }
}

function drawLossGraph() {
  // basic chart box
  const W = canvas.width, H = canvas.height;
  const padL = 46, padR = 12, padT = 12, padB = 34;

  // background
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, W, H);

  // find ranges across all runs (or just selected run if you prefer)
  let minStep = Infinity, maxStep = -Infinity;
  let minLoss = Infinity, maxLoss = -Infinity;

  const usableRuns = runs.filter(r => r.points.length >= 2);
  if (usableRuns.length === 0) {
    ctx.fillStyle = "rgba(15,23,42,0.5)";
    ctx.font = "14px system-ui";
    ctx.fillText("No run data yet. Train to record loss.", 16, 28);
    return;
  }

  for (const r of usableRuns) {
    for (const p of r.points) {
      if (p.step < minStep) minStep = p.step;
      if (p.step > maxStep) maxStep = p.step;
      if (p.loss < minLoss) minLoss = p.loss;
      if (p.loss > maxLoss) maxLoss = p.loss;
    }
  }

  // guard ranges
  if (!Number.isFinite(minStep) || maxStep === minStep) { maxStep = minStep + 1; }
  if (!Number.isFinite(minLoss) || maxLoss === minLoss) { maxLoss = minLoss + 1e-6; }

  // nice loss range (include 0)
  minLoss = Math.min(0, minLoss);

  const x0 = padL, x1 = W - padR;
  const y0 = H - padB, y1 = padT;

  const xMap = (s) => x0 + ((s - minStep) / (maxStep - minStep)) * (x1 - x0);
  const yMap = (l) => y0 - ((l - minLoss) / (maxLoss - minLoss)) * (y0 - y1);

  // axes
  ctx.strokeStyle = "rgba(15,23,42,0.14)";
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(x0, y1); ctx.lineTo(x0, y0); ctx.lineTo(x1, y0); ctx.stroke();

  // labels
  ctx.fillStyle = "rgba(15,23,42,0.55)";
  ctx.font = "12px system-ui";
  ctx.fillText("loss", 10, 18);
  ctx.fillText("steps", W - 48, H - 10);

  // helper to draw one run
  const drawRun = (r, emphasize) => {
    ctx.lineWidth = emphasize ? 2 : 1;
    ctx.strokeStyle = emphasize ? "rgba(37,99,235,0.95)" : "rgba(15,23,42,0.25)";
    ctx.beginPath();

    const pts = r.points;
    ctx.moveTo(xMap(pts[0].step), yMap(pts[0].loss));
    for (let i = 1; i < pts.length; i++) {
      ctx.lineTo(xMap(pts[i].step), yMap(pts[i].loss));
    }
    ctx.stroke();
  };

  // draw non-active first, active last
  for (const r of usableRuns) if (r.id !== activeRunId) drawRun(r, false);
  const active = usableRuns.find(r => r.id === activeRunId);
  if (active) drawRun(active, true);
}

function redraw() {
  // clear only (no gridlines)
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (viewMode === "graph") {
    drawLossGraph();
    return;
  }

  // grid view (your existing logic)
  if (showBackground && model && points.length > 0) drawDecisionBackground();
  drawPoints();
}

function eraseAtEvent(ev) {
  const { x, y } = canvasToXY(ev);

  // Convert brush (pixels) to normalized radius.
  // Use canvas width as reference so radius matches screen feel.
  const rNorm = (brush / canvas.width) * 2.0;
  const r2 = rNorm * rNorm;

  // Keep points that are NOT within radius
  const before = points.length;
  for (let i = points.length - 1; i >= 0; i--) {
    const dx = points[i].x - x;
    const dy = points[i].y - y;
    if (dx * dx + dy * dy <= r2) points.splice(i, 1);
  }

  if (points.length !== before) {
    updateCounts();
    redraw();
  }
}

// ------------------------
// Model + training
// ------------------------
let model = null;
let opt = null;
let trainingHandle = null;

const ui = {
  layersInput: document.getElementById("layersInput"),
  activationSelect: document.getElementById("activationSelect"),
  buildBtn: document.getElementById("buildBtn"),
  resetBtn: document.getElementById("resetBtn"),

  lrInput: document.getElementById("lrInput"),
  momentumInput: document.getElementById("momentumInput"),
  wdInput: document.getElementById("wdInput"),
  l2Input: document.getElementById("l2Input"),
  clipInput: document.getElementById("clipInput"),
  stepsPerTick: document.getElementById("stepsPerTick"),

  trainStepBtn: document.getElementById("trainStepBtn"),
  trainBtn: document.getElementById("trainBtn"),
  stopBtn: document.getElementById("stopBtn"),

  class0Btn: document.getElementById("class0Btn"),
  class1Btn: document.getElementById("class1Btn"),
  clearBtn: document.getElementById("clearBtn"),

  brushInput: document.getElementById("brushInput"),
  pointsCount: document.getElementById("pointsCount"),
  lossValue: document.getElementById("lossValue"),
  statusValue: document.getElementById("statusValue"),

  bgToggleBtn: document.getElementById("bgToggleBtn"),
};

function parseLayers(str) {
  const arr = str.split(",").map(s => parseInt(s.trim(), 10)).filter(n => Number.isFinite(n) && n > 0);
  return arr;
}

function build() {
  const sizes = parseLayers(ui.layersInput.value);

  if (sizes.length < 2 || sizes[0] !== 2 || sizes[sizes.length - 1] !== 1) {
    alert("For this playground: layers must start with 2 and end with 1. Example: 2,8,8,1");
    return;
  }

  model = new MLP(sizes, { activation: ui.activationSelect.value });
  opt = new Optimizer({
    lr: parseFloat(ui.lrInput.value),
    momentum: parseFloat(ui.momentumInput.value),
    weightDecay: parseFloat(ui.wdInput.value),
    l2: parseFloat(ui.l2Input.value),
    clipNorm: parseFloat(ui.clipInput.value),
  });

  setSteps(0);
  setStatus("Built model");
  redraw();
}

function resetWeights() {
  if (!model) return;
  build(); // simplest: rebuild with same UI settings
  setStatus("Reset weights");
}

function setStatus(s) {
  ui.statusValue.textContent = s;
}

function updateCounts() {
  ui.pointsCount.textContent = String(points.length);
}

function sampleToTensor(p) {
  // column vector [2,1]
  return nj.array([[p.x], [p.y]]);
}

function labelToTensor(p) {
  // column vector [1,1]
  return nj.array([[p.label]]);
}

function mseLoss(yPred, yTrue) {
  const err = yPred.subtract(yTrue);
  // 0.5 * err^2 (scalar)
  const val = err.multiply(err).multiply(0.5).tolist()[0][0];
  return val;
}

function trainOneTick(steps = 10) {
  if (!activeRunId) newRun();
  if (!model || !opt) {
    setStatus("Build a model first");
    return;
  }
  if (points.length === 0) {
    setStatus("Add points first");
    return;
  }

  let lossSum = 0;

  for (let s = 0; s < steps; s++) {
    // random point SGD
    const p = points[(Math.random() * points.length) | 0];
    const x = sampleToTensor(p);
    const yTrue = labelToTensor(p);

    const { y, cache } = model.forward(x, { training: true });
    const grads = model.backward(yTrue, cache);
    opt.apply(model, grads);

    lossSum += mseLoss(y, yTrue);
  }
  setSteps(stepCount + steps);

  const r = getActiveRun();
  const avgLoss = lossSum / steps;
  if (r) {
    // record a point every tick (not every SGD step)
    r.points.push({ step: stepCount, loss: avgLoss });
    // cap history so it doesn't grow forever
    if (r.points.length > 10000) r.points.shift();
    renderRunsList(); // keeps meta updated
  }
  ui.lossValue.textContent = avgLoss.toFixed(4);
  setStatus("Training");
  redraw();
}

function startTraining() {
  if (trainingHandle) return;
  if (!activeRunId) newRun();
  const steps = Math.max(1, parseInt(ui.stepsPerTick.value, 10) || 10);

  const loop = () => {
    trainOneTick(steps);
    trainingHandle = requestAnimationFrame(loop);
  };
  trainingHandle = requestAnimationFrame(loop);
}

function stopTraining() {
  if (!trainingHandle) return;
  cancelAnimationFrame(trainingHandle);
  trainingHandle = null;
  setStatus("Stopped");
}

// ------------------------
// Decision background (optional, lightweight)
// ------------------------
function drawDecisionBackground() {
  const N = 80; // higher = smoother (keep reasonable for speed)
  const cellW = canvas.width / N;
  const cellH = canvas.height / N;

  ctx.save();
  ctx.globalAlpha = 1.0; // fully opaque

  const TEMP = 10.0;      // tanh temperature (bigger => more saturated)
  const CONTRAST = 1.15; // optional extra saturation around 0.5

  for (let iy = 0; iy < N; iy++) {
    for (let ix = 0; ix < N; ix++) {
      const x = (ix / (N - 1)) * 2 - 1;
      const y = -((iy / (N - 1)) * 2 - 1);

      const yPred = model.forward(nj.array([[x], [y]]));
      const v = yPred.tolist()[0][0]; // raw output

      // tanh-like saturation instead of sigmoid
      const t = Math.tanh(TEMP * v);     // [-1,1]
      let prob = 0.5 * (t + 1.0);        // [0,1]

      // extra contrast (reduces gray band)
      prob = 0.5 + (prob - 0.5) * CONTRAST;
      prob = Math.max(0, Math.min(1, prob));

      // blend class colors (A=blue, B=red)
      const r = Math.round(90 * (1 - prob) + 235 * prob);
      const g = Math.round(140 * (1 - prob) + 80 * prob);
      const b = Math.round(235 * (1 - prob) + 90 * prob);

      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(ix * cellW, iy * cellH, cellW + 1, cellH + 1);
    }
  }

  ctx.restore();
}

function fitCanvasToCSS() {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.floor(rect.width * dpr));
  const h = Math.max(1, Math.floor(rect.height * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
    redraw();
  }
}

function renderRunsList() {
  if (!ui.runsList) return;
  ui.runsList.innerHTML = "";

  if (runs.length === 0) {
    const empty = document.createElement("div");
    empty.className = "runMeta";
    empty.textContent = "No runs yet";
    ui.runsList.appendChild(empty);
    return;
  }

  for (const r of runs) {
    const item = document.createElement("div");
    item.className = "runItem";
    item.style.outline = (r.id === activeRunId) ? "2px solid rgba(37,99,235,0.25)" : "none";

    const left = document.createElement("div");
    left.className = "runName";
    left.style.cursor = "pointer";
    left.title = "Click to select";
    left.addEventListener("click", () => {
      activeRunId = r.id;
      renderRunsList();
      redraw();
    });

    const name = document.createElement("strong");
    name.textContent = r.name;

    const meta = document.createElement("span");
    meta.className = "runMeta";
    const last = r.points.length ? r.points[r.points.length - 1] : null;
    meta.textContent = last ? `(${last.step} steps, ${last.loss.toFixed(4)})` : "(empty)";

    left.appendChild(name);
    left.appendChild(meta);

    const del = document.createElement("button");
    del.className = "danger";
    del.textContent = "Del";
    del.addEventListener("click", () => deleteRun(r.id));

    item.appendChild(left);
    item.appendChild(del);
    ui.runsList.appendChild(item);
  }
}

// ------------------------
// UI events
// ------------------------
window.addEventListener("resize", fitCanvasToCSS);
requestAnimationFrame(fitCanvasToCSS);

canvas.addEventListener("pointerdown", (e) => {
  const doAction = (ev) => {
    if (toolMode === "erase") {
      eraseAtEvent(ev);
      return;
    }

    // draw
    const { x, y } = canvasToXY(ev);
    points.push({ x, y, label: currentLabel });
    updateCounts();
    redraw();
  };

  doAction(e);

  // Shift+drag: spray (draw) OR scrub (erase)
  if (e.shiftKey) {
    const move = (ev) => doAction(ev);
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  }
});

ui.class0Btn.addEventListener("click", () => {
  toolMode = "draw";
  if (ui.eraseBtn) ui.eraseBtn.classList.remove("active");

  currentLabel = 0;
  ui.class0Btn.classList.add("active");
  ui.class1Btn.classList.remove("active");
});

ui.class1Btn.addEventListener("click", () => {
  toolMode = "draw";
  if (ui.eraseBtn) ui.eraseBtn.classList.remove("active");

  currentLabel = 1;
  ui.class1Btn.classList.add("active");
  ui.class0Btn.classList.remove("active");
});

ui.eraseBtn = document.getElementById("eraseBtn");

ui.eraseBtn.addEventListener("click", () => {
  toolMode = (toolMode === "erase") ? "draw" : "erase";
  ui.eraseBtn.classList.toggle("active", toolMode === "erase");

  // Optional: when erasing, visually de-emphasize class toggles
  const erasing = toolMode === "erase";
  ui.class0Btn.classList.toggle("active", !erasing && currentLabel === 0);
  ui.class1Btn.classList.toggle("active", !erasing && currentLabel === 1);

  setStatus(erasing ? "Erase mode" : "Draw mode");
});

ui.clearBtn.addEventListener("click", () => {
  points.length = 0;
  ui.lossValue.textContent = "—";
  setStatus("Cleared points");
  updateCounts();
  redraw();
});

ui.brushInput.addEventListener("input", () => {
  brush = parseInt(ui.brushInput.value, 10);
  redraw();
});

ui.bgToggleBtn.addEventListener("click", () => {
  showBackground = !showBackground;
  redraw();
});

ui.buildBtn.addEventListener("click", build);
ui.resetBtn.addEventListener("click", resetWeights);

ui.trainStepBtn.addEventListener("click", () => {
  const steps = Math.max(1, parseInt(ui.stepsPerTick.value, 10) || 10);
  trainOneTick(steps);
});

ui.trainBtn.addEventListener("click", startTraining);
ui.stopBtn.addEventListener("click", stopTraining);

ui.viewToggleBtn = document.getElementById("viewToggleBtn");
ui.newRunBtn = document.getElementById("newRunBtn");
ui.runsList = document.getElementById("runsList");

ui.viewToggleBtn.addEventListener("click", () => {
  viewMode = (viewMode === "grid") ? "graph" : "grid";
  ui.viewToggleBtn.textContent = (viewMode === "grid") ? "Graph" : "Grid";
  redraw();
});

ui.newRunBtn.addEventListener("click", () => {
  newRun();
});

// initial draw
updateCounts();
drawGrid();
