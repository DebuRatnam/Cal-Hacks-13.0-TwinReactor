/* Console: core editor, surrogate predictions, maps, the 3-D view and the agent chat.
   Talks to the backend on the same origin: GET /core, POST /predict, POST /chat.
   Everything about the core (grid, materials, rod banks, units, training ranges) comes from
   GET /core, so the page always matches the checkpoint the backend serves. */

let CORE = null, H = 0, W = 0, MATS = {};
// Fuel colors: hue = enrichment (4.2 blue, 4.5 orange), lightness = burnup (fresh brightest).
const FUEL_COLORS = { "4.2": ["#6da7ec", "#3a73c4", "#1f4a85"], "4.5": ["#f39a5e", "#d95926", "#93370f"] };
// Sequential blue, dark-to-light so high values glow on the dark ground.
const BLUE = ["#0d366b","#104281","#184f95","#1c5cab","#256abf","#2a78d6","#3987e5","#5598e7","#6da7ec","#86b6ef","#9ec5f4","#b7d3f6","#cde2fb"];
const VIEWS = [
  { id: "material", label: "Materials" },
  { id: "power", label: "Power", unit: "relative · mean 1 over fuel cells · assembly mean" },
  { id: "thermal", label: "Thermal flux", unit: "10⁻⁵ cm⁻² · ∫(φ₁+φ₂)dA = 1 over quarter core · assembly mean" },
  { id: "fast", label: "Fast flux", unit: "10⁻⁵ cm⁻² · ∫(φ₁+φ₂)dA = 1 over quarter core · assembly mean" },
  { id: "delta", label: "Δ power vs first", unit: "% change in assembly power (surrogate)" },
];
const DEFAULT_CONFIGS = [
  { label: "All rods out", rods: {} },
  { label: "Bank 4 in", rods: { 4: 1 } },
  { label: "Banks 4+8 in", rods: { 4: 1, 8: 1 } },
];

const state = { configs: [], sel: 0, view: "power", paint: null, full: false, busy: false };
const cfg = () => state.configs[state.sel];
const site = (i, j) => CORE.site[i][j];
const bankIds = () => Object.keys(CORE.banks);
function bankAt(i, j) { return bankIds().find((b) => CORE.banks[b].some(([a, c]) => a === i && c === j)) || null; }

// Quarter-core (i, j) for each displayed cell, top-left first. Row 0 / column 0 are half
// assemblies on the symmetry planes, so the full core is (2H-1) x (2W-1) with the centre row and
// column shared by both halves. Every mirror image shares the quarter's state.
function cellsFor(full) {
  const out = [];
  if (!full) { for (let i = H - 1; i >= 0; i--) for (let j = 0; j < W; j++) out.push([i, j]); return out; }
  const N = 2 * H - 1, M = 2 * W - 1;
  for (let R = 0; R < N; R++) for (let C = 0; C < M; C++) out.push([Math.abs(N - 1 - R - (H - 1)), Math.abs(C - (W - 1))]);
  return out;
}

/* ---------- color ---------- */
function hexToRgb(h) { const n = parseInt(h.slice(1), 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; }
function mix(a, b, t) { const A = hexToRgb(a), B = hexToRgb(b); return `rgb(${A.map((v, i) => Math.round(v + (B[i] - v) * t)).join(",")})`; }
function seq(t) { const x = Math.max(0, Math.min(1, t)) * (BLUE.length - 1); const i = Math.min(BLUE.length - 2, Math.floor(x)); return mix(BLUE[i], BLUE[i + 1], x - i); }
function div(t) { return t < 0 ? mix(css("--div-mid"), css("--div-lo"), Math.min(1, -t)) : mix(css("--div-mid"), css("--div-hi"), Math.min(1, t)); }
function lum(c) {
  const rgb = c.startsWith("#") ? hexToRgb(c) : c.match(/\d+/g).map(Number);
  const [r, g, b] = rgb.map((v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}
const textOn = (bg) => lum(bg) < 0.3 ? "#ffffff" : "#0b1220";

/* ---------- backend ---------- */
async function api(path, body) {
  const res = await fetch(path, body ? { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) } : {});
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : `HTTP ${res.status}`);
  return data;
}
const requestFor = (c) => ({ layout: c.layout, rods: c.rods, boron_ppm: c.boron });

async function predict(c) {
  c.result = await api("/predict", requestFor(c));
  c.stale = false;
}
async function runBusy(btn, text, fn) {
  if (state.busy) return;
  state.busy = true;
  const old = btn.textContent; btn.textContent = text;
  document.querySelectorAll("#predict, #staleRun").forEach((b) => b.disabled = true);
  try { await fn(); } catch (e) { toast(e.message, 6000); }
  finally {
    state.busy = false; btn.textContent = old;
    document.querySelectorAll("#predict, #staleRun").forEach((b) => b.disabled = false);
    renderAll();
  }
}

/* ---------- derived fields ---------- */
function fieldFor(c, view) {
  const r = c.result; if (!r || c.stale) return null;
  if (view === "power" || view === "thermal" || view === "fast") return r[view];
  if (view === "delta") {
    const b0 = state.configs[0]; if (!b0.result || b0.stale || c === b0) return null;
    const base = b0.result.power;
    return r.power.map((row, i) => row.map((v, j) => v != null && base[i][j] > 0 ? (v - base[i][j]) / base[i][j] * 100 : null));
  }
  return null;
}
const signed = (view) => view === "delta";

/* ---------- console render ---------- */
function renderConfigs() {
  const el = document.getElementById("cfgList"); el.innerHTML = "";
  state.configs.forEach((c, n) => {
    const b = document.createElement("button"); b.type = "button"; b.className = "cfg"; b.setAttribute("aria-pressed", n === state.sel);
    b.innerHTML = `<span>${esc(c.label)}${c.stale ? " *" : ""}</span><span class="k mono">${c.result && !c.stale ? "k " + fmt(c.result.k_eff, 5) : "no result"}</span>`;
    b.onclick = () => { state.sel = n; renderAll(); };
    el.appendChild(b);
  });
}
function renderBanks() {
  const el = document.getElementById("banks"); el.innerHTML = "";
  bankIds().forEach((b) => {
    const v = Math.round((cfg().rods[b] || 0) * 100);
    const row = document.createElement("div"); row.className = "bank";
    row.innerHTML = `<span class="bank-id ${v > 0 ? "in" : ""}">${b}</span>
      <label for="rod${b}"><small>${CORE.banks[b].map((p) => `(${p})`).join(" ")}</small><input id="rod${b}" type="range" min="0" max="100" step="5" value="${v}"></label>
      <output class="mono" for="rod${b}">${v}%</output>`;
    row.querySelector("input").oninput = (e) => {
      const f = e.target.value / 100;
      if (f > 0) cfg().rods[b] = f; else delete cfg().rods[b];
      row.querySelector("output").textContent = e.target.value + "%";
      row.querySelector(".bank-id").classList.toggle("in", f > 0); markStale();
    };
    el.appendChild(row);
  });
}
function renderPalette() {
  const el = document.getElementById("palette"); el.innerHTML = "";
  Object.values(MATS).filter((m) => m.fuel).forEach((m) => {
    const b = document.createElement("button"); b.type = "button"; b.className = "swatch"; b.setAttribute("aria-pressed", state.paint === m.id);
    b.innerHTML = `<i style="background:${m.color}"></i>${m.short} · ${esc(m.label)}`;
    b.onclick = () => { state.paint = state.paint === m.id ? null : m.id; renderPalette(); };
    el.appendChild(b);
  });
}
function renderViews() {
  const el = document.getElementById("views"); el.innerHTML = "";
  VIEWS.forEach((v) => {
    const b = document.createElement("button"); b.type = "button"; b.textContent = v.label; b.setAttribute("aria-pressed", state.view === v.id);
    b.onclick = () => { state.view = v.id; renderViews(); renderMap(); };
    el.appendChild(b);
  });
}
function paintAt(i, j) {
  if (state.paint == null) return;
  if (site(i, j) !== "fuel") { toast("Only fuel positions can be repainted. The reflector and core boundary are fixed in the training data."); return; }
  cfg().layout[i][j] = state.paint; markStale();
}
function renderMap() {
  const c = cfg(), grid = document.getElementById("grid"); grid.innerHTML = "";
  const f = fieldFor(c, state.view);
  const view = VIEWS.find((v) => v.id === state.view);
  const showAt = (i, j) => f && f[i][j] != null && (MATS[c.layout[i][j]]?.fuel || !["power", "delta"].includes(state.view));
  let lo = Infinity, hi = -Infinity;
  if (f) f.forEach((row, i) => row.forEach((v, j) => { if (showAt(i, j)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }));
  const dmax = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
  const nr = state.full ? 2 * H - 1 : H, nc = state.full ? 2 * W - 1 : W;
  grid.style.gridTemplateColumns = `repeat(${nc}, 1fr)`;
  grid.classList.toggle("full", state.full);
  const q = Array.from({ length: H }, () => Array(W));  // per quarter-core cell, shared with the 3-D view
  for (const [i, j] of cellsFor(state.full)) {
    if (site(i, j) === "void") {
      const empty = document.createElement("div"); empty.setAttribute("aria-hidden", "true");
      grid.appendChild(empty);
      q[i][j] = { bg: null, val: null }; continue;
    }
    const m = MATS[c.layout[i][j]], cell = document.createElement("button");
    cell.type = "button"; cell.className = "cell"; cell.setAttribute("role", "gridcell");
    let bg = m.color, text = m.short, val = null;
    if (state.view !== "material") {
      if (showAt(i, j)) {
        val = f[i][j];
        bg = signed(state.view) ? div(val / dmax) : seq((val - lo) / ((hi - lo) || 1));
        text = signed(state.view) ? (val > 0 ? "+" : "") + fmt(val, Math.abs(val) < 10 ? 1 : 0) : fmt(val, val >= 10 ? 1 : 2);
      } else { bg = css("--surface-2"); text = ""; }
    }
    q[i][j] = { bg, val };
    if (state.full && state.view !== "material") text = "";
    cell.style.background = bg;
    cell.style.color = bg.startsWith("#") || bg.startsWith("rgb") ? textOn(bg) : css("--ink-2");
    cell.textContent = text;
    const b = bankAt(i, j);
    if (b) { const r = document.createElement("span"); r.className = "rod" + ((c.rods[b] || 0) > 0 ? " in" : ""); r.textContent = b; cell.appendChild(r); }
    cell.setAttribute("aria-label", `Row ${i}, column ${j}, ${m.name}${b ? ", rod bank " + b : ""}${val != null ? ", " + view.label + " " + fmt(val, 3) : ""}`);
    cell.onmousemove = (e) => showTip(e, i, j, m, b, val, view);
    cell.onmouseleave = hideTip;
    cell.onclick = () => paintAt(i, j);
    grid.appendChild(cell);
  }
  const yax = document.getElementById("yaxis"), xax = document.getElementById("xaxis");
  const axisLabels = !state.full;
  yax.style.gridTemplateRows = `repeat(${nr}, 1fr)`; xax.style.gridTemplateColumns = `repeat(${nc}, 1fr)`;
  yax.innerHTML = axisLabels ? Array.from({ length: H }, (_, k) => `<span>${H - 1 - k}</span>`).join("") : "";
  xax.innerHTML = axisLabels ? Array.from({ length: W }, (_, k) => `<span>${k}</span>`).join("") : "";
  const pitch = CORE.assembly_pitch_cm;
  document.getElementById("bcNotes").innerHTML = state.full
    ? `<span>Full core, mirrored from the quarter-core model. Dashed lines: symmetry planes</span><span>Edits apply to all mirror images · pitch ${pitch} cm</span>`
    : `<span>Dashed edges: symmetry (reflective); row 0 / col 0 are half assemblies</span><span>Solid edges: vacuum boundary · pitch ${pitch} cm</span>`;
  document.getElementById("mapTitle").textContent = `${c.label} · ${view.label}`;
  renderLegend(f, lo, hi, dmax, view);
  update3D(c, q, hi, dmax, view);
}
function renderLegend(f, lo, hi, dmax, view) {
  const el = document.getElementById("legend");
  if (state.view === "material") {
    el.innerHTML = `<span class="label">Material</span>` + Object.values(MATS).map((m) => `<span class="cat"><i style="background:${m.color}"></i>${m.short || "—"} ${esc(m.label)}</span>`).join("")
      + `<span class="hint">wt% U-235 · GWd/t</span><span class="cat" style="margin-top:6px"><span class="bank-id" style="width:16px;height:16px;font-size:9px">4</span>rod bank</span>`;
    return;
  }
  if (!f) {
    const why = state.view === "delta" && cfg() === state.configs[0] ? "This is the reference configuration." : "No result yet.";
    el.innerHTML = `<span class="label">${view.label}</span><span class="hint">${why}</span>`; return;
  }
  const stops = signed(state.view) ? [1, 0.5, 0, -0.5, -1].map((t) => div(t)) : [1, 0.75, 0.5, 0.25, 0].map((t) => seq(t));
  const ticks = signed(state.view) ? [`+${fmt(dmax, 1)}%`, "0%", `−${fmt(dmax, 1)}%`] : [fmt(hi, 3), fmt((lo + hi) / 2, 3), fmt(lo, 3)];
  el.innerHTML = `<span class="label">${view.label}</span><div class="legend-scale"><div class="ramp" style="background:linear-gradient(${stops.join(",")})"></div><div class="legend-ticks mono">${ticks.map((t) => `<span>${t}</span>`).join("")}</div></div><span class="hint">${view.unit}</span>`;
}
function renderStats() {
  const c = cfg(), el = document.getElementById("stats"), r = c.result;
  if (!r || c.stale) {
    el.innerHTML = ["k_eff", "Reactivity", "Δρ vs first", "Assembly peaking"].map((l) => `<div class="stat"><span class="label">${l}</span><span class="v">—</span><span class="s">run Predict</span></div>`).join(""); return;
  }
  const base = state.configs[0].result, first = c === state.configs[0] || !base || state.configs[0].stale;
  const d = first ? null : rho(r.k_eff) - rho(base.k_eff);
  const mae = CORE.model.test_k_mae_pcm;
  const kNote = `surrogate · test MAE ${mae != null ? fmt(mae, 0) : "?"} pcm`;
  el.innerHTML = `
    <div class="stat"><span class="label">k_eff</span><span class="v">${fmt(r.k_eff, 5)}</span><span class="s">${kNote}</span></div>
    <div class="stat"><span class="label">Reactivity</span><span class="v">${fmt(rho(r.k_eff), 0)}</span><span class="s">pcm, (k−1)/k</span></div>
    <div class="stat"><span class="label">Δρ vs ${esc(state.configs[0].label)}</span><span class="v">${d == null ? "ref" : (d > 0 ? "+" : "−") + fmt(Math.abs(d), 0)}</span><span class="s">${d == null ? "reference configuration" : "pcm (surrogate)"}</span></div>
    <div class="stat"><span class="label">Assembly peaking</span><span class="v">${fmt(r.assembly_peaking, 3)}</span><span class="s">at (${r.assembly_peak_row_col.join(", ")}); mesh-cell max ${fmt(r.node_peaking, 2)}</span></div>`;
}
function renderTable() {
  const base = state.configs[0].result;
  document.getElementById("cmpBody").innerHTML = state.configs.map((c, n) => {
    const r = c.result;
    if (!r || c.stale) return `<tr class="${n === state.sel ? "sel" : ""}"><td>${esc(c.label)}</td><td colspan="5" style="text-align:left;color:var(--ink-3)">no current result</td></tr>`;
    const d = n === 0 || !base ? "—" : fmt(rho(r.k_eff) - rho(base.k_eff), 1);
    return `<tr class="${n === state.sel ? "sel" : ""}"><td>${esc(c.label)}</td><td>${fmt(r.k_eff, 5)}</td><td>${fmt(rho(r.k_eff), 1)}</td><td>${d}</td><td>${fmt(r.assembly_peaking, 3)}</td><td>(${r.assembly_peak_row_col.join(", ")})</td></tr>`;
  }).join("");
}
function renderBanners() {
  const c = cfg();
  document.getElementById("staleBanner").hidden = !c.stale;
  const w = c.result && !c.stale ? c.result.warnings : [];
  const ood = document.getElementById("oodBanner");
  ood.hidden = !w.length;
  ood.innerHTML = w.length ? `<span><b>Check this prediction.</b> ${w.map(esc).join(" ")}</span>` : "";
}
function renderAll() {
  document.getElementById("boron").value = cfg().boron;
  renderConfigs(); renderBanks(); renderMap(); renderStats(); renderTable(); renderBanners();
}
function markStale() { const c = cfg(); c.stale = true; renderConfigs(); renderMap(); renderStats(); renderTable(); renderBanners(); }

/* ---------- tooltips ---------- */
function showTip(e, i, j, m, b, val, view) {
  const c = cfg();
  placeTip(e, `<b>(${i}, ${j})</b>${state.full ? " (quarter-core position)" : ""} · ${esc(m.name)}${b ? `<br>Rod bank ${b}: ${Math.round((c.rods[b] || 0) * 100)}% inserted` : ""}${val != null ? `<br>${view.label}: <span class="mono">${signed(state.view) ? fmt(val, 1) + "%" : fmt(val, 3)}</span>` : ""}${state.paint != null && site(i, j) === "fuel" ? `<br>Click to set ${esc(MATS[state.paint].name)}` : ""}`);
}

/* ---------- 3-D core view ---------- */
// The model is 2-D. Each assembly is drawn as a column of uniform color over an assumed active
// height; nothing here is an axial result. Rods are drawn at their insertion depth for orientation.
let HC = 17;  // display-only active height in pitches; set from the core pitch (366 cm, typical 4-loop PWR)
const V3 = { mode: "core", cut: true, visible: true, api: null };
const mapPane = document.querySelector(".map-pane");
function init3D() {
  HC = 366 / CORE.assembly_pitch_cm;
  const stage = document.getElementById("v3dStage"), canvas = document.getElementById("v3dCanvas");
  const fail = (why) => stage.insertAdjacentHTML("beforeend", `<div class="v3d-fail">3D view unavailable: ${why}. The 2-D map has the same data.</div>`);
  if (typeof THREE === "undefined") return fail("the WebGL library didn't load");
  let renderer;
  try { renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true }); } catch (e) { return fail("WebGL isn't available in this browser"); }
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(30, 1, 0.5, 400);
  scene.add(new THREE.HemisphereLight(0xd6ebff, 0x0a1424, 0.85));
  const sun = new THREE.DirectionalLight(0xffffff, 0.65); sun.position.set(20, 44, 30); scene.add(sun);
  const rim = new THREE.DirectionalLight(0x47b4ff, 0.3); rim.position.set(-30, 12, -24); scene.add(rim);

  // Assemblies: full core mirrored from the quarter, x east, z south, y up; 1 unit = 1 pitch,
  // centre assembly at the origin.
  const box = new THREE.BoxGeometry(0.94, 1, 0.94); box.translate(0, 0.5, 0);
  const cols = [], M = 2 * W - 1;
  cellsFor(true).forEach(([i, j], n) => {
    if (site(i, j) === "void") return;
    const x = n % M - (W - 1), z = Math.floor(n / M) - (H - 1);
    const mesh = new THREE.Mesh(box, new THREE.MeshStandardMaterial({ roughness: 0.6, metalness: 0.05 }));
    mesh.position.set(x, 0, z); mesh.scale.y = 0.01;
    mesh.userData = { i, j, cut: x > 0 && z > 0, h: 0.01, fuel: null };
    scene.add(mesh); cols.push(mesh);
  });
  const R = Math.max(...cols.map((m) => Math.hypot(Math.abs(m.position.x) + 0.5, Math.abs(m.position.z) + 0.5)));

  // Rod cluster control assemblies: 8 rodlets hanging from a spider at the top of the core.
  const rodGeo = new THREE.CylinderGeometry(0.05, 0.05, 1, 6); rodGeo.translate(0, -0.5, 0);
  const shaftGeo = new THREE.CylinderGeometry(0.07, 0.07, 2.6, 8); shaftGeo.translate(0, 1.3, 0);
  const hubGeo = new THREE.BoxGeometry(0.7, 0.08, 0.08);
  const rodMat = {
    in: new THREE.MeshStandardMaterial({ color: 0x47b4ff, emissive: 0x47b4ff, emissiveIntensity: 0.75, roughness: 0.3 }),
    out: new THREE.MeshStandardMaterial({ color: 0x6b7d92, roughness: 0.5, metalness: 0.4 }),
  };
  const rods = [];
  for (const col of cols) {
    const bank = bankAt(col.userData.i, col.userData.j); if (!bank) continue;
    const g = new THREE.Group(); g.position.set(col.position.x, HC, col.position.z);
    const lets = [];
    for (const dx of [-0.28, 0, 0.28]) for (const dz of [-0.28, 0, 0.28]) {
      if (!dx && !dz) continue;
      const r = new THREE.Mesh(rodGeo, rodMat.in); r.position.set(dx, 0, dz); r.scale.y = 0.001; r.visible = false; g.add(r); lets.push(r);
    }
    const shaft = new THREE.Mesh(shaftGeo, rodMat.out), hubA = new THREE.Mesh(hubGeo, rodMat.out), hubB = hubA.clone();
    hubB.rotation.y = Math.PI / 2; hubA.position.y = hubB.position.y = 0.06; g.add(shaft, hubA, hubB);
    scene.add(g); rods.push({ g, bank, lets, parts: [shaft, hubA, hubB], f: 0, target: 0 });
  }

  // Core barrel, vessel outline, floor and the two symmetry planes (dashed, as on the 2-D map).
  const ring = (y, r, color) => {
    const pts = []; for (let a = 0; a <= 72; a++) { const t = a / 72 * 2 * Math.PI; pts.push(new THREE.Vector3(r * Math.cos(t), y, r * Math.sin(t))); }
    return new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color }));
  };
  const rb = R + 0.3, rv = R + 1.2;
  scene.add(ring(0, rb, 0x24344a), ring(HC, rb, 0x24344a), ring(-0.4, rv, 0x172231), ring(HC + 2.6, rv, 0x172231));
  const posts = [];
  for (let a = 0; a < 12; a++) { const t = a / 12 * 2 * Math.PI, x = rv * Math.cos(t), z = rv * Math.sin(t); posts.push(new THREE.Vector3(x, -0.4, z), new THREE.Vector3(x, HC + 2.6, z)); }
  scene.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(posts), new THREE.LineBasicMaterial({ color: 0x172231 })));
  const floor = new THREE.Mesh(new THREE.CircleGeometry(rv, 72), new THREE.MeshStandardMaterial({ color: 0x0e1724, roughness: 1 }));
  floor.rotation.x = -Math.PI / 2; floor.position.y = -0.03; scene.add(floor);
  const sym = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-rv, 0.005, 0), new THREE.Vector3(rv, 0.005, 0), new THREE.Vector3(0, 0.005, -rv), new THREE.Vector3(0, 0.005, rv)]),
    new THREE.LineDashedMaterial({ color: 0xa3dcff, dashSize: 0.45, gapSize: 0.3 }));
  sym.computeLineDistances(); scene.add(sym);

  // Orbit camera. Plain scroll stays with the page; pinch (ctrl+wheel) zooms.
  const HOME = { az: Math.PI / 4, el: 0.5, dist: 64 };
  const view = { ...HOME }, tgt = new THREE.Vector3(0, HC * 0.48, 0);
  let spin = !REDUCE, drag = null;
  const place = () => { const c = Math.cos(view.el); camera.position.set(tgt.x + view.dist * c * Math.sin(view.az), tgt.y + view.dist * Math.sin(view.el), tgt.z + view.dist * c * Math.cos(view.az)); camera.lookAt(tgt); };
  const resize = () => { const w = stage.clientWidth, h = stage.clientHeight; if (!w || !h) return; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); };
  new ResizeObserver(resize).observe(stage);
  let frame = 0;
  const onScreen = () => { const r = stage.getBoundingClientRect(); return r.bottom > 0 && r.top < innerHeight && r.width > 0; };

  const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
  const pick = (e) => {
    const r = canvas.getBoundingClientRect();
    ndc.set((e.clientX - r.left) / r.width * 2 - 1, -(e.clientY - r.top) / r.height * 2 + 1); ray.setFromCamera(ndc, camera);
    const hits = ray.intersectObjects(cols);
    return (hits.find((h) => h.object.userData.fuel) || hits[0] || {}).object;
  };
  canvas.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY, moved: 0 }; spin = false; canvas.setPointerCapture(e.pointerId); stage.classList.add("drag"); hideTip(); });
  canvas.addEventListener("pointermove", (e) => {
    if (drag) {
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.moved += Math.abs(dx) + Math.abs(dy); drag.x = e.clientX; drag.y = e.clientY;
      view.az -= dx * 0.008; view.el = Math.max(0.06, Math.min(1.45, view.el + dy * 0.006)); return;
    }
    const m = pick(e); if (!m) return hideTip();
    const { i, j } = m.userData, cell = V3.q && V3.q[i][j];
    showTip(e, i, j, MATS[cfg().layout[i][j]], bankAt(i, j), cell ? cell.val : null, VIEWS.find((v) => v.id === state.view));
  });
  canvas.addEventListener("pointerup", (e) => {
    if (!drag) return; const click = drag.moved < 5; drag = null; stage.classList.remove("drag");
    if (click && state.paint != null) { const m = pick(e); if (m) paintAt(m.userData.i, m.userData.j); }
  });
  canvas.addEventListener("pointercancel", () => { drag = null; stage.classList.remove("drag"); });
  canvas.addEventListener("pointerleave", () => { if (!drag) hideTip(); });
  canvas.addEventListener("wheel", (e) => { if (!e.ctrlKey && !e.metaKey) return; e.preventDefault(); view.dist = Math.max(24, Math.min(110, view.dist * Math.exp(e.deltaY * 0.01))); }, { passive: false });

  const k = REDUCE ? 1 : 0.14;
  (function tick() {
    requestAnimationFrame(tick);
    if (frame++ % 20 === 0) V3.visible = onScreen();
    if (!V3.visible || mapPane.dataset.dim === "2d") return;
    if (spin) view.az += 0.0018;
    for (const m of cols) m.scale.y += (m.userData.h - m.scale.y) * k;
    for (const r of rods) {
      r.f += (r.target - r.f) * k;
      const s = Math.max(0.001, r.f * HC); r.lets.forEach((l) => { l.scale.y = s; l.visible = r.f > 0.004; });
    }
    place(); renderer.render(scene, camera);
  })();
  V3.api = { cols, rods, rodMat, resize, reset: () => { Object.assign(view, HOME); spin = false; } };
  resize();
}
function update3D(c, q, hi, dmax, view) {
  V3.q = q;
  if (!V3.api) return;
  const relief = V3.mode === "relief", sgn = signed(state.view);
  for (const m of V3.api.cols) {
    const { i, j, cut } = m.userData, fuel = MATS[c.layout[i][j]].fuel, { bg, val } = q[i][j], mat = m.material;
    const flat = V3.cut && cut;
    m.userData.h = flat ? 0.14
      : !relief ? HC
      : val != null ? 0.3 + (sgn ? Math.abs(val) / dmax * 9 : Math.max(0, val) / (hi || 1) * 11)
      : fuel ? 0.5 : 0.14;
    if (m.userData.fuel !== fuel) { mat.transparent = !fuel; mat.depthWrite = fuel; mat.needsUpdate = true; m.userData.fuel = fuel; }
    if (fuel) { mat.color.setStyle(bg); mat.emissive.setStyle(bg); mat.emissiveIntensity = 0.22; mat.opacity = 1; }
    else if (relief && val != null) { mat.color.setStyle(bg); mat.emissive.setStyle(bg); mat.emissiveIntensity = 0.3; mat.opacity = 0.55; }
    else { mat.color.set(0x1d5a8a); mat.emissive.set(0x0c2740); mat.emissiveIntensity = 0.5; mat.opacity = flat ? 0.35 : 0.12; }
  }
  for (const r of V3.api.rods) {
    r.target = c.rods[r.bank] || 0;
    r.g.visible = !relief;  // rod depth has no meaning once column height encodes a value
    r.parts.forEach((p) => { p.material = r.target > 0 ? V3.api.rodMat.in : V3.api.rodMat.out; });
  }
  const ins = bankIds().filter((b) => (c.rods[b] || 0) > 0).map((b) => `${b} ${Math.round(c.rods[b] * 100)}%`).join(" · ") || "none";
  document.getElementById("v3dAxis").innerHTML = `Full core · ${view.label}<br>Banks inserted: ${ins}`;
  document.getElementById("v3dNote").textContent = relief
    ? `Relief: column height is the selected quantity, the same values as the 2-D map (taller = higher${sgn ? " magnitude; color gives the sign" : ""}). Reflector is translucent.`
    : `Display only. The surrogate is 2-D, so each assembly is extruded to an assumed 366 cm active height with uniform color; no axial power shape is computed. Rods are drawn at their insertion depth. ${V3.cut ? "The cutaway quadrant shows the map on the core floor." : ""}`;
}

/* ---------- controls ---------- */
document.querySelectorAll("#dims button").forEach((b) => b.onclick = () => {
  mapPane.dataset.dim = b.dataset.dim;
  document.querySelectorAll("#dims button").forEach((x) => x.setAttribute("aria-pressed", x === b));
  if (V3.api) V3.api.resize();
});
document.querySelectorAll("#v3dMode button").forEach((b) => b.onclick = () => {
  V3.mode = b.dataset.m;
  document.querySelectorAll("#v3dMode button").forEach((x) => x.setAttribute("aria-pressed", x === b));
  if (CORE) renderMap();
});
document.getElementById("v3dCut").onclick = (e) => { V3.cut = !V3.cut; e.currentTarget.setAttribute("aria-pressed", V3.cut); if (CORE) renderMap(); };
document.getElementById("v3dReset").onclick = () => { if (V3.api) V3.api.reset(); };
document.querySelectorAll("#scope button").forEach((b) => b.onclick = () => {
  state.full = b.dataset.full === "1";
  document.querySelectorAll("#scope button").forEach((x) => x.setAttribute("aria-pressed", x === b));
  if (CORE) renderMap();
});
document.getElementById("boron").addEventListener("change", (e) => {
  const v = Number(e.target.value);
  if (!(v >= 0 && v <= 2000)) { toast("Boron must be within the tabulated 0–2000 ppm range."); e.target.value = cfg().boron; return; }
  cfg().boron = v; markStale();
});
const predictBtn = document.getElementById("predict");
const onPredict = () => runBusy(predictBtn, "Predicting…", () => predict(cfg()));
predictBtn.onclick = onPredict;
document.getElementById("staleRun").onclick = onPredict;
document.getElementById("newCfg").onclick = () => {
  const c = cfg();
  state.configs.push({ label: `Config ${state.configs.length + 1}`, rods: { ...c.rods }, layout: clone(c.layout), boron: c.boron, result: null, stale: true });
  state.sel = state.configs.length - 1; renderAll();
};
document.getElementById("export").onclick = () => {
  const out = {
    note: "ReactorTwin research surrogate output.",
    model: CORE.model, units: CORE.units, assembly_pitch_cm: CORE.assembly_pitch_cm, materials: CORE.materials, banks: CORE.banks,
    configurations: state.configs.map((c) => ({ label: c.label, layout: c.layout, rods: c.rods, boron_ppm: c.boron, surrogate: c.stale ? null : c.result })),
  };
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 2)], { type: "application/json" }));
  a.download = "reactortwin-results.json"; a.click(); URL.revokeObjectURL(a.href);
};

/* ---------- chat: the agent runs the same surrogate (POST /chat) ---------- */
const msgs = document.getElementById("msgs");
const CHAT_ID = (crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2));  // one conversation per page load
function addMsg(kind, html) { const d = document.createElement("div"); d.className = "msg " + kind; d.innerHTML = html; msgs.appendChild(d); msgs.scrollTop = msgs.scrollHeight; return d; }
function addTools(calls) {
  if (!calls.length) return;
  const label = (c) => c.name === "evaluate_configurations" ? `evaluate · ${c.input.configurations?.length ?? "?"} configs`
    : c.name === "get_map" ? `get_map · ${c.input.quantity}${c.input.relative_to ? " Δ" : ""}`
    : c.name === "verify_with_reference" ? `verify · ${c.input.label}` : c.name;
  const d = document.createElement("div"); d.className = "tools";
  d.innerHTML = calls.map((c) => `<span class="tool">${esc(label(c))}</span>`).join("");
  msgs.appendChild(d);
}
// Minimal markdown for agent replies: paragraphs, lists, tables, **bold**, `code`, headings.
function md(src) {
  const inline = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/`([^`]+)`/g, "<code>$1</code>");
  const out = [], lines = src.split("\n");
  for (let k = 0; k < lines.length;) {
    const l = lines[k];
    if (/^\s*\|/.test(l)) {
      const rows = []; while (k < lines.length && /^\s*\|/.test(lines[k])) rows.push(lines[k++]);
      const cells = (r) => r.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      const body = rows.filter((r) => !/^\s*\|?\s*:?-{2,}/.test(r));
      out.push(`<div class="table-wrap" style="padding:0"><table><thead><tr>${cells(body[0]).map((c) => `<th>${inline(c)}</th>`).join("")}</tr></thead><tbody>${body.slice(1).map((r) => `<tr>${cells(r).map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`);
    } else if (/^\s*([-*]|\d+\.)\s+/.test(l)) {
      const items = []; while (k < lines.length && /^\s*([-*]|\d+\.)\s+/.test(lines[k])) items.push(lines[k++].replace(/^\s*([-*]|\d+\.)\s+/, ""));
      out.push(`<ul style="margin:0 0 6px;padding-left:18px">${items.map((i) => `<li>${inline(i)}</li>`).join("")}</ul>`);
    } else if (/^#{1,6}\s/.test(l)) { out.push(`<p><b>${inline(l.replace(/^#+\s/, ""))}</b></p>`); k++; }
    else if (!l.trim()) k++;
    else {
      const para = []; while (k < lines.length && lines[k].trim() && !/^\s*(\||[-*]\s|\d+\.\s|#)/.test(lines[k])) para.push(lines[k++]);
      if (!para.length) para.push(lines[k++]);
      out.push(`<p>${inline(para.join(" "))}</p>`);
    }
  }
  return out.join("");
}
// Configurations the agent evaluated join the console list (same label -> replaced), so their
// maps are one click away. They come back with surrogate results already attached.
function adoptConfigs(configs) {
  for (const a of configs) {
    const c = { label: a.label, rods: { ...a.rods }, layout: a.layout, boron: a.boron_ppm, result: a.result, stale: false };
    const n = state.configs.findIndex((x) => x.label === a.label);
    if (n >= 0) state.configs[n] = c; else state.configs.push(c);
  }
  if (configs.length) { state.sel = state.configs.findIndex((x) => x.label === configs[configs.length - 1].label); renderAll(); }
}
addMsg("bot", `<p>Ask about rod worth, boron, power shape or fuel swaps. I run each case through the trained surrogate, starting from the core selected on the left, and the configurations I evaluate appear in the list.</p>`);
const SUGGEST = ["What is each rod bank worth at full insertion?", "Which single bank flattens power the most?", "How does going from 500 to 2000 ppm boron change k_eff and the power shape?"];
document.getElementById("suggest").innerHTML = SUGGEST.map((s) => `<button class="chip" type="button">${s}</button>`).join("");
document.querySelectorAll(".chip").forEach((c) => c.onclick = () => { document.getElementById("ask").value = c.textContent; document.getElementById("ask").focus(); });
let chatBusy = false;
document.getElementById("composer").addEventListener("submit", async (e) => {
  e.preventDefault(); const t = document.getElementById("ask"), q = t.value.trim(); if (!q || chatBusy) return;
  if (!CORE) { toast("The backend isn't connected."); return; }
  chatBusy = true; addMsg("user", esc(q)); t.value = "";
  const wait = addMsg("sys", "Working… (planning cases and running the surrogate)");
  const send = document.querySelector("#composer button"); send.disabled = true;
  try {
    const c = cfg();
    const res = await api("/chat", { session_id: CHAT_ID, message: q, core: { layout: c.layout, rods: c.rods, boron_ppm: c.boron } });
    wait.remove();
    addTools(res.tool_calls);
    addMsg("bot", md(res.reply || "(no answer)"));
    adoptConfigs(res.configs);
  } catch (err) { wait.remove(); addMsg("sys", esc(err.message)); }
  finally { chatBusy = false; send.disabled = false; }
});
document.getElementById("ask").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); document.getElementById("composer").requestSubmit(); } });

/* ---------- start ---------- */
async function initConsole() {
  try { CORE = await api("/core"); }
  catch (e) {
    document.getElementById("modelPill").innerHTML = `<span class="dot"></span>Model: backend unreachable`;
    document.getElementById("sourceNote").textContent = "Couldn't reach the backend. Start it with `uvicorn backend.app:app` and open this page from http://127.0.0.1:8000/.";
    return;
  }
  [H, W] = CORE.grid;
  for (const m of CORE.materials) {
    const k = m.fuel ? CORE.materials.filter((x) => x.fuel && x.enrichment_wt === m.enrichment_wt).indexOf(m) : -1;
    MATS[m.id] = {
      ...m, fuel: m.fuel,
      label: m.fuel ? `${m.enrichment_wt} wt%, ${m.burnup_gwd_t} GWd/t` : "Reflector",
      short: m.fuel ? `${m.enrichment_wt >= 4.5 ? "H" : "L"}${k}` : "",
      color: m.fuel ? FUEL_COLORS[String(m.enrichment_wt)][k] : css("--refl"),
    };
  }
  const md = CORE.model;
  document.getElementById("modelPill").innerHTML = `<span class="dot" style="background:var(--accent)"></span>Model: ${esc(md.name)}`;
  const acc = document.getElementById("accPill"); acc.hidden = false;
  acc.textContent = `Test (${md.test_samples} cores): k MAE ${fmt(md.test_k_mae_pcm, 0)} pcm · p95 ${fmt(md.test_k_p95_pcm, 0)} pcm · power L2 ${fmt(md.test_power_rel_l2 * 100, 1)}%`;
  document.getElementById("sourceNote").textContent =
    `Surrogate: ${md.name}, trained on dataset ${md.dataset} (${md.xs_source}). ${CORE.scope} `
    + `Mesh ${CORE.mesh.shape.join("×")} (${CORE.mesh.cell_cm} cm cells, ${CORE.mesh.cells_per_assembly} per ${CORE.assembly_pitch_cm} cm assembly); maps show assembly means. `
    + `Power: ${CORE.units.power}. Flux: ${CORE.units.flux}, displayed ×10⁵. ${CORE.geometry_note}`;
  state.configs = DEFAULT_CONFIGS.map((d) => ({ label: d.label, rods: { ...d.rods }, layout: clone(CORE.default_layout), boron: CORE.default_boron_ppm, result: null, stale: true }));
  init3D();
  renderViews(); renderPalette(); renderAll();
  try { await Promise.all(state.configs.map(predict)); } catch (e) { toast(e.message, 6000); }
  renderAll();
}
initConsole();
