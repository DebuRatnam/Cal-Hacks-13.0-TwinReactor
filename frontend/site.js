/* Landing-page visuals: hero core twin, speed chart, photos, scroll choreography.
   Loaded before console.js, which reuses the shared helpers below. */

/* ---------- shared helpers ---------- */
const REDUCE = matchMedia("(prefers-reduced-motion: reduce)").matches;
if (!REDUCE) document.documentElement.classList.add("anim");
const clone = (o) => JSON.parse(JSON.stringify(o));
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const rho = (k) => (k - 1) / k * 1e5;
const fmt = (x, d) => x.toFixed(d);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const tip = document.getElementById("tip");
function placeTip(e, html) {
  tip.innerHTML = html; tip.hidden = false;
  const x = Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8), y = Math.min(e.clientY + 14, innerHeight - tip.offsetHeight - 8);
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
function hideTip() { tip.hidden = true; }
let toastTimer;
function toast(msg, ms = 3200) { const t = document.getElementById("toast"); t.textContent = msg; t.hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => t.hidden = true, ms); }

/* ---------- hero twin sample data ----------
   Illustration only: stand-in FV solver with placeholder IAEA 2-D XS (not the trained model's core).
   The console below the fold runs the real surrogate. */
const SAMPLE ={"layout":[["fuel_high","fuel_low","fuel_high","fuel_low","fuel_high","fuel_low","fuel_low","fuel_low","reflector"],["fuel_low","fuel_high","fuel_low","fuel_high","fuel_low","fuel_high","fuel_low","fuel_low","reflector"],["fuel_high","fuel_low","fuel_high","fuel_low","fuel_high","fuel_low","fuel_low","reflector","reflector"],["fuel_low","fuel_high","fuel_low","fuel_high","fuel_low","fuel_low","fuel_low","reflector","reflector"],["fuel_high","fuel_low","fuel_high","fuel_low","fuel_low","fuel_low","reflector","reflector","reflector"],["fuel_low","fuel_high","fuel_low","fuel_low","fuel_low","fuel_low","reflector","reflector","reflector"],["fuel_low","fuel_low","fuel_low","fuel_low","reflector","reflector","reflector","reflector","reflector"],["fuel_low","fuel_low","reflector","reflector","reflector","reflector","reflector","reflector","reflector"],["reflector","reflector","reflector","reflector","reflector","reflector","reflector","reflector","reflector"]],"banks":{"A":[[0,0]],"B":[[0,4],[4,0]],"C":[[2,2],[4,4]],"D":[[2,6],[6,2]]},"pitch_cm":21.5,"results":{"All rods out":{"rods":{},"k_eff":1.07996,"power":[[1.438,1.553,1.395,1.386,1.159,1.078,0.843,0.502,0.0],[1.553,1.437,1.446,1.268,1.184,0.942,0.742,0.446,0.0],[1.395,1.446,1.303,1.262,1.054,0.909,0.646,0.0,0.0],[1.386,1.268,1.262,1.101,1.023,0.788,0.498,0.0,0.0],[1.159,1.184,1.054,1.023,0.863,0.618,0.0,0.0,0.0],[1.078,0.942,0.909,0.788,0.618,0.371,0.0,0.0,0.0],[0.843,0.742,0.646,0.498,0.0,0.0,0.0,0.0,0.0],[0.502,0.446,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],"fast":[[44.962,46.497,43.544,41.522,36.173,32.225,25.09,13.701,1.865],[46.497,44.76,43.379,39.481,35.528,29.342,21.992,10.956,1.383],[43.544,43.379,40.579,37.88,32.83,27.167,17.463,3.801,0.399],[41.522,39.481,37.88,34.272,30.573,23.307,12.249,1.701,0.093],[36.173,35.528,32.83,30.573,25.659,16.685,3.809,0.439,0.028],[32.225,29.342,27.167,23.307,16.685,9.2,1.377,0.085,0.006],[25.09,21.992,17.463,12.249,3.809,1.377,0.265,0.02,0.001],[13.701,10.956,3.801,1.701,0.439,0.085,0.02,0.003,0.0],[1.865,1.383,0.399,0.093,0.028,0.006,0.001,0.0,0.0]],"thermal":[[10.65,11.501,10.337,10.27,8.587,7.988,6.248,3.719,4.583],[11.501,10.646,10.709,9.39,8.77,6.981,5.493,3.304,3.465],[10.337,10.709,9.651,9.351,7.809,6.733,4.785,8.441,1.813],[10.27,9.39,9.351,8.152,7.578,5.835,3.692,4.708,0.63],[8.587,8.77,7.809,7.578,6.389,4.58,8.402,2.08,0.208],[7.988,6.981,6.733,5.835,4.58,2.746,3.868,0.596,0.051],[6.248,5.493,4.785,3.692,8.402,3.868,1.208,0.152,0.012],[3.719,3.304,8.441,4.708,2.08,0.596,0.152,0.023,0.002],[4.583,3.465,1.813,0.63,0.208,0.051,0.012,0.002,0.0]]},"Bank A in":{"rods":{"A":1},"k_eff":1.07517,"power":[[0.283,0.84,1.099,1.318,1.234,1.235,1.006,0.612,0.0],[0.84,0.992,1.235,1.248,1.282,1.086,0.887,0.543,0.0],[1.099,1.235,1.236,1.313,1.175,1.062,0.775,0.0,0.0],[1.318,1.248,1.313,1.21,1.178,0.934,0.601,0.0,0.0],[1.234,1.282,1.175,1.178,1.019,0.743,0.0,0.0,0.0],[1.235,1.086,1.062,0.934,0.743,0.45,0.0,0.0,0.0],[1.006,0.887,0.775,0.601,0.0,0.0,0.0,0.0,0.0],[0.612,0.543,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],"fast":[[12.602,25.414,34.316,39.488,38.513,36.915,29.944,16.7,2.287],[25.414,30.917,37.066,38.884,38.484,33.817,26.299,13.348,1.694],[34.316,37.066,38.513,39.418,36.602,31.739,20.949,4.611,0.487],[39.488,38.884,39.418,37.708,35.202,27.642,14.77,2.062,0.113],[38.513,38.484,36.602,35.202,30.313,20.049,4.606,0.532,0.034],[36.915,33.817,31.739,27.642,20.049,11.165,1.675,0.103,0.007],[29.944,26.299,20.949,14.77,4.606,1.675,0.323,0.025,0.002],[16.7,13.348,4.611,2.062,0.532,0.103,0.025,0.003,0.0],[2.287,1.694,0.487,0.113,0.034,0.007,0.002,0.0,0.0]],"thermal":[[2.099,6.22,8.143,9.763,9.139,9.148,7.453,4.534,5.621],[6.22,7.345,9.147,9.244,9.496,8.042,6.567,4.024,4.245],[8.143,9.147,9.156,9.727,8.703,7.864,5.743,10.241,2.211],[9.763,9.244,9.727,8.966,8.723,6.918,4.453,5.706,0.764],[9.139,9.496,8.703,8.723,7.545,5.504,10.163,2.521,0.252],[9.148,8.042,7.864,6.918,5.504,3.333,4.701,0.724,0.061],[7.453,6.567,5.743,4.453,10.163,4.701,1.472,0.186,0.014],[4.534,4.024,10.241,5.706,2.521,0.724,0.186,0.029,0.003],[5.621,4.245,2.211,0.764,0.252,0.061,0.014,0.003,0.0]]},"Banks A+B in":{"rods":{"A":1,"B":1},"k_eff":1.0628,"power":[[0.314,0.907,1.077,0.978,0.429,0.839,0.882,0.59,0.0],[0.907,1.082,1.303,1.179,1.059,0.967,0.866,0.555,0.0],[1.077,1.303,1.378,1.475,1.304,1.185,0.869,0.0,0.0],[0.978,1.179,1.475,1.474,1.482,1.182,0.754,0.0,0.0],[0.429,1.059,1.304,1.482,1.354,1.003,0.0,0.0,0.0],[0.839,0.967,1.185,1.182,1.003,0.625,0.0,0.0,0.0],[0.882,0.866,0.869,0.754,0.0,0.0,0.0,0.0,0.0],[0.59,0.555,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],"fast":[[13.988,27.485,33.649,29.672,18.866,25.403,26.287,16.08,2.259],[27.485,33.767,39.153,36.787,32.152,30.174,25.708,13.565,1.732],[33.649,39.153,42.981,44.332,40.675,35.457,23.524,5.068,0.519],[29.672,36.787,44.332,45.983,44.357,35.006,18.482,2.524,0.133],[18.866,32.152,40.675,44.357,40.326,27.107,6.127,0.686,0.043],[25.403,30.174,35.457,35.006,27.107,15.534,2.329,0.14,0.01],[26.287,25.708,23.524,18.482,6.127,2.329,0.454,0.034,0.002],[16.08,13.565,5.068,2.524,0.686,0.14,0.034,0.005,0.0],[2.259,1.732,0.519,0.133,0.043,0.01,0.002,0.0,0.0]],"thermal":[[2.328,6.719,7.976,7.247,3.18,6.217,6.536,4.371,5.56],[6.719,8.014,9.652,8.73,7.843,7.162,6.417,4.109,4.337],[7.976,9.652,10.208,10.928,9.661,8.778,6.439,11.313,2.373],[7.247,8.73,10.928,10.922,10.981,8.754,5.583,6.925,0.896],[3.18,7.843,9.661,10.981,10.027,7.431,13.534,3.258,0.317],[6.217,7.162,8.778,8.754,7.431,4.632,6.503,0.98,0.081],[6.536,6.417,6.439,5.583,13.534,6.503,2.06,0.258,0.02],[4.371,4.109,11.313,6.925,3.258,0.98,0.258,0.04,0.004],[5.56,4.337,2.373,0.896,0.317,0.081,0.02,0.004,0.0]]}}};
/* ---------- core twin (canvas) ---------- */
const HERO_KEYS = Object.keys(SAMPLE.results);
const SH = SAMPLE.layout.length, SW = SAMPLE.layout[0].length;
// Full core mirrored from the quarter across its west and south symmetry lines, top-left first.
function heroCells() {
  const out = [];
  for (let R = 0; R < 2 * SH; R++) for (let C = 0; C < 2 * SW; C++) {
    const Y = 2 * SH - 1 - R;
    out.push([Y >= SH ? Y - SH : SH - 1 - Y, C >= SW ? C - SW : SW - 1 - C]);
  }
  return out;
}
const heroBankAt = (i, j) => Object.keys(SAMPLE.banks).find((b) => SAMPLE.banks[b].some(([a, c]) => a === i && c === j)) || null;
const FULL = heroCells(), NF = 2 * SW;
const isFuel = (i, j) => SAMPLE.layout[i][j] !== "reflector";
function heroPeak(r) { let best = -1; r.power.forEach((row, i) => row.forEach((v, j) => { if (isFuel(i, j) && v > best) best = v; })); return best; }
const PMAX = Math.max(...HERO_KEYS.flatMap(k => SAMPLE.results[k].power.flatMap((row, i) => row.filter((_, j) => isFuel(i, j)))));
const HEAT = [[0, [5, 14, 26]], [0.32, [10, 56, 104]], [0.56, [22, 110, 194]], [0.76, [56, 168, 255]], [0.9, [166, 224, 255]], [1, [244, 251, 255]]];
function heat(t) {
  t = Math.max(0, Math.min(1, t));
  for (let k = 1; k < HEAT.length; k++) if (t <= HEAT[k][0]) {
    const [t0, a] = HEAT[k - 1], [t1, b] = HEAT[k], u = (t - t0) / (t1 - t0);
    return a.map((v, n) => Math.round(v + (b[n] - v) * u));
  }
  return HEAT[HEAT.length - 1][1];
}
const hc = document.getElementById("heroCore"), hx = hc.getContext("2d");
const off = document.createElement("canvas"); off.width = off.height = NF; const ox = off.getContext("2d");
const hero = { from: 0, to: 0, t: 1, raf: 0 };
const ease = (u) => u < 0.5 ? 4 * u * u * u : 1 - (-2 * u + 2) ** 3 / 2;
function drawHero() {
  const dpr = Math.min(2, devicePixelRatio || 1), S = Math.max(1, Math.round(hc.getBoundingClientRect().width * dpr));
  if (hc.width !== S) { hc.width = hc.height = S; }
  const A = SAMPLE.results[HERO_KEYS[hero.from]].power, B = SAMPLE.results[HERO_KEYS[hero.to]].power, e = ease(hero.t);
  const img = ox.createImageData(NF, NF);
  FULL.forEach(([i, j], n) => {
    if (!isFuel(i, j)) return;
    const v = A[i][j] + (B[i][j] - A[i][j]) * e, [r, g, b] = heat(v / PMAX);
    img.data.set([r, g, b, 255], n * 4);
  });
  ox.putImageData(img, 0, 0);
  hx.clearRect(0, 0, S, S);
  const pad = S * 0.05, s = S - 2 * pad, c = s / NF, mid = pad + s / 2;
  hx.imageSmoothingEnabled = true; hx.imageSmoothingQuality = "high";
  hx.save(); hx.filter = `blur(${Math.round(S * 0.028)}px)`; hx.globalAlpha = 0.9; hx.drawImage(off, pad, pad, s, s); hx.restore();
  hx.drawImage(off, pad, pad, s, s);
  hx.strokeStyle = "rgba(4,7,12,0.55)"; hx.lineWidth = Math.max(1, dpr);
  FULL.forEach(([i, j], n) => { if (isFuel(i, j)) hx.strokeRect(pad + (n % NF) * c, pad + Math.floor(n / NF) * c, c, c); });
  hx.strokeStyle = "rgba(163,220,255,0.22)"; hx.setLineDash([3 * dpr, 5 * dpr]);
  hx.beginPath(); hx.arc(mid, mid, s / 2 - c * 0.15, 0, Math.PI * 2); hx.stroke();
  hx.beginPath(); hx.moveTo(mid, pad); hx.lineTo(mid, pad + s); hx.moveTo(pad, mid); hx.lineTo(pad + s, mid); hx.stroke();
  hx.setLineDash([]);
  const rods = SAMPLE.results[HERO_KEYS[hero.to]].rods || {};
  FULL.forEach(([i, j], n) => {
    const b = heroBankAt(i, j); if (!b) return;
    const x = pad + (n % NF + 0.5) * c, y = pad + (Math.floor(n / NF) + 0.5) * c, inn = (rods[b] || 0) > 0;
    hx.beginPath(); hx.arc(x, y, c * 0.2, 0, Math.PI * 2);
    hx.fillStyle = inn ? "rgba(4,7,12,0.9)" : "rgba(4,7,12,0.25)"; hx.fill();
    hx.lineWidth = 1.2 * dpr; hx.strokeStyle = inn ? "rgba(244,251,255,0.95)" : "rgba(244,251,255,0.35)"; hx.stroke();
  });
}
function heroReadout(k) {
  const r = SAMPLE.results[HERO_KEYS[k]], base = SAMPLE.results[HERO_KEYS[0]], p = { v: heroPeak(r) };
  const dr = rho(r.k_eff) - rho(base.k_eff), d = k === 0 ? "ref" : (dr > 0 ? "+" : "−") + fmt(Math.abs(dr), 0);
  document.getElementById("hcLabel").textContent = HERO_KEYS[k];
  document.getElementById("hcK").textContent = fmt(r.k_eff, 5);
  document.getElementById("hcD").textContent = `${d} · ${fmt(p.v, 3)}`;
  document.querySelectorAll("#heroCfgs button").forEach((b, n) => b.setAttribute("aria-pressed", n === k));
}
function heroGo(k) {
  if (k === hero.to && hero.t >= 1) return;
  hero.from = hero.t >= 1 ? hero.to : hero.from; hero.to = k; hero.t = REDUCE ? 1 : 0;
  heroReadout(k); cancelAnimationFrame(hero.raf);
  const t0 = performance.now();
  const step = (now) => { hero.t = Math.min(1, (now - t0) / 1100); drawHero(); if (hero.t < 1) hero.raf = requestAnimationFrame(step); };
  if (REDUCE) drawHero(); else hero.raf = requestAnimationFrame(step);
}
document.getElementById("heroCfgs").innerHTML = HERO_KEYS.map((k, n) => `<button type="button" aria-pressed="${n === 0}">${k}</button>`).join("");
document.querySelectorAll("#heroCfgs button").forEach((b, n) => b.onclick = () => heroGo(n));
new ResizeObserver(drawHero).observe(hc);
heroReadout(0); drawHero();

/* ---------- speed chart (log scale) ---------- */
const LO = Math.log10(5e-4), HI = Math.log10(2e4);
const pos = (s) => ((Math.log10(s) - LO) / (HI - LO) * 100).toFixed(2) + "%";
const TICKS = [[1e-3, "1 ms", 0], [1e-2, "10 ms", 1], [1e-1, "100 ms", 1], [1, "1 s", 0], [10, "10 s", 1], [60, "1 min", 0], [600, "10 min", 1], [3600, "1 h", 0]];
const ROWS = [
  { name: "Monte Carlo transport", sub: "OpenMC, continuous energy", lo: 60, hi: 7200, tip: "Minutes to hours per run, plus statistical noise. Used for the validation set." },
  { name: "Deterministic diffusion", sub: "2-group reference solver", lo: 2e-3, hi: 2, pt: 1.0, tip: "Milliseconds to seconds depending on mesh. Dot: ~1 s measured with ndiffusion on the 68×68 quarter-core mesh the surrogate was trained on." },
  { name: "ReactorTwin surrogate", sub: "Fourier neural operator, CPU", pt: 1.6e-3, tip: "1.6 ms median single-case CPU latency (checkpoint fno_softplus, test-set evaluation)." },
];
const gridLines = TICKS.map(([s]) => `<span class="gl" style="left:${pos(s)}"></span>`).join("");
document.getElementById("speedChart").innerHTML = ROWS.map((r, n) => `
  <div class="srow"><div class="name">${r.name}<small>${r.sub}</small></div>
    <div class="track" data-n="${n}">${gridLines}
      ${r.lo ? `<span class="rng" style="left:${pos(r.lo)};width:calc(${pos(r.hi)} - ${pos(r.lo)})"></span>` : ""}
      ${r.pt ? `<span class="pt ${r.target ? "target" : ""}" style="left:${pos(r.pt)}"></span>${r.target ? `<span class="tag" style="left:${pos(r.pt)}">target</span>` : ""}` : ""}
      <span class="hit" style="left:${pos(r.pt || Math.sqrt(r.lo * r.hi))};width:${r.lo ? `calc(${pos(r.hi)} - ${pos(r.lo)})` : "28px"}"></span>
    </div></div>`).join("") + `
  <div class="axis"><span></span><div class="ticks">${TICKS.map(([s, l, minor]) => `<span class="${minor ? "minor" : ""}" style="left:${pos(s)}">${l}</span>`).join("")}</div></div>
  <div class="batch">
    <div class="us"><span>10,000 loading patterns, surrogate at 1.6 ms</span><b>≈ 16 s</b></div>
    <div><span>Same sweep, diffusion at 1 s</span><b>≈ 2.8 h</b></div>
    <div><span>Same sweep, Monte Carlo at 10 min</span><b>≈ 70 days</b></div>
  </div>
  <p class="fine" style="margin-top:12px">Order-of-magnitude comparison for a single 2-D core evaluation on one machine (CPU). Speed is bought with accuracy: the surrogate's test-set k_eff error is ~200 pcm mean absolute, so screen with it and confirm with the reference solver.</p>`;
document.querySelectorAll("#speedChart .track").forEach(t => {
  const r = ROWS[+t.dataset.n], hit = t.querySelector(".hit");
  hit.onmousemove = (e) => placeTip(e, `<b>${r.name}</b><br>${r.tip}`);
  hit.onmouseleave = hideTip;
});

/* ---------- photos & credits ---------- */
const SHOT_CREDITS = [
  "Snow-covered village · Pratihas Sharma, CC BY-SA 4.0",
  "High Flux Isotope Reactor, Oak Ridge · ORNL, CC BY 2.0",
  "Container ship, Felixstowe · John Fielding, CC BY 2.0",
  "Bacuit Bay, Palawan · Vyacheslav Argenberg, CC BY 4.0",
  "Singapore at blue hour · Zexsen Xie, CC BY 2.0",
  "Desperation Lake, Noatak, Alaska · NPS, public domain",
  "The Blue Marble, Apollo 17 · NASA, public domain",
];
const CREDITS = [
  ["Clouds mirrored in a mountain lake", "Ales Krivec", "CC0", "https://commons.wikimedia.org/wiki/File:Clouds_mirrored_in_a_mountain_lake_(Unsplash).jpg"],
  ["City Morning Blue Hour", "Zexsen Xie", "CC BY 2.0", "https://commons.wikimedia.org/wiki/File:City_Morning_Blue_Hour.jpg"],
  ["Village, valleys and mountains covered with snow", "Pratihas Sharma", "CC BY-SA 4.0", "https://commons.wikimedia.org/wiki/File:Village_,_valleys_and_mountains_coved_with_snow_in_winter_season.jpg"],
  ["HFIR Refueling July 2015", "Oak Ridge National Laboratory", "CC BY 2.0", "https://commons.wikimedia.org/wiki/File:HFIR_Refueling_July_2015_(19348392584).jpg"],
  ["Sea coral, clear turquoise water, Bacuit Bay", "Vyacheslav Argenberg", "CC BY 4.0", "https://commons.wikimedia.org/wiki/File:Sea_coral,_clear_turquoise_water,_Bacuit_Bay,_El_Nido,_Palawan,_Philippines.jpg"],
  ["Container ship CSCL Indian Ocean, Felixstowe", "John Fielding", "CC BY 2.0", "https://commons.wikimedia.org/wiki/File:Container_ship_The_CSCL_Indian_Ocean_in_the_Port_of_Felixstowe_(43954818414).jpg"],
  ["Desperation Lake, Noatak National Preserve", "NPS Photo", "Public domain", "https://commons.wikimedia.org/wiki/File:The_Delong_Mountains_are_reflected_in_the_clear,_clean_water_of_Desperation_Lake_in_the_northern_part_of_Noatak_National_(936b4940-1dd8-b71c-070e-3b979ec0a573).jpg"],
  ["The Blue Marble (AS17-148-22727)", "NASA / Apollo 17", "Public domain", "https://commons.wikimedia.org/wiki/File:The_Blue_Marble,_AS17-148-22727.jpg"],
  ["Kernkraftwerk Grafenrheinfeld", "Rainer Lippert", "Public domain", "https://commons.wikimedia.org/wiki/File:Kernkraftwerk_Grafenrheinfeld_1.jpg"],
  ["Limu Pools, Niue", "Pseudopanax", "Public domain", "https://commons.wikimedia.org/wiki/File:Limu_Pools,_looking_towards_sea_track.jpg"],
];
document.getElementById("credits").innerHTML = "Photos via Wikimedia Commons: " + CREDITS.map(([t, a, l, u]) => `<a href="${u}" target="_blank" rel="noopener">${t}</a> (${a}, ${l})`).join(" · ") + ". Images resized.";
const shots = [...document.querySelectorAll(".shot")], tocItems = [...document.querySelectorAll("#toc li")];
function setShot(n) {
  shots.forEach((s, k) => s.classList.toggle("on", k === n));
  tocItems.forEach((s, k) => s.classList.toggle("on", k === n));
  document.getElementById("shotCredit").textContent = "Photo: " + SHOT_CREDITS[n];
}
setShot(0);

/* ---------- scroll choreography ---------- */
// Active chapter = the one whose visible card sits closest to the middle of the viewport.
const chapters = [...document.querySelectorAll(".chapter")], tsteps = [...document.querySelectorAll(".tstep")];
function nearest(els) {
  let best = null, bd = Infinity;
  for (const el of els) {
    const r = el.querySelector(".card").getBoundingClientRect(), d = Math.abs(r.top + r.height / 2 - innerHeight / 2);
    if (d < bd) { bd = d; best = el; }
  }
  return best;
}
let activeShot = 0, activeStep = tsteps[0];
function syncChapters() {
  const n = +nearest(chapters).dataset.shot;
  if (n !== activeShot) { activeShot = n; setShot(n); }
  const s = nearest(tsteps);
  if (s !== activeStep) { activeStep = s; tsteps.forEach(t => t.classList.toggle("on", t === s)); heroGo(+s.dataset.cfg); }
}

// Reveal-once for blocks, plus count-up on figures.
function countUp(el) {
  const to = +el.dataset.count, t0 = performance.now();
  const step = (now) => { const u = Math.min(1, (now - t0) / 1400); el.textContent = Math.round(to * ease(u)); if (u < 1) requestAnimationFrame(step); };
  requestAnimationFrame(step);
}
if (!REDUCE) {
  const io = new IntersectionObserver((es) => es.forEach(e => {
    if (!e.isIntersecting) return;
    e.target.classList.add("in"); io.unobserve(e.target);
    e.target.querySelectorAll("[data-count]").forEach(countUp);
  }), { threshold: 0.12, rootMargin: "0px 0px -6% 0px" });
  document.querySelectorAll("[data-reveal]").forEach(el => io.observe(el));
}

// Mission statement: wrap words so each lights up as the line passes the reading position.
const stmt = document.querySelector("[data-words]");
(function wrapWords(node) {
  [...node.childNodes].forEach(ch => {
    if (ch.nodeType === 3) {
      const frag = document.createDocumentFragment();
      ch.textContent.split(/(\s+)/).forEach(tok => {
        if (!tok) return;
        if (/^\s+$/.test(tok)) frag.appendChild(document.createTextNode(tok));
        else { const s = document.createElement("span"); s.className = "w"; s.textContent = tok; frag.appendChild(s); }
      });
      ch.replaceWith(frag);
    } else if (ch.nodeType === 1) wrapWords(ch);
  });
})(stmt);
const words = [...stmt.querySelectorAll(".w")];

const parallax = [...document.querySelectorAll("[data-parallax]")];
const nav = document.getElementById("nav"), pwrFrame = document.getElementById("pwrFrame");
let ticking = false;
function onScroll() {
  ticking = false;
  const vh = innerHeight;
  nav.classList.toggle("over", scrollY < 40);
  syncChapters();
  if (REDUCE) return;
  for (const img of parallax) {
    const r = img.parentElement.getBoundingClientRect();
    if (r.bottom < -100 || r.top > vh + 100) continue;
    const center = r.top + r.height / 2 - vh / 2;
    img.style.setProperty("--py", (-center * +img.dataset.parallax).toFixed(1) + "px");
  }
  // PWR frame opens from an inset card to full bleed as it rises to the middle of the screen.
  const pr = pwrFrame.getBoundingClientRect();
  const c = Math.max(0, Math.min(1, (pr.top - vh * 0.15) / (vh * 0.7)));
  pwrFrame.style.setProperty("--cy", (c * 7).toFixed(2) + "%");
  pwrFrame.style.setProperty("--cx", (c * 9).toFixed(2) + "%");
  pwrFrame.style.setProperty("--cr", (c * 28).toFixed(1) + "px");
  // Statement words: lit between 85% and 35% of viewport height.
  const sr = stmt.getBoundingClientRect();
  if (sr.bottom > 0 && sr.top < vh) {
    const span = vh * 0.5, start = vh * 0.85;
    const prog = (start - sr.top) / (sr.height + span - vh * 0.15);
    words.forEach((w, k) => {
      const at = k / words.length, o = Math.max(0, Math.min(1, (prog - at) * 6 + 0.5));
      w.style.setProperty("--o", (0.22 + 0.78 * o).toFixed(3));
    });
  }
}
addEventListener("scroll", () => { if (!ticking) { ticking = true; requestAnimationFrame(onScroll); } }, { passive: true });
addEventListener("resize", onScroll);
onScroll();

