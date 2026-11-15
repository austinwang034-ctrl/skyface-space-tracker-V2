/*
 * Skyface Space Tracker: page behaviour.
 *
 * What happens, in order:
 *   1. The start screen offers three ways in: a star's name, coordinates, or the sky map.
 *   2. A name is turned into RA/Dec by the server; then the tracker starts for that patch.
 *   3. We ask the server who is at those coordinates, then start a tracker job.
 *      The server finds SPHEREx images of the patch, lines them up and sorts them by date.
 *   4. We download the frames and play them like a short video, with the year on screen.
 *   5. The other tabs (compare epochs, motion, spectrum, ...) load only when opened.
 *
 * The file is organised top to bottom as: settings, small helpers, set-up, building a tracker,
 * the tracker player, the result tabs, the sky map, spectra, other telescopes, sharing,
 * and finally the button wiring.
 *
 * All data comes from this site's own server under /api (see server.py).
 */

// ===== Settings =====

// Colours drawn on canvases and charts. Keep these in step with styles.css.
const COLOR = {
  text: "#eef4fa",
  muted: "#a9bdd3",
  grid: "#2f5078",
  ink: "#0b1e33", // dark ink, used on the light "plate view"
  target: "#ff6b35", // the tracked object and its path (international orange)
  targetFaint: "rgba(255,107,53,.45)",
  cyan: "#7fe3ff", // a spot the visitor clicked, planets
  gaia: "#9be3b4", // rings around Gaia stars
  wise: "#ff8ad8", // squares around WISE sources
  red: "#ff5d5d", // measured position
  readout: "rgba(11,30,51,.82)", // box behind the year
};

// Font for text drawn on canvases. The page itself is styled in styles.css.
const FONT = {
  text: '"Atkinson Hyperlegible Next", system-ui, sans-serif',
};

// ===== Small helpers =====

// Shorthand for finding one element on the page.
const $ = (s) => document.querySelector(s);

// Everything we know about the tracker currently on screen.
const state = {
  cfg: null,
  job: null,
  mv: null,
  imgs: [],
  k: 0,
  timer: null,
  track: null,
  solved: false,
  overlays: { gaia: null, wise: null }, // catalogue sources to ring on the tracker
  loadId: 0, // counts image loads, so a slow old load cannot overwrite a newer one
};

// Wait a number of milliseconds.
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Make text safe to insert into HTML.
const esc = (s) =>
  String(s ?? "").replace(
    /[&<>"]/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c],
  );

// Format a number with a fixed number of decimals ("–" when missing).
const fmt = (v, d = 2) => (v == null ? "–" : Number(v).toFixed(d));

// Call our server and return the JSON reply. Server error messages become thrown Errors.
async function api(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) {
    let m = r.statusText;
    try {
      m = (await r.json()).detail || m;
    } catch (e) {}
    throw new Error(Array.isArray(m) ? m.map((x) => `${x.loc.slice(-1)}: ${x.msg}`).join("; ") : m);
  }
  return r.json();
}

// Shared look for all Plotly charts.
const PLOT_THEME = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: COLOR.text, family: FONT.text },
  margin: { t: 40, r: 10, b: 50, l: 60 },
};

// Chart layout = shared theme + whatever this chart needs.
const lay = (o = {}) => ({
  ...PLOT_THEME,
  ...o,
  xaxis: { gridcolor: COLOR.grid, ...(o.xaxis || {}) },
  yaxis: { gridcolor: COLOR.grid, ...(o.yaxis || {}) },
});

// Build an HTML table. cols = [key, heading, optional formatter].
const table = (cols, rows) =>
  `<table><tr>${cols.map((c) => `<th>${esc(c[1])}</th>`).join("")}</tr>${rows.map((r) => `<tr>${cols.map((c) => `<td>${c[2] ? c[2](r[c[0]]) : esc(r[c[0]])}</td>`).join("")}</tr>`).join("")}</table>`;

// Update the progress bar and its message.
const setProg = (f, m) => {
  $("#sbar").style.width = 100 * f + "%";
  $("#smsg").textContent = m;
};

// Show (or clear) the red error box.
const showErr = (m) => {
  $("#error").hidden = !m;
  $("#error").textContent = m || "";
};

// ===== Set-up: fill in the form (detector bands, test objects) =====

// The five test objects from the challenge brief.
const EXAMPLES = [
  { n: "Barnard’s Star", ra: "17h57m48.498s", dec: "+04d41m36.11s", ep: 2000, band: "SPHEREx-D2" },
  {
    n: "WISE 0855-0714",
    ra: "08h55m10.831s",
    dec: "-07d14m42.53s",
    ep: 2010.34,
    band: "SPHEREx-D6",
  },
  { n: "Epsilon Indi B", name: "eps Ind Ba", band: "SPHEREx-D2" },
  { n: "RW Aurigae", name: "RW Aur", band: "SPHEREx-D2" },
  { n: "R Draconis", name: "R Dra", band: "SPHEREx-D2" },
];

// Tick one detector band and untick the rest.
const setBand = (b) =>
  document.querySelectorAll("#bands input").forEach((i) => (i.checked = i.value === b));

// Runs once at start: read settings from the server and fill in the form.
async function init() {
  state.cfg = await api("/api/config");
  $("#bands").innerHTML = Object.entries(state.cfg.bands)
    .map(
      ([b, [lo, hi]]) =>
        `<label class="opt"><input type="checkbox" value="${b}"> ${b.slice(-2)} · ${lo}–${hi} µm</label>`,
    )
    .join("");
  setBand("SPHEREx-D2");
  // A small host may cap how many images one tracker uses; keep the slider within that.
  const slider = $("#maxf");
  slider.max = Math.min(+slider.max, state.cfg.max_frames || 300);
  slider.value = Math.min(+slider.value, +slider.max);
  $("#maxv").textContent = slider.value;
  $("#ex").innerHTML = EXAMPLES.map((e, i) => `<button data-i="${i}">${esc(e.n)}</button>`).join(
    "",
  );
  $("#ex").onclick = async (ev) => {
    const e = EXAMPLES[ev.target.dataset.i];
    if (!e) return;
    showErr("");
    setBand(e.band);
    try {
      const r = e.name
        ? await api("/api/resolve?name=" + encodeURIComponent(e.name))
        : { ra: e.ra, dec: e.dec, epoch: e.ep };
      $("#name").value = e.n;
      $("#ra").value = r.ra;
      $("#dec").value = r.dec;
      $("#epoch").value = r.epoch;
    } catch (x) {
      showErr(`Could not look up ${e.n}: ${x.message} Enter its coordinates by hand.`);
    }
  };
}

// ===== Building a tracker: identify the object, start the job, show progress =====

// List of other catalogued objects close to the entered coordinates.
function nearbyHtml(list) {
  return list.length
    ? table(
          [
            ["name", "Object"],
            ["type_label", "Type"],
            ["sep_arcsec", "Distance″", (v) => fmt(v, 1)],
          ],
          list,
        )
    : "";
}

// Show which object sits at the coordinates (name, type, distance, motion).
function renderId(R) {
  const m = R.match,
    c = $("#idcard");
  c.hidden = false;
  if (!m) {
    c.innerHTML =
      `<p class="eyebrow">Who is at these coordinates?</p><div class="warn">${esc(R.note)}</div>` +
      (R.nearby.length ? `<h3>Closest catalogued objects</h3>` + nearbyHtml(R.nearby) : "");
    return;
  }
  const facts = [
    ["Type", `${esc(m.type_label)}${m.otype ? ` (${esc(m.otype)})` : ""}`],
    ["Spectral type", esc(m.sp_type)],
    [
      "Brightness",
      Object.entries(m.mags)
        .map(([b, v]) => `${b} = ${fmt(v, 2)}`)
        .join(", "),
    ],
    [
      "Distance",
      m.distance_pc ? `${fmt(m.distance_ly, 1)} light-years (${fmt(m.distance_pc, 2)} pc)` : "",
    ],
    ["Motion across the sky", m.pm_total ? `${fmt(m.pm_total / 1000, 2)}″ per year` : ""],
    ["Sideways speed", m.vtan_kms ? `${fmt(m.vtan_kms, 1)} km/s` : ""],
    ["Catalogue position (J2000)", esc(m.coords)],
  ].filter((f) => f[1]);
  const source = R.canned
    ? "Demo: an example record, not a live lookup"
    : `Found in the SIMBAD catalogue · <a href="${esc(m.simbad_url)}" target="_blank" rel="noopener">open its SIMBAD page</a>`;
  c.innerHTML =
    `<p class="eyebrow">Who is at these coordinates?</p>` +
    `<div class="object-name">${esc(m.name)}</div>` +
    `<div class="note">${R.canned ? "" : esc(R.note) + " "}${source}</div>` +
    `<dl class="facts">${facts.map((f) => `<div><dt>${f[0]}</dt><dd>${f[1]}</dd></div>`).join("")}</dl>` +
    (R.nearby.length
      ? `<details><summary>${R.nearby.length} other catalogued object${R.nearby.length > 1 ? "s" : ""} nearby</summary>${nearbyHtml(R.nearby)}</details>`
      : "");
}

// Show the same spot in older surveys (DSS2, 2MASS, AllWISE) for reference.
function renderRefs(R) {
  const c = $("#refimgs");
  c.hidden = false;
  c.innerHTML =
    `<h3>The same spot in older surveys</h3><div class="refs">` +
    R.images
      .map(
        (i) =>
          `<figure><div class="ref"><img src="${esc(i.url)}" alt="${esc(i.label)}" onerror="this.replaceWith(Object.assign(document.createElement('span'),{className:'note',textContent:'Image unavailable right now.'}))"></div><figcaption>${esc(i.label)}</figcaption></figure>`,
      )
      .join("") +
    `</div><div class="note">${esc(R.image_note)} North up, East left; the ring marks the centre.</div>`;
}

// The main action: identify the object, start a tracker job, and wait for it to finish.
async function make() {
  stop();
  showErr("");
  ["result", "idcard", "refimgs", "empty"].forEach((i) => ($("#" + i).hidden = true));
  $("#status").hidden = false;
  $("#go").disabled = true;
  setProg(0, "Identifying the object…");
  const demo = $("#s_demo").checked,
    ra = $("#ra").value.trim(),
    dec = $("#dec").value.trim(),
    epoch = parseFloat($("#epoch").value) || 2000,
    fov = +$("#fov").value;
  const post = (b) => ({
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(b),
  });
  try {
    if (!demo && !(ra && dec)) throw new Error("Enter both right ascension and declination.");
    const R = await api(
      "/api/identify",
      post({ ra: demo ? "0" : ra, dec: demo ? "0" : dec, epoch, fov_arcmin: fov, demo }),
    );
    renderId(R);
    renderRefs(R);
    const body = {
      source: demo ? "demo" : "irsa",
      ra,
      dec,
      epoch,
      fov_arcmin: fov,
      max_frames: +$("#maxf").value,
      refine: $("#refine").checked,
      refresh: $("#refresh").checked,
      bands: [...document.querySelectorAll("#bands input:checked")].map((i) => i.value),
    };
    state.lastBody = body;
    let j = await api("/api/jobs", post(body));
    while (j.state === "queued" || j.state === "running") {
      setProg(j.progress, j.message);
      await sleep(700);
      j = await api("/api/jobs/" + j.id);
    }
    if (j.state === "error") throw new Error(j.error);
    state.job = j.id;
    state.track = null;
    state.solved = false;
    await loadMovie();
  } catch (e) {
    showErr(
      (e.message.startsWith("Enter") ? "" : "Could not build the SPHEREx tracker: ") +
        e.message +
        (demo || e.message.startsWith("Enter")
          ? ""
          : " If this is a network error, check the server’s connection to irsa.ipac.caltech.edu. The identification and sky images above are still valid."),
    );
    if (!$("#idcard").innerHTML) $("#empty").hidden = false;
  } finally {
    $("#status").hidden = true;
    $("#go").disabled = false;
  }
}

// Thumbnails of every frame, oldest first.
function buildGallery() {
  const fr = state.mv.frames,
    n = fr.length,
    m = Math.min(24, n);
  const pick = [
    ...new Set(Array.from({ length: m }, (_, i) => Math.round((i * (n - 1)) / Math.max(m - 1, 1)))),
  ];
  $("#gal").innerHTML = pick
    .map(
      (k) =>
        `<figure><img data-k="${k}" src="/api/jobs/${state.job}/frames/${k}.png" alt="Frame ${k + 1}"><figcaption>${esc(fr[k].label)}</figcaption></figure>`,
    )
    .join("");
}

// Fetch the finished tracker description and set up the player and summary numbers.
async function loadMovie() {
  const mv = (state.mv = await api(`/api/jobs/${state.job}/tracker`)),
    n = mv.frames.length;
  const t0 = mv.frames[0],
    t1 = mv.frames[n - 1];
  $("#synth").hidden = !mv.synthetic;
  $("#logbook").hidden = true;
  clearWaveFilter();
  chem.shownFor = null;
  $("#metrics").innerHTML = [
    [n + (mv.synthetic ? "" : ` of ${mv.n_found}`), "Frames"],
    [
      fmt(t1.mjd - t0.mjd, 0) + " days",
      `${t0.time_utc.slice(0, 10)} → ${t1.time_utc.slice(0, 10)}`,
    ],
    [fmt(mv.native_pixel_arcsec) + "″", "Native pixel"],
    [`${fmt(mv.center.ra, 4)}°, ${fmt(mv.center.dec, 4)}°`, "Patch centre"],
    [mv.latest_obs_utc.slice(0, 10), "Newest exposure"],
  ]
    .map((m) => `<div><b>${m[0]}</b><span>${esc(m[1])}</span></div>`)
    .join("");
  $("#refnote").textContent = mv.refine_note;
  $("#bandwarn").hidden = mv.bands.length < 2;
  $("#tsrc").textContent = mv.target.source ? "Target data: " + mv.target.source : "";
  $("#seek").max = n - 1;
  $("#seek").value = 0;
  state.k = 0;
  $("#gif").href = `/api/jobs/${state.job}/tracker.gif`;
  $("#csv").href = `/api/jobs/${state.job}/track.csv`;
  $("#cand").max = n - 1;
  $("#c_meas").checked = false;
  showTab("movie");
  $("#result").hidden = false;
  buildGallery();
  state.click = null;
  state.overlays = { gaia: null, wise: null };
  $("#ov_note").textContent = "";
  delete $("#cmp_band").dataset.ready;
  updateShare();
  await loadImages();
  loadLogbook();
  ["gaia", "wise"].forEach((kind) => $("#c_" + kind).checked && toggleOverlay(kind));
}

// The observer's log: the server's plain-language summary of what was measured.
async function loadLogbook() {
  try {
    const log = await api(`/api/jobs/${state.job}/logbook`);
    $("#log_verdict").textContent = log.verdict;
    $("#log_lines").innerHTML = log.lines
      .map((line) => `<dt>${esc(line.label)}</dt><dd>${esc(line.text)}</dd>`)
      .join("");
    $("#logbook").dataset.kind = log.kind;
    $("#logbook").hidden = false;
  } catch (e) {
    $("#logbook").hidden = true; // the log is a summary; the tabs still hold every number
  }
}

// What the tracker is asked to show: the view, how many neighbours to add, and the stretch.
function viewParams() {
  return new URLSearchParams({
    stretch: $("#c_each").checked ? "frame" : "global",
    view: $("#v_view").value,
    window: $("#v_window").value,
  });
}

// Download every frame image so playback is smooth.
async function loadImages() {
  const q = viewParams(),
    n = state.mv.frames.length,
    id = ++state.loadId;
  let done = 0;
  $("#status").hidden = false;
  const imgs = await Promise.all(
    state.mv.frames.map(
      (f) =>
        new Promise((res) => {
          const im = new Image();
          im.onload = im.onerror = () => {
            if (id === state.loadId) setProg(++done / n, `Loading images ${done}/${n}`);
            res(im);
          };
          im.src = `/api/jobs/${state.job}/frames/${f.k}.png?${q}`;
        }),
    ),
  );
  if (id !== state.loadId) return; // a newer load started meanwhile
  state.imgs = imgs;
  $("#status").hidden = true;
  draw();
}

// ---- Catalogue overlays on the tracker: rings around Gaia stars, squares around WISE sources ----

const OVERLAY_NAMES = { gaia: "Gaia", wise: "WISE" };

// Fetch a catalogue the first time its box is ticked, then redraw.
async function toggleOverlay(kind) {
  const box = $("#c_" + kind),
    note = $("#ov_note");
  if (box.checked && !state.overlays[kind]) {
    note.textContent = `Fetching ${OVERLAY_NAMES[kind]} sources…`;
    try {
      const R = await api(`/api/jobs/${state.job}/overlay/${kind}`);
      state.overlays[kind] = R;
      note.textContent = R.note || `${R.sources.length} ${R.catalogue} sources in this patch.`;
    } catch (e) {
      box.checked = false;
      note.textContent = e.message;
    }
  }
  draw();
  updateShare();
}

// Julian year of a frame, for moving catalogue positions to the frame's date.
const frameYear = (f) => 2000 + (f.mjd - 51544.5) / 365.25;

// Draw one catalogue's sources at their positions on the date of frame f.
function drawOverlay(x, kind, f, pos) {
  const R = state.overlays[kind];
  if (!R || !$("#c_" + kind).checked) return;
  const year = frameYear(f);
  x.strokeStyle = COLOR[kind];
  x.lineWidth = 1.5;
  for (const src of R.sources) {
    const dt = year - (src.epoch ?? R.epoch),
      p = pos(src.e + src.pm_e * dt, src.n + src.pm_n * dt);
    x.beginPath();
    if (kind === "gaia") x.arc(p[0], p[1], 9, 0, 6.2832);
    else x.rect(p[0] - 11, p[1] - 11, 22, 22);
    x.stroke();
  }
}

// ===== The tracker player: draws one frame at a time on the canvas =====

// Draw the current frame: image, position markers, the year, compass and scale bar.
function draw() {
  const mv = state.mv;
  if (!mv) return;
  const g = mv.grid,
    c = $("#cv"),
    x = c.getContext("2d"),
    W = c.width,
    s = W / g.nx,
    f = mv.frames[state.k];
  x.fillStyle = "#000";
  x.fillRect(0, 0, W, W);
  x.imageSmoothingEnabled = $("#c_smooth").checked;
  const im = state.imgs[state.k];
  // Plate view shows the image as a photographic negative: dark stars on a light sky,
  // the way astronomers inspected glass plates. Faint things are easier to see this way.
  // Difference views are already coloured (red brighter, blue fainter), so they are not inverted.
  const plateView = $("#c_neg").checked && $("#v_view").value === "image",
    ink = plateView ? COLOR.ink : COLOR.text;
  x.filter =
    `brightness(${$("#v_bright").value}%) contrast(${$("#v_contrast").value}%)` +
    (plateView ? " invert(1)" : "");
  if (im && im.naturalWidth) x.drawImage(im, 0, 0, W, W);
  x.filter = "none";
  const pos = (e, n) => [
    ((g.nx - 1) / 2 - e / g.scale_arcsec + 0.5) * s,
    (g.ny - 1 - ((g.ny - 1) / 2 + n / g.scale_arcsec) + 0.5) * s,
  ];
  drawOverlay(x, "gaia", f, pos);
  drawOverlay(x, "wise", f, pos);
  if ($("#c_pred").checked) {
    x.strokeStyle = COLOR.targetFaint;
    x.setLineDash([4, 4]);
    x.beginPath();
    mv.frames.forEach((q, i) => {
      const p = pos(...q.pred);
      i ? x.lineTo(...p) : x.moveTo(...p);
    });
    x.stroke();
    x.setLineDash([]);
    const p = pos(...f.pred);
    x.strokeStyle = COLOR.target;
    x.lineWidth = 2;
    x.beginPath();
    x.arc(p[0], p[1], 15, 0, 6.2832);
    x.stroke();
  }
  if ($("#c_meas").checked && state.track) {
    const r = state.track.rows[state.k];
    if (r && r.meas_e != null) {
      const p = pos(r.meas_e, r.meas_n);
      x.strokeStyle = COLOR.red;
      x.lineWidth = 2;
      x.beginPath();
      x.moveTo(p[0] - 8, p[1] - 8);
      x.lineTo(p[0] + 8, p[1] + 8);
      x.moveTo(p[0] + 8, p[1] - 8);
      x.lineTo(p[0] - 8, p[1] + 8);
      x.stroke();
    }
  }
  x.fillStyle = COLOR.readout;
  x.fillRect(8, 8, 330, 78);
  x.lineWidth = 1;
  x.textBaseline = "alphabetic";
  x.fillStyle = "#fff";
  x.font = `700 44px ${FONT.text}`;
  x.fillText(f.time_utc.slice(0, 4), 18, 56);
  x.fillStyle = COLOR.target;
  x.font = `600 16px ${FONT.text}`;
  x.fillText(f.label.replace(" UTC", ""), 18, 78);
  x.fillStyle = ink;
  x.fillText("N ↑   E ←", W - 104, 26);
  const bar = (30 / g.scale_arcsec) * s;
  x.fillRect(12, W - 22, bar, 3);
  x.fillText("30″", 12, W - 30);
  x.textAlign = "right";
  x.fillText(`image ${state.k + 1} of ${mv.frames.length}`, W - 12, W - 16);
  x.textAlign = "left";
  if (state.click) {
    const p = pos(state.click.east, state.click.north);
    x.strokeStyle = COLOR.cyan;
    x.lineWidth = 2;
    x.strokeRect(p[0] - 10, p[1] - 10, 20, 20);
  }
  $("#seek").value = state.k;
  describeFrame(f);
}

// Put the current frame into words for screen-reader users. The canvas label always holds
// it; the live region speaks it only when the visitor steps by hand, not during playback.
function describeFrame(f) {
  const n = state.mv.frames.length,
    text = `Image ${state.k + 1} of ${n}, taken ${f.label.replace(" UTC", " UTC,").replace(" · ", " wavelength ")}.`;
  $("#cv").setAttribute("aria-label", "Tracker image. " + text);
  if (!state.timer) $("#frame_text").textContent = text;
}

// Keyboard control of the tracker: Space plays or pauses, arrow keys step one image.
function trackerKey(e) {
  if (!state.mv) return;
  if (e.key === " ") state.timer ? stop() : play();
  else if (e.key === "ArrowRight") nudge(1);
  else if (e.key === "ArrowLeft") nudge(-1);
  else return;
  e.preventDefault();
}

// Arrow keys move along the row of result tabs, as screen-reader users expect.
function tabKey(e) {
  const tabs = [...document.querySelectorAll(".tabs button[data-tab]")],
    i = tabs.indexOf(document.activeElement);
  if (i < 0) return;
  let j = null;
  if (e.key === "ArrowRight") j = (i + 1) % tabs.length;
  else if (e.key === "ArrowLeft") j = (i - 1 + tabs.length) % tabs.length;
  else if (e.key === "Home") j = 0;
  else if (e.key === "End") j = tabs.length - 1;
  if (j === null) return;
  e.preventDefault();
  showTab(tabs[j].dataset.tab);
  tabs[j].focus();
}

// Advance to the next frame (wraps around at the end).
function step() {
  const n = state.mv.frames.length;
  // With a wavelength filter on, skip ahead to the next image inside the chosen range.
  for (let tries = 0; tries < n; tries++) {
    state.k = (state.k + 1) % n;
    if (!state.waveFilter || state.waveFilter.has(state.k)) break;
  }
  draw();
}

// Step one image forward (+1) or back (-1) by hand; this pauses playback.
function nudge(d) {
  if (!state.mv) return;
  const n = state.mv.frames.length;
  stop();
  state.k = (state.k + d + n) % n;
  draw();
}

// Start playback.
function play() {
  stop();
  state.timer = setInterval(step, 1000 / +$("#fps").value);
  $("#play").textContent = "❚❚ Pause";
}

// Pause playback.
function stop() {
  clearInterval(state.timer);
  state.timer = null;
  $("#play").textContent = "▶ Play";
}

// ===== Result tabs: each one loads its data the first time it is opened =====

// Measured positions and brightness for every frame (fetched once, then reused).
async function getTrack() {
  return (state.track ||= await api(`/api/jobs/${state.job}/track`));
}

// Switch result tab and load that tab's data if needed.
function showTab(t) {
  // Only the open tab is in the keyboard's Tab order; arrow keys move between tabs.
  document.querySelectorAll(".tabs button[data-tab]").forEach((b) => {
    b.setAttribute("aria-selected", b.dataset.tab === t);
    b.tabIndex = b.dataset.tab === t ? 0 : -1;
  });
  ["movie", "gal", "track", "cmp", "spec", "chem", "dq", "solve", "nbr", "more"].forEach(
    (n) => ($("#tab-" + n).hidden = n !== t),
  );
  if (t !== "movie") stop();
  if (t !== "more") nsStop();
  if (t !== "cmp") blinkStop();
  if (t === "more") loadMore().catch((e) => showErr(e.message));
  if (t === "chem") openWavelengths().catch((e) => showErr(e.message));
  if (t === "track") loadTrack().catch((e) => showErr(e.message));
  if (t === "spec") loadSpec().catch((e) => showErr(e.message));
  if (t === "cmp") loadCmp();
  if (t === "dq") loadMeta().catch((e) => showErr(e.message));
  if (t === "solve" && !state.solved) loadSolves().catch((e) => showErr(e.message));
}

// Motion and brightness charts, plus the measured proper motion.
async function loadTrack() {
  const T = await getTrack(),
    ok = T.rows.filter((r) => r.ok),
    tg = state.mv.target;
  if (!ok.length) {
    $("#pmline").className = "warn";
    $("#pmline").textContent =
      "The source was not detected with enough significance in any frame. Try a longer-wavelength band or a larger field, and check the coordinates.";
    $("#plot1").innerHTML = $("#plot2").innerHTML = "";
    return;
  }
  $("#pmline").className = "ok";
  $("#pmline").textContent = T.fit
    ? `Measured proper motion: μα* = ${fmt(T.fit.pmra_masyr, 0)} ± ${fmt(T.fit.pmra_err, 0)}, μδ = ${fmt(T.fit.pmdec_masyr, 0)} ± ${fmt(T.fit.pmdec_err, 0)} mas/yr (catalogue: ${fmt(T.catalogue_pm.pmra, 0)}, ${fmt(T.catalogue_pm.pmdec, 0)}). Used ${T.fit.n_used} frames, RMS ${fmt(T.fit.rms_arcsec)}″.`
    : tg.has_motion
      ? "Not enough baseline or detections for a proper-motion fit (needs 5+ detections over 60+ days)."
      : "";
  const col = {
    color: ok.map((r) => r.wavelength_um),
    colorscale: "Turbo",
    colorbar: { title: "µm" },
    size: 7,
  };
  if (tg.has_motion)
    Plotly.newPlot(
      "plot1",
      [
        {
          x: ok.map((r) => r.time_utc),
          y: ok.map((r) => r.offset_arcsec),
          mode: "markers",
          marker: col,
        },
      ],
      lay({
        title: "Does the source sit where the catalogue says?",
        yaxis: { title: "Measured − predicted (arcsec)" },
        shapes: [
          {
            type: "line",
            xref: "paper",
            x0: 0,
            x1: 1,
            y0: T.native_pixel_arcsec,
            y1: T.native_pixel_arcsec,
            line: { dash: "dot", color: COLOR.muted },
          },
        ],
      }),
      { responsive: true },
    );
  else $("#plot1").innerHTML = "";
  Plotly.newPlot(
    "plot2",
    [
      {
        x: ok.map((r) => r.time_utc),
        y: ok.map((r) => r.flux_mjy),
        error_y: { type: "data", array: ok.map((r) => r.flux_err_mjy), color: "#555" },
        mode: "markers",
        marker: col,
      },
    ],
    lay({ title: "Brightness over time", yaxis: { title: "Aperture flux (mJy, approx.)" } }),
    { responsive: true },
  );
}

// Plate-solving summary: how well each frame matched Gaia.
async function loadSolves() {
  const R = await api(`/api/jobs/${state.job}/solves`);
  state.solved = true;
  if (!R.ran) {
    $("#smetrics").innerHTML = "";
    $("#stable").innerHTML =
      `<div class="note">Plate solving was not run for this tracker. ${esc(R.note)}</div>`;
    return;
  }
  const good = R.frames.filter((f) => f.ok),
    med = (a) => (a.length ? a.sort((p, q) => p - q)[a.length >> 1] : null);
  $("#smetrics").innerHTML = [
    [`${good.length} / ${R.frames.length}`, "Frames solved"],
    [fmt(med(good.map((f) => f.shift_arcsec))) + "″", "Median pointing error removed"],
    [fmt(med(good.map((f) => f.rms_after_arcsec))) + "″", "Median per-star residual"],
  ]
    .map((m) => `<div><b>${m[0]}</b><span>${m[1]}</span></div>`)
    .join("");
  $("#stable").innerHTML = table(
    [
      ["k", "Frame"],
      ["time_utc", "Time (UTC)"],
      ["shift_arcsec", "Shift″", (v) => fmt(v)],
      ["rms_after_arcsec", "Residual″", (v) => fmt(v)],
      ["n_matched", "Gaia matches"],
      ["message", "Status"],
    ],
    R.frames,
  );
  loadCands();
}

// Sources in one frame that have no Gaia counterpart.
async function loadCands() {
  const k = +$("#cand").value;
  $("#candv").textContent = k;
  try {
    const C = await api(`/api/jobs/${state.job}/candidates/${k}`);
    $("#ctable").innerHTML = C.rows.length
      ? `<div class="note">Detected sources with no Gaia counterpart (brightest first):</div>` +
        table(
          [
            ["ra", "RA°", (v) => fmt(v, 5)],
            ["dec", "Dec°", (v) => fmt(v, 5)],
            ["x", "x", (v) => fmt(v, 1)],
            ["y", "y", (v) => fmt(v, 1)],
            ["flux", "Flux", (v) => fmt(v, 1)],
          ],
          C.rows,
        )
      : `<div class="note">No unmatched sources in this frame.</div>`;
  } catch (e) {
    $("#ctable").innerHTML = `<div class="note">${esc(e.message)}</div>`;
  }
}

// Map of neighbouring Gaia stars and where they are heading.
async function nbr() {
  $("#n_go").disabled = true;
  try {
    const q = new URLSearchParams({
      radius_arcmin: $("#n_rad").value,
      mag_limit: $("#n_mag").value,
      year_start: $("#n_y0").value,
      year_end: $("#n_y1").value,
    });
    const N = await api(`/api/jobs/${state.job}/neighborhood?${q}`);
    const size = N.g.map((g) => (g == null ? 4 : Math.max(3, Math.min(16, 14 - 0.9 * (g - 6))))),
      cp = N.pm.map((p) => Math.log10(p + 1));
    const stars = (i) => ({
      x: N.e[i],
      y: N.n[i],
      mode: "markers",
      name: "Gaia stars",
      text: N.ids.map(
        (id, j) => `Gaia ${id}<br>G=${fmt(N.g[j], 1)}<br>PM=${fmt(N.pm[j], 0)} mas/yr`,
      ),
      hoverinfo: "text",
      marker: {
        size,
        color: cp,
        colorscale: "Viridis",
        cmin: 0,
        cmax: 4,
        colorbar: { title: "log10 PM" },
      },
    });
    const last = N.years.length - 1,
      sx = [],
      sy = [];
    N.pm.forEach((p, j) => {
      if (p >= 100) {
        sx.push(N.e[0][j], N.e[last][j], null);
        sy.push(N.n[0][j], N.n[last][j], null);
      }
    });
    const data = [
      stars(0),
      {
        x: sx,
        y: sy,
        mode: "lines",
        name: "Motion of fast movers",
        line: { color: "rgba(255,120,80,.7)", width: 1 },
        hoverinfo: "skip",
      },
    ];
    if (N.target_path)
      data.push({
        x: N.target_path.e,
        y: N.target_path.n,
        mode: "lines+markers",
        name: "Target path",
        line: { color: COLOR.target, width: 2 },
        marker: { symbol: "x", size: 7 },
      });
    await Plotly.newPlot(
      "nplot",
      data,
      lay({
        height: 600,
        xaxis: { title: "ΔRA·cosδ (arcsec, East ←)", autorange: "reversed" },
        yaxis: { title: "ΔDec (arcsec, North ↑)", scaleanchor: "x", scaleratio: 1 },
        legend: { orientation: "h", y: 1.08 },
        sliders: [
          {
            currentvalue: { prefix: "Epoch: " },
            steps: N.years.map((y, i) => ({
              method: "restyle",
              label: y.toFixed(0),
              args: [{ x: [N.e[i]], y: [N.n[i]] }, [0]],
            })),
          },
        ],
      }),
      { responsive: true },
    );
    $("#ntable").innerHTML =
      "<div class='note'>Fastest-moving stars in the field:</div>" +
      table(
        [
          ["source_id", "Gaia source"],
          ["phot_g_mean_mag", "G", (v) => fmt(v, 1)],
          ["pm_total_masyr", "PM (mas/yr)", (v) => fmt(v, 0)],
          ["parallax", "Parallax (mas)", (v) => fmt(v, 1)],
          ["vtan_kms", "v_tan (km/s)", (v) => fmt(v, 1)],
        ],
        N.movers,
      );
  } catch (e) {
    showErr(`Neighbourhood map failed: ${e.message}`);
  } finally {
    $("#n_go").disabled = false;
  }
}

// ===== Sky map: Aladin Lite map, labels, fly-in, and the animated path with the year =====

// Sky-map layers the visitor can switch between.
const LAYERS = [
  {
    label: "All 102 colours",
    url: "https://alasky.cds.unistra.fr/SPHEREx/color",
    note: "SPHEREx all-sky map (NASA/JPL-Caltech, HiPS by CDS): six detector bands combined into one false-colour image of 102 infrared colours, 0.75–5 µm.",
  },
  ...[
    ["D1", "0.75–1.09"],
    ["D2", "1.10–1.62"],
    ["D3", "1.63–2.41"],
    ["D4", "2.42–3.82"],
    ["D5", "3.83–4.41"],
    ["D6", "4.42–5.00"],
  ].map(([d, r]) => ({
    label: `${d} · ${r} µm`,
    url: `https://alasky.cds.unistra.fr/SPHEREx/${d}`,
    note: `SPHEREx detector ${d} (${r} µm): 17 of the 102 colours, shown as a single grey-scale band.`,
  })),
  {
    label: "Optical (DSS2)",
    url: "CDS/P/DSS2/color",
    note: "Digitised Sky Survey plates, 1950s–90s: stars that move fast appear where they were decades ago.",
  },
  { label: "2MASS", url: "CDS/P/2MASS/color", note: "2MASS near-infrared survey, about 1999." },
  {
    label: "AllWISE",
    url: "CDS/P/allWISE/color",
    note: "WISE mid-infrared survey, 2010 (a backup if the SPHEREx layer is slow).",
  },
];

// Everything about the sky-map view.
const sky = {
  al: null,
  sky: null,
  objs: [],
  idx: -1,
  obj: null,
  path: null,
  t: 2000,
  playing: false,
  last: 0,
  speed: 2.5,
  raf: 0,
  view: "map",
  fromObj: false,
  shownYear: null,
};

// True when the visitor asked their device for less animation.
const reduceMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

// Show (or clear) the error box above the map.
const mapErr = (m) => {
  $("#maperr").hidden = !m;
  $("#maperr").textContent = m || "";
};

// Switch between "Track a patch" and "Sky map".
function showView(v) {
  sky.view = v;
  $("#view-map").hidden = v !== "map";
  $("#view-coords").hidden = v !== "coords";
  $("#nav-map").setAttribute("aria-pressed", v === "map");
  $("#nav-coords").setAttribute("aria-pressed", v === "coords");
  if (v === "map") drawOv();
  else stopAnim();
}

// Create the Aladin Lite map (once) and add the object labels.
async function initMap() {
  if (sky.al) return;
  sky.sky = await api("/api/sky");
  sky.objs = sky.sky.objects;
  $("#cards").innerHTML = sky.objs
    .map(
      (o, i) =>
        `<button class="oc" data-i="${i}" aria-pressed="false"><b>${esc(o.name)}</b><span>${esc(o.category)}</span></button>`,
    )
    .join("");
  $("#layers").innerHTML = LAYERS.map(
    (l, i) => `<button data-i="${i}" aria-pressed="${i === 0}">${esc(l.label)}</button>`,
  ).join("");
  $("#laynote").textContent = LAYERS[0].note;
  if (typeof A === "undefined") {
    // The CDS address in index.html did not load (blocked or offline). Try the copy shipped
    // with this site, then two public mirrors. These builds are JavaScript modules, so they
    // are loaded with import() and handed over as the global "A" the rest of the code expects.
    const copies = [
      "/assets/vendor/aladin.js",
      "https://cdn.jsdelivr.net/npm/aladin-lite/dist/aladin.js",
      "https://unpkg.com/aladin-lite/dist/aladin.js",
    ];
    const tried = ["https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.js"];
    for (const url of copies) {
      try {
        const module = await import(url);
        window.A = module.default || module.A;
        if (window.A) break;
      } catch (e) {
        tried.push(url);
      }
    }
    if (typeof A === "undefined") {
      mapErr(
        "The sky viewer could not be loaded. Tried: " +
          tried.join("  |  ") +
          ". Usual causes: an ad blocker or privacy extension, a school/work network filter, or a VPN. Try a private window with extensions off, or another network. You can still pick an object below.",
      );
      return;
    }
  }
  if (!document.createElement("canvas").getContext("webgl2")) {
    mapErr(
      "This browser has WebGL2 switched off, which the sky viewer needs. In Chrome or Edge open Settings > System, turn on “Use graphics acceleration when available”, then restart the browser. You can still pick an object below.",
    );
    return;
  }
  try {
    await Promise.race([
      A.init,
      new Promise((_, no) =>
        setTimeout(
          () =>
            no(new Error("it took longer than 20 seconds to start; check the internet connection")),
          20000,
        ),
      ),
    ]);
  } catch (e) {
    mapErr("The sky viewer could not start: " + e.message);
    return;
  }
  try {
    [500, 2000, 6000].forEach((ms) => setTimeout(labelMapControls, ms));
    sky.al = A.aladin("#aladin", {
      survey: LAYERS[0].url,
      target: "266.4 -29.0",
      fov: 360,
      projection: "AIT",
      cooFrame: "equatorial",
      showCooGrid: false,
      showReticle: false,
      showFullscreenControl: false,
      showLayersControl: false,
      showSimbadPointerControl: false,
      showShareControl: false,
      showSettingsControl: false,
      showContextMenu: false,
    });
    const mk = (name, color, size, font, items) => {
      const c = A.catalog({
        name,
        shape: "circle",
        sourceSize: size,
        color,
        displayLabel: true,
        labelColumn: "name",
        labelColor: color,
        labelFont: font,
      });
      c.addSources(items.map((it) => A.source(it.ra, it.dec, it)));
      sky.al.addCatalog(c);
      return c;
    };
    mk(
      "Stars",
      "#ffffff",
      8,
      `13px ${FONT.text}`,
      sky.sky.stars.map((s) => ({ ...s, kind: "star" })),
    );
    mk(
      "Planets",
      COLOR.cyan,
      10,
      `bold 14px ${FONT.text}`,
      sky.sky.planets.map((p) => ({ ...p, kind: "planet" })),
    );
    mk(
      "Featured",
      COLOR.target,
      14,
      `bold 16px ${FONT.text}`,
      sky.objs.map((o) => ({ name: o.name, ra: o.ra, dec: o.dec, id: o.id, kind: "feat" })),
    );
    sky.al.on("objectClicked", (obj) => {
      const d = obj && obj.data;
      if (!d) return;
      if (d.kind === "feat") selectObj(sky.objs.findIndex((o) => o.id === d.id));
      else
        $("#maphint").textContent =
          `${d.name}: ${d.note}${d.kind === "planet" ? " (its position today; planets move)" : ""}`;
    });
    ["zoomChanged", "positionChanged"].forEach((ev) => {
      try {
        sky.al.on(ev, drawOv);
      } catch (e) {}
    });
    window.addEventListener("resize", drawOv);
  } catch (e) {
    sky.al = null;
    mapErr(
      "The sky viewer failed to start: " +
        e.message +
        " (press F12 and open Console for details). You can still pick an object below.",
    );
  }
}

// The sky-map library draws its own buttons without names. Give them names so screen
// readers can announce them. Called a few times because the library adds controls late.
function labelMapControls() {
  const names = {
    ".aladin-zoom-in": "Zoom in",
    ".aladin-zoom-out": "Zoom out",
    ".aladin-location-copy": "Copy the map position",
    ".aladin-input-select": "Coordinate system",
  };
  for (const [selector, name] of Object.entries(names)) {
    document
      .querySelectorAll("#aladin " + selector)
      .forEach((el) => el.setAttribute("aria-label", name));
  }
  document.querySelectorAll("#aladin .aladin-status-bar-message").forEach((el) => {
    el.tabIndex = 0; // it scrolls, so keyboard users must be able to reach it
    el.setAttribute("role", "status");
    el.setAttribute("aria-label", "Map status");
  });
  // Their icons are decoration; the buttons carry the names.
  document.querySelectorAll("#aladin img:not([alt])").forEach((img) => img.setAttribute("alt", ""));
}

// Smoothly move and zoom the map to a position.
function flyTo(ra, dec, fov, ms = 2800) {
  const al = sky.al;
  if (!al) return Promise.resolve();
  const [r0, d0] = al.getRaDec(),
    fv = al.getFov(),
    f0 = Array.isArray(fv) ? fv[0] : fv,
    dr = ((ra - r0 + 540) % 360) - 180;
  const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2),
    clamp = (t) => Math.max(0, Math.min(1, t));
  if (reduceMotion()) ms = 1;
  return new Promise((done) => {
    const t0 = performance.now();
    const tick = (now) => {
      const t = clamp((now - t0) / ms),
        ep = ease(clamp(t / 0.6)),
        ez = ease(clamp((t - 0.2) / 0.8)); // centre first, then zoom
      al.gotoRaDec((r0 + dr * ep + 360) % 360, d0 + (dec - d0) * ep);
      al.setFoV(Math.exp(Math.log(f0) + (Math.log(fov) - Math.log(f0)) * ez));
      drawOv();
      t < 1 ? requestAnimationFrame(tick) : done();
    };
    requestAnimationFrame(tick);
  });
}

// Sky position to pixel position on the overlay canvas.
const w2p = (ra, dec) => {
  try {
    const p = sky.al.world2pix(ra, dec);
    return p && isFinite(p[0]) && isFinite(p[1]) ? p : null;
  } catch (e) {
    return null;
  }
};

// Where the selected object is along its path at year t.
function pathAt(t) {
  const P = sky.path,
    n = P.years.length,
    f = ((t - P.years[0]) / (P.years[n - 1] - P.years[0])) * (n - 1),
    i = Math.max(0, Math.min(n - 2, Math.floor(f))),
    k = Math.max(0, Math.min(1, f - i));
  return {
    i,
    ra: P.ra[i] + (P.ra[i + 1] - P.ra[i]) * k,
    dec: P.dec[i] + (P.dec[i + 1] - P.dec[i]) * k,
  };
}

// Draw the object's path and current position over the map.
function drawOv() {
  const c = $("#ov");
  if (!c || sky.view !== "map") return;
  const r = c.getBoundingClientRect(),
    d = window.devicePixelRatio || 1,
    W = r.width,
    H = r.height;
  if (c.width !== Math.round(W * d) || c.height !== Math.round(H * d)) {
    c.width = Math.round(W * d);
    c.height = Math.round(H * d);
  }
  const x = c.getContext("2d");
  x.setTransform(d, 0, 0, d, 0, 0);
  x.clearRect(0, 0, W, H);
  if (!sky.al || !sky.path) return;
  const P = sky.path,
    cur = pathAt(sky.t),
    pts = P.years.map((y, i) => w2p(P.ra[i], P.dec[i])),
    line = (arr) => {
      let on = false;
      arr.forEach((p) => {
        if (!p) {
          on = false;
          return;
        }
        on ? x.lineTo(p[0], p[1]) : x.moveTo(p[0], p[1]);
        on = true;
      });
    };
  x.lineWidth = 2;
  x.setLineDash([5, 6]);
  x.strokeStyle = COLOR.targetFaint;
  x.beginPath();
  line(pts);
  x.stroke();
  const q = w2p(cur.ra, cur.dec);
  x.setLineDash([]);
  x.lineWidth = 3.5;
  x.strokeStyle = COLOR.target;
  x.beginPath();
  line(pts.slice(0, cur.i + 1).concat([q]));
  x.stroke();
  x.font = `600 15px ${FONT.text}`;
  x.textBaseline = "middle";
  x.lineJoin = "round";
  P.years.forEach((y, i) => {
    const p = pts[i];
    if (p && y % 10 === 0) {
      x.fillStyle = COLOR.target;
      x.beginPath();
      x.arc(p[0], p[1], 4, 0, 6.2832);
      x.fill();
      x.strokeStyle = "rgba(0,0,0,.8)";
      x.lineWidth = 4;
      x.strokeText(String(y), p[0] + 10, p[1]);
      x.fillStyle = "#fff";
      x.fillText(String(y), p[0] + 10, p[1]);
    }
  });
  if (q) {
    x.strokeStyle = "#fff";
    x.lineWidth = 2.5;
    x.beginPath();
    x.arc(q[0], q[1], 13, 0, 6.2832);
    x.stroke();
  }
}

// Move the animation to a year and update the big year readout.
function setYear(t) {
  const y = Math.floor(t),
    now = Math.floor(sky.sky ? sky.sky.epoch : 2026);
  if (y !== sky.shownYear) {
    sky.shownYear = y;
    $("#yr").textContent = y;
    $("#yrnote").textContent =
      y < 2025 ? "Before SPHEREx launched" : y <= now ? "SPHEREx era" : "Predicted from its motion";
  }
  $("#o_year").value = t;
}

// Start the motion-through-the-years animation.
function startAnim() {
  sky.playing = true;
  sky.last = performance.now();
  $("#o_play").textContent = "❚❚ Pause";
  cancelAnimationFrame(sky.raf);
  sky.raf = requestAnimationFrame(animTick);
}

// Pause it.
function stopAnim() {
  sky.playing = false;
  cancelAnimationFrame(sky.raf);
  $("#o_play").textContent = "▶ Play";
}

// One animation step.
function animTick(now) {
  if (!sky.playing || !sky.path) return;
  const Y = sky.path.years,
    dt = Math.min(0.1, (now - sky.last) / 1000);
  sky.last = now;
  sky.t += dt * sky.speed;
  if (sky.t > Y[Y.length - 1]) sky.t = Y[0];
  setYear(sky.t);
  drawOv();
  sky.raf = requestAnimationFrame(animTick);
}

// Open one of the featured objects: fly to it and show its details.
async function selectObj(i) {
  if (i < 0) return;
  sky.idx = i;
  const meta = sky.objs[i];
  stopAnim();
  showErr("");
  document.querySelectorAll(".oc").forEach((b, j) => b.setAttribute("aria-pressed", j === i));
  $("#maphint").textContent = `Looking up ${meta.name}…`;
  let o;
  try {
    o = await api(`/api/objects/${meta.id}`);
  } catch (e) {
    showErr(`Could not load ${meta.name}: ${e.message}`);
    return;
  }
  sky.obj = o;
  sky.path = o.path;
  sky.t = o.path.years[0];
  sky.shownYear = null;
  setYear(sky.t);
  $("#o_year").min = o.path.years[0];
  $("#o_year").max = o.path.years[o.path.years.length - 1];
  renderOverview(o);
  $("#objpanel").hidden = false;
  $("#yearbox").hidden = false;
  renderSpectrum("op", o, state.track && state.objKey === o.id ? state.track.rows : null);
  $("#maphint").textContent = o.position_is_exact
    ? ""
    : "Offline: this position is approximate until the first online lookup.";
  $("#objpanel").scrollIntoView({ behavior: "smooth", block: "nearest" });
  await flyTo(o.path.center.ra, o.path.center.dec, o.path.fov_deg);
  refreshOverlays();
  if (sky.obj === o && !reduceMotion()) startAnim();
}

// Facts about the selected object.
function renderOverview(o) {
  $("#o_name").textContent = o.name;
  $("#o_cat").textContent = o.category;
  $("#o_head").textContent = o.headline;
  $("#o_over").innerHTML =
    `<table>${o.facts.map((f) => `<tr><td>${esc(f.label)}</td><td class="left">${esc(f.value)}</td></tr>`).join("")}</table>` +
    `<h2>How it shines</h2><p>${esc(o.light)}</p><h2>In SPHEREx</h2><p>${esc(o.spherex)}</p>` +
    `<div class="note">Sources: ` +
    o.sources
      .map((s) => `<a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.title)}</a>`)
      .join(" · ") +
    `</div>`;
}

// ===== Spectra: model curves for featured objects and measured spectra from the tracker =====

// Draw the three spectrum charts: spectrum, brightness-vs-time, and spectrogram.
function renderSpectrum(p, o, rows) {
  const pa = $("#" + p + "1"),
    pb = $("#" + p + "2"),
    pc = $("#" + p + "3"),
    note = $("#" + p + "note");
  [pa, pb, pc].forEach((d) => {
    d.innerHTML = "";
  });
  const meas = (rows || []).filter((r) => r.wavelength_um != null && r.flux_mjy != null);
  if (o) {
    const m = o.model,
      B = Object.entries(o.bands);
    const shapes = B.map(([k, [lo, hi]], i) => ({
      type: "rect",
      xref: "x",
      yref: "paper",
      x0: lo,
      x1: hi,
      y0: 0,
      y1: 1,
      fillcolor: i % 2 ? "rgba(242,177,52,.07)" : "rgba(120,140,255,.08)",
      line: { width: 0 },
      layer: "below",
    }));
    const ann = B.map(([k, [lo, hi]]) => ({
      x: (lo + hi) / 2,
      y: 1,
      yref: "paper",
      text: k.slice(-2),
      showarrow: false,
      font: { size: 11, color: COLOR.muted },
      yanchor: "bottom",
    }));
    o.features.forEach((f) => {
      shapes.push({
        type: "line",
        x0: f.um,
        x1: f.um,
        yref: "paper",
        y0: 0,
        y1: 0.93,
        line: { color: "rgba(255,255,255,.35)", dash: "dot", width: 1 },
      });
      ann.push({
        x: f.um,
        y: 0.02,
        yref: "paper",
        text: f.label,
        textangle: -90,
        showarrow: false,
        font: { size: 11, color: COLOR.text },
        xanchor: "left",
        yanchor: "bottom",
      });
    });
    Plotly.newPlot(
      pa,
      m.components.map((c) => ({
        x: m.wavelength_um,
        y: c.flux,
        mode: "lines",
        name: c.label,
        line: { width: 2.5 },
        hovertemplate: `${esc(c.label)}<br>%{x:.2f} µm<extra></extra>`,
      })),
      lay({
        title: "Where its light comes from (blackbody model: shape only, not real band strengths)",
        height: 400,
        margin: { t: 60, r: 10, b: 50, l: 65 },
        shapes,
        annotations: ann,
        xaxis: { title: "Wavelength (µm), SPHEREx bands D1–D6 shaded", range: [0.7, 5.2] },
        yaxis: { title: "Relative brightness (log)", type: "log", range: [-8, 0.2] },
        legend: { orientation: "h", y: -0.28 },
      }),
      { responsive: true },
    );
  }
  if (meas.length >= 3) {
    const t = meas.map((r) => r.time_utc);
    Plotly.newPlot(
      pb,
      [
        {
          x: t,
          y: meas.map((r) => r.wavelength_um),
          mode: "markers",
          marker: {
            symbol: "square",
            size: 11,
            color: meas.map((r) => r.flux_mjy),
            colorscale: "Viridis",
            colorbar: { title: "mJy" },
          },
          text: meas.map(
            (r) =>
              `${r.time_utc.slice(0, 16)} UTC<br>${fmt(r.wavelength_um, 2)} µm<br>${fmt(r.flux_mjy, 3)} mJy`,
          ),
          hoverinfo: "text",
        },
      ],
      lay({
        title: "Spectrogram: wavelength against time, coloured by brightness",
        height: 360,
        yaxis: { title: "Wavelength (µm)", range: [0.7, 5.2] },
        xaxis: { title: "Observation time (UTC)" },
      }),
      { responsive: true },
    );
    Plotly.newPlot(
      pc,
      [
        {
          x: meas.map((r) => r.wavelength_um),
          y: meas.map((r) => r.flux_mjy),
          error_y: { type: "data", array: meas.map((r) => r.flux_err_mjy), color: "#555" },
          mode: "markers",
          marker: {
            size: 8,
            color: meas.map((r, i) => i),
            colorscale: "Plasma",
            colorbar: { title: "Exposure #<br>(time order)" },
            symbol: meas.map((r) => (r.ok ? "circle" : "circle-open")),
          },
          text: meas.map((r) => r.time_utc.slice(0, 16) + " UTC"),
          hoverinfo: "text+x+y",
        },
      ],
      lay({
        title: "Measured SPHEREx spectrum (open symbols = no clear detection)",
        height: 360,
        xaxis: { title: "Wavelength (µm)", range: [0.7, 5.2] },
        yaxis: { title: "Aperture flux (mJy, approx.)" },
      }),
      { responsive: true },
    );
    note.textContent = "";
  } else if (o) {
    Plotly.newPlot(
      pb,
      [
        {
          z: [o.model.strip_log10],
          x: o.model.channels_um,
          y: ["model"],
          type: "heatmap",
          colorscale: "Inferno",
          colorbar: { title: "log brightness" },
          hovertemplate: "%{x:.2f} µm<br>log10 %{z:.1f}<extra></extra>",
        },
      ],
      lay({
        title: "The 102 SPHEREx colours: how bright this object should look in each (model)",
        height: 210,
        margin: { t: 60, r: 10, b: 50, l: 60 },
        xaxis: { title: "Wavelength (µm), 6 detectors × 17 nominal channels" },
        yaxis: { showticklabels: false },
      }),
      { responsive: true },
    );
    note.textContent =
      "Open the SPHEREx tracker to replace this model strip with the real measured spectrogram (use several bands, or all six, for the full 0.75–5 µm range).";
  }
}

// Spectrum tab of the tracker.
async function loadSpec() {
  const T = await getTrack(),
    rows = state.click ? state.click.rows : T.rows;
  renderSpectrum("sp", state.obj || null, rows);
  $("#spnote").innerHTML = state.click
    ? `Spectrum at the point you clicked (${fmt(state.click.east, 1)}″ E, ${fmt(state.click.north, 1)}″ N; RA ${fmt(state.click.ra, 5)}°, Dec ${fmt(state.click.dec, 5)}°). <button id="sp_reset">Show the tracked object instead</button>`
    : !state.obj && !T.rows.some((r) => r.wavelength_um != null)
      ? "No wavelength information was found in these frames."
      : "";
  const r = $("#sp_reset");
  if (r)
    r.onclick = () => {
      state.click = null;
      draw();
      loadSpec();
    };
}

// Compare-epochs tab: early stack, late stack and their difference.
async function loadCmp() {
  if (!state.job) return;
  const split = +$("#cmp_split").value / 100,
    band = $("#cmp_band").value,
    q = `split=${split}` + (band ? `&band=${encodeURIComponent(band)}` : "");
  try {
    const m = await api(`/api/jobs/${state.job}/compare/meta?${q}`);
    if (!$("#cmp_band").dataset.ready) {
      $("#cmp_band").innerHTML =
        `<option value="">All bands</option>` +
        m.bands.map((b) => `<option>${b}</option>`).join("");
      $("#cmp_band").dataset.ready = 1;
    }
    $("#cmp_info").textContent =
      `Early stack: ${m.n_early} exposures (${m.early_from.slice(0, 10)} to ${m.early_to.slice(0, 10)}). Late stack: ${m.n_late} exposures (${m.late_from.slice(0, 10)} to ${m.late_to.slice(0, 10)}).`;
    const t = Date.now();
    ["early", "late", "diff"].forEach(
      (p) => ($("#cmp_" + p).src = `/api/jobs/${state.job}/compare/${p}.png?${q}&t=${t}`),
    );
    blink.sides = {
      early: {
        src: $("#cmp_early").src,
        tag: `Early: ${m.early_from.slice(0, 10)} to ${m.early_to.slice(0, 10)}`,
      },
      late: {
        src: $("#cmp_late").src,
        tag: `Late: ${m.late_from.slice(0, 10)} to ${m.late_to.slice(0, 10)}`,
      },
    };
    blinkShow("early");
    loadDetectors().catch((e) => ($("#det_note").textContent = e.message));
  } catch (e) {
    $("#cmp_info").textContent = e.message;
  }
}

// One stack per detector: every frame from each of SPHEREx's six detectors added together,
// which lowers the noise. With three or more detectors loaded, they are also combined into
// an infrared colour picture (longest wavelength shown as red, shortest as blue).
async function loadDetectors() {
  if (state.detectorsFor === state.job) return;
  state.detectorsFor = state.job;
  const base = `/api/jobs/${state.job}/detectors`,
    D = await api(base),
    um = (pair) => `${fmt(pair[0], 2)}–${fmt(pair[1], 2)} µm`;
  const figures = D.detectors.map(
    (d) =>
      `<figure><img src="${base}/${d.short}.png" alt="Stack of ${d.n_frames} images from detector ${d.short}" />` +
      `<figcaption>${d.short}: ${um(d.range_um)}, ${d.n_frames} images added</figcaption></figure>`,
  );
  if (D.colour)
    figures.push(
      `<figure><img src="${base}/colour.png" alt="Infrared colour picture from three detectors" />` +
        `<figcaption>Infrared colour picture: red is the longest wavelength, blue the shortest</figcaption></figure>`,
    );
  $("#det_row").innerHTML = figures.join("");
  $("#det_note").textContent = D.colour
    ? "Each detector sees a different slice of infrared light. Objects that change colour between panels are the interesting ones: cold brown dwarfs are bright only at the long end."
    : `This tracker holds ${D.detectors.length === 1 ? "one detector" : "two detectors"}. Tick three or more detector bands under “More options” to compare them side by side and get an infrared colour picture.`;
}

// ---- Wavelengths tab: chemical signatures and temperature ----
//
// Each SPHEREx image of the patch is taken at one wavelength. Choosing a range stacks the
// images inside it, stacks the images just outside it, and subtracts the two, so anything
// that absorbs or gives off light at that wavelength stands out.
const chem = { list: null, picked: null, frames: [], shownFor: null };

async function openWavelengths() {
  if (!chem.list) {
    chem.list = (await api("/api/signatures")).signatures;
    const chip = (sig) =>
      `<button data-sig="${sig.key}" aria-pressed="false">${esc(sig.name)} · ${fmt(sig.lo_um, 2)}–${fmt(sig.hi_um, 2)} µm</button>`;
    $("#chem_molecules").innerHTML = chem.list
      .filter((sig) => sig.kind !== "temperature")
      .map(chip)
      .join("");
    $("#chem_temps").innerHTML = chem.list
      .filter((sig) => sig.kind === "temperature")
      .map(chip)
      .join("");
    document.querySelectorAll("#tab-chem [data-sig]").forEach((button) => {
      button.onclick = () => {
        const sig = chem.list.find((x) => x.key === button.dataset.sig);
        $("#chem_lo").value = sig.lo_um;
        $("#chem_hi").value = sig.hi_um;
        showWavelengths(sig);
      };
    });
  }
  if (chem.shownFor !== state.job) {
    chem.shownFor = state.job;
    $("#chem_images").hidden = true;
    $("#chem_msg").textContent = "";
    $("#chem_about").textContent = "Pick a signature to begin.";
    loadTemperature();
  }
}

// Show one wavelength range. `sig` is a preset, or null for a range the visitor typed.
async function showWavelengths(sig) {
  const lo = parseFloat($("#chem_lo").value),
    hi = parseFloat($("#chem_hi").value),
    msg = $("#chem_msg");
  document
    .querySelectorAll("#tab-chem [data-sig]")
    .forEach((b) => b.setAttribute("aria-pressed", !!sig && b.dataset.sig === sig.key));
  $("#chem_images").hidden = true;
  $("#chem_play").disabled = true;
  msg.className = "note";
  if (!(lo >= 0.5 && hi <= 5.5 && hi > lo)) {
    msg.className = "err";
    msg.textContent = "Enter two wavelengths between 0.75 and 5.0 µm, the shorter one first.";
    return;
  }
  $("#chem_about").textContent = sig ? sig.tells_you : `Your own range: ${lo} to ${hi} µm.`;
  msg.textContent = "Adding images together…";
  try {
    const q = `lo=${lo}&hi=${hi}`,
      base = `/api/jobs/${state.job}/signature`,
      M = await api(`${base}/meta?${q}`),
      nIn = M.frames_in_band.length,
      nOut = M.frames_beside.length;
    chem.frames = M.frames_in_band;
    if (!nIn) {
      const need = M.missing_detectors.length ? M.missing_detectors : M.detectors_needed;
      msg.textContent =
        `This tracker has no images between ${lo} and ${hi} µm. Those wavelengths are seen by detector ${need.join(" and ")}; ` +
        `this tracker holds ${M.detectors_loaded.join(", ")}. Tick ${need.join(" and ")} under “More options” and track the patch again.`;
      return;
    }
    $("#chem_band").src = `${base}/band.png?${q}`;
    $("#chem_band_cap").textContent = `Inside the band: ${nIn} image${nIn > 1 ? "s" : ""} added`;
    ["beside", "diff"].forEach((part) => {
      $("#chem_" + part).closest("figure").hidden = !nOut;
      if (nOut) $("#chem_" + part).src = `${base}/${part}.png?${q}`;
    });
    $("#chem_beside_cap").textContent =
      `Just outside the band: ${nOut} image${nOut > 1 ? "s" : ""} added`;
    $("#chem_images").hidden = false;
    $("#chem_play").disabled = false;

    // What the tracked object itself does at these wavelengths.
    const T = M.target,
      name = state.mv.target.name;
    if (!nOut) {
      msg.textContent = `${nIn} images inside the band. There are none just outside it to compare with, so no difference image.`;
    } else if (!T) {
      msg.textContent = `${nIn} images inside the band and ${nOut} beside it. ${name} could not be measured well enough to compare.`;
    } else {
      const change = `${fmt(Math.abs(T.percent), 0)}% ${T.percent < 0 ? "fainter" : "brighter"}`;
      msg.className = T.verdict === "no clear feature" ? "note" : "ok";
      msg.textContent =
        T.verdict === "absorbs"
          ? `${name} is ${change} inside this band than beside it: it absorbs here.`
          : T.verdict === "emits"
            ? `${name} is ${change} inside this band than beside it: it gives off extra light here.`
            : `${name} is ${change} inside this band, within the scatter between images (±${fmt(T.threshold_percent, 0)}%): no clear feature.`;
    }
  } catch (e) {
    msg.className = "err";
    msg.textContent = e.message;
  }
}

// Play only the frames inside the chosen wavelengths in the main tracker.
function playWavelengths() {
  if (!chem.frames.length) return;
  state.waveFilter = new Set(chem.frames);
  state.k = chem.frames[0];
  const note = $("#filter_note");
  note.textContent = `Showing only the ${chem.frames.length} images taken between ${$("#chem_lo").value} and ${$("#chem_hi").value} µm.`;
  note.hidden = false;
  $("#chem_all").disabled = false;
  showTab("movie");
  draw();
  play();
}

function clearWaveFilter() {
  state.waveFilter = null;
  $("#filter_note").hidden = true;
  $("#chem_all").disabled = true;
}

// Rough temperature from how the object's brightness changes with wavelength.
async function loadTemperature() {
  const out = $("#chem_temp_text");
  out.textContent = "Working it out…";
  try {
    const T = (await api(`/api/jobs/${state.job}/temperature`)).estimate,
      name = state.mv.target.name;
    if (!T) {
      out.textContent =
        "Not enough colour to judge. Temperature shows up as a change in brightness across wavelengths, so this needs images spanning at least 1 µm, which usually means two or more detectors. Tick more detector bands under “More options”.";
      return;
    }
    const kelvin = Math.round(T.temperature_k / 10) * 10,
      celsius = Math.round(kelvin - 273),
      size = T.limit
        ? `${T.limit} about ${kelvin.toLocaleString()} K`
        : `about ${kelvin.toLocaleString()} K (${celsius.toLocaleString()} °C)`;
    out.textContent =
      `${name} has the colour of something ${size}, judged from ${T.n_points} measurements between ${fmt(T.range_um[0], 2)} and ${fmt(T.range_um[1], 2)} µm. ` +
      "This is a colour temperature: it assumes a simple glowing body, so absorption bands and dust can throw it off by hundreds of degrees.";
  } catch (e) {
    out.textContent = e.message;
  }
}

// The blink comparator: flip between the early and late stacks in the same spot, the way
// Clyde Tombaugh found Pluto in 1930. Anything that moved appears to jump.
const blink = { sides: null, side: "early", timer: null };

function blinkShow(side) {
  if (!blink.sides) return;
  blink.side = side;
  $("#blink_img").src = blink.sides[side].src;
  $("#blink_tag").textContent = blink.sides[side].tag;
}

function blinkStop() {
  if (blink.timer) clearInterval(blink.timer);
  blink.timer = null;
  $("#blink_go").textContent = "Start blinking";
}

function blinkToggle() {
  if (blink.timer) return blinkStop();
  if (!blink.sides) return;
  $("#blink_go").textContent = "Stop blinking";
  blink.timer = setInterval(
    () => blinkShow(blink.side === "early" ? "late" : "early"),
    +$("#blink_speed").value,
  );
}

// Ask NASA JPL which known asteroids and comets were near the patch when the frame on
// screen was taken. A mover that is not on this list is not a known small body.
async function checkSmallBodies() {
  const frame = state.mv.frames[state.k],
    c = state.mv.center,
    msg = $("#sb_msg");
  msg.className = "note";
  msg.textContent = "Asking NASA JPL…";
  $("#sb_table").innerHTML = "";
  $("#sb_go").disabled = true;
  try {
    const R = await api(
      `/api/smallbodies?ra=${c.ra}&dec=${c.dec}&mjd=${frame.mjd}&radius_arcmin=15`,
    );
    const when = R.time_utc.slice(0, 16).replace("T", " ");
    if (!R.bodies.length) {
      msg.textContent = `No catalogued asteroid or comet within ${fmt(R.half_width_arcmin, 0)}′ of the patch on ${when} UTC. If something moved here, it is not a known small body.`;
      return;
    }
    msg.className = "ok";
    msg.textContent = `${R.bodies.length} known small ${R.bodies.length > 1 ? "bodies" : "body"} within ${fmt(R.half_width_arcmin, 0)}′ on ${when} UTC, nearest first. Positions are for an observer on the ground and can be a few arcseconds off for SPHEREx.`;
    $("#sb_table").innerHTML = table(
      [
        ["name", "Name"],
        ["offset_arcsec", "From patch centre (″)", (v) => fmt(v, 0)],
        ["vmag", "Visual magnitude", (v) => fmt(v, 1)],
        ["ra", "RA"],
        ["dec", "Dec"],
      ],
      R.bodies,
    );
  } catch (e) {
    msg.className = "err";
    msg.textContent = e.message;
  } finally {
    $("#sb_go").disabled = false;
  }
}

// Data-quality tab: one row of metadata and flags per exposure.
async function loadMeta() {
  const D = await api(`/api/jobs/${state.job}/metadata`);
  $("#dq_legend").textContent =
    "Masked as bad: " +
    D.masked_bits.map((b) => D.flag_legend[b] || "bit " + b).join(", ") +
    ". SOURCE (bit 21) only marks known sources and is not masked. Flags are shown for the target’s pixel; see the SPHEREx Explanatory Supplement for all bits.";
  $("#dq_table").innerHTML = table(
    [
      ["k", "#"],
      ["time_utc", "Time (UTC)"],
      ["band", "Band"],
      ["wavelength_um", "λ (µm)", (v) => fmt(v, 2)],
      ["release", "Release", (v) => esc(String(v || "").toUpperCase())],
      ["version", "Processing"],
      ["finite_pct", "Valid %", (v) => fmt(v, 0)],
      ["flagged_pct", "Masked %", (v) => fmt(v, 1)],
      ["flag_names", "Flags at target", (v) => esc(v.join(", ") || "none")],
    ],
    D.rows,
  );
}

// ===== Other telescopes: NEOSSat images and links to other tools =====

// State of the NEOSSat player.
const neossat = { key: null, rows: [], k: 0, timer: null };

// Centre and size of the patch currently being tracked.
const patch = () => ({
  ra: state.mv.center.ra,
  dec: state.mv.center.dec,
  fov: +$("#fov").value || 6,
});

// Fill in the links and credits for the current patch.
async function loadMore() {
  const c = patch(),
    key = `${c.ra.toFixed(5)},${c.dec.toFixed(5)},${c.fov}`;
  if (neossat.key === key) return;
  neossat.key = key;
  nsStop();
  neossat.rows = [];
  $("#ns_player").hidden = true;
  $("#ns_table").innerHTML = "";
  $("#ns_msg").textContent = "";
  const L = await api(`/api/links?ra=${c.ra}&dec=${c.dec}&fov_arcmin=${c.fov}`),
    groups = [...new Set(L.links.map((l) => l.group))];
  $("#xlinks").innerHTML = groups
    .map(
      (g) =>
        `<div class="xl"><h4>${esc(g)}</h4>${L.links
          .filter((l) => l.group === g)
          .map(
            (l) =>
              `<a href="${esc(l.url)}" target="_blank" rel="noopener">${esc(l.name)}</a><p>${esc(l.why)}</p>`,
          )
          .join("")}</div>`,
    )
    .join("");
  $("#xcredits").innerHTML = L.credits.map((t) => `<li>${esc(t)}</li>`).join("");
}

// Show NEOSSat image number k.
function nsShow(k) {
  const r = neossat.rows[k];
  if (!r) return;
  neossat.k = k;
  $("#ns_seek").value = k;
  const c = patch();
  $("#ns_img").src =
    `/api/neossat/cutout.png?uri=${encodeURIComponent(r.uri)}&ra=${c.ra}&dec=${c.dec}&size_arcmin=${Math.max(c.fov, 4)}`;
  $("#ns_year").textContent = r.iso.slice(0, 4);
  $("#ns_date").textContent =
    r.iso.slice(0, 16).replace("T", " ") + " UTC" + (r.target ? " · " + r.target : "");
}

// Pause the NEOSSat player.
function nsStop() {
  if (neossat.timer) {
    clearInterval(neossat.timer);
    neossat.timer = null;
  }
  const b = $("#ns_play");
  if (b) b.textContent = "▶ Play";
}

// Ask the server for NEOSSat images of this patch.
async function nsSearch() {
  const c = patch(),
    rad = Math.min(30, Math.max(0.5, parseFloat($("#ns_rad").value) || 3)),
    msg = $("#ns_msg");
  nsStop();
  msg.className = "note";
  msg.textContent = "Searching the Canadian Astronomy Data Centre…";
  $("#ns_go").disabled = true;
  try {
    const R = await api(`/api/neossat?ra=${c.ra}&dec=${c.dec}&radius_arcmin=${rad}&max=24`),
      n = R.summary.count;
    if (!n) {
      msg.textContent =
        "NEOSSat has not photographed this patch. It observes chosen targets, not the whole sky, so most positions have no images. Try a larger radius, or one of the featured objects.";
      $("#ns_player").hidden = true;
      $("#ns_table").innerHTML = "";
      return;
    }
    msg.className = "ok";
    msg.textContent =
      `${n} NEOSSat image${n > 1 ? "s" : ""} cover this patch, from ${R.summary.first.slice(0, 10)} to ${R.summary.last.slice(0, 10)} (${fmt(R.summary.span_years, 1)} years).` +
      (n > R.visits.length ? ` Showing ${R.visits.length}, evenly spaced in time.` : "");
    neossat.rows = R.visits.filter((v) => v.has_image);
    $("#ns_player").hidden = !neossat.rows.length;
    if (neossat.rows.length) {
      $("#ns_seek").max = neossat.rows.length - 1;
      nsShow(0);
    }
    $("#ns_table").innerHTML = table(
      [
        ["iso", "Time (UTC)"],
        ["target", "NEOSSat target"],
        ["exposure_s", "Exposure (s)", (v) => fmt(v, 1)],
        ["product", "Processing", (v) => esc(String(v).toUpperCase())],
        [
          "file_url",
          "Full image",
          (v) => `<a href="${esc(v)}" target="_blank" rel="noopener">FITS file</a>`,
        ],
      ],
      R.visits,
    );
  } catch (e) {
    msg.className = "err";
    msg.textContent = e.message;
  } finally {
    $("#ns_go").disabled = false;
  }
}

// On/off display choices that travel in the address: [name in the address, checkbox].
const DISPLAY_FLAGS = [
  ["plate", "#c_neg"],
  ["gaia", "#c_gaia"],
  ["wise", "#c_wise"],
];

// Address that reopens the current tracker.
function shareUrl() {
  const b = state.lastBody;
  if (!b) return "";
  if (b.source === "demo") return location.origin + location.pathname + "?ra=0&dec=0&demo=1";
  const p = new URLSearchParams({ ra: b.ra, dec: b.dec, epoch: b.epoch, fov: b.fov_arcmin });
  if (b.bands.length) p.set("bands", b.bands.map((x) => x.slice(-2)).join(","));
  // How the tracker is displayed, so a shared link opens looking the same.
  if ($("#v_view").value !== "image") p.set("view", $("#v_view").value);
  if ($("#v_window").value !== "1") p.set("window", $("#v_window").value);
  if ($("#fps").value !== "4") p.set("speed", $("#fps").value);
  DISPLAY_FLAGS.forEach(([name, id]) => $(id).checked && p.set(name, "1"));
  return location.origin + location.pathname + "?" + p;
}

// Fill in the share link and embed code.
function updateShare() {
  const u = shareUrl();
  $("#share").hidden = !u;
  if (!u || !state.mv) return;
  $("#share_url").value = u;
  $("#share_embed").value =
    `<iframe src="${u}&embed=1" width="100%" height="900" style="border:0" loading="lazy" title="Skyface Space Tracker"></iframe>`;
  history.replaceState(null, "", u);
}

// Catalogue overlays for the sky map.
const CATALOGUES = {
  gaia: { id: "I/355/gaiadr3", name: "Gaia DR3", color: "#7CFC00", cat: null },
  wise: { id: "II/328/allwise", name: "AllWISE", color: "#ff66cc", cat: null },
};

// Add or remove the Gaia / WISE overlays to match the buttons.
function refreshOverlays() {
  if (!sky.al) return;
  Object.entries(CATALOGUES).forEach(([k, o]) => {
    if (o.cat) {
      try {
        sky.al.removeOverlay(o.cat);
      } catch (e) {}
      o.cat = null;
    }
    if ($(`[data-ov=${k}]`).getAttribute("aria-pressed") !== "true" || !sky.path) return;
    const fv = sky.al.getFov(),
      fov = Array.isArray(fv) ? fv[0] : fv,
      c = sky.path.center;
    try {
      o.cat = A.catalogFromVizieR(
        o.id,
        `${c.ra} ${c.dec}`,
        Math.min(0.5, fov * 0.7),
        {
          name: o.name,
          sourceSize: 11,
          color: o.color,
          shape: "circle",
          limit: 1500,
          onClick: "showTable",
        },
        () => {},
      );
      sky.al.addCatalog(o.cat);
    } catch (e) {
      $("#maphint").textContent = `Could not load the ${o.name} overlay: ${e.message}`;
    }
  });
}

// If the address carries coordinates or an object (?ra=..&dec=..), open it straight away.
async function bootFromUrl() {
  const q = new URLSearchParams(location.search);
  if (q.get("embed")) document.body.classList.add("embed");
  const ra = q.get("ra"),
    dec = q.get("dec"),
    obj = q.get("obj");
  if (!(ra && dec) && !obj) return;
  $("#welcome").classList.add("gone");
  if (ra && dec) {
    showView("coords");
    $("#ra").value = ra;
    $("#dec").value = dec;
    if (q.get("epoch")) $("#epoch").value = q.get("epoch");
    if (q.get("fov")) {
      $("#fov").value = q.get("fov");
      $("#fovv").textContent = q.get("fov");
    }
    if (q.get("bands")) {
      const want = q
        .get("bands")
        .split(",")
        .map((b) => "SPHEREx-" + b.trim().toUpperCase());
      document
        .querySelectorAll("#bands input")
        .forEach((i) => (i.checked = want.includes(i.value)));
    }
    if (q.get("demo")) $("#s_demo").checked = true;
    const pick = (id, value) => {
      if (value && [...$(id).options].some((o) => o.value === value)) $(id).value = value;
    };
    pick("#v_view", q.get("view"));
    pick("#v_window", q.get("window"));
    pick("#fps", q.get("speed"));
    DISPLAY_FLAGS.forEach(([name, id]) => ($(id).checked = q.get(name) === "1"));
    document.body.classList.toggle("plate-view", $("#c_neg").checked);
    if (q.get("autorun") !== "0") $("#go").click();
  } else {
    showView("map");
    await initMap();
    const i = sky.objs.findIndex((o) => o.id === obj);
    if (i >= 0) selectObj(i);
  }
}

// ===== Moving between the sky map and the tracker, and on to the next object =====

// Send the selected sky-map object to the tracker.
async function openTracker(o = sky.obj) {
  if (!o) return;
  sky.fromObj = true;
  state.objKey = o.id;
  state.obj = o;
  state.track = null;
  showView("coords");
  $("#objnav").hidden = false;
  $("#objnavname").textContent = o.name;
  const t = o.tracker;
  setBand(t.band);
  $("#fov").value = t.fov_arcmin;
  $("#fovv").textContent = t.fov_arcmin;
  $("#s_demo").checked = false;
  if (t.ra == null) {
    showErr(
      "This object’s coordinates are not known yet. Connect to the internet once so they can be looked up.",
    );
    return;
  }
  $("#ra").value = t.ra;
  $("#dec").value = t.dec;
  $("#epoch").value = t.epoch;
  await make();
}

// Move to the next (+1) or previous (-1) featured object.
function go(d) {
  if (!sky.objs.length) return;
  const n = sky.objs.length,
    i = (Math.max(sky.idx, 0) + d + n) % n;
  if (sky.view === "coords" && sky.fromObj) {
    sky.idx = i;
    api(`/api/objects/${sky.objs[i].id}`)
      .then((o) => {
        sky.obj = o;
        sky.path = o.path;
        return openTracker(o);
      })
      .catch((e) => showErr(e.message));
  } else {
    if (sky.view !== "map") showView("map");
    selectObj(i);
  }
}

// Leave the current object and return to the whole-sky view.
function backToMap() {
  sky.fromObj = false;
  $("#objnav").hidden = true;
  stopAnim();
  sky.path = null;
  sky.obj = null;
  sky.idx = -1;
  $("#objpanel").hidden = true;
  $("#yearbox").hidden = true;
  document.querySelectorAll(".oc").forEach((b) => b.setAttribute("aria-pressed", false));
  showView("map");
  flyTo(266.4, -29.0, 360, 2200);
  $("#maphint").textContent = "Click a labelled object to zoom in.";
}

// ===== Wiring: connect buttons and sliders to the functions above, then start =====
$("#go").onclick = () => {
  state.obj = null;
  state.objKey = null;
  sky.fromObj = false;
  $("#objnav").hidden = true;
  make();
};
$("#s_demo").onchange = (e) =>
  document.querySelectorAll("#bands input").forEach((i) => (i.disabled = e.target.checked));
$("#gal").onclick = (e) => {
  if (e.target.dataset.k) {
    state.k = +e.target.dataset.k;
    showTab("movie");
    draw();
  }
};
$("#fov").oninput = (e) => ($("#fovv").textContent = e.target.value);
$("#maxf").oninput = (e) => ($("#maxv").textContent = e.target.value);
$("#play").onclick = () => (state.timer ? stop() : play());
$("#seek").oninput = (e) => {
  state.k = +e.target.value;
  draw();
};
$("#prev").onclick = () => nudge(-1);
$("#next").onclick = () => nudge(1);
$("#fps").onchange = () => {
  if (state.timer) play();
  updateShare();
};
["c_pred", "c_smooth"].forEach((id) => ($("#" + id).onchange = draw));
["v_bright", "v_contrast"].forEach((id) => ($("#" + id).oninput = draw));
["c_each", "v_view", "v_window"].forEach(
  (id) =>
    ($("#" + id).onchange = () => {
      if (!state.mv) return;
      updateShare();
      loadImages();
    }),
);
["gaia", "wise"].forEach((kind) => ($("#c_" + kind).onchange = () => toggleOverlay(kind)));
$("#c_meas").onchange = async (e) => {
  if (e.target.checked) {
    try {
      await getTrack();
    } catch (x) {
      showErr(x.message);
      e.target.checked = false;
    }
  }
  draw();
};
document
  .querySelectorAll(".tabs button")
  .forEach((b) => (b.onclick = () => showTab(b.dataset.tab)));
$("#cand").oninput = loadCands;
$("#n_go").onclick = nbr;
$("#ns_go").onclick = nsSearch;
$("#blink_go").onclick = blinkToggle;
$("#blink_speed").onchange = () => {
  if (blink.timer) {
    blinkStop();
    blinkToggle();
  }
};
$("#blink_img").onclick = () => blinkShow(blink.side === "early" ? "late" : "early");
$("#blink_img").onkeydown = (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    $("#blink_img").click();
  }
};
$("#cv").addEventListener("keydown", trackerKey);
document.querySelector("#result .tabs").addEventListener("keydown", tabKey);
$("#sb_go").onclick = checkSmallBodies;
$("#chem_go").onclick = () => showWavelengths(null);
$("#chem_play").onclick = playWavelengths;
$("#chem_all").onclick = () => {
  clearWaveFilter();
  showTab("movie");
};
$("#c_neg").onchange = () => {
  draw();
  document.body.classList.toggle("plate-view", $("#c_neg").checked);
  updateShare();
};
$("#ns_seek").oninput = (e) => {
  nsStop();
  nsShow(+e.target.value);
};
$("#ns_play").onclick = () => {
  if (neossat.timer) return nsStop();
  if (!neossat.rows.length) return;
  $("#ns_play").textContent = "❚❚ Pause";
  neossat.timer = setInterval(() => nsShow((neossat.k + 1) % neossat.rows.length), 900);
};
$("#ns_img").onerror = () => {
  $("#ns_date").textContent = "This image could not be loaded from CADC.";
};
// ---- Start screen: three ways in (a star's name, coordinates, the sky map) ----

const leaveStart = () => $("#welcome").classList.add("gone");

const startMsg = (text) => {
  $("#start_msg").hidden = !text;
  $("#start_msg").textContent = text || "";
};

// Ask SIMBAD where a named object is and put its coordinates in the form.
async function lookupName(name) {
  const r = await api("/api/resolve?name=" + encodeURIComponent(name));
  $("#name").value = name;
  $("#ra").value = r.ra;
  $("#dec").value = r.dec;
  $("#epoch").value = r.epoch;
}

// Leave the start screen and track whatever is now in the form.
function startTracking() {
  leaveStart();
  showView("coords");
  $("#go").click();
}

$("#start_name_form").onsubmit = async (e) => {
  e.preventDefault();
  const name = $("#start_name").value.trim(),
    button = e.target.querySelector("button");
  if (!name) return startMsg("Type a star’s name first, for example Barnard’s Star.");
  startMsg("");
  button.disabled = true;
  try {
    await lookupName(name);
    $("#s_demo").checked = false;
    startTracking();
  } catch (x) {
    startMsg(`Could not find “${name}”: ${x.message} Check the spelling, or enter coordinates.`);
  } finally {
    button.disabled = false;
  }
};
$("#start_coords_form").onsubmit = (e) => {
  e.preventDefault();
  const ra = $("#start_ra").value.trim(),
    dec = $("#start_dec").value.trim();
  if (!(ra && dec)) return startMsg("Enter both a right ascension and a declination.");
  startMsg("");
  $("#name").value = "";
  $("#ra").value = ra;
  $("#dec").value = dec;
  $("#s_demo").checked = false;
  startTracking();
};
$("#start_map").onclick = () => {
  leaveStart();
  showView("map");
  initMap().catch((e) => mapErr("Could not start the sky map: " + e.message));
};
$("#home").onclick = () => {
  stop();
  startMsg("");
  $("#welcome").classList.remove("gone");
  $("#start_name").focus();
};
$("#name_go").onclick = async () => {
  const name = $("#name").value.trim();
  if (!name) return;
  showErr("");
  try {
    await lookupName(name);
  } catch (x) {
    showErr(`Could not find “${name}”: ${x.message}`);
  }
};
$("#name").onkeydown = (e) => e.key === "Enter" && $("#name_go").click();
$("#demo").onclick = () => {
  $("#s_demo").checked = true;
  startTracking();
};
$("#nav-map").onclick = () => {
  showView("map");
  initMap().catch((e) => mapErr(e.message));
};
$("#nav-coords").onclick = () => showView("coords");
$("#cards").onclick = (e) => {
  const b = e.target.closest(".oc");
  if (b) selectObj(+b.dataset.i);
};
$("#layers").onclick = (e) => {
  const b = e.target.closest("button");
  if (!b || !sky.al) return;
  const l = LAYERS[+b.dataset.i];
  sky.al.setBaseImageLayer(l.url);
  [...$("#layers").children].forEach((c) => c.setAttribute("aria-pressed", c === b));
  $("#laynote").textContent = l.note;
};
$("#overlays").onclick = (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  b.setAttribute("aria-pressed", b.getAttribute("aria-pressed") !== "true");
  if (!sky.path)
    $("#maphint").textContent =
      "Zoom in on an object first: catalogue overlays load for the area you are viewing.";
  refreshOverlays();
};
$("#cv").onclick = async (e) => {
  if (!state.mv) return;
  const g = state.mv.grid,
    r = e.target.getBoundingClientRect(),
    s = r.width / g.nx;
  const px = (e.clientX - r.left) / s - 0.5,
    py = g.ny - 1 - ((e.clientY - r.top) / s - 0.5),
    east = ((g.nx - 1) / 2 - px) * g.scale_arcsec,
    north = (py - (g.ny - 1) / 2) * g.scale_arcsec;
  try {
    const R = await api(
      `/api/jobs/${state.job}/spectrum?east=${east.toFixed(2)}&north=${north.toFixed(2)}`,
    );
    state.click = { east, north, ra: R.ra, dec: R.dec, rows: R.rows };
    draw();
    showTab("spec");
  } catch (x) {
    showErr("Could not extract a spectrum there: " + x.message);
  }
};
$("#cmp_split").oninput = (e) => ($("#cmp_sv").textContent = e.target.value);
$("#cmp_split").onchange = $("#cmp_band").onchange = loadCmp;
["share_url", "share_embed"].forEach((id) => ($("#" + id).onfocus = (e) => e.target.select()));
$("#o_play").onclick = () => (sky.playing ? stopAnim() : startAnim());
$("#o_year").oninput = (e) => {
  stopAnim();
  sky.t = +e.target.value;
  setYear(sky.t);
  drawOv();
};
$("#o_speed").onchange = (e) => (sky.speed = +e.target.value);
$("#o_back").onclick = $("#t_back").onclick = backToMap;
$("#o_prev").onclick = $("#t_prev").onclick = () => go(-1);
$("#o_next").onclick = $("#t_next").onclick = () => go(1);
$("#o_track").onclick = () => openTracker();
document.querySelectorAll("[data-ot]").forEach(
  (b) =>
    (b.onclick = () => {
      document
        .querySelectorAll("[data-ot]")
        .forEach((x) => x.setAttribute("aria-selected", x === b));
      $("#o_over").hidden = b.dataset.ot !== "over";
      $("#o_spec").hidden = b.dataset.ot !== "spec";
    }),
);
init()
  .then(bootFromUrl)
  .catch((e) => showErr("Could not reach the server: " + e.message));
