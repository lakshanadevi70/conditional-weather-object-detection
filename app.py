"""Smart Vision - Condition-Aware Object Detection dashboard.



Run with:  streamlit run app.py

"""

import base64

import hashlib

import io

import json
import tempfile
import os

from pathlib import Path



import cv2

import numpy as np

import pandas as pd

import streamlit as st

import streamlit.components.v1 as components

from PIL import Image

from pipeline import detector, enhancers, runner



ROOT = Path(__file__).resolve().parent

CFG = json.loads((ROOT / "config.json").read_text())

CONDITIONS = CFG["classes"]

PAGES = ["Live Detection", "Batch Mode", "Performance", "Sample Gallery", "About Project"]



st.set_page_config(page_title="Smart Vision", page_icon="👁", layout="wide")



# ------------------------------------------------------------------ styling

CSS = """

<style>

@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=DM+Sans:wght@400;500;600&display=swap');



:root {

  color-scheme: dark;

  --bg: #070d19;

  --panel: #101a2b;

  --panel-2: #142238;

  --line: #2a405e;

  --cyan: #35e0f2;

  --blue: #60a5fa;

  --text: #f4f7ff;

  --muted: #c0cce0;

  --ok: #4ade80;

  --warn: #fbbf24;

  --bad: #fb7185;

  --accent: #fb5360;

}



/* App canvas and typography */

.stApp {

  background:

    radial-gradient(1000px 600px at 8% -10%, rgba(59,130,246,.18), transparent 62%),

    radial-gradient(850px 500px at 100% 0%, rgba(34,211,238,.12), transparent 58%),

    var(--bg);

  color: var(--text);

  font-family: 'DM Sans', sans-serif;

}

[data-testid="stToolbar"], footer, #MainMenu { visibility: hidden; }

header[data-testid="stHeader"] { background: transparent; }

.block-container { padding-top: 2rem; max-width: 1400px; }

h1, h2, h3, h4, h5, p, li, label, small { color: var(--text); }

h1, h2, h3, h4 { font-family: 'Sora', sans-serif; }

.stMarkdown, .stMarkdown p, .stCaption, [data-testid="stCaptionContainer"] {

  color: var(--muted);

}

[data-testid="stMetricLabel"], [data-testid="stMetricValue"],

[data-testid="stMetricDelta"] { color: var(--text) !important; }



/* Sidebar: high contrast while preserving Streamlit's native controls */

section[data-testid="stSidebar"] {

  background: #0b1322 !important;

  border-right: 1px solid var(--line);

  min-width: 285px;

}

section[data-testid="stSidebar"] > div { background: transparent; }

section[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: 1.2rem; }

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,

section[data-testid="stSidebar"] label,

section[data-testid="stSidebar"] .stRadio,

section[data-testid="stSidebar"] .stSlider,

section[data-testid="stSidebar"] .stToggle,

section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],

section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {

  color: var(--text) !important;

}

section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {

  font-weight: 600 !important;

}

section[data-testid="stSidebar"] [data-testid="stRadio"] label,

section[data-testid="stSidebar"] [data-testid="stToggle"] label {

  color: #eaf1ff !important;

}

section[data-testid="stSidebar"] [data-testid="stRadio"] label p,

section[data-testid="stSidebar"] [data-testid="stToggle"] label p {

  color: #eaf1ff !important;

}

section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover p {

  color: var(--cyan) !important;

}

section[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBarMin"],

section[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBarMax"] {

  color: var(--muted) !important;

}

section[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stThumbValue"] {

  color: var(--cyan) !important;

}

section[data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] div {

  border-color: var(--accent);

}

section[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {

  background: var(--accent) !important;

  border-color: var(--accent) !important;

}

section[data-testid="stSidebar"] [data-testid="stToggle"] [data-baseweb="switch"] > div {

  background: #64748b;

}

section[data-testid="stSidebar"] [data-testid="stToggle"] [aria-checked="true"] > div {

  background: var(--accent) !important;

}

.sv-brand {

  font-family: 'Sora', sans-serif; font-weight: 700; font-size: 1.35rem;

  letter-spacing: .035em; color: var(--cyan);

}

.sv-side-h {

  color: #a9bdd8 !important; font-size: .86rem; font-weight: 700;

  margin: 1.35rem 0 .55rem; letter-spacing: .02em;

}

.sv-status { display:flex; gap:.55rem; align-items:center; padding:.3rem 0; font-size:.92rem; color:#eaf1ff; }

.sv-status .dot { width:1.1rem; text-align:center; font-weight:700; }

.sv-status.on .dot { color:var(--ok); }

.sv-status.off, .sv-status.off .dot { color:#a9bdd8; }



/* Main heading and descriptive text */

.sv-title {

  font-family:'Sora',sans-serif; font-weight:700; font-size:2.7rem;

  line-height:1.12; margin:0; color:#f8fbff;

  background:linear-gradient(90deg,#ffffff 0%,#8ff3ff 55%,#60a5fa 100%);

  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;

}

.sv-sub1 { font-family:'Sora',sans-serif; font-size:1.15rem; color:var(--cyan) !important; margin:.45rem 0 .25rem; font-weight:600; }

.sv-sub2 { color:#c0cce0 !important; margin-bottom:1.4rem; max-width:70ch; line-height:1.65; }

.sv-card {

  background:linear-gradient(145deg,rgba(20,34,56,.96),rgba(12,22,38,.96));

  border:1px solid #2a405e; border-radius:16px; padding:18px 20px;

  min-height:250px; box-shadow:0 8px 24px rgba(0,0,0,.12);

}

.sv-card.short { min-height:0; }

.sv-card h4 { margin:0 0 .8rem; font-size:1rem; color:#c0d5f1 !important; font-weight:700; }

.sv-big { font-family:'Sora',sans-serif; font-size:2rem; font-weight:700; line-height:1.2; color:#ffffff; }

.sv-row { display:flex; justify-content:space-between; gap:1rem; padding:.45rem 0; border-bottom:1px solid rgba(160,190,230,.14); font-size:.92rem; color:#f4f7ff; }

.sv-row:last-child { border-bottom:0; }

.sv-row span:first-child { color:#b8c8df; }

.sv-row span:last-child { text-align:right; color:#ffffff; }

.sv-bar { height:8px; border-radius:99px; background:#26364d; overflow:hidden; margin:.7rem 0 .3rem; }

.sv-bar>div { height:100%; border-radius:99px; background:linear-gradient(90deg,var(--blue),var(--cyan)); }

.sv-badge { display:inline-block; padding:.3rem .75rem; border-radius:99px; font-weight:700; font-size:.85rem; border:1px solid; }

.sv-badge.ok { color:#6ee7a0; border-color:rgba(74,222,128,.5); background:rgba(74,222,128,.12); }

.sv-badge.warn { color:#fcd34d; border-color:rgba(251,191,36,.5); background:rgba(251,191,36,.12); }

.sv-badge.idle { color:#d1dced; border-color:#526782; background:rgba(148,163,184,.12); }

.sv-note { color:#bdcce0 !important; font-size:.88rem; margin-top:.6rem; line-height:1.5; }

.sv-cap { font-family:'Sora',sans-serif; font-size:.95rem; font-weight:700; margin:0 0 .55rem; color:#eaf5ff !important; }

.sv-steps { display:flex; gap:.55rem; flex-wrap:wrap; margin:.3rem 0 1.4rem; }

.sv-step { flex:1; min-width:130px; padding:.7rem .8rem; border-radius:12px; border:1px solid #2a405e; background:#101b2c; font-size:.87rem; color:#f4f7ff; }

.sv-step b { display:block; font-weight:700; color:#ffffff; }

.sv-step span { color:#bdcce0; font-size:.8rem; }

.sv-step.done { border-color:rgba(53,224,242,.65); background:rgba(34,211,238,.08); }

.sv-step.done span { color:#67e8f9; }

.sv-step.skip { border-style:dashed; }

.sv-step.off { opacity:.85; border-style:dashed; }

.sv-empty { border:1px dashed #3a5577; border-radius:16px; padding:2.2rem; text-align:center; color:#c0cce0 !important; background:rgba(16,26,43,.65); }



/* Native Streamlit widgets: readable labels, inputs, tabs, tables and buttons */

.stTextInput label, .stSelectbox label, .stMultiSelect label, .stNumberInput label,

.stDateInput label, .stTimeInput label, .stFileUploader label, .stCheckbox label,

.stRadio label, .stSlider label, .stToggle label, .stTextArea label {

  color:#f0f5ff !important;

}

input, textarea, [data-baseweb="select"] > div {

  color:#f8fbff !important; background-color:#111d30 !important;

}

[data-testid="stFileUploader"] section {

  background:#111b2b !important; border:1px dashed #35cfe5 !important; border-radius:16px;

}

[data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span {

  color:#dce8f8 !important;

}

[data-testid="stFileUploaderDropzoneInstructions"] div { color:#eaf1ff !important; }

[data-testid="stImage"] img { border-radius:12px; border:1px solid #2a405e; }

.stButton button, .stDownloadButton button {

  color:#06111f !important; background:linear-gradient(90deg,#35e0f2,#60a5fa) !important;

  border:0 !important; border-radius:10px !important; font-weight:700 !important;

}

.stButton button:hover, .stDownloadButton button:hover { filter:brightness(1.08); }

button[kind="secondary"] { color:#f4f7ff !important; }

[data-testid="stTabs"] button { color:#c7d5e9 !important; }

[data-testid="stTabs"] button[aria-selected="true"] {

  color:#55e6f4 !important; border-bottom-color:#35e0f2 !important;

}

[data-testid="stAlert"] { border-radius:12px; }

[data-testid="stDataFrame"], [data-testid="stTable"] { border:1px solid #2a405e; border-radius:10px; }

[data-testid="stExpander"] { border:1px solid #2a405e; border-radius:10px; }

[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p { color:#f4f7ff !important; }

[data-testid="stProgressBar"] > div > div { background:linear-gradient(90deg,#35e0f2,#60a5fa); }

[data-testid="stSpinner"] p { color:#f4f7ff !important; }

.stCaption, [data-testid="stCaptionContainer"] { color:#bdcce0 !important; }

.stRadio [role="radiogroup"] { gap:.25rem; }

.sv-cm { border-collapse:separate; border-spacing:4px; width:100%; font-size:.9rem; }

.sv-cm th { color:#c0cce0; font-weight:600; padding:.45rem .5rem; text-align:center; }

.sv-cm th.r { text-align:right; }

.sv-cm td { text-align:center; padding:.8rem .4rem; border-radius:8px; font-weight:700; }

@media (max-width:900px) {

  .sv-title { font-size:1.9rem; }

  .block-container { padding-top:1.2rem; }

}

</style>

"""

st.markdown(CSS, unsafe_allow_html=True)





@st.cache_resource(show_spinner="Loading models...")

def get_models():

    return runner.load_models(CFG)





# ------------------------------------------------------------------ helpers

@st.cache_resource(show_spinner="Loading SmolVLM the first time...")
def get_vlm():
    """Load SmolVLM only when the user opens the VLM assistant."""
    from pipeline import vlm
    return vlm


def rows_html(pairs):

    return "".join(f'<div class="sv-row"><span>{k}</span><span>{v}</span></div>' for k, v in pairs)





def card(title, body, short=False):

    st.markdown(f'<div class="sv-card{" short" if short else ""}"><h4>{title}</h4>{body}</div>',

                unsafe_allow_html=True)





def header():

    st.markdown(

        '<div class="sv-title">SMART VISION</div>'

        '<div class="sv-sub1">Condition-Aware Object Detection</div>'

        '<div class="sv-sub2">Adaptive image enhancement for reliable object detection in adverse visual conditions</div>',

        unsafe_allow_html=True)





def message(text, kind="info"):

    getattr(st, kind)(text.replace("\n", "  \n"))





def to_png(rgb):

    buf = io.BytesIO()

    Image.fromarray(rgb).save(buf, "PNG")

    return buf.getvalue()





def _b64_jpeg(rgb, maxw=900):

    im = Image.fromarray(rgb)

    if im.width > maxw:

        im = im.resize((maxw, int(im.height * maxw / im.width)))

    buf = io.BytesIO()

    im.save(buf, "JPEG", quality=85)

    return base64.b64encode(buf.getvalue()).decode(), im.size





def compare_slider(left, right, left_label="Original", right_label="Processed"):

    """Drag-to-compare view (pure HTML/JS, no extra package)."""

    h, w = left.shape[:2]

    right = cv2.resize(right, (w, h), interpolation=cv2.INTER_AREA if right.shape[1] > w else cv2.INTER_CUBIC)

    a, (iw, ih) = _b64_jpeg(left)

    b, _ = _b64_jpeg(right)

    html = """

<style>

body{margin*:*0;font-family:sans-serif;color:#e6edf7}

.c{position:relative;max-width:900px;margin:0 auto;border-radius:12px;overflow:hidden;border:1px solid #1d3557}

.c img{display:block;width:100%}

#t{position:absolute;top:0;left:0;clip-path:inset(0 50% 0 0)}

.l{position:absolute;top:10px;padding:3px 10px;border-radius:99px;background:rgba(6,11,22,.75);font-size:12px}

input{display:block;width:100%;max-width:900px;margin:12px auto 0;accent-color:#22d3ee}

</style>

<div class="c"><img src="data:image/jpeg;base64,__B__"><img id="t" src="data:image/jpeg;base64,__A__">

<span class="l" style="left:10px">__L__</span><span class="l" style="right:10px">__R__</span></div>

<input type="range" min="0" max="100" value="50" id="r" aria-label="Compare original and processed">

<script>

const t=document.getElementById('t'),r=document.getElementById('r');

r.oninput=()=>{t.style.clipPath='inset(0 '+(100-r.value)+'% 0 0)'};

</script>

"""

    html = html.replace("__A__", a).replace("__B__", b).replace("__L__", left_label).replace("__R__", right_label)

    components.html(html, height=int(min(iw, 900) * ih / iw) + 60)





# ------------------------------------------------------------------ sidebar

def sidebar():

    models = get_models()

    settings = {}

    with st.sidebar:

        st.markdown('<div class="sv-brand">SMART VISION</div>', unsafe_allow_html=True)

        st.markdown('<div class="sv-side-h">Navigation</div>', unsafe_allow_html=True)

        page = st.radio("Navigation", PAGES, label_visibility="collapsed")



        if page in ("Live Detection", "Batch Mode"):

            st.markdown('<div class="sv-side-h">Pipeline settings</div>', unsafe_allow_html=True)

            settings["threshold"] = st.slider("Gate threshold", 0.50, 0.99,

                                              float(CFG["confidence_threshold"]), 0.01)

            settings["det_conf"] = st.slider("Detector confidence", 0.05, 0.90,

                                             float(CFG["detector"]["conf"]), 0.05)

            esr_ready = models["esrgan"][0] is not None

            settings["use_esrgan"] = st.toggle("ESRGAN super-resolution",

                                               value=CFG["esrgan"].get("enabled", True), disabled=not esr_ready)

            settings["baseline"] = st.toggle("Compare with baseline detection", value=True)



        st.markdown('<div class="sv-side-h">Pipeline status</div>', unsafe_allow_html=True)

        ready = [("Condition Classifier", models["classifier"][0] is not None),

                 ("Confidence Gate", True),

                 ("Adaptive Preprocessing", True),

                 ("ESRGAN", models["esrgan"][0] is not None),

                 ("Object Detector", models["detector"][0] is not None)]

        st.markdown("".join(

            f'<div class="sv-status {"on" if ok else "off"}"><span class="dot">{"✓" if ok else "○"}</span>{n}</div>'

            for n, ok in ready), unsafe_allow_html=True)

        st.markdown('<div class="sv-note">○ means the model file is not connected yet.</div>',

                    unsafe_allow_html=True)

    return page, settings, models





def step_strip(r):

    esr = r["esrgan"]

    rel_ok = r["reliability"] is not None and r["reliability"].status != detector.NOT_CONNECTED

    states = [

        ("Classify", "done" if r["condition"] else "off", "done" if r["condition"] else "unavailable"),

        ("Gate", "done", "done"),

        ("Preprocess", "done" if r["gate"].apply_processing else "skip",

         "done" if r["gate"].apply_processing else "skipped"),

        ("ESRGAN", "done" if esr == "Applied" else ("off" if esr == "Not available" else "skip"),

         "done" if esr == "Applied" else ("unavailable" if esr == "Not available" else "skipped")),

        ("YOLO", "done" if r["detection"] else "off", "done" if r["detection"] else "unavailable"),

        ("Reliability", "done" if rel_ok else "off", "done" if rel_ok else "not connected"),

    ]

    st.markdown('<div class="sv-steps">' + "".join(

        f'<div class="sv-step {c}"><b>{n}</b><span>{t}</span></div>' for n, c, t in states) + '</div>',

        unsafe_allow_html=True)





# ------------------------------------------------------------------ page: live detection

def page_live(settings, models):

    header()

    up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if up is None:

        st.markdown('<div class="sv-empty">Upload a JPG or PNG to run the Smart Vision pipeline.</div>',

                    unsafe_allow_html=True)

        return



    data = up.getvalue()

    key = (hashlib.md5(data).hexdigest(), tuple(sorted(settings.items())))

    runs = st.session_state.setdefault("runs", {})

    if key not in runs:

        pil = Image.open(io.BytesIO(data)).convert("RGB")

        with st.spinner("Running pipeline..."):

            result = runner.run(np.array(pil), pil, models, CFG, settings)

        if len(runs) >= 6:

            runs.pop(next(iter(runs)))

        runs[key] = result

    r = runs[key]



    step_strip(r)



    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown('<div class="sv-cap">Original image</div>', unsafe_allow_html=True)

        st.image(r["original"], width="stretch")

    with c2:

        st.markdown('<div class="sv-cap">Processed image</div>', unsafe_allow_html=True)

        st.image(r["processed"], width="stretch")

    with c3:

        st.markdown('<div class="sv-cap">Detection result</div>', unsafe_allow_html=True)

        if r["detection"] is not None:

            st.image(r["detection"].annotated, width="stretch")

        else:

            message(r["det_error"], "warning")



    if r["condition"] is None:

        message(r["cls_error"], "warning")

    if r.get("esrgan_error"):

        st.caption(f"ESRGAN: {r['esrgan_error']}")



    st.write("")

    k1, k2, k3 = st.columns(3)



    with k1:

        if r["condition"]:

            pct = r["confidence"] * 100

            others = sorted(((k, v) for k, v in r["probs"].items() if k != r["condition"]),

                            key=lambda kv: kv[1], reverse=True)[:2]

            card("Condition result",

                 f'<div class="sv-note" style="margin:0">Predicted condition</div>'

                 f'<div class="sv-big">{r["condition"]}</div>'

                 f'<div class="sv-row" style="margin-top:.8rem"><span>Confidence</span><span>{pct:.1f}%</span></div>'

                 f'<div class="sv-bar"><div style="width:{pct:.1f}%"></div></div>'

                 + rows_html([(f"Next: {k}", f"{v * 100:.1f}%") for k, v in others]))

        else:

            card("Condition result", '<div class="sv-badge idle">Classifier not connected</div>'

                 '<div class="sv-note">Add models/classifier.pth to see the predicted condition and confidence.</div>')



    with k2:

        g = r["gate"]

        card("Confidence gate",

             f'<span class="sv-badge {"ok" if g.apply_processing else "warn"}">{g.label}</span>'

             f'{rows_html([("Threshold", f"{g.threshold:.0%}")])}'

             f'<div class="sv-note">{g.reason}<br>Rule: confidence ≥ threshold and condition ≠ Normal '

             f'→ apply preprocessing; otherwise use the original.</div>')



    with k3:

        used = "Original image"

        if r["esrgan"] == "Applied":

            used = "ESRGAN-enhanced image" if r["gate"].apply_processing else "ESRGAN-enhanced original"

        elif r["gate"].apply_processing:

            used = "Processed image"

        card("Processing",

             rows_html([("Detected condition", r["condition"] or "Unknown"),

                        ("Processing applied", r["processing"]),

                        ("ESRGAN", r["esrgan"]),

                        ("Input to detector", used)]))



    st.write("")

    tabs = st.tabs(["Detection", "Baseline vs Smart Vision", "Before / after", "Stages and timing", "VLM Assistant"])



    with tabs[0]:

        d1, d2 = st.columns([2, 1])

        det = r["detection"]

        with d1:

            if det is None:

                card("Object detection", '<div class="sv-badge idle">Detector not connected</div>'

                     '<div class="sv-note">Add models/detector.pt to run detection.</div>', short=True)

            else:

                card("Object detection",

                     rows_html([("Detector", CFG["detector"]["name"]), ("Objects detected", det.count),

                                ("Inference time", f"{det.inference_ms:.0f} ms")]), short=True)

                if det.count:

                    tbl = det.table.assign(Confidence=det.table["Confidence"] * 100)

                    st.dataframe(tbl, hide_index=True, width="stretch",

                                 column_config={"Confidence": st.column_config.ProgressColumn(

                                     "Confidence", min_value=0, max_value=100, format="%.0f%%")})

                else:

                    st.caption("No objects detected at the current detector confidence.")

        with d2:

            rel = r["reliability"]

            if rel is None:

                card("Detection reliability", '<div class="sv-badge idle">Waiting for detector</div>', short=True)

            else:

                cls = {"GOOD": "ok", "RECOVERY APPLIED": "warn"}.get(rel.status, "idle")

                card("Detection reliability",

                     f'<span class="sv-badge {cls}">{rel.status}</span><div class="sv-note">{rel.message}</div>',

                     short=True)

        st.write("")

        b1, b2, b3 = st.columns(3)

        b1.download_button("Download processed image", to_png(r["processed"]), "processed.png", "image/png")

        if det is not None:

            b2.download_button("Download detection image", to_png(det.annotated), "detection.png", "image/png")

            if det.count:

                b3.download_button("Download detections (CSV)", det.table.to_csv(index=False),

                                   "detections.csv", "text/csv")



    with tabs[1]:

        base, det = r["baseline"], r["detection"]

        if det is None:

            message(r["det_error"], "warning")

        elif base is None:

            st.info("Baseline comparison is turned off in the sidebar.")

        else:

            x, y = st.columns(2)

            with x:

                st.markdown('<div class="sv-cap">Baseline: YOLO on the original image</div>', unsafe_allow_html=True)

                st.image(base.annotated, width="stretch")

            with y:

                st.markdown('<div class="sv-cap">Smart Vision: YOLO on the processed image</div>', unsafe_allow_html=True)

                st.image(det.annotated, width="stretch")

            m1, m2, m3 = st.columns(3)

            m1.metric("Baseline objects", base.count)

            m2.metric("Smart Vision objects", det.count, delta=det.count - base.count)

            mean = lambda d: d.table["Confidence"].mean() * 100 if d.count else None

            mb, ms = mean(base), mean(det)

            m3.metric("Mean confidence", f"{ms:.1f}%" if ms is not None else "n/a",

                      delta=f"{ms - mb:+.1f} pts" if (ms is not None and mb is not None) else None)

            st.caption("These are detection counts and confidences for this one image, not accuracy. "

                       "Use the Performance page (mAP on the test set) for the real comparison.")



    with tabs[2]:

        compare_slider(r["original"], r["processed"])



    with tabs[3]:

        s1, s2, s3 = st.columns(3)

        shape = lambda im: f"{im.shape[1]}×{im.shape[0]}px"

        with s1:

            st.markdown(f'<div class="sv-cap">Original ({shape(r["original"])})</div>', unsafe_allow_html=True)

            st.image(r["original"], width="stretch")

        with s2:

            st.markdown(f'<div class="sv-cap">After preprocessing ({shape(r["pre"])})</div>', unsafe_allow_html=True)

            st.image(r["pre"], width="stretch")

        with s3:

            st.markdown(f'<div class="sv-cap">After ESRGAN ({shape(r["processed"])})</div>', unsafe_allow_html=True)

            if r["esrgan"] == "Applied":

                st.image(r["processed"], width="stretch")

            else:

                st.markdown(f'<div class="sv-empty">ESRGAN: {r["esrgan"]}</div>', unsafe_allow_html=True)

        names = {"classify": "Condition classifier", "preprocess": "Preprocessing", "esrgan": "ESRGAN",

                 "detect": "YOLO detection", "baseline": "Baseline YOLO (original)"}

        rows = [{"Stage": names[k], "Time (ms)": round(v, 1)} for k, v in r["timings"].items()]

        if rows:

            tdf = pd.DataFrame(rows)

            t1, t2 = st.columns(2)

            t1.dataframe(tdf, hide_index=True, width="stretch")

            t2.bar_chart(tdf.set_index("Stage"))

            st.caption("Times are measured on the first run for this image; repeat runs with new "

                       "slider settings reuse cached stages.")

        else:

            st.caption("Timings appear once the models are connected.")



    with tabs[4]:
        st.subheader("SmolVLM scene explanation and Q&A")
        st.caption("SmolVLM explains the scene using the uploaded image and YOLO detections. YOLO performs object detection; VLM does not replace it.")

        det_for_vlm = r.get("detection")
        detection_records = []
        if det_for_vlm is not None and not det_for_vlm.table.empty:
            for row in det_for_vlm.table.to_dict(orient="records"):
                detection_records.append({
                    "class": str(row.get("Object", "unknown")),
                    "confidence": float(row["Confidence"]) if row.get("Confidence") is not None else None,
                })

        if det_for_vlm is None:
            st.warning("YOLO detection is unavailable. VLM can still describe the image, but detection data will be missing.")
        else:
            counts = det_for_vlm.table["Object"].value_counts().to_dict() if not det_for_vlm.table.empty else {}
            m1, m2 = st.columns(2)
            m1.metric("Detected objects", int(det_for_vlm.count))
            m2.metric("Object categories", len(counts))
            if counts:
                st.dataframe(
                    pd.DataFrame([{"Object": name, "Count": count} for name, count in counts.items()]),
                    hide_index=True, width="stretch"
                )
            else:
                st.info("No objects were detected at the current confidence threshold.")

        question = st.text_input(
            "Ask a question about this image",
            value="Explain the environmental condition and detected objects.",
            placeholder="Example: How might fog affect object visibility?",
            key="vlm_question",
        )
        if st.button("Generate VLM explanation", type="primary", key="run_vlm"):
            if not question.strip():
                st.warning("Enter a question first.")
            else:
                temp_path = None
                try:
                    vlm_module = get_vlm()
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                        temp_file.write(to_png(r["original"]))
                        temp_path = temp_file.name
                    with st.spinner("SmolVLM is analysing the image. The first run may take longer..."):
                        answer = vlm_module.answer_question(
                            image_path=temp_path,
                            question=question.strip(),
                            condition=r.get("condition") or "Unknown",
                            detections=detection_records,
                            condition_confidence=r.get("confidence"),
                        )
                    st.markdown("#### VLM response")
                    st.write(answer)
                except Exception as exc:
                    st.error(f"VLM could not generate a response: {exc}")
                    st.info("Check that transformers, torch, and the SmolVLM model are installed and available. The main detection dashboard should remain usable.")
                finally:
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except OSError:
                            pass

    if r["probs"]:

        with st.expander("Class probabilities"):

            st.bar_chart(pd.Series(r["probs"], name="Probability"))





# ------------------------------------------------------------------ page: batch mode

def page_batch(settings, models):

    header()

    files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if not files:

        st.markdown('<div class="sv-empty">Upload several images to evaluate them in one go.<br>'

                    'You get a results table you can download as CSV.</div>', unsafe_allow_html=True)

        return

    if st.button(f"Run pipeline on {len(files)} image(s)", type="primary"):

        rows, bar = [], st.progress(0.0)

        for i, f in enumerate(files):

            pil = Image.open(f).convert("RGB")

            r = runner.run(np.array(pil), pil, models, CFG, {**settings, "baseline": True})

            rows.append({

                "Image": f.name,

                "Condition": r["condition"],

                "Confidence (%)": round(r["confidence"] * 100, 1) if r["confidence"] is not None else None,

                "Gate": r["gate"].label,

                "Processing": r["processing"],

                "ESRGAN": r["esrgan"],

                "Baseline objects": r["baseline"].count if r["baseline"] else None,

                "Smart Vision objects": r["detection"].count if r["detection"] else None,

                "Time (ms)": round(sum(r["timings"].values()), 0) if r["timings"] else None,

            })

            bar.progress((i + 1) / len(files))

        bar.empty()

        st.session_state["batch_df"] = pd.DataFrame(rows)



    df = st.session_state.get("batch_df")

    if df is None:

        return

    m1, m2, m3 = st.columns(3)

    m1.metric("Images processed", len(df))

    m2.metric("Processing applied", int((df["Gate"] == "PROCESSING APPLIED").sum()))

    m3.metric("Fell back to original", int((df["Gate"] == "FALLBACK TO ORIGINAL").sum()))

    st.dataframe(df, hide_index=True, width="stretch")

    if df["Condition"].notna().any():

        st.markdown("**Predicted conditions**")

        st.bar_chart(df["Condition"].value_counts())

    st.download_button("Download results (CSV)", df.to_csv(index=False), "batch_results.csv", "text/csv")





# ------------------------------------------------------------------ page: performance

def load_csv(key, **kw):

    path = ROOT / CFG["results"].get(key, "")

    if not path.is_file():

        return None

    try:

        df = pd.read_csv(path, **kw)

        return df if not df.empty else None

    except Exception:

        return None





def _norm(s):

    return str(s).strip().lower().replace(" ", "_").replace("-", "_")





def pick(df, names):

    lookup = {_norm(c): c for c in df.columns}

    for n in names:

        if _norm(n) in lookup:

            return lookup[_norm(n)]

    return None





def fmt_metric(v):

    try:

        v = float(v)

    except (TypeError, ValueError):

        return str(v)

    return f"{v * 100:.2f}%" if v <= 1 else f"{v:.2f}%"





CLS_METRICS = {"Accuracy": ["accuracy", "acc"], "Precision": ["precision"], "Recall": ["recall"],

               "F1-score": ["f1", "f1_score", "f1-score"]}

DET_COLS = {

    "Model": ["model", "detector", "name"],

    "Precision": ["precision", "p"],

    "Recall": ["recall", "r"],

    "mAP@0.5": ["map50", "map@0.5", "map_0.5", "map@50", "map_50"],

    "mAP@0.5:0.95": ["map50_95", "map@0.5:0.95", "map50-95", "map_50_95", "map@50-95"],

    "Inference (ms)": ["inference_ms", "inference_time", "inference_time_ms", "time_ms"],

    "FPS": ["fps"],

    "Model size (MB)": ["model_size_mb", "size_mb", "model_size", "size"],

}





def confusion_matrix_view():

    st.subheader("Confusion matrix")

    png = ROOT / CFG["results"].get("confusion_png", "")

    cm = load_csv("confusion_csv", index_col=0)

    if cm is None and not png.is_file():

        st.info("No confusion matrix available yet.")

        return

    if cm is not None:

        try:

            vals = cm.values.astype(float)

            rs = vals.sum(axis=1, keepdims=True)

            norm = np.divide(vals, rs, out=np.zeros_like(vals), where=rs > 0)

            head = '<tr><th></th>' + "".join(f"<th>{c}</th>" for c in cm.columns) + "</tr>"

            body = "".join(

                f'<tr><th class="r">{idx}</th>' + "".join(

                    f'<td style="background:rgba(34,211,238,{0.06 + 0.8 * n:.2f});color:{"#041018" if n > .55 else "#e6edf7"}">{int(v)}</td>'

                    for v, n in zip(row, nrow)) + "</tr>"

                for idx, row, nrow in zip(cm.index, vals, norm))

            st.markdown(f'<table class="sv-cm">{head}{body}</table>', unsafe_allow_html=True)

            st.caption("Rows: true condition. Columns: predicted condition. Shading is per-row share.")

            return

        except Exception:

            st.dataframe(cm, width="stretch")

            return

    st.image(str(png), width="stretch")





def page_performance():

    header()

    st.subheader("Condition classifier performance")

    cdf = load_csv("classifier_csv")

    if cdf is None:

        st.info("No performance results available yet.")

    else:

        row = None

        if len(cdf) == 1:

            row = cdf.iloc[0]

        else:

            label_col = pick(cdf, ["class", "label", "metric", "split"]) or cdf.columns[0]

            hit = cdf[cdf[label_col].astype(str).str.lower().isin(

                ["overall", "all", "macro avg", "macro", "weighted avg", "test"])]

            if not hit.empty:

                row = hit.iloc[0]

        for col, (name, aliases) in zip(st.columns(4), CLS_METRICS.items()):

            src = pick(cdf, aliases)

            col.metric(name, fmt_metric(row[src]) if (row is not None and src) else "n/a")

        st.dataframe(cdf, hide_index=True, width="stretch")

        f1c, lab = pick(cdf, CLS_METRICS["F1-score"]), pick(cdf, ["class", "label"])

        if f1c and lab and len(cdf) > 1:

            st.bar_chart(cdf.set_index(lab)[f1c])

    st.write("")

    confusion_matrix_view()



    st.write("")

    st.subheader("Detector comparison")

    ddf = load_csv("detector_csv")

    if ddf is None:

        st.info("Results not available yet.")

    else:

        shown = {n: ddf[src] for n, a in DET_COLS.items() if (src := pick(ddf, a)) is not None}

        table = pd.DataFrame(shown)

        if "Model" not in table:

            table.insert(0, "Model", [f"Model {i + 1}" for i in range(len(table))])

        indexed = table.set_index("Model")



        best = []

        if "mAP@0.5" in indexed:

            best.append(("Highest mAP@0.5", indexed["mAP@0.5"].idxmax(), indexed["mAP@0.5"].max()))

        if "FPS" in indexed:

            best.append(("Fastest (FPS)", indexed["FPS"].idxmax(), indexed["FPS"].max()))

        if "Model size (MB)" in indexed:

            best.append(("Smallest model", indexed["Model size (MB)"].idxmin(), indexed["Model size (MB)"].min()))

        for col, (title, name, val) in zip(st.columns(max(len(best), 1)), best):

            col.metric(title, str(name), f"{val:.3f}" if val <= 1 else f"{val:.1f}", delta_color="off")



        st.dataframe(table, hide_index=True, width="stretch")

        groups = [("Accuracy metrics", ["Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"]),

                  ("Inference time (ms)", ["Inference (ms)"]), ("Speed (FPS)", ["FPS"]),

                  ("Model size (MB)", ["Model size (MB)"])]

        groups = [(t, [c for c in cs if c in indexed.columns]) for t, cs in groups]

        groups = [(t, cs) for t, cs in groups if cs]

        for i in range(0, len(groups), 2):

            for col, (title, cs) in zip(st.columns(2), groups[i:i + 2]):

                with col:

                    st.markdown(f"**{title}**")

                    st.bar_chart(indexed[cs])



    st.write("")

    st.subheader("Detector results by condition")

    pdf = load_csv("per_condition_csv")

    if pdf is None:

        st.info("Results not available yet. Add results/per_condition_results.csv with columns: "

                "model, condition, precision, recall, map50, map50_95 (optional: setting).")

        return

    cond_col, model_col = pick(pdf, ["condition", "class", "weather"]), pick(pdf, DET_COLS["Model"])

    set_col = pick(pdf, ["setting", "pipeline", "variant", "mode"])

    metrics = {n: src for n, a in DET_COLS.items()

               if n in ("Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95") and (src := pick(pdf, a))}

    if not cond_col or not model_col or not metrics:

        st.dataframe(pdf, hide_index=True, width="stretch")

        return

    choice = st.selectbox("Metric", list(metrics))

    series = pdf[model_col].astype(str) + (" (" + pdf[set_col].astype(str) + ")" if set_col else "")

    pivot = pdf.assign(_s=series).pivot_table(index=cond_col, columns="_s", values=metrics[choice])

    st.bar_chart(pivot)

    st.dataframe(pivot, width="stretch")





# ------------------------------------------------------------------ page: gallery

def find_sample(folder: Path, stem: str):

    for ext in ("jpg", "jpeg", "png"):

        p = folder / f"{stem}.{ext}"

        if p.exists():

            return p

    return None





def page_gallery():

    header()

    base = ROOT / CFG["samples_dir"]

    for tab, cond in zip(st.tabs(CONDITIONS), CONDITIONS):

        slug = cond.lower()

        with tab:

            found = False

            for col, (stem, title) in zip(st.columns(3), [("original", "Original"), ("processed", "Processed"),

                                                          ("detection", "Detection")]):

                with col:

                    st.markdown(f'<div class="sv-cap">{title}</div>', unsafe_allow_html=True)

                    p = find_sample(base / slug, stem)

                    if p:

                        found = True

                        st.image(str(p), width="stretch")

                    else:

                        st.markdown('<div class="sv-empty">Not added yet</div>', unsafe_allow_html=True)

            if not found:

                st.caption(f"Run \`python make_samples.py <image_folder>\` to fill this automatically, "

                           f"or add original/processed/detection images to samples/{slug}/.")





# ------------------------------------------------------------------ page: about

def page_about():

    header()

    a, b = st.columns(2)

    with a:

        card("Project", rows_html([("Project", "Smart Vision"), ("Title", "Condition-Aware Object Detection")]))

    with b:

        card("Conditions", "".join(f'<span class="sv-badge idle" style="margin:0 .4rem .4rem 0">{c}</span>'

                                   for c in CONDITIONS))

    st.write("")

    a, b = st.columns(2)

    with a:

        card("Problem", "<p>Object detection performance can degrade under difficult visual conditions "

             "such as low light, fog, rain, snow and sand/dust.</p>")

    with b:

        card("Solution", "<p>Smart Vision first identifies the visual condition, applies an appropriate "

             "preprocessing method when confidence is sufficient, performs ESRGAN super-resolution, "

             "and then performs object detection.</p>")

    st.write("")

    stages = ["Image", "Condition Classification", "Confidence Gate", "Adaptive Preprocessing",

              "ESRGAN", "YOLO", "Reliability Check", "Final Detection"]

    st.markdown("#### Pipeline")

    st.markdown('<div class="sv-steps">' + "".join(f'<div class="sv-step done"><b>{s}</b></div>' for s in stages)

                + '</div>', unsafe_allow_html=True)

    st.markdown("#### Condition-specific processing")

    st.dataframe(pd.DataFrame({"Condition": list(enhancers.PROCESSING_LABELS),

                               "Processing": list(enhancers.PROCESSING_LABELS.values())}),

                 hide_index=True, width="stretch")





# ------------------------------------------------------------------ router

page, settings, models = sidebar()

if page == "Live Detection":

    page_live(settings, models)

elif page == "Batch Mode":

    page_batch(settings, models)

elif page == "Performance":

    page_performance()

elif page == "Sample Gallery":

    page_gallery()

else:

    page_about()
