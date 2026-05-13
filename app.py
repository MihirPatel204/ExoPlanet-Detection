from __future__ import annotations

from pathlib import Path

import warnings
warnings.filterwarnings("ignore", message=".*tpfmodel.*oktopus.*")

import joblib
import lightkurve as lk
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model_utils import build_feature_frame

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "Data"
MODEL_PATH = ROOT_DIR / "Models" / "best_model.joblib"

st.set_page_config(page_title="Exoplanet Finder", page_icon="🛰️", layout="wide")

# ── Dark Vintage Theme ──────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700;800&family=DM+Sans:wght@400;500;600;700&display=swap');

    :root {
        --bg-deep:    #0f1015;
        --bg-mid:     #14151b;
        --bg-card:    #1b1b22;
        --bg-card-hi: #23232b;
        --warm-100:   #f2e9da;
        --warm-200:   #ddd2c2;
        --warm-300:   #c1b19a;
        --warm-400:   #a9967f;
        --warm-500:   #8c7f6c;
        --copper:     #b7a48a;
        --amber:      #a9967f;
        --gold:       #d6b57a;
        --rust:       #7a5a4b;
        --cream:      #f2e9da;
        --text-main:  #f2e9da;
        --text-dim:   #b7a48a;
        --text-muted: #8d826f;
        --line:       rgba(183, 164, 138, 0.14);
        --line-hi:    rgba(183, 164, 138, 0.25);
        --shadow:     0 20px 50px rgba(0, 0, 0, 0.55);
        --glow:       0 0 60px rgba(183, 164, 138, 0.10);
    }

    /* ── Base ─────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--text-main);
    }
    html, body {
        height: 100%;
        overflow-y: auto;
    }
    div[data-testid="stAppViewContainer"],
    section.main {
        overflow-y: auto;
    }
    h1, h2, h3, h4 {
        font-family: 'Playfair Display', serif;
        letter-spacing: -0.01em;
        color: var(--cream) !important;
    }

    /* ── App background ─────────────────────────────── */
    .stApp {
        background: var(--bg-deep);
        color: var(--text-main);
    }
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        background:
            radial-gradient(700px 500px at 12% 6%,  rgba(183,164,138,0.06), transparent 60%),
            radial-gradient(600px 400px at 88% 85%, rgba(125,117,103,0.05), transparent 55%),
            radial-gradient(400px 350px at 50% 40%, rgba(242,233,218,0.03), transparent 50%);
        pointer-events: none;
        z-index: 0;
    }
    .stApp > div {
        position: relative;
        z-index: 1;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* ── Reduce Streamlit's excessive padding ────────── */
    section.main > div {
        padding-top: 1.4rem;
    }
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 100% !important;
    }
    /* Reduce gap between Streamlit elements */
    .stElementContainer {
        margin-bottom: 0 !important;
    }
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.55rem !important;
    }

    /* ── Style Streamlit columns as cards ─────────────── */
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
        background: var(--bg-card);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1.2rem 1.4rem 1.4rem;
        box-shadow: 0 8px 28px rgba(0,0,0,0.35);
        transition: border-color 0.3s ease;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:hover {
        border-color: var(--line-hi);
    }
    /* Avoid card-in-card when using columns inside a column */
    div[data-testid="stColumn"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
        background: transparent;
        border: none;
        box-shadow: none;
        padding: 0;
    }

    /* ── Streamlit widget overrides ──────────────────── */
    .stSelectbox label,
    .stTextInput label,
    .stSlider label,
    .stCheckbox label {
        color: var(--text-dim) !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.02em;
    }
    .stSubheader {
        color: var(--cream) !important;
        font-family: 'Playfair Display', serif !important;
        padding-bottom: 0.1rem !important;
        border-bottom: none !important;
        margin-bottom: 0.4rem !important;
    }
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    div[data-baseweb="textarea"] > div {
        background: var(--bg-card-hi) !important;
        border: 1px solid var(--line-hi) !important;
        border-radius: 10px !important;
        color: var(--text-main) !important;
    }
    div[data-baseweb="input"] input,
    div[data-baseweb="select"] input {
        color: var(--text-main) !important;
    }
    /* Selected option text color */
    div[data-baseweb="select"] span {
        color: var(--text-main) !important;
    }
    div[data-baseweb="popover"] > div,
    div[data-baseweb="menu"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--line-hi) !important;
    }
    div[data-baseweb="menu"] li {
        color: var(--text-main) !important;
    }
    div[data-baseweb="menu"] li:hover {
        background: var(--bg-card-hi) !important;
    }
    .stSlider > div[data-baseweb="slider"] {
        background: var(--bg-card-hi);
        border-radius: 10px;
        padding: 0.7rem 0.5rem;
        border: 1px solid var(--line);
    }
    .stSlider [role="slider"] {
        background: var(--copper) !important;
    }
    .stSlider div[data-testid="stTickBarMin"],
    .stSlider div[data-testid="stTickBarMax"] {
        color: var(--text-dim) !important;
    }
    .stCheckbox span {
        color: var(--text-main) !important;
    }

    /* ── Hero banner ────────────────────────────────── */
    .hero {
        position: relative;
        overflow: hidden;
        padding: 1.8rem 2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #14151b 0%, #1d1e26 40%, #242631 100%);
        color: var(--cream);
        box-shadow: var(--shadow), var(--glow);
        border: 1px solid var(--line-hi);
        margin-bottom: 0.3rem;
    }
    .hero::after {
        content: "";
        position: absolute;
        width: 300px;
        height: 300px;
        right: -120px;
        top: -90px;
        background: radial-gradient(circle, rgba(214,181,122,0.20), transparent 70%);
        filter: blur(8px);
    }
    .hero::before {
        content: "";
        position: absolute;
        width: 180px;
        height: 180px;
        left: -50px;
        bottom: -70px;
        background: radial-gradient(circle, rgba(183,164,138,0.12), transparent 70%);
        filter: blur(6px);
    }
    .hero-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 1.8rem;
        align-items: center;
    }
    .hero-text {
        flex: 1 1 320px;
        min-width: 260px;
    }
    .hero h1 {
        font-weight: 800;
        margin: 0.3rem 0;
        font-size: 2.4rem;
        background: linear-gradient(135deg, var(--gold) 0%, var(--copper) 60%, var(--warm-300) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero p {
        margin: 0;
        opacity: 0.85;
        font-size: 0.95rem;
        line-height: 1.55;
        color: var(--warm-200);
    }
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        font-size: 0.62rem;
        font-weight: 700;
        color: var(--copper);
        border: 1px solid rgba(183,164,138,0.30);
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        background: rgba(183,164,138,0.06);
    }
    .stat-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 1rem;
    }
    .stat {
        padding: 0.45rem 0.75rem;
        border-radius: 10px;
        background: rgba(183,164,138,0.07);
        border: 1px solid rgba(183,164,138,0.16);
        font-size: 0.78rem;
        color: var(--warm-200);
    }
    .stat span {
        display: block;
        opacity: 0.5;
        font-size: 0.62rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.1rem;
    }
    .hero-panel {
        flex: 1 1 240px;
        min-width: 220px;
        padding: 1rem 1.1rem;
        border-radius: 14px;
        background: rgba(183,164,138,0.06);
        border: 1px solid rgba(183,164,138,0.16);
    }
    .hero-panel h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1.05rem;
        color: var(--gold) !important;
    }
    .hero-panel ul {
        margin: 0;
        padding-left: 1rem;
        color: var(--warm-200);
        line-height: 1.6;
        font-size: 0.88rem;
    }
    .hero-panel li {
        margin-bottom: 0.2rem;
    }
    .hero-panel li::marker {
        color: var(--copper);
    }

    /* ── Section titles ─────────────────────────────── */
    .section-title {
        margin: 1rem 0 0.6rem 0;
        font-family: 'Playfair Display', serif;
        font-size: 1.25rem;
        color: var(--warm-300) !important;
        font-weight: 700;
        position: relative;
        padding-left: 0.9rem;
    }
    .section-title::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0.15em;
        bottom: 0.15em;
        width: 3px;
        border-radius: 2px;
        background: linear-gradient(180deg, var(--copper), var(--gold));
    }

    /* ── Steps ──────────────────────────────────────── */
    .steps {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.8rem;
        margin-top: 0.6rem;
    }
    .step {
        padding: 1rem 1.1rem;
        border-radius: 14px;
        background: var(--bg-card);
        border: 1px solid var(--line);
        box-shadow: 0 6px 18px rgba(0,0,0,0.25);
        transition: transform 0.25s ease, border-color 0.3s ease, box-shadow 0.25s ease;
    }
    .step:hover {
        transform: translateY(-2px);
        border-color: var(--line-hi);
        box-shadow: 0 10px 24px rgba(0,0,0,0.4), var(--glow);
    }
    .step-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 26px;
        height: 26px;
        border-radius: 8px;
        background: rgba(183,164,138,0.16);
        border: 1px solid rgba(183,164,138,0.28);
        color: var(--copper);
        font-size: 0.72rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .step h4 {
        margin: 0 0 0.35rem 0;
        font-size: 0.95rem;
        color: var(--gold) !important;
    }
    .step p {
        color: var(--text-dim);
        font-size: 0.84rem;
        line-height: 1.45;
        margin: 0;
    }

    /* ── Metric overrides ───────────────────────────── */
    div[data-testid="stMetric"] {
        background: transparent;
        border-radius: 14px;
        padding: 0.2rem 0;
        border: none;
        box-shadow: none;
    }
    div[data-testid="stMetric"] label {
        color: var(--text-dim) !important;
        font-size: 0.78rem !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: var(--gold) !important;
        font-family: 'Playfair Display', serif !important;
    }

    /* ── Button ─────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, var(--copper) 0%, var(--amber) 50%, var(--gold) 100%);
        color: #111217;
        border: none;
        border-radius: 999px;
        padding: 0.55rem 1.4rem;
        font-weight: 700;
        font-size: 0.85rem;
        font-family: 'DM Sans', sans-serif;
        box-shadow: 0 8px 24px rgba(183,164,138,0.25);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        letter-spacing: 0.02em;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(183,164,138,0.35);
    }

    /* ── Caption / footer ───────────────────────────── */
    
    .stCaption, .stMarkdown small, div[data-testid="stCaptionContainer"] {
        color: var(--text-muted) !important;
    }
    .footer-bar {
        margin-top: 1.5rem;
        padding: 0.8rem 0;
        border-top: 1px solid var(--line);
        text-align: center;
        color: var(--text-muted);
        font-size: 0.75rem;
        letter-spacing: 0.03em;
    }
    .footer-bar a {
        color: var(--copper);
        text-decoration: none;
    }

    /* ── Info / Error / Warning boxes ────────────────── */
    .stAlert {
        background: var(--bg-card-hi) !important;
        border-radius: 12px !important;
        color: var(--text-main) !important;
    }

    /* ── Fade-in animation ──────────────────────────── */
    .fade-in {
        animation: fadeUp 0.5s ease both;
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ── Scrollbar ───────────────────────────────────── */
    ::-webkit-scrollbar { width: 7px; height: 7px; }
    ::-webkit-scrollbar-track { background: var(--bg-deep); }
    ::-webkit-scrollbar-thumb { background: var(--warm-500); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--warm-400); }

    /* ── Divider ──────────────────────────────────────── */
    hr { border-color: var(--line) !important; }

    /* ── Responsive ──────────────────────────────────── */
    @media (max-width: 768px) {
        .steps { grid-template-columns: 1fr; }
        .hero h1 { font-size: 1.8rem; }
        .hero-grid { gap: 1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Hero Section ────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero fade-in">
        <div class="hero-grid">
            <div class="hero-text">
                <span class="badge">✦ Kepler Signal Lab</span>
                <h1>Exoplanet Finder</h1>
                <p>Choose a Kepler target, pull its light curve, and run the trained classifier to flag planet-like transit dips.</p>
                <div class="stat-row">
                    <div class="stat">
                        <span>Pipeline</span>
                        Lightkurve + feature extraction
                    </div>
                    <div class="stat">
                        <span>Output</span>
                        Probability + label
                    </div>
                    <div class="stat">
                        <span>Focus</span>
                        Long-cadence Kepler
                    </div>
                </div>
            </div>
            <div class="hero-panel">
                <h3>Workflow</h3>
                <ul>
                    <li>Pick a star from the catalog or enter a KIC ID.</li>
                    <li>Fetch a stitched light curve in one click.</li>
                    <li>Inspect the curve and prediction confidence.</li>
                </ul>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Model & Data Loading ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_catalog() -> pd.DataFrame | None:
    catalog_path = DATA_DIR / "exo_list.csv"
    if not catalog_path.exists():
        return None
    return pd.read_csv(catalog_path)


def get_catalog_actual_label(
    catalog: pd.DataFrame | None, kic_id: str
) -> tuple[str | None, str | None]:
    if catalog is None or not kic_id:
        return None, None
    match = catalog[catalog["kepid"].astype(str) == str(kic_id)]
    if match.empty:
        return None, None

    dispositions = (
        match["koi_disposition"].dropna().astype(str).unique().tolist()
    )
    if any(d in ("CONFIRMED", "CANDIDATE") for d in dispositions):
        return "Exoplanet", "CONFIRMED/CANDIDATE"
    if "FALSE POSITIVE" in dispositions:
        return "No Planet", "FALSE POSITIVE"

    if dispositions:
        return "Unknown", ", ".join(dispositions)
    return "Unknown", "Unknown"


@st.cache_data(show_spinner=False)
def download_lightcurve_flux(kic_id: str) -> tuple[np.ndarray, np.ndarray | None]:
    name = f"KIC {kic_id}"
    search_result = lk.search_lightcurve(
        name, mission="Kepler", author="Kepler", cadence="long"
    )
    if len(search_result) == 0:
        raise ValueError("No Kepler light curve found for that ID.")

    lcc = search_result.download_all()
    if lcc is None:
        raise ValueError("Light curve download failed.")

    lc = lcc.stitch().remove_nans()
    flux = np.array(lc.flux.value, dtype=np.float64)
    time = np.array(lc.time.value, dtype=np.float64) if hasattr(lc, "time") else None
    return flux, time


model_pack = load_model()

# ── Signal Explorer ─────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Signal Explorer</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    st.subheader("Star Selection")

    catalog = load_catalog()
    selected_kic = ""

    if catalog is not None:
        catalog = catalog.copy()

        # Filter by disposition
        disp_options = ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE", "All"]
        disp_filter = st.selectbox("Filter by disposition", disp_options, index=0)

        if disp_filter != "All":
            filtered = catalog[catalog["koi_disposition"] == disp_filter].copy()
        else:
            filtered = catalog.copy()

        # Build display names safely
        def _make_display_name(row):
            name = row.get("kepler_name")
            if pd.isna(name):
                name = row.get("kepoi_name")
            if pd.isna(name):
                name = "Unknown"
            return f"KIC {row.kepid} — {name}"

        filtered["display_name"] = filtered.apply(_make_display_name, axis=1)

        if len(filtered) > 0:
            options = filtered["display_name"].tolist()
            selected_label = st.selectbox("Pick a star", options, index=0)

            match = filtered.loc[
                filtered["display_name"] == selected_label, "kepid"
            ]
            if len(match) > 0:
                selected_kic = str(match.iloc[0])
        else:
            st.warning("No stars match this filter.")
    else:
        st.info("Catalog not found — enter a KIC ID manually below.")

    manual_kic = st.text_input("Or enter KIC ID", value=selected_kic)
    threshold = st.slider("Prediction threshold", 0.1, 0.9, 0.5, 0.05)
    show_curve = st.checkbox("Show light curve", value=True)

    if st.button("Clear download cache"):
        download_lightcurve_flux.clear()
        st.success("Cache cleared — next prediction will re-download.")

# Variables to hold chart data for full-width rendering outside columns
_chart_flux = None
_chart_time = None
_chart_show = False

with col_right:
    st.subheader("Prediction")

    if model_pack is None:
        st.error(
            "Model artifact not found. "
            "Run `scripts/train_best_model.py` first to train and save the model."
        )
    else:
        if st.button("Download + Predict", type="primary"):
            kic_id = manual_kic.strip()
            if not kic_id:
                st.error("Please provide a KIC ID.")
            else:
                with st.spinner("Downloading Kepler light curve…"):
                    try:
                        flux, time = download_lightcurve_flux(kic_id)
                        # Ensure plain numpy arrays (cache may return astropy types)
                        flux = np.asarray(flux, dtype=np.float64)
                        if time is not None:
                            time = np.asarray(time, dtype=np.float64)
                    except Exception as exc:
                        flux, time = None, None
                        st.error(f"Download failed: {exc}")

                if flux is not None:
                    # Defensively extract model parameters
                    if isinstance(model_pack, dict):
                        n_points = int(model_pack.get("n_points", len(flux)))
                        sigma = int(model_pack.get("sigma", 10))
                        feature_names = model_pack.get("feature_names")
                        scaler = model_pack.get("scaler")
                        model = model_pack.get("model")
                    else:
                        n_points = len(flux)
                        sigma = 10
                        feature_names = None
                        scaler = None
                        model = model_pack

                    try:
                        features = build_feature_frame(
                            flux, n_points, sigma, feature_names
                        )

                        if scaler is not None:
                            X_scaled = scaler.transform(features)
                        else:
                            X_scaled = features.values

                        probability = float(model.predict_proba(X_scaled)[0, 1])
                        prediction = (
                            "Exoplanet" if probability >= threshold else "No Planet"
                        )

                        col_m1, col_m2 = st.columns(2)
                        col_m1.metric("Prediction", prediction)
                        col_m2.metric("Exoplanet Probability", f"{probability:.3f}")
                        st.caption("Label 1 = exoplanet · Label 0 = no planet")

                        actual_label, actual_source = get_catalog_actual_label(
                            catalog, kic_id
                        )
                        if actual_label is None:
                            actual_label = "Unknown / Not available"
                            actual_source = "Catalog: not found"
                            match_status = "N/A"
                        elif actual_label == "Unknown":
                            match_status = "N/A"
                        else:
                            match_status = (
                                "Match" if actual_label == prediction else "Mismatch"
                            )

                        compare_df = pd.DataFrame(
                            [
                                {"Metric": "Prediction", "Value": prediction},
                                {
                                    "Metric": "Actual (catalog)",
                                    "Value": actual_label,
                                },
                                {
                                    "Metric": "Disposition", 
                                    "Value": actual_source,
                                },
                                {"Metric": "Match", "Value": match_status},
                            ]
                        )
                        st.table(compare_df)
                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")
                        flux = None  # skip chart

                    if flux is not None and show_curve:
                        _chart_flux = flux
                        _chart_time = time
                        _chart_show = True

# ── Full-width Light Curve Chart ────────────────────────────────────────────
if _chart_show and _chart_flux is not None:
    st.markdown('<div class="section-title">Light Curve Preview</div>', unsafe_allow_html=True)

    # Downsample to max 5000 points (WebGL handles this efficiently)
    MAX_CHART_PTS = 5000
    if len(_chart_flux) > MAX_CHART_PTS:
        idx = np.linspace(0, len(_chart_flux) - 1, MAX_CHART_PTS, dtype=int)
    else:
        idx = np.arange(len(_chart_flux))

    curve_data = _chart_flux[idx]

    # Interactive Plotly chart
    if _chart_time is not None and len(_chart_time) >= len(_chart_flux):
        x_vals = _chart_time[idx]
        x_label = "Time (BKJD)"
    else:
        x_vals = np.arange(len(curve_data))
        x_label = "Index"

    # Auto-zoom Y-axis to show transit dips
    p1 = np.percentile(curve_data, 0.5)
    p99 = np.percentile(curve_data, 99.5)
    margin = (p99 - p1) * 0.15

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=x_vals,
        y=curve_data,
        mode="lines",
        line=dict(color="#b7a48a", width=1.2),
        fill="tozeroy",
        fillcolor="rgba(183,164,138,0.08)",
        hovertemplate=(
            f"<b>{x_label}</b>: %{{x:.2f}}<br>"
            "<b>Flux</b>: %{y:.6f}<extra></extra>"
        ),
    ))
    fig.update_layout(
        height=370,
        margin=dict(l=50, r=20, t=10, b=45),
        paper_bgcolor="#1b1b22",
        plot_bgcolor="#14151b",
        xaxis=dict(
            title=x_label,
            color="#b7a48a",
            gridcolor="rgba(183,164,138,0.12)",
            zeroline=False,
        ),
        yaxis=dict(
            title="Normalized Flux",
            color="#b7a48a",
            gridcolor="rgba(183,164,138,0.12)",
            zeroline=False,
            range=(
                [p1 - margin, p99 + margin]
                if margin > 0
                else None
            ),
        ),
        font=dict(family="DM Sans", color="#b7a48a"),
        hovermode="x unified",
        dragmode="zoom",
    )
    st.plotly_chart(fig, width="stretch", key="lc_chart")

# ── How It Works ────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="section-title">How It Works</div>
    <div class="steps">
        <div class="step">
            <div class="step-num">1</div>
            <h4>Target Selection</h4>
            <p>Pick from the curated Kepler catalog or paste a custom KIC ID to begin.</p>
        </div>
        <div class="step">
            <div class="step-num">2</div>
            <h4>Feature Pipeline</h4>
            <p>Flux is normalized, smoothed, and summarized into model-ready features.</p>
        </div>
        <div class="step">
            <div class="step-num">3</div>
            <h4>Model Verdict</h4>
            <p>Predict a planet probability and preview the light curve transit segment.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="footer-bar">Powered by <strong>Lightkurve</strong> · '
    "Model trained in main.ipynb · Kepler Space Telescope data</div>",
    unsafe_allow_html=True,
)
