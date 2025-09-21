# app.py
# Zeus Polyimide Process Suite
# Wire Diameter Predictor with GitHub hosted CSV read and write
# Five required inputs, Predict works, Quick Calibrate Alpha updates automatically
# History loads from GitHub CSV, Save Run appends back to GitHub CSV via API
# If GitHub secrets are missing, falls back to local CSV next to app.py

import base64
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.isotonic import IsotonicRegression

st.set_page_config(
    page_title="Zeus Polyimide Process Suite",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================
# Config via Secrets
# =========================================
# Set these in Streamlit Cloud: Settings -> Secrets
# [gh]
# token = "ghp_xxx"
# owner = "TheJason11"
# repo = "polyimide-calculators"
# branch = "main"
# path = "wire_diameter_runs.csv"
def _get_secret(path, default=None):
    try:
        return st.secrets
    except Exception:
        return {}

GH = st.secrets.get("gh", {})
GH_TOKEN = GH.get("token", "")
GH_OWNER = GH.get("owner", "")
GH_REPO = GH.get("repo", "")
GH_BRANCH = GH.get("branch", "main")
GH_PATH = GH.get("path", "wire_diameter_runs.csv")

RAW_URL = ""
API_URL = ""
if GH_OWNER and GH_REPO and GH_PATH:
    RAW_URL = f"https://raw.githubusercontent.com/{GH_OWNER}/{GH_REPO}/{GH_BRANCH}/{GH_PATH}"
    API_URL = f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/contents/{GH_PATH}"

# Local fallback filename if no secrets are set or API write fails
LOCAL_CSV = Path("wire_diameter_runs.csv")

# =========================================
# Session State Defaults
# =========================================
def _ensure_state():
    if "alpha" not in st.session_state:
        st.session_state.alpha = 1.0
    if "use_isotonic" not in st.session_state:
        st.session_state.use_isotonic = True

_ensure_state()

# =========================================
# Modeling Core
# =========================================
BASELINE_ACTIVATION_F = 650.0

def base_unscaled_stretch_in(start_od_in: float,
                             passes: int,
                             speed_fpm: float,
                             anneal_height_ft: float,
                             anneal_temp_f: float) -> float:
    mech = 0.00002 * passes * (1.0 + 0.1 * math.log(max(speed_fpm, 1.0)))
    act = max(0.0, anneal_temp_f - BASELINE_ACTIVATION_F)
    therm = (0.0000006 * passes * anneal_height_ft * act) / max(speed_fpm, 1.0)
    return mech + therm

def predict_final_od(start_od_in: float,
                     passes: int,
                     speed_fpm: float,
                     anneal_height_ft: float,
                     anneal_temp_f: float,
                     alpha: float) -> float:
    stretch_unscaled = base_unscaled_stretch_in(start_od_in, passes, speed_fpm, anneal_height_ft, anneal_temp_f)
    stretch_scaled = alpha * stretch_unscaled
    final_od = max(0.0, start_od_in - stretch_scaled)
    return final_od

def isotonic_correct(raw_pred: np.ndarray, actual: np.ndarray, new_raw_pred: float) -> float:
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(raw_pred, actual)
    return float(ir.predict([new_raw_pred])[0])

# =========================================
# History IO
# =========================================
HISTORY_COLUMNS = [
    "date_time",
    "starting_diameter_in",
    "passes",
    "line_speed_fpm",
    "annealer_height_ft",
    "annealer_temp_F",
    "pred_final_od_in",
    "actual_final_od_in",
    "alpha_used",
    "notes",
]

def load_history_from_github() -> pd.DataFrame | None:
    if not RAW_URL:
        return None
    try:
        df = pd.read_csv(RAW_URL, na_values=["", " ", "NA", "N/A", "None", "null"])
        missing = set(HISTORY_COLUMNS) - set(df.columns)
        if missing:
            return None
        return df
    except Exception:
        return None

def load_history_local() -> pd.DataFrame | None:
    if not LOCAL_CSV.exists():
        return None
    try:
        df = pd.read_csv(LOCAL_CSV, na_values=["", " ", "NA", "N/A", "None", "null"])
        missing = set(HISTORY_COLUMNS) - set(df.columns)
        if missing:
            return None
        return df
    except Exception:
        return None

def github_get_file_sha() -> str | None:
    if not API_URL or not GH_TOKEN:
        return None
    headers = {"Authorization": f"token {GH_TOKEN}"}
    r = requests.get(API_URL, headers=headers)
    if r.status_code == 200:
        return r.json().get("sha")
    return None

def github_upsert_csv(new_df: pd.DataFrame, commit_message: str) -> bool:
    if not API_URL or not GH_TOKEN:
        return False
    headers = {"Authorization": f"token {GH_TOKEN}"}
    existing_sha = github_get_file_sha()
    content_bytes = new_df.to_csv(index=False).encode("utf-8")
    payload = {
        "message": commit_message,
        "content": base64.b64encode(content_bytes).decode("utf-8"),
        "branch": GH_BRANCH,
    }
    if existing_sha:
        payload["sha"] = existing_sha
    r = requests.put(API_URL, headers=headers, data=json.dumps(payload))
    return r.status_code in (200, 201)

def save_run_row(row: dict, commit_message: str = "append wire diameter run") -> bool:
    # Try GitHub first if configured
    hist = load_history_from_github()
    if hist is None:
        # Try local fallback existence
        if LOCAL_CSV.exists():
            hist = load_history_local()
    # Build the new row DataFrame with correct column order
    row_df = pd.DataFrame([{k: row.get(k, "") for k in HISTORY_COLUMNS}])
    if hist is None:
        combined = row_df
    else:
        combined = pd.concat([hist, row_df], ignore_index=True)
    # Attempt GitHub write if token present
    if GH_TOKEN and API_URL:
        ok = github_upsert_csv(combined, commit_message)
        if ok:
            return True
    # Fallback to local file write
    try:
        if not LOCAL_CSV.exists():
            combined.to_csv(LOCAL_CSV, index=False)
        else:
            combined.to_csv(LOCAL_CSV, index=False)
        return True
    except Exception:
        return False

# =========================================
# Layout
# =========================================
st.title("Zeus Polyimide Process Suite")

tabs = st.tabs([
    "Wire Diameter Predictor",
    "Other Modules",
])

with tabs[0]:
    st.subheader("Wire Diameter Predictor")

    src_col, iso_col = st.columns([1.6, 1.0])
    with src_col:
        st.write("Data source")
        if RAW_URL:
            st.caption(f"GitHub CSV: {RAW_URL}")
        else:
            st.caption("Local CSV fallback: wire_diameter_runs.csv")

    with iso_col:
        st.session_state.use_isotonic = st.checkbox(
            "Use isotonic correction when history has actuals",
            value=st.session_state.use_isotonic,
        )

    left, right = st.columns([1.2, 1.0])

    with left:
        st.markdown("**Inputs**")
        start_od_in = st.number_input("Starting diameter in", min_value=0.00000, value=0.04210, step=0.00010, format="%.5f")
        passes = st.number_input("Number of passes", min_value=1, value=30, step=1)
        speed_fpm = st.number_input("Line speed fpm", min_value=1.0, value=18.0, step=1.0, format="%.2f")
        anneal_height_ft = st.number_input("Annealer height ft", min_value=1.0, value=14.0, step=1.0, format="%.2f")
        anneal_temp_f = st.number_input("Annealer temperature F", min_value=200.0, value=800.0, step=5.0, format="%.1f")

        st.markdown("**Model**")
        alpha = st.number_input(
            "Alpha stretch factor",
            min_value=0.000,
            value=float(st.session_state.alpha),
            step=0.010,
            format="%.3f",
            key="alpha_input",
        )

        col_pred, col_save = st.columns([1, 1])
        with col_pred:
            predict_click = st.button("Predict Final Diameter", use_container_width=True)
        with col_save:
            save_click = st.button("Save Run", type="secondary", use_container_width=True)

    with right:
        st.markdown("**Quick calibrate alpha from an actual final OD**")
        actual_for_cal = st.number_input(
            "Actual final OD in",
            min_value=0.00000,
            value=0.04100,
            step=0.00010,
            format="%.5f",
            help="Enter a known final OD from a real run to solve the alpha",
        )
        solve_alpha_click = st.button("Solve Alpha from Actual", use_container_width=True)

        st.markdown("**Notes for save**")
        user_notes = st.text_area("Notes", value="", height=100)

        st.markdown("**Recent history**")
        hist_df = load_history_from_github() or load_history_local()
        if hist_df is not None and len(hist_df) > 0:
            show_cols = [
                "date_time",
                "starting_diameter_in",
                "passes",
                "line_speed_fpm",
                "annealer_height_ft",
                "annealer_temp_F",
                "pred_final_od_in",
                "actual_final_od_in",
                "alpha_used",
            ]
            st.dataframe(hist_df.tail(15)[show_cols], use_container_width=True, height=360)
        else:
            st.info("No history yet")

    # Actions
    if solve_alpha_click:
        try:
            stretch_unscaled = base_unscaled_stretch_in(
                float(start_od_in), int(passes), float(speed_fpm), float(anneal_height_ft), float(anneal_temp_f)
            )
            if stretch_unscaled <= 0:
                st.error("Cannot solve alpha because base stretch is zero. Check inputs.")
            else:
                solved_alpha = (float(start_od_in) - float(actual_for_cal)) / float(stretch_unscaled)
                st.session_state.alpha = float(solved_alpha)
                st.session_state.alpha_input = float(solved_alpha)
                st.success(f"Solved alpha set to {solved_alpha:.5f}")
        except Exception as e:
            st.error(f"Alpha solve failed: {e}")

    if predict_click:
        try:
            alpha_val = float(st.session_state.get("alpha_input", st.session_state.alpha))
            raw_pred = predict_final_od(
                start_od_in=float(start_od_in),
                passes=int(passes),
                speed_fpm=float(speed_fpm),
                anneal_height_ft=float(anneal_height_ft),
                anneal_temp_f=float(anneal_temp_f),
                alpha=alpha_val,
            )

            corrected = raw_pred
            if st.session_state.use_isotonic and hist_df is not None:
                valid = hist_df.dropna(subset=["pred_final_od_in", "actual_final_od_in"])
                if len(valid) >= 5:
                    try:
                        corrected = isotonic_correct(
                            raw_pred=valid["pred_final_od_in"].to_numpy(dtype=float),
                            actual=valid["actual_final_od_in"].to_numpy(dtype=float),
                            new_raw_pred=float(raw_pred),
                        )
                    except Exception:
                        pass

            st.subheader("Prediction")
            c1, c2 = st.columns(2)
            with c1:
                st.metric(label="Raw predicted final OD in", value=f"{raw_pred:.5f}")
            with c2:
                st.metric(label="Corrected final OD in", value=f"{corrected:.5f}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    if save_click:
        try:
            alpha_val = float(st.session_state.get("alpha_input", st.session_state.alpha))
            pred_val = predict_final_od(
                start_od_in=float(start_od_in),
                passes=int(passes),
                speed_fpm=float(speed_fpm),
                anneal_height_ft=float(anneal_height_ft),
                anneal_temp_f=float(anneal_temp_f),
                alpha=alpha_val,
            )

            row = {
                "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "starting_diameter_in": float(start_od_in),
                "passes": int(passes),
                "line_speed_fpm": float(speed_fpm),
                "annealer_height_ft": float(anneal_height_ft),
                "annealer_temp_F": float(anneal_temp_f),
                "pred_final_od_in": float(pred_val),
                "actual_final_od_in": float(actual_for_cal) if actual_for_cal > 0 else "",
                "alpha_used": float(alpha_val),
                "notes": user_notes,
            }
            ok = save_run_row(row, commit_message="append wire diameter run")
            if ok:
                st.success("Saved run to GitHub CSV" if GH_TOKEN else "Saved run to local CSV")
            else:
                st.error("Save failed")
        except Exception as e:
            st.error(f"Save failed: {e}")

with tabs[1]:
    st.subheader("Other Modules")
    st.info("Placeholders for other tools.")
