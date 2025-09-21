# app.py
# Zeus Polyimide Process Suite
# Full file replacement focused on a reliable Wire Diameter Predictor
# Five required inputs
# Predict works
# Quick Calibrate Alpha updates the model value automatically
# Save Run creates and appends to a CSV with a fixed header

import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.isotonic import IsotonicRegression

st.set_page_config(
    page_title="Zeus Polyimide Process Suite",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================
# Session State Defaults
# =========================================
def _ensure_state():
    if "alpha" not in st.session_state:
        st.session_state.alpha = 1.0
    if "default_csv_path" not in st.session_state:
        st.session_state.default_csv_path = "wire_diameter_runs.csv"
    if "use_isotonic" not in st.session_state:
        st.session_state.use_isotonic = True

_ensure_state()

# =========================================
# Modeling Core
# =========================================
BASELINE_ACTIVATION_F = 650.0

# The base unscaled stretch captures:
# 1. More passes increase stretch
# 2. Higher speed reduces thermal dwell which reduces stretch
# 3. Taller annealer and hotter temperature increase stretch
# Alpha scales the total effect and is calibrated from real runs
def base_unscaled_stretch_in(start_od_in: float,
                             passes: int,
                             speed_fpm: float,
                             anneal_height_ft: float,
                             anneal_temp_f: float) -> float:
    # Mechanical part grows with passes and weakly with speed
    mech = 0.00002 * passes * (1.0 + 0.1 * math.log(max(speed_fpm, 1.0)))

    # Thermal activation only above baseline
    act = max(0.0, anneal_temp_f - BASELINE_ACTIVATION_F)

    # Thermal part increases with passes, annealer height, and activation
    # and decreases with speed due to shorter dwell
    therm = (0.0000006 * passes * anneal_height_ft * act) / max(speed_fpm, 1.0)

    # Total base stretch before alpha scaling in inches of diameter reduction
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

# Optional isotonic bias correction
# Trains a monotonic mapping from raw predicted to actual using historical rows
def isotonic_correct(raw_pred: np.ndarray, actual: np.ndarray, new_raw_pred: float) -> float:
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(raw_pred, actual)
    return float(ir.predict([new_raw_pred])[0])

# Load historical data if present
def load_history(csv_path: Path) -> pd.DataFrame | None:
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            needed = {
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
            }
            if needed.issubset(set(df.columns)):
                return df
        except Exception:
            return None
    return None

# Save a single row to CSV with consistent header and ordering
def save_run_row(csv_path: Path, row: dict):
    columns = [
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
    df_row = pd.DataFrame([{k: row.get(k, "") for k in columns}])
    if not csv_path.exists():
        df_row.to_csv(csv_path, index=False)
    else:
        df_row.to_csv(csv_path, mode="a", header=False, index=False)

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

    # File location and options
    cfg_col, iso_col = st.columns([1.5, 1])
    with cfg_col:
        csv_path_str = st.text_input(
            "CSV path for saved runs",
            value=st.session_state.default_csv_path,
            help="File is created on first save then appended"
        )
    with iso_col:
        st.session_state.use_isotonic = st.checkbox(
            "Use isotonic correction when history has actuals",
            value=st.session_state.use_isotonic
        )

    csv_path = Path(csv_path_str)

    left, right = st.columns([1.2, 1.0])

    with left:
        st.markdown("**Inputs**")
        start_od_in = st.number_input("Starting diameter in", min_value=0.00000, value=0.04200, step=0.00010, format="%.5f")
        passes = st.number_input("Number of passes", min_value=1, value=30, step=1)
        speed_fpm = st.number_input("Line speed fpm", min_value=1.0, value=18.0, step=1.0, format="%.2f")
        anneal_height_ft = st.number_input("Annealer height ft", min_value=1.0, value=14.0, step=1.0, format="%.2f")
        anneal_temp_f = st.number_input("Annealer temperature F", min_value=200.0, value=800.0, step=5.0, format="%.1f")

        st.markdown("**Model**")
        alpha = st.number_input("Alpha stretch factor", min_value=0.000, value=float(st.session_state.alpha), step=0.010, format="%.3f", key="alpha_input")

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
            help="Enter a known final OD from a real run to solve the alpha"
        )
        solve_alpha_click = st.button("Solve Alpha from Actual", use_container_width=True)

        st.markdown("**Notes for save**")
        user_notes = st.text_area("Notes", value="", height=100)

        # History preview
        st.markdown("**Recent history**")
        hist_df = load_history(csv_path)
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
    # Solve alpha
    if solve_alpha_click:
        try:
            stretch_unscaled = base_unscaled_stretch_in(start_od_in, int(passes), float(speed_fpm), float(anneal_height_ft), float(anneal_temp_f))
            if stretch_unscaled <= 0:
                st.error("Cannot solve alpha because base stretch is zero. Check inputs")
            else:
                solved_alpha = (start_od_in - actual_for_cal) / stretch_unscaled
                # Update both session state and the visible input
                st.session_state.alpha = float(solved_alpha)
                st.session_state.alpha_input = float(solved_alpha)
                st.success(f"Solved alpha set to {solved_alpha:.5f}")
        except Exception as e:
            st.error(f"Alpha solve failed: {e}")

    # Predict
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
            # Apply isotonic only if enabled and if we have rows with both raw pred and actual
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
                if corrected != raw_pred:
                    st.metric(label="Isotonic corrected final OD in", value=f"{corrected:.5f}")
                else:
                    st.metric(label="Corrected final OD in", value=f"{corrected:.5f}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # Save
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
            save_run_row(Path(csv_path), row)
            st.success(f"Saved to {csv_path}")
        except Exception as e:
            st.error(f"Save failed: {e}")

with tabs[1]:
    st.subheader("Other Modules")
    st.info("Placeholders for other tools. This tab remains unchanged in this version.")
