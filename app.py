import math
import re
import base64
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import requests
from sklearn.neighbors import NearestNeighbors
from sklearn.isotonic import IsotonicRegression

st.set_page_config(
    page_title="Zeus Polyimide Process Suite",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================================
# CONSTANTS
# ================================
COPPER_DENSITY_LB_PER_IN3 = 0.323
PI_DENSITY_G_PER_CM3_DEFAULT = 1.42

IN3_TO_CM3 = 16.387064
G_PER_LB = 453.59237
LB_PER_G = 1.0 / G_PER_LB
IN_PER_FT = 12.0
PI = math.pi

COPPER_ALPHA_PER_C = 16.5e-6
PI_ALPHA_PER_C = 20e-6

BASELINE_ACTIVATION_F = 650.0  # anneal activation floor

# Tension model defaults (you can expose in an "advanced" expander if you want)
SIGMA_REF_PSI = 1500.0   # stress where tension factor ~= 1
TENSION_P = 1.0          # exponent for mechanical scaling
THERM_Q = 0.5            # exponent for thermal scaling (creep-like)
EPS_THERM_FLOOR = 0.05   # small thermal floor at zero tension
SPOOL_MULT_LARGE = 1.0
SPOOL_MULT_22 = 1.6      # 22" tends to deliver higher effective line tension

# ================================
# DECIMAL INPUT
# ================================
DECIMAL_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*$")

def _parse_decimal(s: str):
    if s is None:
        return None
    s = str(s)
    if s == "":
        return None
    m = DECIMAL_RE.match(s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def decimal_input(label, default, key, min_value=None, max_value=None, help=None):
    if key not in st.session_state:
        st.session_state[key] = f"{default}"
    s = st.text_input(label, value=st.session_state[key], key=f"{key}_text", help=help)
    v = _parse_decimal(s)
    if v is None:
        st.caption("Enter a decimal like 0.08 or .08")
        return None
    if (min_value is not None and v < min_value) or (max_value is not None and v > max_value):
        st.warning(f"Value must be between {min_value} and {max_value}.")
    st.session_state[key] = s
    return v

# ================================
# UTILITIES
# ================================
def awg_to_diameter_inches(awg):
    if isinstance(awg, str):
        special = {"4/0": 0.4600, "3/0": 0.4096, "2/0": 0.3648, "1/0": 0.3249}
        if awg in special:
            return special[awg]
        if awg.strip() == "0":
            return 0.3249
        awg = int(awg)
    if awg == 36:
        return 0.0050
    if awg == 0:
        return 0.3249
    return 0.005 * 92 ** ((36 - awg) / 39)

def diameter_inches_to_awg(d):
    if d is None or d <= 0:
        return None
    return round(36 - 39 * math.log(d / 0.005, 92))

def circle_area(d):
    return PI * (d ** 2) / 4.0

def annulus_area(id_in, wall_in):
    od = id_in + 2.0 * wall_in
    return PI * (od ** 2 - id_in ** 2) / 4.0

def wire_volume_in3(diameter, length_ft):
    return circle_area(diameter) * length_ft * IN_PER_FT

def thermal_strain(delta_f, material="copper"):
    delta_c = delta_f * 5.0 / 9.0
    if material == "copper":
        return COPPER_ALPHA_PER_C * delta_c
    if material == "polyimide":
        return PI_ALPHA_PER_C * delta_c
    return 0.0

# ================================
# GitHub CSV helpers
# ================================
GH = st.secrets.get("gh", {})
GH_TOKEN = GH.get("token", "")
GH_OWNER = GH.get("owner", "")
GH_REPO = GH.get("repo", "")
GH_BRANCH = GH.get("branch", "main")

def _gh_api_url(path: str) -> str:
    return f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/contents/{path}"

def _gh_raw_url(path: str) -> str:
    return f"https://raw.githubusercontent.com/{GH_OWNER}/{GH_REPO}/{GH_BRANCH}/{path}"

def _gh_get_sha(path: str) -> str | None:
    if not GH_TOKEN:
        return None
    r = requests.get(_gh_api_url(path), headers={"Authorization": f"token {GH_TOKEN}"})
    if r.status_code == 200:
        return r.json().get("sha")
    return None

def _gh_upsert_csv(path: str, df: pd.DataFrame, message: str) -> bool:
    if not GH_TOKEN:
        return False
    sha = _gh_get_sha(path)
    content_bytes = df.to_csv(index=False).encode("utf-8")
    payload = {
        "message": message,
        "content": base64.b64encode(content_bytes).decode("utf-8"),
        "branch": GH_BRANCH,
    }
    if sha:
        payload["sha"] = sha
    r = requests.put(_gh_api_url(path), headers={"Authorization": f"token {GH_TOKEN}"}, data=json.dumps(payload))
    return r.status_code in (200, 201)

def save_with_retry(path: str, df: pd.DataFrame, message: str, max_retries: int = 3) -> bool:
    ok = False
    for attempt in range(max_retries):
        try:
            ok = _gh_upsert_csv(path, df, message)
            if ok:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2 ** attempt)
    return ok

def load_csv(path: str, local_fallback: Path, required_cols: list[str]) -> pd.DataFrame | None:
    try:
        if GH_OWNER and GH_REPO:
            df = pd.read_csv(_gh_raw_url(path), na_values=["", " ", "NA", "N/A", "None", "null"])
            if set(required_cols).issubset(df.columns):
                return df
        if local_fallback.exists():
            df = pd.read_csv(local_fallback, na_values=["", " ", "NA", "N/A", "None", "null"])
            if set(required_cols).issubset(df.columns):
                return df
    except Exception:
        return None
    return None

def save_row(path: str, local_fallback: Path, cols: list[str], row: dict, commit_message: str) -> bool:
    row_df = pd.DataFrame([{k: row.get(k, "") for k in cols}])
    hist = load_csv(path, local_fallback, cols)
    combined = row_df if hist is None else pd.concat([hist, row_df], ignore_index=True)

    remote_ok = False
    if GH_TOKEN:
        remote_ok = save_with_retry(path, combined, commit_message, max_retries=3)

    local_ok = False
    try:
        combined.to_csv(local_fallback, index=False)
        local_ok = True
    except Exception as e:
        st.error(f"Local save failed: {e}")

    if not remote_ok and GH_TOKEN:
        st.warning("Remote save to GitHub did not confirm. Local file was written.")

    return local_ok or remote_ok

# =========================================================
# WIRE DIAMETER PREDICTOR
# =========================================================
WIRE_FILE = "wire_diameter_runs.csv"
WIRE_LOCAL = Path("wire_diameter_runs.csv")
WIRE_COLS = [
    "date_time",
    "starting_diameter_in",
    "passes",
    "line_speed_fpm",
    "annealer_height_ft",
    "annealer_temp_F",
    "payoff_type",
    "payoff_tension_lb",
    "pred_final_od_in",
    "actual_final_od_in",
    "alpha_used",
    "notes",
]

# Session defaults
if "alpha" not in st.session_state:
    st.session_state["alpha"] = 0.10  # you will recalibrate
if "use_residual_learning" not in st.session_state:
    st.session_state["use_residual_learning"] = False

def _effective_tension(tension_lb: float, payoff_type: str) -> float:
    if payoff_type == '22" Spool':
        return tension_lb * SPOOL_MULT_22
    return tension_lb * SPOOL_MULT_LARGE

def _tension_factors(d0_in: float, tension_lb: float, payoff_type: str):
    """Return (phi_mech, phi_therm) given start diameter, indicated tension, and payoff type."""
    area_in2 = max(circle_area(max(d0_in, 1e-6)), 1e-9)
    T_eff = _effective_tension(max(tension_lb, 0.0), payoff_type)
    sigma = T_eff / area_in2  # psi

    # Mechanical scaling
    phi_mech = (sigma / SIGMA_REF_PSI) ** TENSION_P if sigma > 0 else 0.0
    # Thermal scaling with floor
    phi_therm = EPS_THERM_FLOOR + (1.0 - EPS_THERM_FLOOR) * (phi_mech ** THERM_Q)
    # Clamp to avoid blow-ups
    phi_mech = float(np.clip(phi_mech, 0.0, 3.0))
    phi_therm = float(np.clip(phi_therm, EPS_THERM_FLOOR, 3.0))
    return phi_mech, phi_therm

def _base_unscaled_stretch_in(d0_in: float, passes: int, speed_fpm: float, ht_ft: float, temp_f: float,
                              tension_lb: float, payoff_type: str) -> float:
    """Baseline stretch before alpha. Includes passes, dwell, temperature AND tension."""
    # Original shape
    mech0 = 0.000012 * passes * (1.0 + 0.08 * math.log(max(speed_fpm, 1.0)))
    act = max(0.0, temp_f - BASELINE_ACTIVATION_F)
    therm0 = (0.00000035 * passes * ht_ft * act) / max(speed_fpm, 1.0)

    # Tension scaling
    phi_mech, phi_therm = _tension_factors(d0_in, tension_lb, payoff_type)

    mech = mech0 * phi_mech
    therm = therm0 * phi_therm

    return mech + therm

def _predict_final_od(d0_in: float, passes: int, speed_fpm: float, ht_ft: float, temp_f: float,
                      tension_lb: float, payoff_type: str, alpha: float) -> float:
    raw_stretch = _base_unscaled_stretch_in(d0_in, passes, speed_fpm, ht_ft, temp_f, tension_lb, payoff_type)
    dfinal = max(0.0, d0_in - alpha * raw_stretch)
    return dfinal

def _reverse_required_start(target_final_in: float, passes: int, speed_fpm: float, ht_ft: float, temp_f: float,
                            tension_lb: float, payoff_type: str, alpha: float,
                            max_iter: int = 8, tol: float = 1e-6) -> float:
    """Solve d0 such that predicted_final(d0)=target_final. Needs iteration because stress depends on d0."""
    d0 = max(target_final_in + 0.001, 1e-5)
    for _ in range(max_iter):
        base = _base_unscaled_stretch_in(d0, passes, speed_fpm, ht_ft, temp_f, tension_lb, payoff_type)
        new_d0 = target_final_in + alpha * base
        if abs(new_d0 - d0) < tol:
            d0 = new_d0
            break
        d0 = new_d0
    return max(d0, target_final_in)  # monotone

def wire_diameter_predictor_page():
    st.title("Wire Diameter Predictor")

    left, right = st.columns([1.25, 1.0])

    with left:
        st.markdown("**Inputs**")
        d0 = decimal_input("Starting Diameter (in)", 0.04210, "wd_d0", min_value=0.001, max_value=0.200)
        passes = int(st.number_input("Number of Passes", min_value=1, max_value=200, value=30, step=1, key="wd_passes"))
        speed = decimal_input("Line Speed (FPM)", 18.0, "wd_speed", min_value=1.0, max_value=100.0)
        annealer_ht = decimal_input("Annealer Height (ft)", 14.0, "wd_ht", min_value=1.0, max_value=50.0)
        anneal_temp = decimal_input("Anneal Temperature (°F)", 825.0, "wd_temp", min_value=400.0, max_value=1200.0)

        payoff_type = st.selectbox("Payoff Type", ['Large', '22" Spool'], index=0, key="wd_payoff")
        max_tension = 4.0 if payoff_type == 'Large' else 15.0
        payoff_tension = decimal_input("Payoff Tension (lbf)", 2.0 if payoff_type=='Large' else 10.0,
                                       "wd_tension", min_value=0.0, max_value=max_tension,
                                       help="Large: 0–4 lbf. 22\" Spool: 0–15 lbf.")

        st.markdown("**Model**")
        alpha = decimal_input("Alpha stretch factor", st.session_state["alpha"], "wd_alpha",
                              min_value=1e-6, max_value=2.0,
                              help="Scale on total stretch. Use Quick Calibrate to set it from a known run.")

        with st.expander("Tension model (advanced)"):
            st.caption("These tune tension sensitivity. Defaults are usually fine.")
            st.write(f"σ_ref={SIGMA_REF_PSI:.0f} psi, p={TENSION_P}, q={THERM_Q}, ε={EPS_THERM_FLOOR}, spool_mult(22\")={SPOOL_MULT_22}")

        cols = st.columns(3)
        do_predict = cols[0].button("Predict Final Diameter", use_container_width=True)
        do_save = cols[1].button("Save Run", type="secondary", use_container_width=True)
        do_toggle_resid = cols[2].toggle("Residual learning", key="use_residual_learning", help="Use history to correct bias when enough similar runs exist.")

        st.markdown("**Reverse Calculator**")
        target_final = decimal_input("Desired Final Diameter (in)", 0.04100, "wd_target", min_value=0.001, max_value=0.200,
                                     help="Compute required starting diameter to hit this final under the inputs above.")
        do_reverse = st.button("Compute Required Start Diameter", use_container_width=True, key="wd_reverse")

    with right:
        st.markdown("**Quick Calibrate Alpha (from a real run)**")
        cal_d0 = decimal_input("Known Start OD (in)", 0.04210, "cal_d0", min_value=0.001, max_value=0.200)
        cal_df = decimal_input("Known Final OD (in)", 0.04100, "cal_df", min_value=0.001, max_value=0.200)
        cal_speed = decimal_input("Speed FPM", 18.0, "cal_speed", min_value=1.0, max_value=100.0)
        cal_ht = decimal_input("Annealer Height ft", 14.0, "cal_ht", min_value=1.0, max_value=50.0)
        cal_temp = decimal_input("Anneal Temp °F", 825.0, "cal_temp", min_value=400.0, max_value=1200.0)
        cal_passes = int(st.number_input("Passes", min_value=1, max_value=200, value=30, step=1, key="cal_passes"))
        cal_payoff_type = st.selectbox("Payoff Type", ['Large', '22" Spool'], index=0, key="cal_payoff")
        cal_max_tension = 4.0 if cal_payoff_type == 'Large' else 15.0
        cal_tension = decimal_input("Payoff Tension (lbf)",
                                    2.0 if cal_payoff_type=='Large' else 10.0,
                                    "cal_tension", min_value=0.0, max_value=cal_max_tension)

        do_solve = st.button("Solve Alpha From This Run", use_container_width=True)

        st.markdown("**Recent history**")
        hist_df = load_csv(WIRE_FILE, WIRE_LOCAL, WIRE_COLS)
        if hist_df is not None and len(hist_df) > 0:
            show_cols = [
                "date_time",
                "starting_diameter_in",
                "passes",
                "line_speed_fpm",
                "annealer_height_ft",
                "annealer_temp_F",
                "payoff_type",
                "payoff_tension_lb",
                "pred_final_od_in",
                "actual_final_od_in",
                "alpha_used",
            ]
            st.dataframe(hist_df.tail(15)[show_cols], use_container_width=True, height=360)
        else:
            st.info("No history yet (wire_diameter_runs.csv)")

    # Quick calibrate
    if do_solve:
        if None in (cal_d0, cal_df, cal_speed, cal_ht, cal_temp, cal_tension):
            st.error("Please complete all calibration inputs.")
        else:
            base_stretch = _base_unscaled_stretch_in(cal_d0, cal_passes, cal_speed, cal_ht, cal_temp,
                                                     cal_tension, cal_payoff_type)
            if base_stretch <= 0:
                st.error("Cannot solve alpha because base stretch is zero. Check inputs.")
            else:
                solved_alpha = (cal_d0 - cal_df) / base_stretch
                st.session_state["alpha"] = solved_alpha
                st.session_state["wd_alpha"] = str(solved_alpha)
                st.success(f"Solved alpha = {solved_alpha:.6f}. The Alpha field has been updated.")

    # Predict forward
    if do_predict:
        if None in (d0, speed, annealer_ht, anneal_temp, alpha, payoff_tension):
            st.error("Please complete all inputs.")
        else:
            try:
                raw_pred = _predict_final_od(d0, passes, speed, annealer_ht, anneal_temp, payoff_tension, payoff_type, alpha)
                # Optional residual correction from history
                correction, n_neigh = 0.0, 0
                if st.session_state.get("use_residual_learning", False):
                    correction, n_neigh = residual_correction_from_history(
                        d0, passes, speed, annealer_ht, anneal_temp, payoff_tension, payoff_type, alpha
                    )
                    raw_pred = float(np.clip(raw_pred + correction, 0.0, 1.0))

                st.subheader("Prediction")
                st.metric("Predicted Final Diameter", f"{raw_pred:.5f} in")
                base = _base_unscaled_stretch_in(d0, passes, speed, annealer_ht, anneal_temp, payoff_tension, payoff_type)
                a, b, c = st.columns(3)
                a.metric("Base Unscaled Stretch", f"{base:.5f} in")
                b.metric("Alpha Used", f"{alpha:.4f}")
                c.metric("Predicted Stretch", f"{(d0 - raw_pred):.5f} in")
                if st.session_state.get("use_residual_learning", False):
                    st.caption(f"Residual correction {correction:+.5f} in applied from {n_neigh} similar runs.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # Reverse calculator
    if do_reverse:
        if None in (target_final, speed, annealer_ht, anneal_temp, alpha, payoff_tension):
            st.error("Please complete inputs above and desired final diameter.")
        else:
            try:
                required_start = _reverse_required_start(target_final, passes, speed, annealer_ht, anneal_temp,
                                                         payoff_tension, payoff_type, alpha)
                st.subheader("Reverse Calculator")
                r1, r2 = st.columns(2)
                r1.metric("Required Start Diameter", f"{required_start:.5f} in")
                r2.metric("Delta Start minus Final", f"{(required_start - target_final):.5f} in")
            except Exception as e:
                st.error(f"Reverse calculation failed: {e}")

    # Save
    if do_save:
        if None in (d0, speed, annealer_ht, anneal_temp, alpha, payoff_tension):
            st.error("Please complete all inputs before saving.")
        else:
            try:
                pred_val = _predict_final_od(d0, passes, speed, annealer_ht, anneal_temp, payoff_tension, payoff_type, alpha)
                # Optional input to capture actual final now (or leave blank)
                actual_optional = st.session_state.get("wd_actual_text", "")
                row = {
                    "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "starting_diameter_in": float(d0),
                    "passes": int(passes),
                    "line_speed_fpm": float(speed),
                    "annealer_height_ft": float(annealer_ht),
                    "annealer_temp_F": float(anneal_temp),
                    "payoff_type": payoff_type,
                    "payoff_tension_lb": float(payoff_tension),
                    "pred_final_od_in": float(pred_val),
                    "actual_final_od_in": actual_optional,
                    "alpha_used": float(alpha),
                    "notes": "",
                }
                ok = save_row(WIRE_FILE, WIRE_LOCAL, WIRE_COLS, row, "append wire diameter run")
                if ok:
                    st.success("Saved run to wire_diameter_runs.csv")
                else:
                    st.error("Save failed")
            except Exception as e:
                st.error(f"Save failed: {e}")

# ---------- Residual “learning” layer (optional) ----------
def _features_for_residual(d0, passes, speed_fpm, ht_ft, temp_f, tension_lb, payoff_type):
    """Return feature vector for kNN residuals."""
    dwell_s = (ht_ft / max(speed_fpm, 1e-6)) * 60.0
    T_eff = _effective_tension(tension_lb, payoff_type)
    return np.array([
        math.log(max(d0, 1e-6)),
        float(passes),
        math.log(max(dwell_s, 1e-6)),
        float(temp_f),
        math.log(max(T_eff, 1e-6)),
        1.0 if payoff_type == '22" Spool' else 0.0
    ], dtype=float)

def residual_correction_from_history(d0, passes, speed_fpm, ht_ft, temp_f, tension_lb, payoff_type, alpha,
                                     k: int = 15, gate_p75: float = 1.5):
    """kNN residual (actual - base_pred) from previously saved runs."""
    hist = load_csv(WIRE_FILE, WIRE_LOCAL, WIRE_COLS)
    if hist is None or len(hist) < 12:
        return 0.0, 0
    # Filter rows with actuals and needed fields
    need = ["starting_diameter_in", "passes", "line_speed_fpm", "annealer_height_ft", "annealer_temp_F",
            "payoff_type", "payoff_tension_lb", "actual_final_od_in"]
    for c in need:
        if c not in hist.columns:
            return 0.0, 0
    hist = hist.dropna(subset=need)
    if len(hist) < 12:
        return 0.0, 0

    # Build features and residuals
    X_list, res_list = [], []
    for _, r in hist.iterrows():
        try:
            d0_i = float(r["starting_diameter_in"])
            p_i = int(r["passes"])
            s_i = float(r["line_speed_fpm"])
            h_i = float(r["annealer_height_ft"])
            t_i = float(r["annealer_temp_F"])
            pt_i = str(r["payoff_type"]) if pd.notna(r["payoff_type"]) else "Large"
            T_i = float(r["payoff_tension_lb"])
            actual_i = float(r["actual_final_od_in"])
            pred_i = _predict_final_od(d0_i, p_i, s_i, h_i, t_i, T_i, pt_i, alpha)
            X_list.append(_features_for_residual(d0_i, p_i, s_i, h_i, t_i, T_i, pt_i))
            res_list.append(actual_i - pred_i)
        except Exception:
            continue

    if len(X_list) < 12:
        return 0.0, 0

    X = np.vstack(X_list)
    y_res = np.array(res_list, dtype=float)

    # Standardize
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd

    q = _features_for_residual(d0, passes, speed_fpm, ht_ft, temp_f, tension_lb, payoff_type)
    qz = (q - mu) / sd

    k_eff = min(k, len(Z))
    nn = NearestNeighbors(n_neighbors=k_eff, metric='euclidean').fit(Z)
    D, I = nn.kneighbors(qz.reshape(1, -1))
    D = D.flatten(); I = I.flatten()

    # Gate: if neighbors are far in z-space, return no correction
    if len(D) == 0:
        return 0.0, 0
    if np.percentile(D, 75) > gate_p75:
        return 0.0, len(D)

    # Gaussian weights
    bw = np.median(D) if np.median(D) > 0 else (np.mean(D) + 1e-8)
    w = np.exp(-0.5 * (D / (bw + 1e-8)) ** 2)
    w = w / (w.sum() + 1e-12)
    correction = float(np.sum(w * y_res[I]))
    return correction, len(D)

# =========================================================
# RUNTIME CALCULATOR PAGE (unchanged except earlier fix)
# =========================================================
def runtime_calculator_page():
    st.title("Production Runtime Calculator")
    with st.form("runtime_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            length_ft = decimal_input("Finished Feet (ft)", 12000.0, "rt_len", min_value=0.0)
        with c2:
            speed_fpm = decimal_input("Line Speed (FPM)", 18.0, "rt_speed", min_value=1.0)
        with c3:
            efficiency_pct = decimal_input("Process Efficiency (%)", 85.0, "rt_eff", min_value=50.0, max_value=100.0)
        c4, c5 = st.columns(2)
        with c4:
            startup_min = decimal_input("Startup Time (min)", 30.0, "rt_start", min_value=0.0)
        with c5:
            shutdown_min = decimal_input("Shutdown Time (min)", 15.0, "rt_stop", min_value=0.0)
        go_btn = st.form_submit_button("Calculate Runtime")
    if go_btn:
        if None in (length_ft, speed_fpm, efficiency_pct, startup_min, shutdown_min):
            st.error("Please complete all inputs.")
            return
        prod_min = length_ft / speed_fpm
        eff_prod_min = prod_min / (efficiency_pct / 100.0)
        total_min = eff_prod_min + startup_min + shutdown_min
        total_hr = total_min / 60.0
        eff_speed = speed_fpm * (efficiency_pct / 100.0)
        hr_out = eff_speed * 60.0
        day_out = hr_out * 8.0
        a, b, c, d = st.columns(4)
        a.metric("Production Time", f"{prod_min:.1f} min")
        b.metric("Setup Shutdown", f"{startup_min + shutdown_min:.0f} min")
        c.metric("Total Runtime", f"{total_hr:.2f} hours")
        d.metric("Total minutes", f"{total_min:.1f} min")
        a, b, c = st.columns(3)
        a.metric("Effective Speed", f"{eff_speed:.1f} FPM")
        b.metric("Hourly Output", f"{hr_out:,.0f} ft hr")
        c.metric("Daily Output", f"{day_out:,.0f} ft day")
    st.markdown("Typical line speeds  \nFine <0.010 in 15 to 25 FPM  0.010 to 0.050 in 12 to 20 FPM  Larger than 0.050 in 8 to 15 FPM.")

# =========================================================
# COPPER WIRE CONVERTER PAGE (unchanged)
# =========================================================
def copper_wire_converter_page():
    st.title("Copper Wire Length and Weight")
    mode = st.radio("Choose converter", ["Feet to Pounds", "Pounds to Feet"], horizontal=True, index=0)

    c1, c2 = st.columns(2)
    with c1:
        d_in = decimal_input("Wire Diameter (in)", 0.0500, "cw_d", min_value=0.0001, max_value=1.0)
    with c2:
        if d_in:
            area_in2 = circle_area(d_in)
            st.info(f"Cross Section Area {area_in2:.6f} in²")
        else:
            area_in2 = None

    if mode == "Feet to Pounds":
        feet = decimal_input("Length (ft)", 12000.0, "cw_len_ft", min_value=0.0)
        if st.button("Calculate", key="cw_calc_ft_to_lb"):
            if None in (d_in, feet):
                st.error("Please complete all inputs.")
                return
            length_in = feet * IN_PER_FT
            volume_in3 = area_in2 * length_in
            pounds = volume_in3 * COPPER_DENSITY_LB_PER_IN3
            st.subheader("Results")
            st.metric("Estimated Weight", f"{pounds:,.2f} lb")
            st.caption(f"Linear Density {(pounds/feet) if feet>0 else 0:.5f} lb per ft")
    else:
        pounds = decimal_input("Weight (lb)", 54.0, "cw_lb", min_value=0.0)
        if st.button("Calculate", key="cw_calc_lb_to_ft"):
            if None in (d_in, pounds):
                st.error("Please complete all inputs.")
                return
            if area_in2 is None or area_in2 <= 0:
                st.error("Diameter must be greater than zero.")
                return
            length_in = pounds / (COPPER_DENSITY_LB_PER_IN3 * area_in2)
            feet = length_in / IN_PER_FT
            st.subheader("Results")
            st.metric("Estimated Length", f"{feet:,.0f} ft")
            st.caption(f"Linear Density {(pounds/feet) if feet>0 else 0:.5f} lb per ft")

# =========================================================
# COATED COPPER CONVERTER PAGE (unchanged)
# =========================================================
def coated_copper_converter_page():
    st.title("Coated Copper Length and Weight")
    mode = st.radio("Choose converter", ["Feet to Pounds", "Pounds to Feet"], horizontal=True, index=0)

    c1, c2, c3 = st.columns(3)
    with c1:
        id_in = decimal_input("Bare Copper Diameter ID (in)", 0.0500, "cc_id", min_value=0.0001, max_value=1.0)
    with c2:
        wall_in = decimal_input("Coating Wall (in)", 0.0015, "cc_wall", min_value=0.0, max_value=0.1000)
    with c3:
        coat_density = decimal_input("Coating Density (lb in³)", 0.0513, "cc_rho", min_value=0.0100, max_value=0.0800)

    if None in (id_in, wall_in, coat_density):
        st.info("Enter all inputs to calculate.")
        return

    area_cu_in2 = circle_area(id_in)
    area_coat_in2 = annulus_area(id_in, wall_in)
    lin_den_lb_per_ft = IN_PER_FT * (area_cu_in2 * COPPER_DENSITY_LB_PER_IN3 + area_coat_in2 * coat_density)

    if mode == "Feet to Pounds":
        feet = decimal_input("Length (ft)", 1500.0, "cc_len_ft", min_value=0.0)
        if st.button("Calculate", key="cc_ft_to_lb"):
            if feet is None:
                st.error("Please enter length.")
                return
            pounds = feet * lin_den_lb_per_ft
            st.subheader("Results")
            st.metric("Linear Density", f"{lin_den_lb_per_ft:,.5f} lb per ft")
            st.metric("Estimated Weight", f"{pounds:,.3f} lb")
    else:
        gross_lb = decimal_input("Gross Spool Weight (lb)", 12.0, "cc_gross", min_value=0.0)
        tare_lb = decimal_input("Spool Tare (lb)", 0.0, "cc_tare", min_value=0.0)
        if st.button("Calculate", key="cc_lb_to_ft"):
            if None in (gross_lb, tare_lb):
                st.error("Please enter weights.")
                return
            net_lb = max(gross_lb - tare_lb, 0.0)
            feet = (net_lb / lin_den_lb_per_ft) if lin_den_lb_per_ft > 0 else 0.0
            st.subheader("Results")
            st.metric("Linear Density", f"{lin_den_lb_per_ft:,.5f} lb per ft")
            st.metric("Net Wire Weight", f"{net_lb:,.3f} lb")
            st.metric("Estimated Length", f"{feet:,.0f} ft")

# =========================================================
# PAA USAGE PAGE (unchanged from your last working version)
# =========================================================
def paa_usage_page():
    st.title("PAA Usage Calculator")

    # Always visible core inputs
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        id_in = decimal_input("ID (in)", 0.0160, "paa_id", min_value=0.0001, max_value=1.0)
    with c2:
        wall_in = decimal_input("Wall (in)", 0.0010, "paa_wall", min_value=0.0001, max_value=0.0500)
    with c3:
        length_ft = decimal_input("Finished Length (ft)", 1500.0, "paa_len", min_value=0.0)
    with c4:
        solids_frac = decimal_input("Solids Fraction", 0.15, "paa_solids", min_value=0.01, max_value=1.0)
    with c5:
        soln_density_g_cm3 = decimal_input("Solution Density (g cm³)", 1.06, "paa_soln_rho", min_value=0.80, max_value=1.50)

    # Toggle to reveal additional losses and allowance
    with st.expander("Additional losses and allowance"):
        c6, c7, c8 = st.columns(3)
        with c6:
            startup_ft = decimal_input("Startup Scrap (ft)", 150.0, "paa_startup", min_value=0.0)
        with c7:
            shutdown_ft = decimal_input("Shutdown Scrap (ft)", 50.0, "paa_shutdown", min_value=0.0)
        with c8:
            allowance_frac = decimal_input("Allowance Fraction", 0.05, "paa_allow", min_value=0.0, max_value=0.50)

        c9, c10 = st.columns(2)
        with c9:
            hold_up_cm3 = decimal_input("Hold up Volume (cm³)", 400.0, "paa_holdup", min_value=0.0)
        with c10:
            heel_cm3 = decimal_input("Heel Volume (cm³)", 120.0, "paa_heel", min_value=0.0)

    # Calculate button
    if st.button("Calculate PAA Usage", key="paa_calc"):
        # Defaults for hidden fields in case expander was not opened
        startup_ft = float(st.session_state.get("paa_startup_text", 150.0)) if 'paa_startup_text' in st.session_state else 150.0
        shutdown_ft = float(st.session_state.get("paa_shutdown_text", 50.0)) if 'paa_shutdown_text' in st.session_state else 50.0
        allowance_frac = float(st.session_state.get("paa_allow_text", 0.05)) if 'paa_allow_text' in st.session_state else 0.05
        hold_up_cm3 = float(st.session_state.get("paa_holdup_text", 400.0)) if 'paa_holdup_text' in st.session_state else 400.0
        heel_cm3 = float(st.session_state.get("paa_heel_text", 120.0)) if 'paa_heel_text' in st.session_state else 120.0

        if None in (id_in, wall_in, length_ft, solids_frac, soln_density_g_cm3):
            st.error("Please complete all required inputs.")
            return

        length_in = length_ft * IN_PER_FT
        A_wall_in2 = annulus_area(id_in, wall_in)
        V_in3 = A_wall_in2 * length_in
        V_cm3 = V_in3 * IN3_TO_CM3

        mass_PI_g = V_cm3 * PI_DENSITY_G_PER_CM3_DEFAULT
        mass_PI_lb = mass_PI_g * LB_PER_G
        solution_for_polymer_lb = mass_PI_lb / solids_frac if solids_frac > 0 else 0.0

        scrap_total_ft = startup_ft + shutdown_ft
        scrap_solution_lb = solution_for_polymer_lb * (scrap_total_ft / length_ft) if length_ft > 0 else 0.0

        hold_up_mass_lb = (hold_up_cm3 * soln_density_g_cm3) * LB_PER_G
        heel_mass_lb = (heel_cm3 * soln_density_g_cm3) * LB_PER_G

        subtotal_lb = solution_for_polymer_lb + scrap_solution_lb + hold_up_mass_lb + heel_mass_lb
        total_with_allowance_lb = subtotal_lb * (1.0 + allowance_frac)

        st.subheader("Results")
        a, b, c = st.columns(3)
        a.metric("Solution for Finished Length", f"{solution_for_polymer_lb:,.4f} lb")
        b.metric("Startup plus Shutdown Scrap", f"{scrap_solution_lb:,.4f} lb")
        c.metric("Subtotal", f"{subtotal_lb:,.4f} lb")
        a, b, c = st.columns(3)
        a.metric("Hold up Mass", f"{hold_up_mass_lb:,.3f} lb")
        b.metric("Heel Mass", f"{heel_mass_lb:,.3f} lb")
        c.metric("Total with Allowance", f"{total_with_allowance_lb:,.4f} lb")

# =========================================================
# ANNEAL TEMP ESTIMATOR PAGE (unchanged from your improved version)
# =========================================================
def anneal_temp_estimator_page():
    st.title("Advanced Annealing Temperature Estimator")
    st.markdown(
        "kNN on log space features ln diameter and minus ln dwell with z score standardization and Gaussian weighting. "
        "Backbone with non negative slopes. A local isotonic correction enforces that as speed increases dwell decreases and required temperature does not decrease."
    )

    @st.cache_data
    def load_and_clean_data(filepath):
        try:
            df = pd.read_csv(filepath)
            df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
            req = {'wire_dia', 'speed', 'annealer_ht', 'anneal_t'}
            if not req.issubset(df.columns):
                st.error(f"CSV missing required columns {req}")
                return None
            df['wire_dia_in'] = df['wire_dia']
            df['speed_fpm'] = df['speed']
            df['annealer_ht_ft'] = df['annealer_ht']
            df['anneal_temp_f'] = df['anneal_t']
            df = df[(df['wire_dia_in'] > 0) & (df['speed_fpm'] > 0) & (df['annealer_ht_ft'] > 0)]
            df['dwell_s'] = (df['annealer_ht_ft'] / df['speed_fpm']) * 60.0
            df = df[(df['anneal_temp_f'] > 500) & (df['anneal_temp_f'] < 1200)]

            y = df['anneal_temp_f'].values
            med = np.median(y)
            mad = np.median(np.abs(y - med))
            if mad == 0:
                df['is_outlier'] = False
            else:
                mz = 0.6745 * (y - med) / mad
                df['is_outlier'] = np.abs(mz) > 2.5
            return df
        except Exception as e:
            st.error(f"Error loading data {e}")
            return None

    df = load_and_clean_data("annealing_dataset_clean.csv")
    if df is None or len(df) < 8:
        st.error("Not enough valid data to build a model.")
        return

    with st.expander("Data Quality Overview"):
        a, b, c, d = st.columns(4)
        a.metric("Total Records", len(df))
        b.metric("Clean Records", len(df[~df['is_outlier']]))
        c.metric("Outliers Detected", int(df['is_outlier'].sum()))
        d.metric("Temperature Range", f"{df['anneal_temp_f'].min():.0f} to {df['anneal_temp_f'].max():.0f} °F")

    st.subheader("Process Parameters")
    a, b, c = st.columns(3)
    with a:
        wire_dia = decimal_input("Wire Diameter (in)", 0.0250, key="an_d", min_value=0.001, max_value=0.200)
    with b:
        speed = decimal_input("Line Speed (FPM)", 18.0, key="an_s", min_value=1.0, max_value=100.0)
    with c:
        height = decimal_input("Annealer Height (ft)", 14.0, key="an_h", min_value=1.0, max_value=50.0)

    st.subheader("Model Configuration")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        include_outliers = st.checkbox("Include Outliers", value=False)
    clean_df = df if include_outliers else df[~df['is_outlier']]

    with m2:
        max_k = min(60, len(clean_df))
        default_k = min(35, max_k) if max_k >= 35 else max_k
        k_neighbors = st.slider("Number of Neighbors k", min_value=3, max_value=max_k, value=default_k)
    with m3:
        weight_method = st.selectbox("Weighting Method", ["Gaussian", "Distance", "Uniform"], index=0)
    with m4:
        oor_gate = st.slider("Out of Range Gate p75 dist z space", min_value=0.5, max_value=3.0, value=1.25, step=0.05)

    if st.button("Predict Temperature", key="anneal_predict"):
        if None in (wire_dia, speed, height):
            st.error("Please fill all inputs.")
            return

        dwell_s = (height / speed) * 60.0
        ln_d = np.log(clean_df['wire_dia_in'].values)
        ln_dw = np.log(clean_df['dwell_s'].values)
        X = np.column_stack([ln_d, -ln_dw]).astype(float)
        y = clean_df['anneal_temp_f'].values.astype(float)

        mu = X.mean(axis=0)
        sd = X.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        Z = (X - mu) / sd

        q_raw = np.array([math.log(wire_dia), -math.log(dwell_s)], dtype=float)
        qz = (q_raw - mu) / sd

        nn = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
        nn.fit(Z)
        D, I = nn.kneighbors(qz.reshape(1, -1))
        D = D.flatten()
        I = I.flatten()
        neigh_temps = y[I]

        eps = 1e-8
        if weight_method == "Distance":
            w = 1.0 / (D + eps)
        elif weight_method == "Gaussian":
            bw = np.median(D) if np.median(D) > 0 else (np.mean(D) + eps)
            w = np.exp(-0.5 * (D / (bw + eps)) ** 2)
        else:
            w = np.ones_like(D)
        w = w / (w.sum() + eps)
        T_knn = float(np.sum(w * neigh_temps))

        lam = 10.0
        Xg = np.column_stack([np.ones(len(X)), X])
        G = Xg.T @ Xg + lam * np.eye(Xg.shape[1])
        beta = np.linalg.solve(G, Xg.T @ y)
        b0, b_d, b_dwneg = beta.tolist()
        b_d = max(0.0, b_d)
        b_dwneg = max(0.0, b_dwneg)
        mean_y = float(y.mean())
        mean_X = X.mean(axis=0)
        b0 = mean_y - b_d * mean_X[0] - b_dwneg * mean_X[1]
        T_ridge = float(b0 + b_d * q_raw[0] + b_dwneg * q_raw[1])

        pr75 = float(np.percentile(D, 75))
        out_of_range = pr75 > oor_gate

        ln_d_all = np.log(clean_df['wire_dia_in'].values)
        ln_dw_all = np.log(clean_df['dwell_s'].values)
        temps_all = clean_df['anneal_temp_f'].values

        def isotonic_predict(ln_d_q, ln_dw_q, band=0.06, min_n=12):
            cur_band = band
            for _ in range(5):
                mask = np.abs(ln_d_all - ln_d_q) <= cur_band
                x = -ln_dw_all[mask]
                yv = temps_all[mask]
                if x.size >= min_n and np.unique(x).size >= 3:
                    order = np.argsort(x)
                    ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
                    ir.fit(x[order], yv[order])
                    return float(ir.predict([-ln_dw_q])[0])
                cur_band *= 1.5
            return None

        T_iso = isotonic_predict(q_raw[0], -q_raw[1])
        if T_iso is None:
            x_local_raw = X[I, 1]
            y_local = neigh_temps
            if np.unique(x_local_raw).size >= 3:
                order = np.argsort(x_local_raw)
                ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
                ir.fit(x_local_raw[order], y_local[order])
                T_iso = float(ir.predict([q_raw[1]])[0])

        if T_iso is not None:
            T_local = 0.7 * T_iso + 0.3 * T_knn
        else:
            T_local = T_knn

        w_local = 0.90 if not out_of_range else 0.70
        T_mix = w_local * T_local + (1.0 - w_local) * T_ridge
        T_final = float(np.clip(T_mix, 650.0, 1200.0))

        st.success("Prediction complete.")
        a, b, c, d = st.columns(4)
        a.metric("Predicted Temperature", f"{T_final:.1f} °F")
        b.metric("kNN Core", f"{T_knn:.1f} °F")
        c.metric("Backbone NNLS", f"{T_ridge:.1f} °F")
        d.metric("Dwell Time", f"{dwell_s:.1f} s")
        if T_iso is not None:
            st.caption("Monotonic guard active.")
        st.caption(
            f"Distance stats z space median {np.median(D):.2f}  p75 {pr75:.2f}  out of range {out_of_range}  k {k_neighbors}  weighting {weight_method}"
        )

        try:
            weighted_mean = T_knn
            weighted_var = float(np.sum(w * (neigh_temps - weighted_mean) ** 2))
            weighted_sd = float(np.sqrt(max(weighted_var, 0.0)))
            q25, q75 = np.percentile(neigh_temps, [25, 75])
            u1, u2, u3 = st.columns(3)
            u1.metric("Weighted SD", f"±{weighted_sd:.1f} °F")
            u2.metric("Neighbor IQR", f"{q25:.0f} to {q75:.0f} °F")
            u3.metric("Neighbor Count", f"{len(neigh_temps)}")
        except Exception:
            st.caption("Uncertainty metrics unavailable for this query.")

        with st.expander("Closest Historical Runs"):
            neighbor_display = clean_df.iloc[I][['wire_dia_in', 'speed_fpm', 'annealer_ht_ft', 'dwell_s', 'anneal_temp_f']].copy()
            neighbor_display['Distance'] = D
            neighbor_display['Weight'] = w
            neighbor_display = neighbor_display.rename(columns={
                'wire_dia_in': 'Dia in', 'speed_fpm': 'Speed FPM', 'annealer_ht_ft': 'Height ft',
                'dwell_s': 'Dwell s', 'anneal_temp_f': 'Temp °F'
            })
            st.dataframe(
                neighbor_display.style.format({
                    'Dia in': '{:.4f}', 'Speed FPM': '{:.1f}', 'Height ft': '{:.0f}',
                    'Dwell s': '{:.1f}', 'Temp °F': '{:.0f}', 'Distance': '{:.3f}', 'Weight': '{:.3f}'
                })
            )

# =========================================================
# MAIN APP
# =========================================================
def main():
    st.sidebar.title("Zeus Polyimide Process Suite")
    page = st.sidebar.radio(
        "Choose a module",
        [
            "Wire Diameter Predictor",
            "Runtime Calculator",
            "Copper Wire Converter",
            "Coated Copper Converter",
            "PAA Usage",
            "Anneal Temp Estimator",
        ],
        index=0,
    )

    if page == "Wire Diameter Predictor":
        wire_diameter_predictor_page()
    elif page == "Runtime Calculator":
        runtime_calculator_page()
    elif page == "Copper Wire Converter":
        copper_wire_converter_page()
    elif page == "Coated Copper Converter":
        coated_copper_converter_page()
    elif page == "PAA Usage":
        paa_usage_page()
    elif page == "Anneal Temp Estimator":
        anneal_temp_estimator_page()

    st.sidebar.markdown("---")
    st.sidebar.info("""
Zeus Polyimide Process Suite

• Wire Diameter Predictor now includes Payoff Type and Payoff Tension, keeps passes, and uses a stress-based scale.
• Reverse calculator solves iteratively since tension depends on start OD.
• Optional residual “learning” correction leverages saved runs (off by default).
• GitHub saves retry and always write a local fallback to protect history.
• Other modules unchanged.
    """)

if __name__ == "__main__":
    main()
