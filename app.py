import math
import re
import numpy as np
import pandas as pd
import streamlit as st
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
E_COPPER_PSI = 16_000_000
NU_COPPER = 0.34
PI_DENSITY_G_PER_CM3_DEFAULT = 1.42

IN3_TO_CM3 = 16.387064
G_PER_LB = 453.59237
LB_PER_G = 1.0 / G_PER_LB
MIL_PER_IN = 1000.0
IN_PER_MIL = 1.0 / MIL_PER_IN
IN_PER_FT = 12.0
PI = math.pi

YIELD_STRESS_ANNEALED_MIN = 10_000
YIELD_STRESS_ANNEALED_MAX = 20_000
YIELD_STRESS_HARD_MIN = 30_000
YIELD_STRESS_HARD_MAX = 45_000

COPPER_ALPHA_PER_C = 16.5e-6
PI_ALPHA_PER_C = 20e-6

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
        special = {"2/0": 0.3648, "3/0": 0.4096, "4/0": 0.4600}
        if awg in special:
            return special[awg]
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
# NEW: WIRE DIAMETER PREDICTOR
#   Goal: predict final diameter from starting diameter and process stretch
#   Model: conservation of volume + empirical plastic strain
#   plastic_strain = alpha * passes * dwell_s * s_factor(T)
#   s_factor(T) = max(0, (T - T0) / Tscale) with defaults T0=600 F, Tscale=400 F
#   Final diameter d_f = d_0 / sqrt(1 + plastic_strain)
#   Optional thermal component can be toggled but is off by default
# ================================
def wire_diameter_predictor_page():
    st.title("Wire Diameter Predictor")

    # global calibration params in session
    if "alpha" not in st.session_state:
        st.session_state["alpha"] = 4.0e-5   # default chosen to be close to Jason’s example
    if "T0" not in st.session_state:
        st.session_state["T0"] = 600.0
    if "Tscale" not in st.session_state:
        st.session_state["Tscale"] = 400.0

    c1, c2, c3 = st.columns(3)
    with c1:
        d0 = decimal_input("Starting Diameter or ID (in)", 0.0422, "wd_d0", min_value=0.001, max_value=0.200)
    with c2:
        passes = int(st.number_input("Number of Passes", min_value=1, max_value=200, value=30, step=1))
    with c3:
        length_ft = decimal_input("Reference Length for Reporting (ft)", 100.0, "wd_len", min_value=1.0, max_value=100000.0)

    a1, a2, a3 = st.columns(3)
    with a1:
        speed = decimal_input("Line Speed (FPM)", 18.0, "wd_speed", min_value=1.0, max_value=100.0)
    with a2:
        annealer_ht = decimal_input("Annealer Height (ft)", 14.0, "wd_ht", min_value=1.0, max_value=50.0)
    with a3:
        anneal_temp = decimal_input("Anneal Temperature (°F)", 800.0, "wd_temp", min_value=400.0, max_value=1200.0)

    with st.expander("Model Settings"):
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            alpha = decimal_input("Alpha stretch factor", st.session_state["alpha"], "wd_alpha", min_value=1e-7, max_value=1e-3,
                                  help="Empirical plastic stretch factor. Calibrate below.")
        with b2:
            T0 = decimal_input("Activation baseline T0 (°F)", st.session_state["T0"], "wd_T0", min_value=300.0, max_value=900.0)
        with b3:
            Tscale = decimal_input("Activation scale (°F)", st.session_state["Tscale"], "wd_Tscale", min_value=100.0, max_value=800.0)
        with b4:
            include_thermal = st.checkbox("Include thermal stretch component", value=False,
                                          help="Adds reversible thermal strain during hot running. Usually off for final OD after cool-down.")

    calc = st.button("Predict Final Diameter", key="wd_predict")

    if calc:
        if None in (d0, speed, annealer_ht, anneal_temp, alpha, T0, Tscale, length_ft):
            st.error("Please complete all inputs.")
            return

        dwell_s = (annealer_ht / speed) * 60.0
        s_factor = max(0.0, (anneal_temp - T0) / Tscale)
        plastic_strain = alpha * passes * dwell_s * s_factor

        thermal_pct = 0.0
        if include_thermal:
            thermal_pct = thermal_strain(anneal_temp - 77.0, "copper")

        total_strain = plastic_strain + thermal_pct  # thermal is reversible, so treat carefully
        # For final diameter prediction after cool-down, use plastic component only
        length_ratio_plastic = 1.0 + plastic_strain
        dfinal = d0 / math.sqrt(max(1e-12, length_ratio_plastic))

        # Reporting
        a, b, c, d = st.columns(4)
        a.metric("Dwell per Pass", f"{dwell_s:.1f} s")
        b.metric("Plastic Elongation", f"{plastic_strain*100:.2f} %")
        c.metric("Activation Factor f(T)", f"{s_factor:.3f}")
        d.metric("Predicted Final Diameter", f"{dfinal:.5f} in")

        a, b, c = st.columns(3)
        a.metric("Area Reduction", f"{(1.0 - (dfinal/d0)**2)*100:.2f} %")
        b.metric("Length Increase from Plastic", f"{length_ratio_plastic*100:.2f} % of start")
        c.metric("Optional Thermal Strain", f"{thermal_pct*100:.3f} %")

        # Save tuned params back
        st.session_state["alpha"] = alpha
        st.session_state["T0"] = T0
        st.session_state["Tscale"] = Tscale

    st.divider()
    st.subheader("Quick Calibrate Alpha")
    st.caption("Use a known run to solve alpha so the model matches your observed start to final diameter under given conditions.")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        cal_d0 = decimal_input("Known Start Diameter (in)", 0.0422, "cal_d0", min_value=0.001, max_value=0.200)
    with k2:
        cal_df = decimal_input("Known Final Diameter (in)", 0.0410, "cal_df", min_value=0.001, max_value=0.200)
    with k3:
        cal_speed = decimal_input("Speed FPM", 18.0, "cal_speed", min_value=1.0, max_value=100.0)
    with k4:
        cal_ht = decimal_input("Annealer Height ft", 14.0, "cal_ht", min_value=1.0, max_value=50.0)

    k5, k6 = st.columns(2)
    with k5:
        cal_temp = decimal_input("Anneal Temp °F", 800.0, "cal_temp", min_value=400.0, max_value=1200.0)
    with k6:
        cal_passes = int(st.number_input("Passes", min_value=1, max_value=200, value=30, step=1, key="cal_passes"))

    if st.button("Solve Alpha From This Run", key="solve_alpha"):
        if None in (cal_d0, cal_df, cal_speed, cal_ht, cal_temp):
            st.error("Please complete all calibration inputs.")
        else:
            dwell_s = (cal_ht / cal_speed) * 60.0
            # From volume conservation: Lf/L0 = (d0/df)^2
            length_ratio = (cal_d0 / cal_df) ** 2
            target_plastic = max(0.0, length_ratio - 1.0)
            s_factor = max(0.0, (cal_temp - st.session_state["T0"]) / st.session_state["Tscale"])
            denom = max(1e-12, cal_passes * dwell_s * s_factor)
            alpha_solved = target_plastic / denom
            st.session_state["alpha"] = alpha_solved
            st.success(f"Solved alpha = {alpha_solved:.6e}. Model updated.")
            st.caption("Re run prediction above with your new alpha.")

# ================================
# PAGES FROM PRIOR BUILD
# ================================
def runtime_calculator_page():
    st.title("Production Runtime Calculator")
    with st.form("runtime_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            length_ft = decimal_input("Wire Length (ft)", 12000.0, "rt_len", min_value=0.0)
        with c2:
            speed_fpm = decimal_input("Line Speed (FPM)", 18.0, "rt_speed", min_value=1.0)
        with c3:
            efficiency_pct = decimal_input("Process Efficiency (%)", 85.0, "rt_eff", min_value=50.0, max_value=100.0)
        c4, c5, c6 = st.columns(3)
        with c4:
            startup_min = decimal_input("Startup Time (min)", 30.0, "rt_start", min_value=0.0)
        with c5:
            shutdown_min = decimal_input("Shutdown Time (min)", 15.0, "rt_stop", min_value=0.0)
        with c6:
            passes = int(st.number_input("Number of Passes", min_value=1, value=1, step=1))
        go_btn = st.form_submit_button("Calculate Runtime")
    if go_btn:
        if None in (length_ft, speed_fpm, efficiency_pct, startup_min, shutdown_min):
            st.error("Please complete all inputs.")
            return
        prod_min = (length_ft * passes) / speed_fpm
        eff_prod_min = prod_min / (efficiency_pct / 100.0)
        total_min = eff_prod_min + startup_min + shutdown_min
        total_hr = total_min / 60.0
        eff_speed = speed_fpm * (efficiency_pct / 100.0)
        hr_out = eff_speed * 60.0
        day_out = hr_out * 8.0
        a,b,c,d = st.columns(4)
        a.metric("Production Time", f"{prod_min:.1f} min")
        b.metric("Setup Shutdown", f"{startup_min + shutdown_min:.0f} min")
        c.metric("Total Runtime", f"{total_hr:.2f} hours")
        d.metric("Total minutes", f"{total_min:.1f} min")
        a,b,c = st.columns(3)
        a.metric("Effective Speed", f"{eff_speed:.1f} FPM")
        b.metric("Hourly Output", f"{hr_out:,.0f} ft hr")
        c.metric("Daily Output", f"{day_out:,.0f} ft day")
    st.markdown("Typical line speeds  \nFine <0.010 in 15 to 25 FPM  0.010 to 0.050 in 12 to 20 FPM  Larger than 0.050 in 8 to 15 FPM.")

def copper_wire_converter_page():
    st.title("Copper Wire Weight Calculator")
    method = st.radio("Input Method", ["Diameter", "AWG"], horizontal=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        if method == "AWG":
            awg = st.selectbox("AWG Size", options=list(range(50, -1, -1)) + ["2/0","3/0","4/0"], index=20)
            diameter_in = awg_to_diameter_inches(awg)
            st.info(f"Diameter {diameter_in:.4f} in")
        else:
            diameter_in = decimal_input("Wire Diameter (in)", 0.0100, "cw_d", min_value=0.0001, max_value=1.0)
            if diameter_in:
                eq = diameter_inches_to_awg(diameter_in)
                if eq is not None:
                    st.info(f"Closest AWG {eq}")
    with c2:
        length_ft = decimal_input("Wire Length (ft)", 1000.0, "cw_len", min_value=0.0)
    with c3:
        rho = decimal_input("Copper Density (lb in³)", COPPER_DENSITY_LB_PER_IN3, "cw_rho", min_value=0.300, max_value=0.350)
    if st.button("Calculate", key="cw_calc"):
        if None in (diameter_in, length_ft, rho):
            st.error("Please complete all inputs.")
            return
        area = circle_area(diameter_in)
        vol = wire_volume_in3(diameter_in, length_ft)
        wt = vol * rho
        wpf = wt / length_ft if length_ft > 0 else 0
        a,b,c = st.columns(3)
        a.metric("Wire Weight", f"{wt:.3f} lbs")
        b.metric("Weight per Foot", f"{wpf:.5f} lb ft")
        c.metric("Cross Section", f"{area:.6f} in²")

def coated_copper_converter_page():
    st.title("Coated Wire Weight Calculator")
    with st.form("coat_form"):
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            d_cu = decimal_input("Copper Diameter (in)", 0.0100, "coat_d", min_value=0.001, max_value=0.200)
        with c2:
            t_mil = decimal_input("Coating per Side (mil)", 1.0, "coat_t", min_value=0.1, max_value=10.0)
        with c3:
            L = decimal_input("Wire Length (ft)", 1000.0, "coat_len", min_value=0.0)
        with c4:
            layers = int(st.number_input("Number of Layers", min_value=1, max_value=20, value=1, step=1))
        with st.expander("Advanced"):
            a,b,c = st.columns(3)
            with a:
                rho_pi = decimal_input("Polyimide Density (g cm³)", PI_DENSITY_G_PER_CM3_DEFAULT, "coat_rho", min_value=1.0, max_value=2.0)
            with b:
                eff = decimal_input("Layer Efficiency (%)", 95.0, "coat_eff", min_value=80.0, max_value=100.0)
            with c:
                scrap = decimal_input("Scrap Rate (%)", 5.0, "coat_scrap", min_value=0.0, max_value=20.0)
        go = st.form_submit_button("Calculate Coated Wire Properties")
    if go:
        if None in (d_cu, t_mil, L, rho_pi, eff, scrap):
            st.error("Please complete all inputs.")
            return
        build_in = (t_mil * layers * eff / 100.0) / MIL_PER_IN
        d_final = d_cu + 2*build_in
        v_cu = wire_volume_in3(d_cu, L)
        v_total = wire_volume_in3(d_final, L)
        v_pi = v_total - v_cu
        wt_cu = v_cu * COPPER_DENSITY_LB_PER_IN3
        wt_pi_g = (v_pi * IN3_TO_CM3) * rho_pi
        wt_pi = wt_pi_g * LB_PER_G
        wt_total = wt_cu + wt_pi
        wt_scrap = wt_total * (1 + scrap/100.0)
        a,b,c,d = st.columns(4)
        a.metric("Starting OD", f"{d_cu:.4f} in")
        b.metric("Coating Build", f"{build_in*MIL_PER_IN:.1f} mil side")
        c.metric("Final OD", f"{d_final:.4f} in")
        d.metric("OD Increase", f"{(d_final/d_cu - 1)*100:.1f}%")
        a,b,c,d = st.columns(4)
        a.metric("Copper Weight", f"{wt_cu:.3f} lbs")
        b.metric("Coating Weight", f"{wt_pi:.3f} lbs")
        c.metric("Total Weight", f"{wt_total:.3f} lbs")
        d.metric("With Scrap", f"{wt_scrap:.3f} lbs")

# ================================
# ANNEAL TEMP ESTIMATOR
# ================================
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
        speed = decimal_input("Line Speed (FPM)", 18.0, key="an_s", min_value=_
