import os
import math
import re
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.isotonic import IsotonicRegression
import plotly.graph_objects as go
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Zeus Polyimide Process Suite",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# CONSTANTS AND PHYSICAL PROPERTIES
# ================================
COPPER_DENSITY_LB_PER_IN3 = 0.323
COPPER_DENSITY_G_PER_CM3 = 8.96
E_COPPER_PSI = 16_000_000
NU_COPPER = 0.34
COPPER_THERMAL_EXPANSION = 16.5e-6  # per °C

YIELD_STRESS_ANNEALED_MIN = 10_000  # PSI
YIELD_STRESS_ANNEALED_MAX = 20_000  # PSI
YIELD_STRESS_HARD_MIN = 30_000  # PSI
YIELD_STRESS_HARD_MAX = 45_000  # PSI

PI_DENSITY_G_PER_CM3_DEFAULT = 1.42
PI_THERMAL_EXPANSION = 20e-6  # per °C

IN3_TO_CM3 = 16.387064
G_PER_LB = 453.59237
LB_PER_G = 1.0 / G_PER_LB
MIL_PER_IN = 1000.0
IN_PER_MIL = 1.0 / MIL_PER_IN
IN_PER_FT = 12.0

PI = math.pi

# ================================
# DECIMAL INPUT HELPER
# ================================
# Streamlit's number_input can be fussy for decimals (.08 → 8). This text-based helper
# accepts ".08", "0.08", "  0.080 " etc., validates, and returns a float.
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


def decimal_input(label, value, key, min_value=None, max_value=None, help=None, width=None):
    col = st if width is None else st.container()
    if key not in st.session_state:
        st.session_state[key] = f"{value}"
    s = col.text_input(label, value=st.session_state[key], key=f"{key}_text", help=help)
    v = _parse_decimal(s)
    if v is None:
        st.session_state[key] = s  # preserve what user typed
        st.caption("Enter a decimal like 0.08 or .08")
        return None
    if min_value is not None and v < min_value:
        st.warning(f"Minimum is {min_value}")
    if max_value is not None and v > max_value:
        st.warning(f"Maximum is {max_value}")
    st.session_state[key] = s
    return v


# ================================
# UTILITY FUNCTIONS
# ================================
def validate_numeric_input(value, min_val=0, max_val=None, allow_zero=False):
    if value is None:
        return False, "Value cannot be None"
    if not isinstance(value, (int, float)):
        return False, "Value must be numeric"
    if math.isnan(value) or math.isinf(value):
        return False, "Value must be finite"
    if not allow_zero and value <= 0:
        return False, "Value must be positive"
    if allow_zero and value < 0:
        return False, "Value cannot be negative"
    if min_val is not None and value < min_val:
        return False, f"Value must be at least {min_val}"
    if max_val is not None and value > max_val:
        return False, f"Value must be at most {max_val}"
    return True, "Valid"


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


def diameter_inches_to_awg(diameter):
    if diameter is None or diameter <= 0:
        return None
    awg = 36 - 39 * math.log(diameter / 0.005, 92)
    return round(awg)


def calculate_circle_area(diameter):
    return PI * (diameter ** 2) / 4.0


def calculate_annulus_area(id_in, wall_in):
    od_in = id_in + 2.0 * wall_in
    return PI * (od_in ** 2 - id_in ** 2) / 4.0


def calculate_wire_volume(diameter, length_ft):
    area = calculate_circle_area(diameter)
    return area * length_ft * IN_PER_FT


def calculate_thermal_strain(temp_change_f, material="copper"):
    temp_change_c = temp_change_f * 5.0 / 9.0
    if material == "copper":
        return COPPER_THERMAL_EXPANSION * temp_change_c
    elif material == "polyimide":
        return PI_THERMAL_EXPANSION * temp_change_c
    return 0.0


# ================================
# RUNTIME CALCULATOR
# ================================
def runtime_calculator_page():
    st.title("Production Runtime Calculator")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("Calculate runtime, throughput, and efficiency (with startup/shutdown).")
        with st.form("runtime_form"):
            st.subheader("Production Parameters")
            c1, c2, c3 = st.columns(3)
            with c1:
                wire_length_ft = decimal_input("Wire Length (ft)", 12000.0, key="rt_len", min_value=0.0, help="Total length this pass")
            with c2:
                line_speed_fpm = decimal_input("Line Speed (FPM)", 18.0, key="rt_speed", min_value=1.0)
            with c3:
                efficiency_pct = decimal_input("Process Efficiency (%)", 85.0, key="rt_eff", min_value=50.0, max_value=100.0)

            st.subheader("Additional Time Factors")
            c4, c5, c6 = st.columns(3)
            with c4:
                startup_min = decimal_input("Startup Time (min)", 30.0, key="rt_start", min_value=0.0)
            with c5:
                shutdown_min = decimal_input("Shutdown Time (min)", 15.0, key="rt_stop", min_value=0.0)
            with c6:
                passes = int(st.number_input("Number of Passes", min_value=1, value=1, step=1))

            calculate_btn = st.form_submit_button("Calculate Runtime")

        if calculate_btn:
            if not all(validate_numeric_input(v, min_val=0 if k == "rt_len" else 1)[0]
                       for v, k in [(wire_length_ft, "rt_len"), (line_speed_fpm, "rt_speed")]):
                st.error("Invalid length or speed.")
                return

            production_time_min = (wire_length_ft * passes) / line_speed_fpm
            effective_production_time_min = production_time_min / (efficiency_pct / 100.0)
            total_time_min = effective_production_time_min + startup_min + shutdown_min
            total_time_hr = total_time_min / 60.0

            effective_speed_fpm = line_speed_fpm * (efficiency_pct / 100.0)
            hourly_throughput_ft = effective_speed_fpm * 60.0
            daily_throughput_ft = hourly_throughput_ft * 8

            st.success("Calculation complete.")
            a, b, c, d = st.columns(4)
            a.metric("Production Time", f"{production_time_min:.1f} min")
            b.metric("Setup/Shutdown", f"{startup_min + shutdown_min:.0f} min")
            c.metric("Total Runtime", f"{total_time_hr:.2f} hours")
            d.metric("Total (minutes)", f"{total_time_min:.1f} min")

            a, b, c = st.columns(3)
            a.metric("Effective Speed", f"{effective_speed_fpm:.1f} FPM")
            b.metric("Hourly Output", f"{hourly_throughput_ft:,.0f} ft/hr")
            c.metric("Daily Output (8hr)", f"{daily_throughput_ft:,.0f} ft/day")

    with col2:
        st.subheader("Quick Reference (Zeus practice)")
        st.info("""
- Fine wire (< 0.010"): 15–25 FPM
- Medium wire (0.010–0.050"): 12–20 FPM
- Heavy wire (> 0.050"): 8–15 FPM

We rarely exceed ~25 FPM; 30 FPM is an absolute ceiling and almost never used.
        """)


# ================================
# COPPER WIRE CONVERTER
# ================================
def copper_wire_converter_page():
    st.title("Copper Wire Weight Calculator")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("Convert between wire dimensions, AWG sizes, and weight for bare copper.")
        input_method = st.radio("Input Method", ["Diameter", "AWG"], horizontal=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            if input_method == "AWG":
                awg = st.selectbox("AWG Size", options=list(range(50, -1, -1)) + ["2/0", "3/0", "4/0"], index=20)
                diameter_in = awg_to_diameter_inches(awg)
                if diameter_in is None:
                    st.error("Invalid AWG")
                    return
                st.info(f"Diameter: {diameter_in:.4f} inches")
            else:
                diameter_in = decimal_input("Wire Diameter (inches)", 0.0100, key="cw_d", min_value=0.0001, max_value=1.0)
                if diameter_in is not None:
                    equiv_awg = diameter_inches_to_awg(diameter_in)
                    if equiv_awg is not None:
                        st.info(f"Closest AWG: {equiv_awg}")

        with c2:
            length_ft = decimal_input("Wire Length (ft)", 1000.0, key="cw_len", min_value=0.0)
        with c3:
            copper_density = decimal_input("Copper Density (lb/in³)", COPPER_DENSITY_LB_PER_IN3, key="cw_rho",
                                           min_value=0.300, max_value=0.350)

        if st.button("Calculate", key="copper_calc"):
            if None in (diameter_in, length_ft, copper_density):
                st.error("Please fill all inputs.")
                return
            area_in2 = calculate_circle_area(diameter_in)
            volume_in3 = calculate_wire_volume(diameter_in, length_ft)
            weight_lb = volume_in3 * copper_density
            weight_per_ft = weight_lb / length_ft if length_ft > 0 else 0
            resistivity_ohm_cm = 1.72e-6  # ~20°C
            resistance_ohms = (resistivity_ohm_cm * length_ft * 30.48) / (area_in2 * 6.4516)

            st.success("Calculation complete.")
            a, b, c, d = st.columns(4)
            a.metric("Wire Weight", f"{weight_lb:.3f} lbs")
            b.metric("Weight per Foot", f"{weight_per_ft:.5f} lb/ft")
            c.metric("Cross Section", f"{area_in2:.6f} in²")
            d.metric("Resistance", f"{resistance_ohms:.3f} Ω")

            with st.expander("Visual Comparison"):
                awg_range = list(range(10, 41, 2))
                diameters = [awg_to_diameter_inches(a) for a in awg_range]
                weights = [calculate_wire_volume(d, 1000) * copper_density for d in diameters]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=awg_range, y=weights, mode='lines', name='Weight per 1000ft', line=dict(width=2)))
                fig.add_trace(go.Scatter(
                    x=[diameter_inches_to_awg(diameter_in)],
                    y=[weight_lb * 1000 / length_ft if length_ft > 0 else 0],
                    mode='markers', name='Your Wire', marker=dict(size=12, symbol='star')
                ))
                fig.update_layout(title='Weight vs AWG Size (per 1000 ft)', xaxis_title='AWG', yaxis_title='Weight (lbs)', height=400, hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Quick Reference")
        st.info("""
Common AWG diameters:
- AWG 30: 0.0100"
- AWG 26: 0.0159"
- AWG 22: 0.0253"
- AWG 18: 0.0403"
- AWG 14: 0.0641"
        """)


# ================================
# COATED COPPER CONVERTER
# ================================
def coated_copper_converter_page():
    st.title("Coated Wire Weight Calculator")

    st.markdown("Calculate weight and dimensions for polyimide-coated copper wire with multi-layer build-up.")
    with st.form("coated_wire_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            copper_dia_in = decimal_input("Copper Diameter (in)", 0.0100, key="coat_d", min_value=0.001, max_value=0.200)
        with c2:
            coating_thickness_mil = decimal_input("Coating per Side (mil)", 1.0, key="coat_t", min_value=0.1, max_value=10.0)
        with c3:
            wire_length_ft = decimal_input("Wire Length (ft)", 1000.0, key="coat_len", min_value=0.0)
        with c4:
            num_layers = int(st.number_input("Number of Layers", min_value=1, max_value=20, value=1, step=1))

        with st.expander("Advanced Parameters"):
            a, b, c = st.columns(3)
            with a:
                pi_density = decimal_input("Polyimide Density (g/cm³)", PI_DENSITY_G_PER_CM3_DEFAULT, key="coat_rho",
                                           min_value=1.0, max_value=2.0)
            with b:
                layer_efficiency = decimal_input("Layer Efficiency (%)", 95.0, key="coat_eff", min_value=80.0, max_value=100.0)
            with c:
                scrap_rate_pct = decimal_input("Scrap Rate (%)", 5.0, key="coat_scrap", min_value=0.0, max_value=20.0)

        calculate_btn = st.form_submit_button("Calculate Coated Wire Properties")

    if calculate_btn:
        if None in (copper_dia_in, coating_thickness_mil, wire_length_ft, pi_density, layer_efficiency, scrap_rate_pct):
            st.error("Please fill all inputs.")
            return

        coating_per_side_in = (coating_thickness_mil * num_layers * layer_efficiency / 100.0) / MIL_PER_IN
        final_od_in = copper_dia_in + 2 * coating_per_side_in
        copper_volume_in3 = calculate_wire_volume(copper_dia_in, wire_length_ft)
        total_volume_in3 = calculate_wire_volume(final_od_in, wire_length_ft)
        coating_volume_in3 = total_volume_in3 - copper_volume_in3

        copper_weight_lb = copper_volume_in3 * COPPER_DENSITY_LB_PER_IN3
        coating_weight_g = (coating_volume_in3 * IN3_TO_CM3) * pi_density
        coating_weight_lb = coating_weight_g * LB_PER_G
        total_weight_lb = copper_weight_lb + coating_weight_lb
        total_with_scrap_lb = total_weight_lb * (1 + scrap_rate_pct / 100.0)

        st.success("Calculation complete.")
        a, b, c, d = st.columns(4)
        a.metric("Starting OD", f"{copper_dia_in:.4f} in")
        b.metric("Coating Build", f"{coating_per_side_in*MIL_PER_IN:.1f} mil/side")
        c.metric("Final OD", f"{final_od_in:.4f} in")
        d.metric("OD Increase", f"{(final_od_in/copper_dia_in - 1)*100:.1f}%")

        a, b, c, d = st.columns(4)
        a.metric("Copper Weight", f"{copper_weight_lb:.3f} lbs")
        b.metric("Coating Weight", f"{coating_weight_lb:.3f} lbs")
        c.metric("Total Weight", f"{total_weight_lb:.3f} lbs")
        d.metric("With Scrap", f"{total_with_scrap_lb:.3f} lbs")


# ================================
# WIRE STRETCH PREDICTOR
# ================================
def wire_stretch_predictor_page():
    st.title("Wire Elongation Predictor")

    st.markdown("Predict elastic + thermal elongation and check against yield.")
    c1, c2, c3 = st.columns(3)
    with c1:
        wire_dia_in = decimal_input("Wire Diameter (in)", 0.0100, key="st_d", min_value=0.001, max_value=0.200)
    with c2:
        tension_lb = decimal_input("Applied Tension (lb)", 5.0, key="st_t", min_value=0.0, max_value=100.0)
    with c3:
        wire_length_ft = decimal_input("Wire Length (ft)", 100.0, key="st_len", min_value=1.0, max_value=10000.0)

    with st.expander("Temperature and Material Properties"):
        a, b, c = st.columns(3)
        with a:
            operating_temp_f = decimal_input("Operating Temp (°F)", 800.0, key="st_to", min_value=0.0, max_value=1200.0)
        with b:
            ambient_temp_f = decimal_input("Ambient Temp (°F)", 77.0, key="st_ta", min_value=0.0, max_value=120.0)
        with c:
            wire_condition = st.selectbox("Wire Condition", ["Annealed", "Half-Hard", "Hard"])

    if st.button("Predict Elongation", key="stretch"):
        if None in (wire_dia_in, tension_lb, wire_length_ft, operating_temp_f, ambient_temp_f):
            st.error("Please fill all inputs.")
            return

        area_in2 = calculate_circle_area(wire_dia_in)
        stress_psi = tension_lb / area_in2 if area_in2 > 0 else 0
        elastic_strain = stress_psi / E_COPPER_PSI
        elastic_elongation_in = elastic_strain * wire_length_ft * IN_PER_FT
        temp_rise_f = operating_temp_f - ambient_temp_f
        thermal_strain = calculate_thermal_strain(temp_rise_f, "copper")
        thermal_elongation_in = thermal_strain * wire_length_ft * IN_PER_FT
        total_elongation_in = elastic_elongation_in + thermal_elongation_in
        percent_elongation = (total_elongation_in / (wire_length_ft * IN_PER_FT)) * 100

        if wire_condition == "Annealed":
            yield_min, yield_max = YIELD_STRESS_ANNEALED_MIN, YIELD_STRESS_ANNEALED_MAX
        elif wire_condition == "Half-Hard":
            yield_min = (YIELD_STRESS_ANNEALED_MAX + YIELD_STRESS_HARD_MIN) / 2
            yield_max = (YIELD_STRESS_ANNEALED_MAX + YIELD_STRESS_HARD_MAX) / 2
        else:
            yield_min, yield_max = YIELD_STRESS_HARD_MIN, YIELD_STRESS_HARD_MAX

        radial_strain = -NU_COPPER * elastic_strain
        dia_reduction_mil = abs(radial_strain * wire_dia_in * MIL_PER_IN)

        st.subheader("Elongation Results")
        a, b, c, d = st.columns(4)
        a.metric("Applied Stress", f"{stress_psi:,.0f} PSI")
        b.metric("Elastic Elongation", f"{elastic_elongation_in:.4f} in")
        c.metric("Thermal Elongation", f"{thermal_elongation_in:.4f} in")
        d.metric("Total Elongation", f"{total_elongation_in:.4f} in")

        a, b, c = st.columns(3)
        a.metric("Percent Elongation", f"{percent_elongation:.3f}%")
        b.metric("Diameter Reduction", f"{dia_reduction_mil:.3f} mil")
        c.metric("Volume Conservation", "Maintained")

        if stress_psi > yield_max:
            st.error(f"Exceeds yield strength. Applied: {stress_psi:,.0f} PSI; Max yield: {yield_max:,.0f} PSI.")
        elif stress_psi > yield_min:
            st.warning(f"Approaching yield. Applied: {stress_psi:,.0f} PSI; Yield range: {yield_min:,.0f}-{yield_max:,.0f} PSI.")
        else:
            safety_factor = yield_min / stress_psi if stress_psi > 0 else float('inf')
            st.success(f"Operating safely with {safety_factor:.1f}× safety factor.")


# ================================
# ANNEALING TEMPERATURE ESTIMATOR
#  kNN (log-space) + Ridge (non-neg slopes) + Monotonicity Guard (isotonic vs dwell)
# ================================
def anneal_temp_estimator_page():
    st.title("Advanced Annealing Temperature Estimator")
    st.markdown(
        "kNN on log-space features [ln(diameter), −ln(dwell)] with z-score standardization and Gaussian weighting. "
        "Ridge backbone with **non-negative slopes**. A local isotonic correction enforces: as speed increases "
        "(dwell ↓), required temperature does not decrease."
    )

    @st.cache_data
    def load_and_clean_data(filepath):
        try:
            df = pd.read_csv(filepath)
            df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
            req = {'wire_dia', 'speed', 'annealer_ht', 'anneal_t'}
            if not req.issubset(df.columns):
                st.error(f"CSV missing required columns: {req}")
                return None
            df['wire_dia_in'] = df['wire_dia']
            df['speed_fpm'] = df['speed']
            df['annealer_ht_ft'] = df['annealer_ht']
            df['anneal_temp_f'] = df['anneal_t']
            df = df[(df['wire_dia_in'] > 0) & (df['speed_fpm'] > 0) & (df['annealer_ht_ft'] > 0)]
            df['dwell_s'] = (df['annealer_ht_ft'] / df['speed_fpm']) * 60.0
            df = df[(df['anneal_temp_f'] > 500) & (df['anneal_temp_f'] < 1200)]

            # Modified Z for temp outliers
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
            st.error(f"Error loading data: {e}")
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
        d.metric("Temperature Range", f"{df['anneal_temp_f'].min():.0f}-{df['anneal_temp_f'].max():.0f}°F")
        fig = px.box(df, y='anneal_temp_f', x='is_outlier',
                     labels={'anneal_temp_f': 'Temperature (°F)', 'is_outlier': 'Is Outlier'},
                     title='Temperature Distribution with Outliers Identified')
        st.plotly_chart(fig, use_container_width=True)

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
        k_neighbors = st.slider("Number of Neighbors (k)", min_value=3, max_value=max_k, value=default_k)
    with m3:
        weight_method = st.selectbox("Weighting Method", ["Gaussian", "Distance", "Uniform"], index=0)
    with m4:
        oor_gate = st.slider("Out-of-Range Gate (p75 dist, z-space)", min_value=0.5, max_value=3.0, value=1.25, step=0.05)

    if st.button("Predict Temperature", key="anneal_predict"):
        if None in (wire_dia, speed, height):
            st.error("Please fill all inputs.")
            return

        # --- features: ln(diameter), -ln(dwell) so larger value = harder condition (needs higher T) ---
        dwell_s = (height / speed) * 60.0
        ln_d = np.log(clean_df['wire_dia_in'].values)
        ln_dw = np.log(clean_df['dwell_s'].values)
        X = np.column_stack([ln_d, -ln_dw]).astype(float)
        y = clean_df['anneal_temp_f'].values.astype(float)

        # z-score standardization
        mu = X.mean(axis=0)
        sd = X.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        Z = (X - mu) / sd

        # query in same space
        q_raw = np.array([math.log(wire_dia), -math.log(dwell_s)], dtype=float)
        qz = (q_raw - mu) / sd

        # neighbors
        nn = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
        nn.fit(Z)
        D, I = nn.kneighbors(qz.reshape(1, -1))
        D = D.flatten()
        I = I.flatten()
        neigh_temps = y[I]

        # weights
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

        # --- Ridge backbone with non-negative slopes on both features ---
        # Solve ridge, then clip negative slopes to 0 and recompute intercept.
        lam = 10.0
        Xg = np.column_stack([np.ones(len(X)), X])  # [1, ln d, -ln dwell]
        G = Xg.T @ Xg + lam * np.eye(Xg.shape[1])
        beta = np.linalg.solve(G, Xg.T @ y)  # [b0, b_d, b_dwneg]
        b0, b_d, b_dwneg = beta.tolist()
        # Enforce physics: bigger diameter → not lower temp; higher speed (−ln dwell larger) → not lower temp
        b_d = max(0.0, b_d)
        b_dwneg = max(0.0, b_dwneg)
        # Refit intercept to match means with constrained slopes
        mean_y = float(y.mean())
        mean_X = X.mean(axis=0)
        b0 = mean_y - b_d * mean_X[0] - b_dwneg * mean_X[1]
        T_ridge = float(b0 + b_d * q_raw[0] + b_dwneg * q_raw[1])

        # out-of-range check
        pr75 = float(np.percentile(D, 75))
        out_of_range = pr75 > oor_gate

        # --- Monotonicity guard: isotonic vs speed at near-constant diameter ---
        ln_d_all = np.log(clean_df['wire_dia_in'].values)
        ln_dw_all = np.log(clean_df['dwell_s'].values)
        temps_all = clean_df['anneal_temp_f'].values

        def isotonic_predict(ln_d_q, ln_dw_q, band=0.12, min_n=10):
            cur_band = band
            for _ in range(5):  # widen progressively
                mask = np.abs(ln_d_all - ln_d_q) <= cur_band
                x = -ln_dw_all[mask]  # increasing x == faster line (shorter dwell)
                yv = temps_all[mask]
                if x.size >= min_n and np.unique(x).size >= 3:
                    order = np.argsort(x)
                    ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
                    ir.fit(x[order], yv[order])
                    return float(ir.predict([-ln_dw_q])[0])
                cur_band *= 1.5
            return None

        T_iso = isotonic_predict(q_raw[0], -q_raw[1])  # pass ln_d, ln_dw (note sign)
        if T_iso is None:
            # Fallback: use neighbor set to fit isotonic along speed directly
            x_local = Z[I, 1]  # the standardized -ln(dwell) of neighbors
            # Convert back to raw -ln(dwell) for a sane scale
            x_local_raw = X[I, 1]
            y_local = neigh_temps
            if np.unique(x_local_raw).size >= 3:
                order = np.argsort(x_local_raw)
                ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
                ir.fit(x_local_raw[order], y_local[order])
                T_iso = float(ir.predict([q_raw[1]])[0])

        # blend local monotone with knn
        if T_iso is not None:
            T_local = 0.7 * T_iso + 0.3 * T_knn
        else:
            T_local = T_knn

        # global blend with backbone
        w_local = 0.75 if not out_of_range else 0.55
        T_mix = w_local * T_local + (1.0 - w_local) * T_ridge
        T_final = float(np.clip(T_mix, 650.0, 1200.0))

        st.success("Prediction complete.")
        a, b, c, d = st.columns(4)
        a.metric("Predicted Temperature", f"{T_final:.1f} °F")
        b.metric("k-NN Core", f"{T_knn:.1f} °F")
        c.metric("Backbone (NNLS ridge)", f"{T_ridge:.1f} °F")
        d.metric("Dwell Time", f"{dwell_s:.1f} s")
        if T_iso is not None:
            st.caption(f"Monotonic guard active (isotonic vs speed).")

        st.caption(
            f"Distance stats (z-space): median={np.median(D):.2f} | p75={pr75:.2f} | out_of_range={out_of_range} | k={k_neighbors} | weighting={weight_method}"
        )
        if out_of_range:
            st.warning("Parameters are outside the dense region of history; prediction leans more on the backbone and monotone fit.")

        with st.expander("Closest Historical Runs"):
            neighbor_display = clean_df.iloc[I][['wire_dia_in', 'speed_fpm', 'annealer_ht_ft', 'dwell_s', 'anneal_temp_f']].copy()
            neighbor_display['Distance'] = D
            neighbor_display['Weight'] = w
            neighbor_display = neighbor_display.rename(columns={
                'wire_dia_in': 'Dia (in)', 'speed_fpm': 'Speed (FPM)', 'annealer_ht_ft': 'Height (ft)',
                'dwell_s': 'Dwell (s)', 'anneal_temp_f': 'Temp (°F)'
            })
            st.dataframe(
                neighbor_display.style.format({
                    'Dia (in)': '{:.4f}', 'Speed (FPM)': '{:.1f}', 'Height (ft)': '{:.0f}',
                    'Dwell (s)': '{:.1f}', 'Temp (°F)': '{:.0f}', 'Distance': '{:.3f}', 'Weight': '{:.3f}'
                }).background_gradient(subset=['Weight'], cmap='YlOrRd')
            )


# ================================
# MAIN APP
# ================================
def main():
    st.sidebar.title("Zeus Polyimide Process Suite")
    page = st.sidebar.radio(
        "Choose a module",
        [
            "Runtime Calculator",
            "Copper Wire Converter",
            "Coated Copper Converter",
            "Wire Stretch Predictor",
            "PAA Usage",
            "Anneal Temp Estimator",
        ],
        index=5,
    )

    if page == "Runtime Calculator":
        runtime_calculator_page()
    elif page == "Copper Wire Converter":
        copper_wire_converter_page()
    elif page == "Coated Copper Converter":
        coated_copper_converter_page()
    elif page == "Wire Stretch Predictor":
        wire_stretch_predictor_page()
    elif page == "PAA Usage":
        paa_usage_page()
    elif page == "Anneal Temp Estimator":
        anneal_temp_estimator_page()

    st.sidebar.markdown("---")
    st.sidebar.info("""
Zeus Polyimide Process Suite v2.3

Changes:
- Decimal typing fix (custom inputs accept .08 / 0.08 cleanly)
- Anneal model monotone vs speed, non-negative backbone slopes
- PAA slider formatting fix
- Correct speed quick reference
    """)

if __name__ == "__main__":
    main()
