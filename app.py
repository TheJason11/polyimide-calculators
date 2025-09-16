import os
import math
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Polyimide Calculators", page_icon="🧮", layout="centered")

# -----------------------------
# Constants and conversions
# -----------------------------
COPPER_DENSITY_LB_PER_IN3 = 0.323
E_COPPER_PSI = 16_000_000
NU_COPPER = 0.34
YIELD_WARN_LOW = 10_000
YIELD_WARN_HIGH = 20_000

PI_DENSITY_G_PER_CM3_DEFAULT = 1.42
IN3_TO_CM3 = 16.387064
G_PER_LB = 453.59237
PI_CONST = math.pi
R_GAS = 8.314  # J/mol-K

# -----------------------------
# Sidebar navigation
# -----------------------------
st.sidebar.title("Polyimide Calculators")
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
    index=0,
)

# -----------------------------
# Helpers
# -----------------------------
def inches_from_feet(feet: float) -> float:
    return feet * 12.0

def circle_area_in2(d_in: float) -> float:
    return PI_CONST * (d_in**2) / 4.0

def annulus_area_in2(id_in: float, wall_in: float) -> float:
    od_in = id_in + 2.0 * wall_in
    return PI_CONST * (od_in**2 - id_in**2) / 4.0

def float_input(label, default):
    val_str = st.text_input(label, str(default))
    try:
        return float(val_str)
    except ValueError:
        st.error(f"Invalid number for {label}")
        return default

# =============================
# Module: Runtime Calculator
# =============================
if page == "Runtime Calculator":
    st.title("Job Runtime Calculator")
    st.caption("Compute runtime, footage, or rate from feet, time, and speed")

    mode = st.radio("Choose calculator",
                    ["Feet and Speed → Runtime", "Time and Speed → Feet", "Feet and Time → Rate"], index=0)

    if mode == "Feet and Speed → Runtime":
        c1, c2 = st.columns(2)
        with c1:
            feet = float_input("Total feet", 12000.0)
        with c2:
            fpm = float_input("Line speed (FPM)", 18.0)
        run_minutes = feet / fpm if fpm > 0 else 0.0
        run_hours = run_minutes / 60.0
        st.subheader("Results")
        st.write(f"Runtime: **{run_minutes:,.2f} minutes**  |  **{run_hours:,.2f} hours**")
        st.write(f"Throughput: **{fpm*60:,.0f} ft per hour**")

    elif mode == "Time and Speed → Feet":
        c1, c2, c3 = st.columns(3)
        with c1:
            hours = float_input("Hours", 3.0)
        with c2:
            minutes = float_input("Minutes", 0.0)
        with c3:
            fpm = float_input("Line speed (FPM)", 18.0)
        total_minutes = hours * 60.0 + minutes
        feet = fpm * total_minutes
        ft_per_hour = fpm * 60.0
        st.subheader("Results")
        st.write(f"Total minutes: **{total_minutes:,.2f}**")
        st.write(f"Feet possible: **{feet:,.0f} ft**")
        st.write(f"Throughput: **{ft_per_hour:,.0f} ft per hour**")

    elif mode == "Feet and Time → Rate":
        c1, c2 = st.columns(2)
        with c1:
            feet_run = float_input("Feet run", 600.0)
        with c2:
            minutes_run = float_input("Minutes", 60.0)
        if minutes_run > 0:
            fpm_calc = feet_run / minutes_run
            st.subheader("Results")
            st.write(f"Calculated rate: **{fpm_calc:,.2f} FPM**")
            st.write(f"Throughput: **{fpm_calc*60:,.0f} ft per hour**")
        else:
            st.info("Enter minutes greater than zero")

# =============================
# Module: Copper Wire Converter
# =============================
elif page == "Copper Wire Converter":
    st.title("Copper Wire Length ↔ Weight")
    mode = st.radio("Choose converter", ["Feet → Pounds", "Pounds → Feet"], index=0)

    c1, c2 = st.columns(2)
    with c1:
        d_in = float_input("Wire diameter (in)", 0.0500)
    with c2:
        area_in2 = circle_area_in2(d_in)
        st.write(f"Cross section area: **{area_in2:.6f} in²**")

    if mode == "Feet → Pounds":
        feet = float_input("Length (ft)", 12000.0)
        length_in = inches_from_feet(feet)
        volume_in3 = area_in2 * length_in
        pounds = volume_in3 * COPPER_DENSITY_LB_PER_IN3
        st.subheader("Results")
        st.write(f"Estimated weight: **{pounds:,.2f} lb**")
    else:
        pounds = float_input("Weight (lb)", 54.0)
        length_in = pounds / (COPPER_DENSITY_LB_PER_IN3 * area_in2) if area_in2 > 0 else 0.0
        feet = length_in / 12.0
        st.subheader("Results")
        st.write(f"Estimated length: **{feet:,.0f} ft**")

# =============================
# Module: Coated Copper Converter
# =============================
elif page == "Coated Copper Converter":
    st.title("Coated Copper Length ↔ Weight")
    mode = st.radio("Choose converter", ["Feet → Pounds", "Pounds → Feet"], index=0)

    c1, c2, c3 = st.columns(3)
    with c1:
        id_in = float_input("Bare copper diameter, ID (in)", 0.0500)
    with c2:
        wall_in = float_input("Coating wall (in)", 0.0015)
    with c3:
        coat_density = float_input("Coating density (lb/in³)", 0.0513)

    area_cu_in2 = circle_area_in2(id_in)
    area_coat_in2 = annulus_area_in2(id_in, wall_in)
    lin_den_lb_per_ft = 12.0 * (area_cu_in2 * COPPER_DENSITY_LB_PER_IN3 + area_coat_in2 * coat_density)

    if mode == "Feet → Pounds":
        feet = float_input("Length (ft)", 1500.0)
        pounds = feet * lin_den_lb_per_ft
        st.subheader("Results")
        st.write(f"Linear density: **{lin_den_lb_per_ft:,.5f} lb/ft**")
        st.write(f"Estimated weight: **{pounds:,.3f} lb**")
    else:
        gross_lb = float_input("Gross spool weight (lb)", 12.0)
        tare_lb = float_input("Spool tare (lb)", 0.0)
        net_lb = max(gross_lb - tare_lb, 0.0)
        feet = (net_lb / lin_den_lb_per_ft) if lin_den_lb_per_ft > 0 else 0.0
        st.subheader("Results")
        st.write(f"Linear density: **{lin_den_lb_per_ft:,.5f} lb/ft**")
        st.write(f"Net wire weight: **{net_lb:,.3f} lb**")
        st.write(f"Estimated length: **{feet:,.0f} ft**")

# =============================
# Module: Wire Stretch Predictor
# =============================
elif page == "Wire Stretch Predictor":
    st.title("Wire Stretch Predictor")
    st.caption("Elastic estimate based on copper properties and applied tension")

    c1, c2, c3 = st.columns(3)
    with c1:
        d0_in = float_input("Starting diameter (in)", 0.0500)
    with c2:
        tension_lbf = float_input("Applied tension (lbf)", 15.0)
    with c3:
        passes = st.number_input("Number of passes", min_value=1, value=10, step=1, format="%d")

    c4, c5, c6 = st.columns(3)
    with c4:
        anneal_temp_f = float_input("Anneal temp (°F)", 700.0)
    with c5:
        oven_height_ft = float_input("Oven height (ft)", 12.0)
    with c6:
        line_speed_fpm = float_input("Line speed (FPM)", 18.0)

    cal = st.slider("Calibration factor", min_value=0.50, max_value=1.50, value=1.00, step=0.01)

    area_in2 = circle_area_in2(d0_in)
    sigma_psi = tension_lbf / area_in2 if area_in2 > 0 else 0.0
    elastic_strain = sigma_psi / E_COPPER_PSI if E_COPPER_PSI > 0 else 0.0
    radial_frac = NU_COPPER * elastic_strain
    d_loaded = d0_in * (1.0 - radial_frac) * cal

    st.subheader("Results")
    st.write(f"Axial stress: **{sigma_psi:,.0f} psi**")
    st.write(f"Axial strain: **{elastic_strain:.6e}**")
    st.write(f"Predicted diameter under load: **{d_loaded:.6f} in**")

    if sigma_psi > YIELD_WARN_LOW:
        st.warning(f"Stress exceeds {YIELD_WARN_LOW:,} psi. Yield onset between {YIELD_WARN_LOW:,}–{YIELD_WARN_HIGH:,} psi.")

# =============================
# Module: PAA Usage
# =============================
elif page == "PAA Usage":
    st.title("PAA Usage Calculator")

    c1, c2, c3 = st.columns(3)
    with c1:
        id_in = float_input("ID (in)", 0.0160)
    with c2:
        wall_in = float_input("Wall (in)", 0.0010)
    with c3:
        length_ft = float_input("Finished length (ft)", 1500.0)

    c4, c5, c6 = st.columns(3)
    with c4:
        solids_frac = float_input("Solids fraction", 0.15)
    with c5:
        startup_ft = float_input("Startup scrap (ft)", 150.0)
    with c6:
        shutdown_ft = float_input("Shutdown scrap (ft)", 50.0)

    c7, c8, c9 = st.columns(3)
    with c7:
        hold_up_cm3 = float_input("Hold up volume (cm³)", 400.0)
    with c8:
        heel_cm3 = float_input("Heel volume (cm³)", 120.0)
    with c9:
        soln_density_g_cm3 = float_input("Solution density (g/cm³)", 1.06)

    allowance_frac = float_input("Allowance fraction", 0.05)

    length_in = inches_from_feet(length_ft)
    A_wall_in2 = annulus_area_in2(id_in, wall_in)
    V_in3 = A_wall_in2 * length_in
    V_cm3 = V_in3 * IN3_TO_CM3
    mass_PI_g = V_cm3 * PI_DENSITY_G_PER_CM3_DEFAULT
    mass_PI_lb = mass_PI_g / G_PER_LB
    solution_for_polymer_lb = mass_PI_lb / solids_frac if solids_frac > 0 else 0.0

    scrap_total_ft = startup_ft + shutdown_ft
    scrap_solution_lb = solution_for_polymer_lb * (scrap_total_ft / length_ft) if length_ft > 0 else 0.0

    hold_up_mass_lb = (hold_up_cm3 * soln_density_g_cm3) / G_PER_LB
    heel_mass_lb = (heel_cm3 * soln_density_g_cm3) / G_PER_LB

    subtotal_lb = solution_for_polymer_lb + scrap_solution_lb + hold_up_mass_lb + heel_mass_lb
    total_with_allowance_lb = subtotal_lb * (1.0 + allowance_frac)

    st.subheader("Results")
    st.write(f"Solution for finished length: **{solution_for_polymer_lb:,.4f} lb**")
    st.write(f"Startup + shutdown scrap: **{scrap_solution_lb:,.4f} lb**")
    st.write(f"Hold up mass: **{hold_up_mass_lb:,.3f} lb**")
    st.write(f"Heel mass: **{heel_mass_lb:,.3f} lb**")
    st.write(f"Subtotal: **{subtotal_lb:,.4f} lb**")
    st.write(f"Total with allowance: **{total_with_allowance_lb:,.4f} lb**")

# =============================
# Module: Anneal Temp Estimator
# =============================
elif page == "Anneal Temp Estimator":
    st.title("Anneal Temperature Estimator (Local + Global + Physics Blend)")

    # Inputs
    c1, c2, c3 = st.columns(3)
    with c1:
        diameter_in = float_input("Wire diameter (in)", 0.0500)
    with c2:
        speed_fpm = float_input("Line speed (FPM)", 18.0000)
    with c3:
        height_ft = float_input("Annealer height (ft)", 8.0000)

    with st.expander("Advanced physics baseline"):
        x_um = float_input("Target oxide thickness (µm)", 0.2000)
        k0 = st.number_input("Parabolic rate K0 (m²/s)", value=1e-10, step=1e-11, format="%.1e")
        ea_kj = float_input("Activation energy Ea (kJ/mol)", 120.0000)
        alpha = float_input("Thermal diffusivity α (m²/s)", 1.11e-4)
        beta = float_input("Heat up multiplier β", 1.0000)
        T_ambient_F = float_input("Ambient (°F)", 75.0)

    def physics_temp_required(d_in, speed, height, x_um, k0, ea_kj, alpha, beta, T_amb_F):
        dwell_s = (height / speed) * 60.0
        d_m = d_in * 0.0254
        t_heat = beta * (d_m**2) / (alpha * (math.pi**2))
        t_eff = max(1e-3, dwell_s - t_heat)
        x_m = x_um * 1e-6
        Ea = ea_kj * 1000.0
        denom = math.log(max(1e-30, (k0 * t_eff) / (x_m**2)))
        if denom <= 0:
            T_K = 1100.0
        else:
            T_K = (Ea / R_GAS) / denom
        T_F = (T_K - 273.15) * 9.0/5.0 + 32.0
        T_F = max(T_F, T_amb_F + 50.0)
        return T_F, dwell_s

    T_phys_F, dwell_s = physics_temp_required(
        diameter_in, speed_fpm, height_ft, x_um, k0, ea_kj, alpha, beta, T_ambient_F
    )

    csv_path = "annealing_dataset_clean.csv"
    if not os.path.exists(csv_path):
        st.error(f"Dataset not found: {csv_path}. Add it to the repo.")
    else:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            df = None
            st.error(f"Could not read {csv_path}: {e}")

        if df is not None:
            rename_map = {
                "Wire Dia": "wire_dia", "Speed": "speed_fpm",
                "Annealer Ht": "annealer_ht_ft", "Anneal T": "anneal_temp_f",
                "wire_dia": "wire_dia", "speed_fpm": "speed_fpm",
                "annealer_ht_ft": "annealer_ht_ft", "anneal_temp_f": "anneal_temp_f",
            }
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
            need = ["wire_dia", "speed_fpm", "annealer_ht_ft", "anneal_temp_f"]
            if not set(need).issubset(df.columns):
                st.error(f"CSV must contain columns: {need}")
            else:
                df = df.dropna(subset=need)
                df = df[(df["wire_dia"] > 0) & (df["speed_fpm"] > 0) & (df["annealer_ht_ft"] > 0)]
                if len(df) < 8:
                    st.error("Not enough rows to fit a stable model. Add more historical runs.")
                else:
                    df["dwell_s"] = (df["annealer_ht_ft"] / df["speed_fpm"]) * 60.0
                    df["ln_d"] = np.log(df["wire_dia"])
                    df["ln_dw"] = np.log(df["dwell_s"])
                    y = df["anneal_temp_f"].values

                    # Global ridge
                    Xg = np.column_stack([np.ones(len(df)), df["ln_d"].values, df["ln_dw"].values])
                    lam = 10.0
                    G = Xg.T @ Xg + lam * np.eye(Xg.shape[1])
                    cg = np.linalg.solve(G, Xg.T @ y)

                    # Local neighborhood
                    ln_d_all = df["ln_d"].values
                    ln_dw_all = df["ln_dw"].values
                    mu = np.array([ln_d_all.mean(), ln_dw_all.mean()])
                    sd = np.array([ln_d_all.std(ddof=1), ln_dw_all.std(ddof=1)])
                    sd[sd == 0] = 1.0

                    q_raw = np.array([math.log(diameter_in), math.log(dwell_s)])
                    qz = (q_raw - mu) / sd
                    Zz = (df[["ln_d","ln_dw"]].values - mu) / sd

                    from numpy.linalg import norm
                    dist = norm(Zz - qz, axis=1)

                    k = min(40, len(df))
                    idx = np.argsort(dist)[:k]
                    neigh = df.iloc[idx].copy()

                    Xl = np.column_stack([np.ones(len(neigh)), neigh["ln_d"].values, neigh["ln_dw"].values])
                    yl = neigh["anneal_temp_f"].values
                    cl, *_ = np.linalg.lstsq(Xl, yl, rcond=None)
                    a, b, c = float(cl[0]), float(cl[1]), float(cl[2])

                    b = max(b, 1.0)
                    c = min(c, -1.0)

                    ln_d = q_raw[0]; ln_dw = q_raw[1]
                    T_local = a + b * ln_d + c * ln_dw
                    T_global = float(cg[0] + cg[1]*ln_d + cg[2]*ln_dw)

                    pr75 = float(np.percentile(dist[idx], 75))
                    out_of_range = pr75 > 1.25

                    w_local = 0.75 if not out_of_range else 0.50
                    T_mix = w_local * T_local + (1.0 - w_local) * T_global

                    physics_w = 0.00 if not out_of_range else 0.20
                    T_blend = (1.0 - physics_w) * T_mix + physics_w * T_phys_F

                    T_final = float(np.clip(T_blend, 650.0, 1200.0))

                    st.subheader("Estimated anneal temperature")
                    st.write(f"**{T_final:,.1f} °F**")

                    st.caption(
                        f"dwell: {dwell_s:.1f} s  |  local slopes: d → +{b:.2f} °F/ln(in), "
                        f"dwell → {c:.2f} °F/ln(s)  |  global: {T_global:,.1f} °F  |  "
                        f"physics: {T_phys_F:,.1f} °F  |  oor={out_of_range}  |  k={k}  |  p75 dist={pr75:.2f}"
                    )

                    with st.expander("Closest historical runs"):
                        neigh["_dist"] = dist[idx]
                        st.dataframe(
                            neigh.sort_values("_dist").head(12)[
                                ["wire_dia","speed_fpm","annealer_ht_ft","dwell_s","anneal_temp_f","_dist"]
                            ].rename(columns={
                                "wire_dia":"dia (in)","speed_fpm":"speed (FPM)",
                                "annealer_ht_ft":"height (ft)","dwell_s":"dwell (s)",
                                "anneal_temp_f":"temp (°F)","_dist":"distance"
                            })
                        )

                    with st.expander("Data QA"):
                        bad_numeric = df[need].isna().sum().sum()
                        st.write(f"Rows: {len(df)}")
                        st.write(f"Empty cells across required columns: {bad_numeric}")
                        st.write("Tip: keep height consistent with your history when testing a neighborhood.")
