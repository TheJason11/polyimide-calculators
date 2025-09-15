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

def num_input(label, value, step, minv, maxv=None, fmt="%.4f"):
    return st.number_input(label, value=float(value), step=float(step),
                           min_value=float(minv), max_value=None if maxv is None else float(maxv),
                           format=fmt)


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
            feet = num_input("Total feet", 12000.0, 100.0, 0.0, fmt="%.2f")
        with c2:
            fpm = num_input("Line speed (FPM)", 18.0, 0.5, 0.0, fmt="%.2f")
        run_minutes = feet / fpm if fpm > 0 else 0.0
        run_hours = run_minutes / 60.0
        st.subheader("Results")
        st.write(f"Runtime: **{run_minutes:,.2f} minutes**  |  **{run_hours:,.2f} hours**")
        st.write(f"Throughput: **{fpm*60:,.0f} ft per hour**")

    elif mode == "Time and Speed → Feet":
        c1, c2, c3 = st.columns(3)
        with c1:
            hours = num_input("Hours", 3.0, 0.5, 0.0, fmt="%.2f")
        with c2:
            minutes = num_input("Minutes", 0.0, 5.0, 0.0, fmt="%.2f")
        with c3:
            fpm = num_input("Line speed (FPM)", 18.0, 0.5, 0.0, fmt="%.2f")
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
            feet_run = num_input("Feet run", 600.0, 50.0, 0.0, fmt="%.2f")
        with c2:
            minutes_run = num_input("Minutes", 60.0, 1.0, 0.0, fmt="%.2f")
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
        d_in = num_input("Wire diameter (in)", 0.0500, 0.0010, 0.0001)
    with c2:
        area_in2 = circle_area_in2(d_in)
        st.write(f"Cross section area: **{area_in2:.6f} in²**")

    if mode == "Feet → Pounds":
        feet = num_input("Length (ft)", 12000.0, 100.0, 0.0, fmt="%.2f")
        length_in = inches_from_feet(feet)
        volume_in3 = area_in2 * length_in
        pounds = volume_in3 * COPPER_DENSITY_LB_PER_IN3
        st.subheader("Results")
        st.write(f"Estimated weight: **{pounds:,.2f} lb**")
    else:
        pounds = num_input("Weight (lb)", 54.0, 0.5, 0.0, fmt="%.2f")
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
        id_in = num_input("Bare copper diameter, ID (in)", 0.0500, 0.0010, 0.0001)
    with c2:
        wall_in = num_input("Coating wall (in)", 0.0015, 0.0001, 0.0)
    with c3:
        coat_density = num_input("Coating density (lb/in³)", 0.0513, 0.0001, 0.01, 0.08, fmt="%.4f")

    area_cu_in2 = circle_area_in2(id_in)
    area_coat_in2 = annulus_area_in2(id_in, wall_in)
    lin_den_lb_per_ft = 12.0 * (area_cu_in2 * COPPER_DENSITY_LB_PER_IN3 + area_coat_in2 * coat_density)

    if mode == "Feet → Pounds":
        feet = num_input("Length (ft)", 1500.0, 50.0, 0.0, fmt="%.2f")
        pounds = feet * lin_den_lb_per_ft
        st.subheader("Results")
        st.write(f"Linear density: **{lin_den_lb_per_ft:,.5f} lb/ft**")
        st.write(f"Estimated weight: **{pounds:,.3f} lb**")
    else:
        gross_lb = num_input("Gross spool weight (lb)", 12.0, 0.1, 0.0, fmt="%.2f")
        tare_lb = num_input("Spool tare (lb)", 0.0, 0.1, 0.0, fmt="%.2f")
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
        d0_in = num_input("Starting diameter (in)", 0.0500, 0.0010, 0.0001)
    with c2:
        tension_lbf = num_input("Applied tension (lbf)", 15.0, 1.0, 0.0, fmt="%.2f")
    with c3:
        passes = st.number_input("Number of passes", min_value=1, value=10, step=1)

    c4, c5, c6 = st.columns(3)
    with c4:
        anneal_temp_f = num_input("Anneal temp (°F)", 700.0, 10.0, 0.0, fmt="%.1f")
    with c5:
        oven_height_ft = num_input("Oven height (ft)", 12.0, 1.0, 1.0, fmt="%.0f")
    with c6:
        line_speed_fpm = num_input("Line speed (FPM)", 18.0, 0.5, 0.1, fmt="%.2f")

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
        id_in = num_input("ID (in)", 0.0160, 0.0005, 0.0001)
    with c2:
        wall_in = num_input("Wall (in)", 0.0010, 0.0001, 0.0001)
    with c3:
        length_ft = num_input("Finished length (ft)", 1500.0, 50.0, 0.0, fmt="%.2f")

    c4, c5, c6 = st.columns(3)
    with c4:
        solids_frac = num_input("Solids fraction", 0.15, 0.01, 0.01, 1.0, fmt="%.2f")
    with c5:
        startup_ft = num_input("Startup scrap (ft)", 150.0, 10.0, 0.0, fmt="%.1f")
    with c6:
        shutdown_ft = num_input("Shutdown scrap (ft)", 50.0, 10.0, 0.0, fmt="%.1f")

    c7, c8, c9 = st.columns(3)
    with c7:
        hold_up_cm3 = num_input("Hold-up volume (cm³)", 400.0, 10.0, 0.0, fmt="%.1f")
    with c8:
        heel_cm3 = num_input("Heel volume (cm³)", 120.0, 5.0, 0.0, fmt="%.1f")
    with c9:
        soln_density_g_cm3 = num_input("Solution density (g/cm³)", 1.06, 0.01, 0.80, 1.50, fmt="%.2f")

    allowance_frac = num_input("Allowance fraction", 0.05, 0.01, 0.0, 0.5, fmt="%.2f")

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
    st.write(f"Hold-up mass: **{hold_up_mass_lb:,.3f} lb**")
    st.write(f"Heel mass: **{heel_mass_lb:,.3f} lb**")
    st.write(f"Subtotal: **{subtotal_lb:,.4f} lb**")
    st.write(f"Total with allowance: **{total_with_allowance_lb:,.4f} lb**")


# =============================
# Module: Anneal Temp Estimator (HYBRID: physics baseline + data correction)
# =============================
elif page == "Anneal Temp Estimator":
    st.title("Anneal Temperature Estimator (Hybrid)")

    # ---- Inputs with decimal-friendly steps
    c1, c2, c3 = st.columns(3)
    with c1:
        diameter_in = num_input("Wire diameter (in)", 0.0500, 0.0010, 0.001)
    with c2:
        speed_fpm = num_input("Line speed (FPM)", 18.0, 0.5, 0.1, fmt="%.2f")
    with c3:
        height_ft = num_input("Annealer height (ft)", 14.0, 1.0, 1.0, fmt="%.0f")

    with st.expander("Advanced (oxide target and kinetics)"):
        # Parabolic oxide growth: x^2 = K0 * exp(-Ea/(R T)) * t
        # Defaults are conservative; you can tune later with your lab data.
        x_um = num_input("Target oxide thickness (µm)", 0.20, 0.05, 0.02, 5.0, fmt="%.2f")
        k0 = num_input("Parabolic rate K0 (m²/s)", 1e-10, 1e-11, 1e-12, 1e-8, fmt="%.1e")
        ea_kj = num_input("Activation energy Ea (kJ/mol)", 120.0, 5.0, 40.0, 200.0, fmt="%.1f")
        t_heat_margin_s = num_input("Heat-up allowance (s)", 6.0, 1.0, 0.0, 60.0, fmt="%.1f")

    def physics_temp_required(d_in, speed, height, x_um, k0, ea_kj, t_margin_s):
        # Dwell time
        dwell_s = (height / speed) * 60.0
        # Simple thermal penalty for large diameters (lumped lag ~ d^2 scaling)
        # Empirical: subtract a small heat-up margin that grows with d^2
        t_eff = max(1e-3, dwell_s - (t_margin_s * (d_in / 0.05)**2))
        # Convert target thickness to meters
        x_m = x_um * 1e-6
        # Solve x^2 = K0 * exp(-Ea/RT) * t  =>  T = Ea/R / ln( K0 * t / x^2 )
        Ea = ea_kj * 1000.0  # J/mol
        num = Ea / R_GAS
        denom = math.log(max(1e-20, (k0 * t_eff) / (x_m**2)))
        if denom <= 0:
            # If ln is non-positive (impossible physically), push to high temperature bound
            T_K = 1100.0  # ~ 827 °C = 1100 K (placeholder)
        else:
            T_K = num / denom
        T_F = (T_K - 273.15) * 9.0/5.0 + 32.0
        return T_F, dwell_s

    # Physics baseline
    T_phys_F, dwell_s = physics_temp_required(diameter_in, speed_fpm, height_ft, x_um, k0, ea_kj, t_heat_margin_s)

    # ---- Data correction from repo CSV (if available)
    csv_path = "annealing_dataset_clean.csv"
    coef = None
    mae = rmse = None
    used_rows = None

    if os.path.exists(csv_path):
        try:
            hist = pd.read_csv(csv_path)
            # Normalize headers
            rename_map = {
                "Wire Dia": "wire_dia", "Speed": "speed_fpm",
                "Annealer Ht": "annealer_ht_ft", "Anneal T": "anneal_temp_f",
                "wire_dia": "wire_dia", "speed_fpm": "speed_fpm",
                "annealer_ht_ft": "annealer_ht_ft", "anneal_temp_f": "anneal_temp_f",
            }
            hist = hist.rename(columns={k: v for k, v in rename_map.items() if k in hist.columns})
            need = ["wire_dia", "speed_fpm", "annealer_ht_ft", "anneal_temp_f"]
            if set(need).issubset(hist.columns):
                hist = hist.dropna(subset=need)
                hist = hist[(hist["wire_dia"] > 0) & (hist["speed_fpm"] > 0) & (hist["annealer_ht_ft"] > 0)]
                hist["dwell_s"] = (hist["annealer_ht_ft"] / hist["speed_fpm"]) * 60.0

                # Compute physics baseline for each historical row (same advanced settings)
                def phys_row(row):
                    tF, _ = physics_temp_required(row["wire_dia"], row["speed_fpm"], row["annealer_ht_ft"],
                                                  x_um, k0, ea_kj, t_heat_margin_s)
                    return tF
                hist["T_phys_F"] = hist.apply(phys_row, axis=1)

                # Hybrid fit: actual_T ≈ a + b*T_phys + c*ln(d) + d*ln(dwell)
                X = np.column_stack([
                    np.ones(len(hist)),
                    hist["T_phys_F"].values,
                    np.log(hist["wire_dia"].values),
                    np.log(hist["dwell_s"].values)
                ])
                y = hist["anneal_temp_f"].values
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                pred_hist = X @ coef
                mae = float(np.mean(np.abs(pred_hist - y)))
                rmse = float(np.sqrt(np.mean((pred_hist - y) ** 2)))

                # Keep a few nearest rows for display
                ln_d = math.log(max(diameter_in, 1e-6))
                ln_dw = math.log(max(dwell_s, 1e-6))
                hist["_dist"] = np.sqrt((np.log(hist["wire_dia"]) - ln_d)**2 + (np.log(hist["dwell_s"]) - ln_dw)**2)
                used_rows = hist.sort_values("_dist").head(5)[
                    ["wire_dia", "speed_fpm", "annealer_ht_ft", "dwell_s", "anneal_temp_f", "T_phys_F"]
                ]
        except Exception as e:
            st.error(f"Could not read/fit dataset: {e}")

    # Prediction
    if coef is not None:
        Xq = np.array([1.0, T_phys_F, math.log(max(diameter_in, 1e-6)), math.log(max(dwell_s, 1e-6))])
        T_hybrid = float(Xq @ coef)
        # Guardrails
        T_final = float(np.clip(T_hybrid, 500.0, 1100.0))
        model_note = f"hybrid = a + b·T_phys + c·ln(d) + d·ln(dwell)  |  fit MAE {mae:.1f} °F, RMSE {rmse:.1f} °F"
    else:
        # No data available; fall back to physics
        T_final = float(np.clip(T_phys_F, 500.0, 1100.0))
        model_note = "physics only (no repo data)"

    st.subheader("Estimated anneal temperature")
    st.write(f"**{T_final:,.1f} °F**")
    st.caption(f"dwell: {dwell_s:.1f} s  |  physics baseline: {T_phys_F:,.1f} °F  |  {model_note}")

    if used_rows is not None:
        st.write("Closest historical runs (for context):")
        st.dataframe(
            used_rows.rename(columns={
                "wire_dia": "dia (in)", "speed_fpm": "speed (FPM)",
                "annealer_ht_ft": "height (ft)", "dwell_s": "dwell (s)",
                "anneal_temp_f": "actual (°F)", "T_phys_F": "physics (°F)"
            })
        )
