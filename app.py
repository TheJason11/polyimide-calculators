import math
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
        "Coated Copper Converter",  # NEW
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

def k_to_f(t_k: float) -> float:
    return (t_k - 273.15) * 9.0/5.0 + 32.0

# -----------------------------
# Module: Runtime Calculator
# -----------------------------
if page == "Runtime Calculator":
    st.title("Job Runtime Calculator")
    st.caption("Compute runtime, footage, or rate from feet, time, and speed")

    mode = st.radio(
        "Choose calculator",
        ["Feet and Speed → Runtime", "Time and Speed → Feet", "Feet and Time → Rate"],
        index=0,
    )

    if mode == "Feet and Speed → Runtime":
        c1, c2 = st.columns(2)
        with c1:
            feet = st.number_input("Total feet", min_value=0.0, value=12000.0, step=100.0)
        with c2:
            fpm = st.number_input("Line speed (FPM)", min_value=0.0, value=18.0, step=0.5)

        run_minutes = feet / fpm if fpm > 0 else 0.0
        run_hours = run_minutes / 60.0

        st.subheader("Results")
        st.write(f"Runtime: **{run_minutes:,.2f} minutes**  |  **{run_hours:,.2f} hours**")
        st.write(f"Throughput: **{fpm*60:,.0f} ft per hour**")

    elif mode == "Time and Speed → Feet":
        c1, c2, c3 = st.columns(3)
        with c1:
            hours = st.number_input("Hours", min_value=0.0, value=3.0, step=0.5)
        with c2:
            minutes = st.number_input("Minutes", min_value=0.0, value=0.0, step=5.0)
        with c3:
            fpm = st.number_input("Line speed (FPM)", min_value=0.0, value=18.0, step=0.5)

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
            feet_run = st.number_input("Feet run", min_value=0.0, value=600.0, step=50.0)
        with c2:
            minutes_run = st.number_input("Minutes", min_value=0.0, value=60.0, step=1.0)

        if minutes_run > 0:
            fpm_calc = feet_run / minutes_run
            st.subheader("Results")
            st.write(f"Calculated rate: **{fpm_calc:,.2f} FPM**")
            st.write(f"Throughput: **{fpm_calc*60:,.0f} ft per hour**")
        else:
            st.info("Enter minutes greater than zero")

# -----------------------------
# Module: Copper Wire Converter
# -----------------------------
elif page == "Copper Wire Converter":
    st.title("Copper Wire Length ↔ Weight")

    mode = st.radio("Choose converter", ["Feet → Pounds", "Pounds → Feet"], index=0)

    c1, c2 = st.columns(2)
    with c1:
        d_in = st.number_input("Wire diameter (in)", min_value=0.0001, value=0.0500, step=0.0010)
    with c2:
        area_in2 = circle_area_in2(d_in)
        st.write(f"Cross section area: **{area_in2:.6f} in²**")

    if mode == "Feet → Pounds":
        feet = st.number_input("Length (ft)", min_value=0.0, value=12000.0, step=100.0)
        length_in = inches_from_feet(feet)
        volume_in3 = area_in2 * length_in
        pounds = volume_in3 * COPPER_DENSITY_LB_PER_IN3
        st.subheader("Results")
        st.write(f"Estimated weight: **{pounds:,.2f} lb**")

    elif mode == "Pounds → Feet":
        pounds = st.number_input("Weight (lb)", min_value=0.0, value=54.0, step=0.5)
        length_in = pounds / (COPPER_DENSITY_LB_PER_IN3 * area_in2) if area_in2 > 0 else 0.0
        feet = length_in / 12.0
        st.subheader("Results")
        st.write(f"Estimated length: **{feet:,.0f} ft**")

# -----------------------------
# Module: Coated Copper Converter (NEW)
# -----------------------------
elif page == "Coated Copper Converter":
    st.title("Coated Copper Length ↔ Weight")

    mode = st.radio("Choose converter", ["Feet → Pounds", "Pounds → Feet"], index=0)

    c1, c2, c3 = st.columns(3)
    with c1:
        id_in = st.number_input("Bare copper diameter, ID (in)", min_value=0.0001, value=0.0500, step=0.0010)
    with c2:
        wall_in = st.number_input("Coating wall (in)", min_value=0.0000, value=0.0015, step=0.0001)
    with c3:
        coat_density = st.number_input("Coating density (lb/in³)", min_value=0.0100, max_value=0.0800, value=0.0513, step=0.0001)

    # Cross-sections
    area_cu_in2 = circle_area_in2(id_in)
    area_coat_in2 = annulus_area_in2(id_in, wall_in)

    # Linear density (lb/ft)
    lin_den_lb_per_ft = 12.0 * (area_cu_in2 * COPPER_DENSITY_LB_PER_IN3 + area_coat_in2 * coat_density)

    if mode == "Feet → Pounds":
        feet = st.number_input("Length (ft)", min_value=0.0, value=1500.0, step=50.0)
        pounds = feet * lin_den_lb_per_ft
        st.subheader("Results")
        st.write(f"Linear density: **{lin_den_lb_per_ft:,.5f} lb/ft**")
        st.write(f"Estimated weight: **{pounds:,.3f} lb**")

    elif mode == "Pounds → Feet":
        gross_lb = st.number_input("Gross spool weight (lb)", min_value=0.0, value=12.0, step=0.1)
        tare_lb = st.number_input("Spool tare (lb)", min_value=0.0, value=0.0, step=0.1)
        net_lb = max(gross_lb - tare_lb, 0.0)
        feet = (net_lb / lin_den_lb_per_ft) if lin_den_lb_per_ft > 0 else 0.0
        st.subheader("Results")
        st.write(f"Linear density: **{lin_den_lb_per_ft:,.5f} lb/ft**")
        st.write(f"Net wire weight: **{net_lb:,.3f} lb**")
        st.write(f"Estimated length: **{feet:,.0f} ft**")

# -----------------------------
# Module: Wire Stretch Predictor
# -----------------------------
elif page == "Wire Stretch Predictor":
    st.title("Wire Stretch Predictor")
    st.caption("Elastic estimate based on copper properties and applied tension")

    c1, c2, c3 = st.columns(3)
    with c1:
        d0_in = st.number_input("Starting diameter (in)", min_value=0.0001, value=0.0500, step=0.0010)
    with c2:
        tension_lbf = st.number_input("Applied tension (lbf)", min_value=0.0, value=15.0, step=1.0)
    with c3:
        passes = st.number_input("Number of passes", min_value=1, value=10, step=1)

    c4, c5, c6 = st.columns(3)
    with c4:
        anneal_temp_f = st.number_input("Anneal temp (°F)", min_value=0.0, value=700.0, step=10.0)
    with c5:
        oven_height_ft = st.number_input("Oven height (ft)", min_value=1.0, value=12.0, step=1.0)
    with c6:
        line_speed_fpm = st.number_input("Line speed (FPM)", min_value=0.1, value=18.0, step=0.5)

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

# -----------------------------
# Module: PAA Usage
# -----------------------------
elif page == "PAA Usage":
    st.title("PAA Usage Calculator")

    c1, c2, c3 = st.columns(3)
    with c1:
        id_in = st.number_input("ID (in)", min_value=0.0001, value=0.0160, step=0.0005)
    with c2:
        wall_in = st.number_input("Wall (in)", min_value=0.0001, value=0.0010, step=0.0001)
    with c3:
        length_ft = st.number_input("Finished length (ft)", min_value=0.0, value=1500.0, step=50.0)

    c4, c5, c6 = st.columns(3)
    with c4:
        solids_frac = st.number_input("Solids fraction", min_value=0.01, max_value=1.0, value=0.15, step=0.01)
    with c5:
        startup_ft = st.number_input("Startup scrap (ft)", min_value=0.0, value=150.0, step=10.0)
    with c6:
        shutdown_ft = st.number_input("Shutdown scrap (ft)", min_value=0.0, value=50.0, step=10.0)

    c7, c8, c9 = st.columns(3)
    with c7:
        hold_up_cm3 = st.number_input("Hold-up volume (cm³)", min_value=0.0, value=400.0, step=10.0)
    with c8:
        heel_cm3 = st.number_input("Heel volume (cm³)", min_value=0.0, value=120.0, step=5.0)
    with c9:
        soln_density_g_cm3 = st.number_input("Solution density (g/cm³)", min_value=0.80, max_value=1.50, value=1.06, step=0.01)

    allowance_frac = st.number_input("Allowance fraction", min_value=0.0, max_value=0.50, value=0.05, step=0.01)

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

# -----------------------------
# Module: Anneal Temp Estimator
# -----------------------------
elif page == "Anneal Temp Estimator":
    st.title("Anneal Temperature Estimator")

    c1, c2, c3 = st.columns(3)
    with c1:
        wire_d_in = st.number_input("Wire diameter (in)", min_value=0.0005, value=0.0500, step=0.0010)
    with c2:
        height_ft = st.number_input("Annealer height (ft)", min_value=1.0, value=12.0, step=1.0)
    with c3:
        fpm = st.number_input("Line speed (FPM)", min_value=0.1, value=18.0, step=0.5)

    c4, c5, c6 = st.columns(3)
    with c4:
        target_oxide_um = st.number_input("Target oxide thickness (µm)", min_value=0.01, value=0.20, step=0.01)
    with c5:
        k0_ox = st.number_input("Oxide k0 (m²/s)", min_value=1e-20, value=1e-10, step=1e-10, format="%.1e")
    with c6:
        Ea_ox_kJ = st.number_input("Oxide Ea (kJ/mol)", min_value=20.0, value=120.0, step=5.0)

    dwell_s = (height_ft / fpm) * 60.0
    x_m = target_oxide_um * 1e-6
    Ea_ox_J = Ea_ox_kJ * 1000.0

    ox_possible = (k0_ox * dwell_s) > (x_m**2)
    T_ox_K = None
    if ox_possible:
        T_ox_K = Ea_ox_J / (R_GAS * math.log((k0_ox * dwell_s) / (x_m**2)))

    st.subheader("Results")
    st.write(f"Dwell time: **{dwell_s:,.1f} s**")
    if T_ox_K:
        st.write(f"Required oven set: **{k_to_f(T_ox_K):,.1f} °F**")
    else:
        st.error("Target not reachable with current inputs")
