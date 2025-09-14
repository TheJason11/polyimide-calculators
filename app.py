import math
import streamlit as st

# =========================
# App config
# =========================
st.set_page_config(page_title="Polyimide Calculators", page_icon="🧮", layout="centered")

# =========================
# Constants (imperial + conversions)
# =========================
COPPER_DENSITY_LB_PER_IN3 = 0.323        # annealed copper working value
E_COPPER_PSI = 16_000_000                # Young's modulus
NU_COPPER = 0.34                         # Poisson ratio
YIELD_WARN_LOW = 10_000                  # psi
YIELD_WARN_HIGH = 20_000                 # psi

# Polyimide & general conversions
PI_DENSITY_G_PER_CM3_DEFAULT = 1.42
IN3_TO_CM3 = 16.387064
G_PER_LB = 453.59237
PI_CONST = math.pi

# Gas constant
R_GAS = 8.314  # J/mol-K

# =========================
# Sidebar navigation
# =========================
st.sidebar.title("Polyimide Calculators")
page = st.sidebar.radio(
    "Choose a module",
    [
        "Runtime Calculator",
        "Copper Wire Converter",
        "Wire Stretch Predictor",
        "PAA Usage",
        "Anneal Temp Estimator",   # new physics-based estimator
    ],
    index=0,
)

# =========================
# Shared helpers
# =========================
def inches_from_feet(feet: float) -> float:
    return feet * 12.0

def circle_area_in2(d_in: float) -> float:
    return PI_CONST * (d_in**2) / 4.0

def annulus_area_in2(id_in: float, wall_in: float) -> float:
    """Area of tube wall = π/4 * (OD^2 - ID^2) where OD = ID + 2*wall."""
    od_in = id_in + 2.0 * wall_in
    return PI_CONST * (od_in**2 - id_in**2) / 4.0

def f_to_k(t_f: float) -> float:
    return (t_f - 32.0) * 5.0/9.0 + 273.15

def k_to_f(t_k: float) -> float:
    return (t_k - 273.15) * 9.0/5.0 + 32.0

# =========================
# Module 3: Runtime Calculator
# =========================
if page == "Runtime Calculator":
    st.title("Job Runtime Calculator")
    st.caption("Compute runtime, footage, or rate from feet, time, and speed")

    mode = st.radio(
        "Choose calculator",
        [
            "Feet and Speed → Runtime",
            "Time and Speed → Feet",
            "Feet and Time → Rate",
        ],
        index=0,
    )

    with st.expander("Notes", expanded=False):
        st.write(
            """
            • Runtime minutes = Feet ÷ FPM  
            • Feet = FPM × Minutes  
            • FPM = Feet ÷ Minutes  
            • Ft per hour = FPM × 60
            """
        )

    if mode == "Feet and Speed → Runtime":
        c1, c2 = st.columns(2)
        with c1:
            feet = st.number_input("Total feet", min_value=0.0, value=12000.0, step=100.0, format="%.2f")
        with c2:
            fpm = st.number_input("Line speed (FPM)", min_value=0.0, value=18.0, step=0.5, format="%.2f")

        run_minutes = feet / fpm if fpm > 0 else 0.0
        run_hours = run_minutes / 60.0

        st.subheader("Results")
        st.write(f"Runtime: **{run_minutes:,.2f} minutes**  |  **{run_hours:,.2f} hours**")
        st.write(f"Throughput: **{fpm*60:,.0f} ft per hour**")

    elif mode == "Time and Speed → Feet":
        c1, c2, c3 = st.columns(3)
        with c1:
            hours = st.number_input("Hours", min_value=0.0, value=3.0, step=0.5, format="%.2f")
        with c2:
            minutes = st.number_input("Minutes", min_value=0.0, value=0.0, step=5.0, format="%.2f")
        with c3:
            fpm = st.number_input("Line speed (FPM)", min_value=0.0, value=18.0, step=0.5, format="%.2f")

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
            feet_run = st.number_input("Feet run", min_value=0.0, value=600.0, step=50.0, format="%.2f")
        with c2:
            minutes_run = st.number_input("Minutes", min_value=0.0, value=60.0, step=1.0, format="%.1f")

        if minutes_run > 0:
            fpm_calc = feet_run / minutes_run
            st.subheader("Results")
            st.write(f"Calculated rate: **{fpm_calc:,.2f} FPM**")
            st.write(f"Throughput: **{fpm_calc*60:,.0f} ft per hour**")
        else:
            st.info("Enter minutes greater than zero")

# =========================
# Module 4: Copper Wire Converter
# =========================
elif page == "Copper Wire Converter":
    st.title("Copper Wire Length ↔ Weight")

    mode = st.radio("Choose converter", ["Feet → Pounds", "Pounds → Feet"], index=0)

    c1, c2 = st.columns(2)
    with c1:
        d_in = st.number_input("Wire diameter (in)", min_value=0.0001, value=0.0500, step=0.0010, format="%.4f")
    with c2:
        area_in2 = circle_area_in2(d_in)
        st.write(f"Cross section area: **{area_in2:.6f} in²**")

    if mode == "Feet → Pounds":
        feet = st.number_input("Length (ft)", min_value=0.0, value=12000.0, step=100.0, format="%.2f")
        length_in = inches_from_feet(feet)
        volume_in3 = area_in2 * length_in
        pounds = volume_in3 * COPPER_DENSITY_LB_PER_IN3

        st.subheader("Results")
        st.write(f"Estimated weight: **{pounds:,.2f} lb**")

    elif mode == "Pounds → Feet":
        pounds = st.number_input("Weight (lb)", min_value=0.0, value=54.0, step=0.5, format="%.2f")
        length_in = pounds / (COPPER_DENSITY_LB_PER_IN3 * area_in2) if area_in2 > 0 else 0.0
        feet = length_in / 12.0

        st.subheader("Results")
        st.write(f"Estimated length: **{feet:,.0f} ft**")

# =========================
# Module 5: Wire Stretch Predictor (Elastic baseline + placeholders)
# =========================
elif page == "Wire Stretch Predictor":
    st.title("Wire Stretch Predictor")
    st.caption("Elastic estimate using tension and copper properties. Extra inputs collected now; calibration factor lets us fit later when you have data.")

    # --- Inputs (collect everything we may need) ---
    c1, c2, c3 = st.columns(3)
    with c1:
        d0_in = st.number_input("Starting diameter (in)", min_value=0.0001, value=0.0500, step=0.0010, format="%.4f")
    with c2:
        tension_lbf = st.number_input("Applied tension (lbf)", min_value=0.0, value=15.0, step=1.0, format="%.1f")
    with c3:
        passes = st.number_input("Number of passes", min_value=1, value=10, step=1, format="%d")

    c4, c5, c6 = st.columns(3)
    with c4:
        anneal_temp_f = st.number_input("Anneal temp (°F)", min_value=0.0, value=700.0, step=10.0, format="%.0f")
    with c5:
        oven_height_ft = st.number_input("Oven height (ft)", min_value=1.0, value=12.0, step=1.0, format="%.0f")
    with c6:
        line_speed_fpm = st.number_input("Line speed (FPM)", min_value=0.1, value=18.0, step=0.5, format="%.1f")

    # Calibration knob (future data fit)
    cal = st.slider("Calibration factor (future fit from history)", min_value=0.50, max_value=1.50, value=1.00, step=0.01)

    # --- Physics baseline (elastic only under load) ---
    area_in2 = circle_area_in2(d0_in)
    sigma_psi = tension_lbf / area_in2 if area_in2 > 0 else 0.0
    elastic_strain = sigma_psi / E_COPPER_PSI if E_COPPER_PSI > 0 else 0.0
    radial_frac = NU_COPPER * elastic_strain
    d_loaded_baseline = d0_in * (1.0 - radial_frac)

    # Apply calibration factor (kept at 1.0 until we fit)
    d_loaded = d_loaded_baseline * cal

    st.subheader("Results (baseline elastic)")
    st.write(f"Axial stress: **{sigma_psi:,.0f} psi**")
    st.write(f"Axial strain: **{elastic_strain:.6e}**")
    st.write(f"Predicted diameter under load: **{d_loaded:.6f} in**  (calibration ×{cal:.2f})")

    if sigma_psi > YIELD_WARN_LOW:
        st.warning(f"Stress exceeds {YIELD_WARN_LOW:,} psi. Stay well below yield ({YIELD_WARN_LOW:,}–{YIELD_WARN_HIGH:,} psi) for elastic-only prediction.")

    with st.expander("Notes on extra inputs", expanded=False):
        st.write(
            """
            • **Passes, anneal temp, oven height, speed** strongly affect *permanent* set and thermal softening.  
            • Baseline above is **elastic-only** under current tension.  
            • We’ve added a **calibration factor** so when you share history (passes, temps, speeds, final measured ID), we’ll fit a model that uses these variables.
            """
        )

    st.divider()
    st.subheader("What tension hits a chosen diameter (elastic-only)")
    target_d_in = st.number_input("Target diameter under load (in)", min_value=0.0, value=0.0000, step=0.0005, format="%.4f")
    if 0 < target_d_in < d0_in:
        # From d_loaded = d0*(1 - nu*eps) ⇒ eps_req = (d0 - target)/ (nu*d0)
        eps_req = (d0_in - target_d_in) / (NU_COPPER * d0_in)
        sigma_req = eps_req * E_COPPER_PSI
        T_req = sigma_req * area_in2
        st.write(f"Required tension: **{T_req:,.1f} lbf**  |  required stress **{sigma_req:,.0f} psi**")
    else:
        st.info("Enter a target diameter smaller than the starting diameter to get a required tension estimate.")

# =========================
# Module: PAA Usage (full procedure)
# =========================
elif page == "PAA Usage":
    st.title("PAA Usage Calculator")
    st.caption("Checks BOM sufficiency using geometry + scrap + system hold-up + heel + allowance. Solids by weight (e.g., 0.15 for 15%).")

    with st.expander("Variables (enter what’s on the job packet + your line constants)", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            id_in = st.number_input("ID (in)", min_value=0.0001, value=0.0160, step=0.0005, format="%.4f")
        with c2:
            wall_in = st.number_input("Wall (in)", min_value=0.0001, value=0.0010, step=0.0001, format="%.4f")
        with c3:
            length_ft = st.number_input("Finished length (ft)", min_value=0.0, value=1500.0, step=50.0, format="%.0f")

        c4, c5, c6 = st.columns(3)
        with c4:
            solids_frac = st.number_input("Solids fraction (wt)", min_value=0.01, max_value=1.0, value=0.15, step=0.01, format="%.2f")
        with c5:
            startup_ft = st.number_input("Startup scrap (ft)", min_value=0.0, value=150.0, step=10.0, format="%.0f")
        with c6:
            shutdown_ft = st.number_input("Shutdown scrap (ft)", min_value=0.0, value=50.0, step=10.0, format="%.0f")

        c7, c8, c9 = st.columns(3)
        with c7:
            hold_up_cm3 = st.number_input("Hold-up volume (cm³)", min_value=0.0, value=400.0, step=10.0, format="%.0f")
        with c8:
            heel_cm3 = st.number_input("Heel volume (cm³)", min_value=0.0, value=120.0, step=5.0, format="%.0f")
        with c9:
            soln_density_g_cm3 = st.number_input("Solution density (g/cm³)", min_value=0.80, max_value=1.50, value=1.06, step=0.01, format="%.2f")

        c10, c11 = st.columns(2)
        with c10:
            pi_density_g_cm3 = st.number_input("Cured PI density (g/cm³)", min_value=1.30, max_value=1.60, value=PI_DENSITY_G_PER_CM3_DEFAULT, step=0.01, format="%.2f")
        with c11:
            allowance_frac = st.number_input("Allowance fraction", min_value=0.0, max_value=0.50, value=0.05, step=0.01, format="%.02f")

    # --- Core geometry → polymer mass for finished footage ---
    length_in = inches_from_feet(length_ft)
    A_wall_in2 = annulus_area_in2(id_in, wall_in)
    V_in3 = A_wall_in2 * length_in
    V_cm3 = V_in3 * IN3_TO_CM3
    mass_PI_g = V_cm3 * pi_density_g_cm3
    mass_PI_lb = mass_PI_g / G_PER_LB
    solution_for_polymer_lb = mass_PI_lb / solids_frac if solids_frac > 0 else 0.0

    # --- Startup + shutdown scrap (treated same as finished product) ---
    scrap_total_ft = startup_ft + shutdown_ft
    scrap_solution_lb = solution_for_polymer_lb * (scrap_total_ft / length_ft) if length_ft > 0 else 0.0

    # --- Hold-up & heel from volumes and solution density ---
    hold_up_mass_lb = (hold_up_cm3 * soln_density_g_cm3) / G_PER_LB
    heel_mass_lb = (heel_cm3 * soln_density_g_cm3) / G_PER_LB

    subtotal_lb = solution_for_polymer_lb + scrap_solution_lb + hold_up_mass_lb + heel_mass_lb
    total_with_allowance_lb = subtotal_lb * (1.0 + allowance_frac)

    st.subheader("Results")
    st.write(f"Solution for **finished length** (at solids): **{solution_for_polymer_lb:,.4f} lb**")
    st.write(f"Startup + shutdown **scrap solution**: **{scrap_solution_lb:,.4f} lb**  (scrap total: {scrap_total_ft:,.0f} ft)")
    st.write(f"**Hold-up** mass from {hold_up_cm3:,.0f} cm³ @ ρ={soln_density_g_cm3:.2f}: **{hold_up_mass_lb:,.3f} lb**")
    st.write(f"**Heel** mass from {heel_cm3:,.0f} cm³ @ ρ={soln_density_g_cm3:.2f}: **{heel_mass_lb:,.3f} lb**")
    st.write(f"**Subtotal** (no allowance): **{subtotal_lb:,.4f} lb**")
    st.write(f"**Total with allowance** ({allowance_frac:.0%}): **{total_with_allowance_lb:,.4f} lb**")

    with st.expander("Quick procedure / audit checklist", expanded=False):
        st.write(
            """
            1) Compute solution for finished footage from geometry & solids.  
            2) Add startup + shutdown scrap (treated like finished).  
            3) Add fixed hold-up + heel (convert cm³ with measured density).  
            4) Add allowance for mid-run events.  
            5) Compare to BOM; request adjustment if short.
            """
        )

# =========================
# Module: Anneal Temp Estimator (physics-based)
# =========================
elif page == "Anneal Temp Estimator":
    st.title("Anneal Temperature Estimator")
    st.caption("Predict oven set temp needed at given height & FPM to hit target oxide thickness and anneal severity. Tunable kinetics—calibrate later.")

    # --- Inputs ---
    c1, c2, c3 = st.columns(3)
    with c1:
        wire_d_in = st.number_input("Wire diameter (in)", min_value=0.0005, value=0.0500, step=0.0010, format="%.4f")
    with c2:
        height_ft = st.number_input("Annealer height (ft)", min_value=1.0, value=12.0, step=1.0, format="%.0f")
    with c3:
        fpm = st.number_input("Line speed (FPM)", min_value=0.1, value=18.0, step=0.5, format="%.1f")

    c4, c5, c6 = st.columns(3)
    with c4:
        target_oxide_um = st.number_input("Target oxide thickness (µm)", min_value=0.01, value=0.20, step=0.01, format="%.2f")
    with c5:
        # Oxide growth: x^2 = k0 * exp(-Ea/(R T)) * t
        k0_ox = st.number_input("Oxide k0 (m²/s)", min_value=1e-20, value=1e-10, step=1e-10, format="%.1e")
    with c6:
        Ea_ox_kJ = st.number_input("Oxide Ea (kJ/mol)", min_value=20.0, value=120.0, step=5.0, format="%.0f")

    c7, c8, c9 = st.columns(3)
    with c7:
        # Anneal severity: k_a * t >= 1, with k_a = k0_a * exp(-Ea_a/(R T))
        k0_a = st.number_input("Anneal k0 (1/s)", min_value=1e-6, value=1e3, step=1.0, format="%.1e")
    with c8:
        Ea_a_kJ = st.number_input("Anneal Ea (kJ/mol)", min_value=20.0, value=90.0, step=5.0, format="%.0f")
    with c9:
        anneal_target = st.number_input("Anneal target index (≥1)", min_value=0.10, value=1.00, step=0.10, format="%.2f")

    # Heating correction (effective dwell scaling, accounts for thermal mass/entry ramps)
    eff_time_scale = st.slider("Effective dwell scale (heating correction)", 0.50, 1.50, 1.00, 0.05)

    # --- Dwell time ---
    dwell_s = (height_ft / fpm) * 60.0
    t_eff = dwell_s * eff_time_scale

    # --- Solve for T from oxide criterion ---
    # x^2 = k0*exp(-Ea/RT)*t  =>  T = Ea / (R * ln(k0*t/x^2))
    x_m = target_oxide_um * 1e-6
    x2 = x_m**2
    Ea_ox_J = Ea_ox_kJ * 1000.0
    T_ox_K = None
    ox_possible = (k0_ox * t_eff) > x2
    if ox_possible:
        T_ox_K = Ea_ox_J / (R_GAS * math.log((k0_ox * t_eff) / x2))

    # --- Solve for T from anneal severity criterion ---
    # require k_a * t >= anneal_target  => k0_a*exp(-Ea_a/RT)*t >= A  => T = Ea_a / (R * ln(k0_a*t/A))
    Ea_a_J = Ea_a_kJ * 1000.0
    an_possible = (k0_a * t_eff) > max(anneal_target, 1e-12)
    T_an_K = None
    if an_possible:
        T_an_K = Ea_a_J / (R_GAS * math.log((k0_a * t_eff) / anneal_target))

    # --- Choose the higher temperature need (oxide vs anneal) ---
    temps = []
    labels = []
    if T_ox_K is not None and T_ox_K > 0:
        temps.append(T_ox_K)
        labels.append("oxide")
    if T_an_K is not None and T_an_K > 0:
        temps.append(T_an_K)
        labels.append("anneal")

    st.subheader("Results")
    st.write(f"Dwell time (geom): **{dwell_s:,.1f} s**  | Effective dwell (scaled): **{t_eff:,.1f} s**")

    if len(temps) == 0:
        st.error("Given the current constants, the targets are not reachable at any finite temperature (ln term ≤ 0). "
                 "Try reducing target oxide, increasing dwell (slower FPM or taller annealer), or increasing k0 values.")
    else:
        T_need_K = max(temps)
        which = labels[temps.index(T_need_K)]
        T_need_F = k_to_f(T_need_K)
        st.write(f"Required oven set (meets **{which}** target, higher of two): **{T_need_F:,.1f} °F**")

        if T_ox_K:
            st.write(f"  • Oxide-only requirement: **{k_to_f(T_ox_K):,.1f} °F**")
        else:
            st.write("  • Oxide-only requirement: not reachable with current k0/Ea/targets")

        if T_an_K:
            st.write(f"  • Anneal-only requirement: **{k_to_f(T_an_K):,.1f} °F**")
        else:
            st.write("  • Anneal-only requirement: not reachable with current k0/Ea/targets")

    with st.expander("Notes & tuning", expanded=False):
        st.write(
            """
            **Model**  
            • Oxide growth: parabolic \(x^2 = k_0 e^{-E_a/(RT)} t\).  
            • Anneal severity: \(k_a t \ge A\) with \(k_a = k_{0,a} e^{-E_{a,a}/(RT)}\).  
            • We solve each for \(T\) and take the **higher** temperature needed.

            **Units**  
            • \(k_0\) is in **m²/s**, oxide \(x\) in **µm** (converted internally), time in **s**.  
            • Anneal \(k_{0,a}\) is **1/s**; target index \(A\) is dimensionless.

            **Feasibility check**  
            • Expressions require \(\ln(k_0 t / x^2) > 0\) and \(\ln(k_{0,a} t / A) > 0\).  
              If not, the target is **not reachable** with current dwell & constants.

            **Calibration later**  
            • Keep these defaults to start. When you have history (temps, FPM, height, oxide thickness, hardness), 
              we’ll fit \(k_0, E_a, k_{0,a}, E_{a,a}\) and the heating scale.
            """
        )
