import os
import math
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Zeus Polyimide Process Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# CONSTANTS AND PHYSICAL PROPERTIES
# ================================
# Copper properties
COPPER_DENSITY_LB_PER_IN3 = 0.323
COPPER_DENSITY_G_PER_CM3 = 8.96
E_COPPER_PSI = 16_000_000  # Young's modulus
NU_COPPER = 0.34  # Poisson's ratio
COPPER_THERMAL_EXPANSION = 16.5e-6  # per °C
COPPER_THERMAL_DIFFUSIVITY = 1.11e-4  # m²/s

# Copper yield stress ranges (annealed)
YIELD_STRESS_ANNEALED_MIN = 10_000  # PSI
YIELD_STRESS_ANNEALED_MAX = 20_000  # PSI
YIELD_STRESS_HARD_MIN = 30_000  # PSI
YIELD_STRESS_HARD_MAX = 45_000  # PSI

# Polyimide properties
PI_DENSITY_G_PER_CM3_DEFAULT = 1.42
PI_DENSITY_LB_PER_IN3_DEFAULT = 0.0513
PI_THERMAL_EXPANSION = 20e-6  # per °C

# Unit conversions
IN3_TO_CM3 = 16.387064
CM3_TO_IN3 = 1.0 / IN3_TO_CM3
G_PER_LB = 453.59237
LB_PER_G = 1.0 / G_PER_LB
MIL_PER_IN = 1000.0
IN_PER_MIL = 1.0 / MIL_PER_IN
FT_PER_IN = 1.0 / 12.0
IN_PER_FT = 12.0

# Mathematical constants
PI = math.pi

# Process constants
DEFAULT_AMBIENT_TEMP_F = 77.0
DEFAULT_HUMIDITY_PCT = 50.0
R_GAS = 8.314  # J/mol-K

# ================================
# UTILITY FUNCTIONS
# ================================
def validate_numeric_input(value, min_val=0, max_val=None, allow_zero=False):
    """Comprehensive input validation with informative error messages"""
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
    """
    Convert AWG to diameter in inches.
    Accepts integers (e.g., 30) or special strings '2/0','3/0','4/0'.
    """
    if isinstance(awg, str):
        special = {"2/0": 0.3648, "3/0": 0.4096, "4/0": 0.4600}
        if awg in special:
            return special[awg]
        try:
            awg = int(awg)
        except Exception:
            return None
    if awg == 36:
        return 0.0050
    if awg == 0:
        return 0.3249
    # Standard AWG formula
    return 0.005 * 92 ** ((36 - awg) / 39)

def diameter_inches_to_awg(diameter):
    """Convert diameter in inches to nearest AWG (approximate)."""
    if diameter <= 0:
        return None
    awg = 36 - 39 * math.log(diameter / 0.005, 92)
    return round(awg)

def calculate_circle_area(diameter):
    """Area of a circle given diameter (in²)."""
    return PI * (diameter ** 2) / 4.0

def calculate_annulus_area(id_in, wall_in):
    """Area of an annulus (ring) given ID and wall thickness (in²)."""
    od_in = id_in + 2.0 * wall_in
    return PI * (od_in ** 2 - id_in ** 2) / 4.0

def calculate_wire_volume(diameter, length_ft):
    """Volume of wire in cubic inches."""
    area = calculate_circle_area(diameter)
    return area * length_ft * IN_PER_FT

def calculate_thermal_strain(temp_change_f, material="copper"):
    """Thermal strain for temperature change (unitless)."""
    temp_change_c = temp_change_f * 5.0 / 9.0
    if material == "copper":
        return COPPER_THERMAL_EXPANSION * temp_change_c
    elif material == "polyimide":
        return PI_THERMAL_EXPANSION * temp_change_c
    return 0.0

def estimate_heating_time(wire_dia_in, target_temp_f, ambient_temp_f=DEFAULT_AMBIENT_TEMP_F):
    """Rough time constant-based estimate for wire heating."""
    wire_dia_m = wire_dia_in * 0.0254
    tau = (wire_dia_m ** 2) / (4 * COPPER_THERMAL_DIFFUSIVITY)
    return 3 * tau  # ~95% of target

# ================================
# ENHANCED CALCULATOR PAGES
# ================================
def runtime_calculator_page():
    """Enhanced runtime calculator with additional metrics"""
    st.title("🕐 Production Runtime Calculator")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        This calculator helps you determine production runtime, throughput, and efficiency metrics 
        for your wire coating process. It accounts for startup, production, and shutdown phases.
        """)

        with st.form("runtime_form"):
            st.subheader("Production Parameters")

            c1, c2, c3 = st.columns(3)
            with c1:
                wire_length_ft = st.number_input(
                    "Wire Length (ft)",
                    min_value=0.0,
                    value=12000.0,
                    step=100.0,
                    help="Total length of wire to process"
                )
            with c2:
                line_speed_fpm = st.number_input(
                    "Line Speed (FPM)",
                    min_value=1.0,
                    value=18.0,
                    step=0.5,
                    help="Feet per minute processing speed"
                )
            with c3:
                efficiency_pct = st.number_input(
                    "Process Efficiency (%)",
                    min_value=50.0,
                    max_value=100.0,
                    value=85.0,
                    step=1.0,
                    help="Account for stops, changeovers, etc."
                )

            st.subheader("Additional Time Factors")

            c4, c5, c6 = st.columns(3)
            with c4:
                startup_min = st.number_input(
                    "Startup Time (min)",
                    min_value=0.0,
                    value=30.0,
                    step=5.0,
                    help="Time to reach steady state"
                )
            with c5:
                shutdown_min = st.number_input(
                    "Shutdown Time (min)",
                    min_value=0.0,
                    value=15.0,
                    step=5.0,
                    help="Cool down and cleaning time"
                )
            with c6:
                passes = st.number_input(
                    "Number of Passes",
                    min_value=1,
                    value=1,
                    step=1,
                    help="For multi-pass coating"
                )

            calculate_btn = st.form_submit_button("Calculate Runtime", type="primary")

        if calculate_btn:
            valid, msg = validate_numeric_input(wire_length_ft, min_val=1)
            if not valid:
                st.error(f"Invalid wire length: {msg}")
                return

            production_time_min = (wire_length_ft * passes) / line_speed_fpm
            effective_production_time_min = production_time_min / (efficiency_pct / 100.0)
            total_time_min = effective_production_time_min + startup_min + shutdown_min
            total_time_hr = total_time_min / 60.0

            effective_speed_fpm = line_speed_fpm * (efficiency_pct / 100.0)
            hourly_throughput_ft = effective_speed_fpm * 60.0
            daily_throughput_ft = hourly_throughput_ft * 8

            st.success("✅ Calculation Complete")

            st.subheader("Runtime Breakdown")
            met_col1, met_col2, met_col3, met_col4 = st.columns(4)
            met_col1.metric("Production Time", f"{production_time_min:.1f} min")
            met_col2.metric("Setup/Shutdown", f"{startup_min + shutdown_min:.0f} min")
            met_col3.metric("Total Runtime", f"{total_time_hr:.2f} hours")
            met_col4.metric("Total with Efficiency", f"{total_time_min:.1f} min")

            st.subheader("Throughput Metrics")
            thr_col1, thr_col2, thr_col3 = st.columns(3)
            thr_col1.metric("Effective Speed", f"{effective_speed_fpm:.1f} FPM")
            thr_col2.metric("Hourly Output", f"{hourly_throughput_ft:,.0f} ft/hr")
            thr_col3.metric("Daily Output (8hr)", f"{daily_throughput_ft:,.0f} ft/day")

    with col2:
        st.subheader("Quick Reference")
        st.info("""
        **Typical Line Speeds:**
        - Fine wire (< 0.010"): 8-15 FPM
        - Medium wire (0.010-0.050"): 15-25 FPM
        - Heavy wire (> 0.050"): 20-40 FPM

        **Efficiency Factors:**
        - New setup: 70-80%
        - Mature process: 85-95%
        - Include changeovers: -10%
        """)

def copper_wire_converter_page():
    """Enhanced copper wire converter with AWG support and visualization"""
    st.title("🔌 Copper Wire Weight Calculator")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("Convert between wire dimensions, AWG sizes, and weight for bare copper wire.")

        input_method = st.radio(
            "Input Method",
            ["Diameter", "AWG"],
            horizontal=True,
            help="Choose how to specify wire size"
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            if input_method == "AWG":
                awg = st.selectbox(
                    "AWG Size",
                    options=list(range(50, -1, -1)) + ["2/0", "3/0", "4/0"],
                    index=20,
                    help="American Wire Gauge standard sizes"
                )
                diameter_in = awg_to_diameter_inches(awg)
                if diameter_in is None:
                    st.error("Invalid AWG")
                    return
                st.info(f"Diameter: {diameter_in:.4f} inches")
            else:
                diameter_in = st.number_input(
                    "Wire Diameter (inches)",
                    min_value=0.0001,
                    max_value=1.0,
                    value=0.0100,
                    step=0.0001,
                    format="%.4f",
                    help="Enter diameter in decimal inches"
                )
                equiv_awg = diameter_inches_to_awg(diameter_in)
                if equiv_awg is not None:
                    st.info(f"Closest AWG: {equiv_awg}")

        with c2:
            length_ft = st.number_input(
                "Wire Length (ft)",
                min_value=0.0,
                value=1000.0,
                step=10.0,
                help="Total length of copper wire"
            )

        with c3:
            copper_density = st.number_input(
                "Copper Density (lb/in³)",
                min_value=0.300,
                max_value=0.350,
                value=COPPER_DENSITY_LB_PER_IN3,
                step=0.001,
                format="%.3f",
                help="Adjust for alloy if needed"
            )

        if st.button("Calculate", type="primary"):
            area_in2 = calculate_circle_area(diameter_in)
            volume_in3 = calculate_wire_volume(diameter_in, length_ft)
            weight_lb = volume_in3 * copper_density
            weight_per_ft = weight_lb / length_ft if length_ft > 0 else 0

            # approximate resistance at 20°C
            resistivity_ohm_cm = 1.72e-6
            resistance_ohms = (resistivity_ohm_cm * length_ft * 30.48) / (area_in2 * 6.4516)

            st.success("✅ Calculation Complete")

            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            res_col1.metric("Wire Weight", f"{weight_lb:.3f} lbs")
            res_col2.metric("Weight per Foot", f"{weight_per_ft:.5f} lb/ft")
            res_col3.metric("Cross Section", f"{area_in2:.6f} in²")
            res_col4.metric("Resistance", f"{resistance_ohms:.3f} Ω")

            with st.expander("Visual Comparison"):
                awg_range = list(range(10, 41, 2))
                diameters = [awg_to_diameter_inches(a) for a in awg_range]
                weights = [calculate_wire_volume(d, 1000) * copper_density for d in diameters]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=awg_range,
                    y=weights,
                    mode='lines',
                    name='Weight per 1000ft',
                    line=dict(width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=[diameter_inches_to_awg(diameter_in)],
                    y=[weight_lb * 1000 / length_ft if length_ft > 0 else 0],
                    mode='markers',
                    name='Your Wire',
                    marker=dict(size=12, symbol='star')
                ))
                fig.update_layout(
                    title='Weight vs AWG Size (per 1000 ft)',
                    xaxis_title='AWG',
                    yaxis_title='Weight (lbs)',
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Quick Reference")
        st.info("""
        **Common AWG Sizes:**
        - AWG 30: 0.0100" dia
        - AWG 26: 0.0159" dia
        - AWG 22: 0.0253" dia
        - AWG 18: 0.0403" dia
        - AWG 14: 0.0641" dia

        **Copper Alloys:**
        - ETP: 0.323 lb/in³
        - OFHC: 0.323 lb/in³
        - Beryllium: 0.298 lb/in³
        """)

def coated_copper_converter_page():
    """Enhanced coated wire calculator with build-up visualization"""
    st.title("🎯 Coated Wire Weight Calculator")

    st.markdown("Calculate weight and dimensions for polyimide-coated copper wire with multi-layer build-up.")

    with st.form("coated_wire_form"):
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            copper_dia_in = st.number_input(
                "Copper Diameter (in)",
                min_value=0.001,
                max_value=0.200,
                value=0.0100,
                step=0.001,
                format="%.4f",
                help="Bare copper wire diameter"
            )

        with c2:
            coating_thickness_mil = st.number_input(
                "Coating per Side (mil)",
                min_value=0.1,
                max_value=10.0,
                value=1.0,
                step=0.1,
                help="Single side coating thickness"
            )

        with c3:
            wire_length_ft = st.number_input(
                "Wire Length (ft)",
                min_value=0.0,
                value=1000.0,
                step=10.0,
                help="Total coated wire length"
            )

        with c4:
            num_layers = st.number_input(
                "Number of Layers",
                min_value=1,
                max_value=20,
                value=1,
                step=1,
                help="Coating layers per side"
            )

        with st.expander("Advanced Parameters"):
            ac1, ac2, ac3 = st.columns(3)
            with ac1:
                pi_density = st.number_input(
                    "Polyimide Density (g/cm³)",
                    min_value=1.0,
                    max_value=2.0,
                    value=PI_DENSITY_G_PER_CM3_DEFAULT,
                    step=0.01,
                    format="%.2f"
                )
            with ac2:
                layer_efficiency = st.number_input(
                    "Layer Efficiency (%)",
                    min_value=80.0,
                    max_value=100.0,
                    value=95.0,
                    step=1.0,
                    help="Accounts for coating compression"
                )
            with ac3:
                scrap_rate_pct = st.number_input(
                    "Scrap Rate (%)",
                    min_value=0.0,
                    max_value=20.0,
                    value=5.0,
                    step=0.5,
                    help="Expected material loss"
                )

        calculate_btn = st.form_submit_button("Calculate Coated Wire Properties", type="primary")

    if calculate_btn:
        coating_per_side_in = (coating_thickness_mil * num_layers * layer_efficiency / 100.0) / MIL_PER_IN
        final_od_in = copper_dia_in + 2 * coating_per_side_in

        copper_volume_in3 = calculate_wire_volume(copper_dia_in, wire_length_ft)
        total_volume_in3 = calculate_wire_volume(final_od_in, wire_length_ft)
        coating_volume_in3 = total_volume_in3 - copper_volume_in3

        copper_weight_lb = copper_volume_in3 * COPPER_DENSITY_LB_PER_IN3
        coating_volume_cm3 = coating_volume_in3 * IN3_TO_CM3
        coating_weight_g = coating_volume_cm3 * pi_density
        coating_weight_lb = coating_weight_g * LB_PER_G
        total_weight_lb = copper_weight_lb + coating_weight_lb

        total_with_scrap_lb = total_weight_lb * (1 + scrap_rate_pct / 100.0)
        linear_density_lb_per_ft = total_weight_lb / wire_length_ft if wire_length_ft > 0 else 0

        st.success("✅ Calculation Complete")

        st.subheader("Wire Dimensions")
        dim_col1, dim_col2, dim_col3, dim_col4 = st.columns(4)
        dim_col1.metric("Starting OD", f"{copper_dia_in:.4f} in")
        dim_col2.metric("Coating Build", f"{coating_per_side_in*MIL_PER_IN:.1f} mil/side")
        dim_col3.metric("Final OD", f"{final_od_in:.4f} in")
        dim_col4.metric("OD Increase", f"{(final_od_in/copper_dia_in - 1)*100:.1f}%")

        st.subheader("Weight Breakdown")
        wt_col1, wt_col2, wt_col3, wt_col4 = st.columns(4)
        wt_col1.metric("Copper Weight", f"{copper_weight_lb:.3f} lbs")
        wt_col2.metric("Coating Weight", f"{coating_weight_lb:.3f} lbs")
        wt_col3.metric("Total Weight", f"{total_weight_lb:.3f} lbs")
        wt_col4.metric("With Scrap", f"{total_with_scrap_lb:.3f} lbs")

        with st.expander("Coating Build-Up Visualization"):
            fig = go.Figure()
            theta = np.linspace(0, 2*np.pi, 100)

            r_copper = copper_dia_in / 2
            x_copper = r_copper * np.cos(theta)
            y_copper = r_copper * np.sin(theta)
            fig.add_trace(go.Scatter(
                x=x_copper, y=y_copper,
                fill='toself',
                fillcolor='rgba(184, 115, 51, 0.5)',
                line=dict(),
                name='Copper Core',
                hovertemplate='Copper: %{text}<extra></extra>',
                text=[f'Dia: {copper_dia_in:.4f}"'] * len(theta)
            ))

            for i in range(1, num_layers + 1):
                layer_radius = copper_dia_in/2 + (coating_thickness_mil * i * layer_efficiency/100.0)/MIL_PER_IN
                x_layer = layer_radius * np.cos(theta)
                y_layer = layer_radius * np.sin(theta)
                opacity = 0.3 + 0.5 * (i / num_layers)
                fig.add_trace(go.Scatter(
                    x=x_layer, y=y_layer,
                    fill='toself',
                    fillcolor=f'rgba(255, 165, 0, {opacity})',
                    line=dict(dash='dot' if i < num_layers else 'solid'),
                    name=f'Layer {i}',
                    hovertemplate='Layer %{text}<extra></extra>',
                    text=[f'{i}: Dia {layer_radius*2:.4f}"'] * len(theta)
                ))

            fig.update_layout(
                title='Cross-Section View (Not to Scale)',
                xaxis=dict(scaleanchor='y', scaleratio=1, showgrid=False, visible=False),
                yaxis=dict(showgrid=False, visible=False),
                height=400,
                showlegend=True,
                hovermode='closest'
            )

            st.plotly_chart(fig, use_container_width=True)

def wire_stretch_predictor_page():
    """Enhanced wire stretch predictor with thermal effects"""
    st.title("📏 Wire Elongation Predictor")

    st.markdown("""
    Predict wire elongation under tension, including elastic deformation, thermal expansion, 
    and onset of plastic yield. Critical for maintaining dimensional control.
    """)

    c1, c2, c3 = st.columns(3)

    with c1:
        wire_dia_in = st.number_input(
            "Wire Diameter (in)",
            min_value=0.001,
            max_value=0.200,
            value=0.0100,
            step=0.001,
            format="%.4f"
        )

    with c2:
        tension_lb = st.number_input(
            "Applied Tension (lb)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.5
        )

    with c3:
        wire_length_ft = st.number_input(
            "Wire Length (ft)",
            min_value=1.0,
            max_value=10000.0,
            value=100.0,
            step=10.0
        )

    with st.expander("Temperature and Material Properties"):
        t1, t2, t3 = st.columns(3)
        with t1:
            operating_temp_f = st.number_input(
                "Operating Temp (°F)",
                min_value=0.0,
                max_value=1200.0,
                value=800.0,
                step=10.0
            )
        with t2:
            ambient_temp_f = st.number_input(
                "Ambient Temp (°F)",
                min_value=0.0,
                max_value=120.0,
                value=77.0,
                step=1.0
            )
        with t3:
            wire_condition = st.selectbox(
                "Wire Condition",
                ["Annealed", "Half-Hard", "Hard"],
                help="Affects yield strength"
            )

    if st.button("Predict Elongation", type="primary"):
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
            yield_min = YIELD_STRESS_ANNEALED_MIN
            yield_max = YIELD_STRESS_ANNEALED_MAX
        elif wire_condition == "Half-Hard":
            yield_min = (YIELD_STRESS_ANNEALED_MAX + YIELD_STRESS_HARD_MIN) / 2
            yield_max = (YIELD_STRESS_ANNEALED_MAX + YIELD_STRESS_HARD_MAX) / 2
        else:
            yield_min = YIELD_STRESS_HARD_MIN
            yield_max = YIELD_STRESS_HARD_MAX

        radial_strain = -NU_COPPER * elastic_strain
        dia_reduction_mil = abs(radial_strain * wire_dia_in * MIL_PER_IN)

        st.subheader("Elongation Results")
        e_col1, e_col2, e_col3, e_col4 = st.columns(4)
        e_col1.metric("Applied Stress", f"{stress_psi:,.0f} PSI")
        e_col2.metric("Elastic Elongation", f"{elastic_elongation_in:.4f} in")
        e_col3.metric("Thermal Elongation", f"{thermal_elongation_in:.4f} in")
        e_col4.metric("Total Elongation", f"{total_elongation_in:.4f} in")

        st.subheader("Secondary Effects")
        s_col1, s_col2, s_col3 = st.columns(3)
        s_col1.metric("Percent Elongation", f"{percent_elongation:.3f}%")
        s_col2.metric("Diameter Reduction", f"{dia_reduction_mil:.3f} mil")
        s_col3.metric("Volume Conservation", "✓ Maintained")

        if stress_psi > yield_max:
            st.error(f"""
            ⚠️ **CRITICAL: Exceeds yield strength!**
            - Applied stress: {stress_psi:,.0f} PSI
            - Maximum yield: {yield_max:,.0f} PSI
            - Wire will experience permanent deformation
            - Reduce tension immediately
            """)
        elif stress_psi > yield_min:
            st.warning(f"""
            ⚠️ **WARNING: Approaching yield point**
            - Applied stress: {stress_psi:,.0f} PSI
            - Yield range: {yield_min:,.0f} - {yield_max:,.0f} PSI
            - Risk of permanent deformation
            """)
        else:
            safety_factor = yield_min / stress_psi if stress_psi > 0 else float('inf')
            st.success(f"✅ Operating safely with {safety_factor:.1f}x safety factor")

        with st.expander("Stress-Strain Visualization"):
            strain_range = np.linspace(0, 0.002, 100)
            stress_range = E_COPPER_PSI * strain_range

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=strain_range * 100,
                y=stress_range,
                mode='lines',
                name='Elastic Region',
                line=dict(width=2)
            ))
            fig.add_trace(go.Scatter(
                x=[elastic_strain * 100],
                y=[stress_psi],
                mode='markers',
                name='Operating Point',
                marker=dict(size=12, symbol='star')
            ))
            fig.add_hrect(
                y0=yield_min, y1=yield_max,
                fillcolor="orange", opacity=0.2,
                line_width=0,
                annotation_text="Yield Zone"
            )
            fig.update_layout(
                title='Stress-Strain Relationship',
                xaxis_title='Strain (%)',
                yaxis_title='Stress (PSI)',
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

def paa_usage_page():
    """Enhanced PAA usage calculator with batch management"""
    st.title("🧪 PAA Solution Requirements Calculator")

    st.markdown("""
    Calculate the precise amount of polyamic acid solution needed for your production run, 
    including startup losses, heel volume, and safety factors.
    """)

    with st.form("paa_calculator"):
        st.subheader("Product Specifications")

        p1, p2, p3 = st.columns(3)
        with p1:
            target_id_in = st.number_input(
                "Target ID (in)",
                min_value=0.001,
                max_value=0.200,
                value=0.0160,
                step=0.001,
                format="%.4f"
            )
        with p2:
            wall_thickness_mil = st.number_input(
                "Wall Thickness (mil)",
                min_value=0.1,
                max_value=10.0,
                value=1.0,
                step=0.1
            )
        with p3:
            finished_length_ft = st.number_input(
                "Finished Length (ft)",
                min_value=0.0,
                value=1500.0,
                step=10.0
            )

        st.subheader("Solution Properties")

        s1, s2, s3 = st.columns(3)
        with s1:
            paa_solids_pct = st.number_input(
                "PAA Solids (%)",
                min_value=5.0,
                max_value=40.0,
                value=15.0,
                step=0.5
            )
        with s2:
            solution_density = st.number_input(
                "Solution Density (g/mL)",
                min_value=0.8,
                max_value=1.5,
                value=1.06,
                step=0.01,
                format="%.2f"
            )
        with s3:
            viscosity_cp = st.number_input(
                "Viscosity (cP)",
                min_value=100.0,
                max_value=10000.0,
                value=2500.0,
                step=100.0
            )

        st.subheader("Process Losses")

        l1, l2, l3, l4 = st.columns(4)
        with l1:
            startup_scrap_ft = st.number_input(
                "Startup Scrap (ft)",
                min_value=0.0,
                value=150.0,
                step=10.0
            )
        with l2:
            shutdown_scrap_ft = st.number_input(
                "Shutdown Scrap (ft)",
                min_value=0.0,
                value=50.0,
                step=10.0
            )
        with l3:
            holdup_volume_ml = st.number_input(
                "System Holdup (mL)",
                min_value=0.0,
                value=400.0,
                step=10.0
            )
        with l4:
            heel_volume_ml = st.number_input(
                "Tank Heel (mL)",
                min_value=0.0,
                value=120.0,
                step=10.0
            )

        safety_factor = st.slider(
            "Safety Factor",
            min_value=0.0,
            max_value=0.30,
            value=0.10,
            step=0.01,
            format="%.0%"
        )

        calculate_btn = st.form_submit_button("Calculate PAA Requirements", type="primary")

    if calculate_btn:
        wall_in = wall_thickness_mil / MIL_PER_IN
        coating_volume_in3 = calculate_annulus_area(target_id_in, wall_in) * finished_length_ft * IN_PER_FT
        coating_volume_cm3 = coating_volume_in3 * IN3_TO_CM3

        pi_weight_g = coating_volume_cm3 * PI_DENSITY_G_PER_CM3_DEFAULT

        solution_for_product_g = pi_weight_g / (paa_solids_pct / 100.0)
        solution_for_product_ml = solution_for_product_g / solution_density

        total_scrap_ft = startup_scrap_ft + shutdown_scrap_ft
        scrap_fraction = total_scrap_ft / finished_length_ft if finished_length_ft > 0 else 0
        solution_for_scrap_ml = solution_for_product_ml * scrap_fraction

        base_solution_ml = solution_for_product_ml + solution_for_scrap_ml
        total_solution_ml = base_solution_ml + holdup_volume_ml + heel_volume_ml
        total_with_safety_ml = total_solution_ml * (1 + safety_factor)

        total_liters = total_with_safety_ml / 1000.0
        total_gallons = total_liters * 0.264172

        st.success("✅ Calculation Complete")

        st.subheader("PAA Solution Requirements")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Product Solution", f"{solution_for_product_ml:.0f} mL")
        r2.metric("Scrap Solution", f"{solution_for_scrap_ml:.0f} mL")
        r3.metric("Process Losses", f"{holdup_volume_ml + heel_volume_ml:.0f} mL")
        r4.metric("Safety Buffer", f"{total_solution_ml * safety_factor:.0f} mL")

        st.subheader("Total Requirements")
        t1, t2, t3 = st.columns(3)
        t1.metric("Total Volume", f"{total_liters:.2f} Liters")
        t2.metric("Total Volume", f"{total_gallons:.2f} Gallons")
        t3.metric("Total Weight", f"{total_with_safety_ml * solution_density:.0f} grams")

        with st.expander("Cost Estimation"):
            cost_per_liter = st.number_input(
                "PAA Cost ($/Liter)",
                min_value=0.0,
                value=150.0,
                step=10.0
            )
            total_cost = total_liters * cost_per_liter
            cost_per_ft = total_cost / finished_length_ft if finished_length_ft > 0 else 0
            c1, c2 = st.columns(2)
            c1.metric("Total Material Cost", f"${total_cost:,.2f}")
            c2.metric("Cost per Foot", f"${cost_per_ft:.4f}")

# ================================
# ANNEALING TEMPERATURE ESTIMATOR (Upgraded kNN + Ridge Backbone)
# ================================
def anneal_temp_estimator_page():
    st.title("🔥 Advanced Annealing Temperature Estimator")

    st.markdown("""
    Uses k-Nearest Neighbors on **log-space features** (ln diameter, ln dwell) with **z-score standardization**,
    **Gaussian weighting**, and a small **ridge backbone** for stability when you're in sparse regions.
    Outliers are flagged and (optionally) excluded.
    """)

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
            # Temperature sanity
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

    data_file = "annealing_dataset_clean.csv"
    df = load_and_clean_data(data_file)
    if df is None or len(df) < 8:
        st.error("Not enough valid data to build a model.")
        return

    with st.expander("📊 Data Quality Overview"):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", len(df))
        col2.metric("Clean Records", len(df[~df['is_outlier']]))
        col3.metric("Outliers Detected", int(df['is_outlier'].sum()))
        col4.metric("Temperature Range", f"{df['anneal_temp_f'].min():.0f}-{df['anneal_temp_f'].max():.0f}°F")
        fig = px.box(df, y='anneal_temp_f', x='is_outlier',
                     labels={'anneal_temp_f': 'Temperature (°F)', 'is_outlier': 'Is Outlier'},
                     title='Temperature Distribution with Outliers Identified')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Process Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        wire_dia = st.number_input("Wire Diameter (in)", min_value=0.001, max_value=0.200, value=0.0250, step=0.001, format="%.4f")
    with col2:
        speed = st.number_input("Line Speed (FPM)", min_value=1.0, max_value=100.0, value=18.0, step=0.5)
    with col3:
        height = st.number_input("Annealer Height (ft)", min_value=1.0, max_value=50.0, value=14.0, step=1.0)

    st.subheader("Model Configuration")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        include_outliers = st.checkbox("Include Outliers", value=False, help="Include outlier rows in the neighbor set.")
    clean_df = df if include_outliers else df[~df['is_outlier']]

    with m2:
        max_k = min(60, len(clean_df))
        default_k = min(35, max_k) if max_k >= 35 else max_k
        k_neighbors = st.slider("Number of Neighbors (k)", min_value=3, max_value=max_k, value=default_k, help="Larger k reduces variance.")
    with m3:
        weight_method = st.selectbox("Weighting Method", ["Gaussian", "Distance", "Uniform"], index=0,
                                     help="Gaussian is smoothest; Distance is 1/(d+ε); Uniform is equal weights.")
    with m4:
        oor_gate = st.slider("Out-of-Range Gate (p75 dist)", min_value=0.5, max_value=3.0, value=1.25, step=0.05,
                             help="If the 75th-percentile neighbor distance (in z-scored log-space) exceeds this, treat as out-of-range.")

    if st.button("🔮 Predict Temperature", type="primary"):
        if len(clean_df) < k_neighbors:
            st.error(f"Not enough rows after filtering for k={k_neighbors}. Have {len(clean_df)}.")
            return

        # ===== Features in LOG SPACE =====
        Xd = np.log(clean_df['wire_dia_in'].values)
        Xdw = np.log(clean_df['dwell_s'].values)
        X = np.column_stack([Xd, Xdw]).astype(float)
        y = clean_df['anneal_temp_f'].values.astype(float)

        # Z-score standardization
        mu = X.mean(axis=0)
        sd = X.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        Z = (X - mu) / sd

        # Query point
        dwell_s = (height / speed) * 60.0
        q_raw = np.array([math.log(wire_dia), math.log(dwell_s)], dtype=float)
        qz = (q_raw - mu) / sd

        # Neighbors in standardized space
        nn = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
        nn.fit(Z)
        distances, indices = nn.kneighbors(qz.reshape(1, -1))
        D = distances.flatten()
        I = indices.flatten()

        neigh_temps = y[I]

        # ===== Weights =====
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

        # ===== Tiny Ridge Backbone on same features =====
        # Solve (X'X + λI)β = X'y  for β on [1, ln_d, ln_dw]
        lam = 10.0
        Xg = np.column_stack([np.ones(len(X)), X])   # note: X already log features
        G = Xg.T @ Xg + lam * np.eye(Xg.shape[1])
        beta = np.linalg.solve(G, Xg.T @ y)
        T_ridge = float(beta[0] + beta[1]*q_raw[0] + beta[2]*q_raw[1])

        # ===== Out-of-range test (based on local distances) =====
        pr75 = float(np.percentile(D, 75))
        out_of_range = pr75 > oor_gate

        # Blend knn with backbone
        w_local = 0.75 if not out_of_range else 0.50
        T_mix = w_local * T_knn + (1.0 - w_local) * T_ridge

        # Final clip
        T_final = float(np.clip(T_mix, 650.0, 1200.0))

        # ===== Output =====
        st.success("✅ Prediction Complete")
        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        res_col1.metric("Predicted Temperature", f"{T_final:.1f} °F")
        res_col2.metric("k-NN Core", f"{T_knn:.1f} °F")
        res_col3.metric("Ridge Backbone", f"{T_ridge:.1f} °F")
        res_col4.metric("Dwell Time", f"{dwell_s:.1f} s")

        conf = "High" if not out_of_range else "Lower (out-of-range)"
        st.caption(
            f"Distance stats (z-space): median={np.median(D):.2f}  |  p75={pr75:.2f}  |  out_of_range={out_of_range}  |  k={k_neighbors}  |  weighting={weight_method}"
        )
        if out_of_range:
            st.warning("⚠️ Parameters sit outside the dense region of your history. Prediction relies more on the global backbone.")

        with st.expander("📈 Analysis Details"):
            # Neighbor table
            neighbor_display = clean_df.iloc[I][['wire_dia_in', 'speed_fpm', 'annealer_ht_ft', 'dwell_s', 'anneal_temp_f']].copy()
            neighbor_display['Distance'] = D
            neighbor_display['Weight'] = w
            neighbor_display = neighbor_display.rename(columns={
                'wire_dia_in': 'Dia (in)',
                'speed_fpm': 'Speed (FPM)',
                'annealer_ht_ft': 'Height (ft)',
                'dwell_s': 'Dwell (s)',
                'anneal_temp_f': 'Temp (°F)'
            })
            st.dataframe(
                neighbor_display.style.format({
                    'Dia (in)': '{:.4f}',
                    'Speed (FPM)': '{:.1f}',
                    'Height (ft)': '{:.0f}',
                    'Dwell (s)': '{:.1f}',
                    'Temp (°F)': '{:.0f}',
                    'Distance': '{:.3f}',
                    'Weight': '{:.3f}'
                }).background_gradient(subset=['Weight'], cmap='YlOrRd')
            )

            # Contribution plot
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(range(1, len(neigh_temps) + 1)),
                y=neigh_temps,
                text=[f"{t:.0f}°F<br>w={wt:.3f}" for t, wt in zip(neigh_temps, w)],
                textposition='outside',
                marker_color=w,
                marker_colorscale='YlOrRd',
                showlegend=False
            ))
            fig.add_hline(y=T_final, line_dash="dash", annotation_text=f"Predicted: {T_final:.1f}°F")
            fig.update_layout(
                title='Neighbor Contributions to Prediction',
                xaxis_title='Neighbor Rank',
                yaxis_title='Temperature (°F)',
                height=380
            )
            st.plotly_chart(fig, use_container_width=True)

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
        index=5,  # default to Anneal estimator
    )

    st.markdown("""
    <style>
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin: 5px 0; }
    </style>
    """, unsafe_allow_html=True)

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
    **Zeus Polyimide Process Suite v2.1**

    Upgrades:
    - kNN on log-space (ln d, ln dwell)
    - Z-score standardization
    - Gaussian weighting with adaptive bandwidth
    - Ridge backbone blending when out-of-range
    - Outlier QA and diagnostics

    © 2024 Zeus Industrial Products
    """)

if __name__ == "__main__":
    main()
