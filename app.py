import streamlit as st

# -------------------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Biophysics Suite | DFOC",
    page_icon-"🧫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------------
# MODULE IMPORTS
# -------------------------------------------------------------------------
# We use underscores because Python cannot import files with hyphens.
modules = {}

try:
    from simulations import growth_sim
    modules["Bacterial Growth"] = growth_sim
except ImportError:
    pass

try:
    from simulations import lotka_volterra
    modules["Lotka-Volterra"] = lotka_volterra
except ImportError:
    pass

try:
    from simulations import mega_plate
    modules["Mega Plate Evolution"] = mega_plate
except ImportError:
    pass

try:
    from simulations import rps_sim
    modules["Rock-Paper-Scissors"] = rps_sim
except ImportError:
    pass

# -------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------------------------
st.sidebar.title("Biophysics Suite")
st.sidebar.markdown("**Databased Fall of Code '25**")

# Define the navigation options
options = ["Home"] + list(modules.keys())
page = st.sidebar.radio("Select Simulation:", options)

st.sidebar.markdown("---")
st.sidebar.caption("Project Team:")
st.sidebar.info(
    "**Aaitijhya Goswami**\n*Simulation & Modeling*\n\n"
    "**Ritaja Dutta**\n*Theoretical Framework*"
)

# -------------------------------------------------------------------------
# MAIN ROUTING
# -------------------------------------------------------------------------
if page == "Home":
    st.title("Computational Biophysics Simulation Suite")
    st.markdown("### Stochastic & Deterministic Modeling of Biological Dynamics")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Welcome.** This interactive dashboard explores the intersection of statistical physics and biology.
        Developed as part of the **Databased Fall of Code** initiative, these modules visualize how simple mathematical rules give rise to complex biological phenomena.
        
        #### The Modules:
        
        **1. Bacterial Growth (growth_sim)**
        * *Concept:* Reaction-Diffusion systems with stochastic noise.
        * *Visuals:* Real-time colony morphology and nutrient depletion fields.
        
        **2. Lotka-Volterra (lotka_volterra)**
        * *Concept:* Coupled differential equations on a spatial grid.
        * *Visuals:* Phase-space dynamics and oscillating populations.
        
        **3. The Mega Plate (mega_plate)**
        * *Concept:* Spatial evolutionary dynamics.
        * *Visuals:* Adaptation of bacteria to increasing antibiotic gradients (inspired by the Kishony Lab).
        
        **4. Rock-Paper-Scissors (rps_sim)**
        * *Concept:* Non-transitive cyclic dominance in ecosystems.
        * *Visuals:* Biodiversity preservation through spatial spiral waves.
        """)
        
        st.info("👈 **Select a simulation from the sidebar to begin.**")

    with col2:
        st.markdown("### Project Status")
        st.success("✅ **Simulations Live**")
        st.warning("📝 **Reports In Progress**")
        st.write("Full theoretical derivations and research reports are currently being drafted.")

else:
    # Run the selected simulation module
    if page in modules:
        modules[page].app()
