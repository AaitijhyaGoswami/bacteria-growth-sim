import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

def app():
    # -----------------------
    # 0. CONFIGURATION
    # -----------------------
    # Fixed Physics Constants
    D_DIFFUSION = 0.15   # Slower diffusion preserves local waves
    DECAY_RATE = 0.05    # Chemicals break down over time
    
    def laplacian(field):
        """Discrete Laplace operator for diffusion"""
        return (
            np.roll(field, 1, axis=0) + 
            np.roll(field, -1, axis=0) + 
            np.roll(field, 1, axis=1) + 
            np.roll(field, -1, axis=1) - 
            4 * field
        )

    st.title("Chemically Mediated Lotka-Volterra")
    st.markdown("""
    **The "Poison-Excreta" Cycle:**
    This model creates predator-prey oscillations not through direct contact, but through chemical fields.
    
    1.  <span style='color:#FF4444'>**Producer (A)**</span>: Grows freely. **Secretes Food (X)**.
    2.  <span style='color:#44FF44'>**Consumer (B)**</span>: Eats Food (X). **Secretes Poison (Y)**.
    3.  **Dynamics**: B chases A's food trail, while A runs away from B's poison.
    """, unsafe_allow_html=True)

    # -----------------------
    # 1. PARAMETERS
    # -----------------------
    with st.sidebar:
        st.header("Dynamics Controls")
        
        STEPS_PER_FRAME = st.slider("Simulation Speed", 1, 20, 5)
        GRID_SIZE = 120
        
        st.subheader("Species A (The Producer)")
        growth_a = st.slider("A Growth Rate (Alpha)", 0.0, 1.0, 0.1, help="Intrinsic growth rate of A into empty space.")
        prod_x = st.slider("Production of Food X", 0.1, 1.0, 0.5, help="How much food A provides for B.")
        
        st.subheader("Species B (The Consumer)")
        growth_b = st.slider("B Efficiency (Delta)", 0.0, 2.0, 0.8, help="How well B converts Food X into new B cells.")
        death_b = st.slider("B Starvation Rate (Gamma)", 0.0, 0.1, 0.02, help="Rate at which B dies without food.")
        
        st.subheader("Chemical Warfare")
        prod_y = st.slider("Production of Poison Y", 0.1, 1.0, 0.5, help="How much poison B creates.")
        toxicity = st.slider("Lethality of Y (Beta)", 0.0, 2.0, 0.8, help="How effectively Poison Y kills A.")

        if st.button("Reset System"):
            st.session_state.cf_initialized = False
            st.rerun()

    # -----------------------
    # 2. INITIALIZATION
    # -----------------------
    if 'cf_initialized' not in st.session_state:
        st.session_state.cf_initialized = False

    def init_simulation():
        # Grid: 0=Empty, 1=Species A, 2=Species B
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        
        # Random initialization (Salt and Pepper)
        r = np.random.random((GRID_SIZE, GRID_SIZE))
        grid[r < 0.05] = 1 # A
        grid[(r > 0.05) & (r < 0.1)] = 2 # B
        
        # Fields: X (Food), Y (Poison)
        field_x = np.zeros((GRID_SIZE, GRID_SIZE))
        field_y = np.zeros((GRID_SIZE, GRID_SIZE))

        hist = {"time": [], "pop_a": [], "pop_b": []}

        st.session_state.cf_grid = grid
        st.session_state.cf_x = field_x
        st.session_state.cf_y = field_y
        st.session_state.cf_time = 0
        st.session_state.cf_hist = hist
        st.session_state.cf_initialized = True

    if not st.session_state.cf_initialized:
        init_simulation()

    # -----------------------
    # 3. SIMULATION LOGIC
    # -----------------------
    def step_simulation():
        grid = st.session_state.cf_grid
        X = st.session_state.cf_x # Food (made by A)
        Y = st.session_state.cf_y # Poison (made by B)
        
        # --- A. Reaction-Diffusion of Chemicals ---
        
        # 1. Production
        mask_A = (grid == 1)
        mask_B = (grid == 2)
        
        X += prod_x * mask_A # A makes X
        Y += prod_y * mask_B # B makes Y
        
        # 2. Diffusion
        X += D_DIFFUSION * laplacian(X)
        Y += D_DIFFUSION * laplacian(Y)
        
        # 3. Decay/Consumption
        # B consumes X to grow, but we model simple decay + uptake here
        consumption_X = mask_B * X * 0.2
        X -= (DECAY_RATE * X) + consumption_X
        Y -= DECAY_RATE * Y
        
        # Clamp
        X = np.clip(X, 0, 10)
        Y = np.clip(Y, 0, 10)

        # --- B. Biological Dynamics (Stochastic) ---
        
        rand_birth = np.random.random(grid.shape)
        rand_death = np.random.random(grid.shape)
        
        # --- DEATH RULES ---
        
        # A dies due to Poison Y (Predation Term)
        # Probability of death = Toxicity * Concentration of Y
        prob_death_A = toxicity * Y * 0.1
        kill_A = mask_A & (rand_death < prob_death_A)
        
        # B dies due to natural decay (Starvation Term)
        prob_death_B = death_b
        kill_B = mask_B & (rand_death < prob_death_B)
        
        grid[kill_A] = 0
        grid[kill_B] = 0
        
        # --- BIRTH RULES ---
        
        # Shift for neighbor checking (von Neumann neighborhood)
        shifts = [(0,1), (0,-1), (1,0), (-1,0)]
        shift_idx = np.random.randint(0, 4)
        sx, sy = shifts[shift_idx]
        neighbor_grid = np.roll(grid, sx, axis=0)
        neighbor_grid = np.roll(neighbor_grid, sy, axis=1)
        
        mask_empty = (grid == 0)
        
        # Growth of A (Producer)
        # Grows naturally into empty space (Logistic Growth Term)
        birth_A = mask_empty & (neighbor_grid == 1) & (rand_birth < growth_a)
        
        # Growth of B (Consumer)
        # Grows into empty space ONLY if Food X is present (Growth Term)
        # Probability = Efficiency * Concentration of X
        prob_growth_B = growth_b * X
        birth_B = mask_empty & (neighbor_grid == 2) & (rand_birth < prob_growth_B)
        
        # Apply updates
        grid[birth_A] = 1
        grid[birth_B] = 2
        
        # State Update
        st.session_state.cf_grid = grid
        st.session_state.cf_x = X
        st.session_state.cf_y = Y
        st.session_state.cf_time += 1
        
        if st.session_state.cf_time % 5 == 0:
            hist = st.session_state.cf_hist
            hist["time"].append(st.session_state.cf_time)
            hist["pop_a"].append(int(np.sum(grid == 1)))
            hist["pop_b"].append(int(np.sum(grid == 2)))

    # -----------------------
    # 4. VISUALIZATION
    # -----------------------
    col_main, col_plots = st.columns([1.5, 1])
    
    with col_main:
        st.write("### Spatial Battleground")
        legend = """
        <div style='display: flex; gap: 15px; font-size: 14px; margin-bottom:10px;'>
            <div><span style='color:#FF4444; font-size:20px'>■</span> <b>Producer A</b></div>
            <div><span style='color:#44FF44; font-size:20px'>■</span> <b>Consumer B</b></div>
            <div><span style='color:#8888FF; font-size:20px'>☁</span> <b>Poison Y Field</b></div>
        </div>
        """
        st.markdown(legend, unsafe_allow_html=True)
        dish_container = st.empty()

    with col_plots:
        st.write("### Population Cycles")
        chart_container = st.empty()
        
    run_sim = st.toggle("Run Simulation", value=False)
    
    if run_sim:
        for _ in range(STEPS_PER_FRAME):
            step_simulation()
        st.rerun()
        
    # Construct Image
    grid = st.session_state.cf_grid
    Y = st.session_state.cf_y
    
    img = np.zeros((GRID_SIZE, GRID_SIZE, 3))
    
    # Red for A
    img[grid == 1] = [1.0, 0.2, 0.2]
    # Green for B
    img[grid == 2] = [0.2, 1.0, 0.2]
    
    # Visualizing the Poison Cloud (Blue/Purple tint) in empty space
    mask_empty = (grid == 0)
    # Normalize poison for display
    poison_intensity = np.clip(Y / 5.0, 0, 0.8)
    img[mask_empty, 2] = poison_intensity[mask_empty] # Blue channel
    img[mask_empty, 0] = poison_intensity[mask_empty] * 0.5 # Add a bit of red -> Purple
    
    dish_container.image(img, use_column_width=True, clamp=True)
    
    # Chart
    hist = st.session_state.cf_hist
    if len(hist["time"]) > 2:
        df = pd.DataFrame({
            "Time": hist["time"],
            "Producer (A)": hist["pop_a"],
            "Consumer (B)": hist["pop_b"]
        })
        melted = df.melt("Time", var_name="Species", value_name="Population")
        
        c = alt.Chart(melted).mark_line().encode(
            x="Time", 
            y="Population", 
            color=alt.Color("Species", scale=alt.Scale(range=['#44FF44', '#FF4444']))
        ).properties(height=250)
        
        chart_container.altair_chart(c, use_container_width=True)

if __name__ == "__main__":
    app()
