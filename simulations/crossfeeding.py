import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

def app():
    # -----------------------
    # 0. CONFIGURATION
    # -----------------------
    # Fixed Physics Constants
    D_DIFFUSION = 0.15   
    DECAY_RATE = 0.05    
    
    def laplacian(field):
        """Discrete Laplace operator for diffusion"""
        return (
            np.roll(field, 1, axis=0) + 
            np.roll(field, -1, axis=0) + 
            np.roll(field, 1, axis=1) + 
            np.roll(field, -1, axis=1) - 
            4 * field
        )

    st.title("Chemically Mediated Cross-freeding")
    st.markdown("""
    **The Metabolyte Consumption Cycle:**
    1.  <span style='color:#FF4444'>**Producer (A)**</span>: Grows freely. **Secretes Food (X)**.
    2.  <span style='color:#44FF44'>**Consumer (B)**</span>: Eats Food (X). **Secretes Poison (Y)**.
    3.  **Dynamics**: B follows A's metabolyte trail, while A avoids B's poison.
    """, unsafe_allow_html=True)

    # -----------------------
    # 1. PARAMETERS
    # -----------------------
    with st.sidebar:
        st.header("Dynamics Controls")
        
        # PARAMETERS DICTIONARY
        params = {}
        
        params['STEPS'] = st.slider("Simulation Speed", 1, 20, 5)
        params['GRID'] = 300 
        
        st.subheader("Species A (The Producer)")
        params['growth_a'] = st.slider("A Growth Rate (Alpha)", 0.0, 1.0, 0.1, step=0.01)
        params['prod_x'] = st.slider("Production of Food X", 0.0, 1.0, 0.5, step=0.01)
        
        st.subheader("Species B (The Consumer)")
        # FIX: Added format="%.2f" and ensuring clear zero handling
        params['growth_b'] = st.slider("B Efficiency (Delta)", 0.0, 2.0, 0.8, step=0.01)
        params['death_b'] = st.slider("B Starvation Rate (Gamma)", 0.0, 0.1, 0.02, step=0.001, format="%.3f")
        
        st.subheader("Chemical Warfare")
        params['prod_y'] = st.slider("Production of Poison Y", 0.0, 1.0, 0.5, step=0.01)
        params['toxicity'] = st.slider("Lethality of Y (Beta)", 0.0, 2.0, 0.8, step=0.01)

        if st.button("Reset Simulation"):
            st.session_state.cf_initialized = False
            st.rerun()

    # -----------------------
    # 2. INITIALIZATION
    # -----------------------
    if 'cf_initialized' not in st.session_state:
        st.session_state.cf_initialized = False

    def init_simulation():
        GRID_SIZE = 300
        # Grid: 0=Empty, 1=Species A, 2=Species B
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
        
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
    # 3. PURE SIMULATION FUNCTION
    # -----------------------
    def step_simulation(grid, X, Y, p):
        """
        Pure function: takes current state and parameters, returns new state.
        This ensures parameters p are always fresh from the slider.
        """
        
        # --- A. Reaction-Diffusion of Chemicals ---
        
        mask_A = (grid == 1)
        mask_B = (grid == 2)
        
        # 1. Production
        if p['prod_x'] > 0: X += p['prod_x'] * mask_A
        if p['prod_y'] > 0: Y += p['prod_y'] * mask_B
        
        # 2. Diffusion
        X += D_DIFFUSION * laplacian(X)
        Y += D_DIFFUSION * laplacian(Y)
        
        # 3. Decay/Consumption
        # FIX: Link Consumption to Efficiency
        # High Efficiency (growth_b) -> Low Consumption factor
        # Low Efficiency -> High Consumption factor
        # Formula: Base Consumption / (Efficiency + 0.2)
        consumption_factor = 0.2 / (p['growth_b'] + 0.2)
        
        consumption_X = mask_B * X * consumption_factor
        X -= (DECAY_RATE * X) + consumption_X
        Y -= DECAY_RATE * Y
        
        # Clamp
        X = np.clip(X, 0, 10)
        Y = np.clip(Y, 0, 10)

        # --- B. Biological Dynamics (Stochastic) ---
        
        rand_birth = np.random.random(grid.shape)
        rand_death = np.random.random(grid.shape)
        
        # --- DEATH RULES ---
        
        # A dies due to Poison Y
        prob_death_A = p['toxicity'] * Y * 0.1
        kill_A = mask_A & (rand_death < prob_death_A)
        
        # B dies due to starvation
        prob_death_B = p['death_b']
        kill_B = mask_B & (rand_death < prob_death_B)
        
        grid[kill_A] = 0
        grid[kill_B] = 0
        
        # --- BIRTH RULES ---
        
        # Neighborhood check
        shifts = [(0,1), (0,-1), (1,0), (-1,0)]
        sx, sy = shifts[np.random.randint(0, 4)]
        neighbor_grid = np.roll(grid, sx, axis=0)
        neighbor_grid = np.roll(neighbor_grid, sy, axis=1)
        
        mask_empty = (grid == 0)
        
        # Growth of A (Producer)
        if p['growth_a'] > 0:
            birth_A = mask_empty & (neighbor_grid == 1) & (rand_birth < p['growth_a'])
            grid[birth_A] = 1
        
        # Growth of B (Consumer)
        # FIX: Hard check for near-zero values to prevent float overshoot
        if p['growth_b'] > 1e-6:
            # Probability = Efficiency * Concentration of X
            prob_growth_B = p['growth_b'] * X
            birth_B = mask_empty & (neighbor_grid == 2) & (rand_birth < prob_growth_B)
            # B writes over A's birth if collision occurs (Competition advantage to predator)
            grid[birth_B] = 2 
            
        return grid, X, Y

    # -----------------------
    # 4. RUNNER LOOP
    # -----------------------
    col_main, col_plots = st.columns([1.5, 1])
    
    with col_main:
        st.write("### Spatial Species Distribution")
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
        grid = st.session_state.cf_grid
        X = st.session_state.cf_x
        Y = st.session_state.cf_y
        
        # Run steps
        for _ in range(params['STEPS']):
            grid, X, Y = step_simulation(grid, X, Y, params)
            st.session_state.cf_time += 1
            
            # Record Stats sparingly
            if st.session_state.cf_time % 5 == 0:
                hist = st.session_state.cf_hist
                hist["time"].append(st.session_state.cf_time)
                hist["pop_a"].append(int(np.sum(grid == 1)))
                hist["pop_b"].append(int(np.sum(grid == 2)))
        
        # Update State
        st.session_state.cf_grid = grid
        st.session_state.cf_x = X
        st.session_state.cf_y = Y
        st.rerun()
        
    # Construct Image
    grid = st.session_state.cf_grid
    Y = st.session_state.cf_y
    
    img = np.zeros((300, 300, 3))
    
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
