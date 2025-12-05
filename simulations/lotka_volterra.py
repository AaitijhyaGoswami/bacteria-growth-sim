import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import time

def app():
    st.title("Spatial Lotka-Volterra (Predator-Prey)")
    st.markdown("""
    **Simulation Details:**
    * **Model:** Reaction-Diffusion on a 2D Petri dish.
    * **Dynamics:** Prey (Blue) consume Nutrient (Green); Predators (Red) consume Prey.
    * **Logic:** Finite Difference Method (Laplacian diffusion).
    """)

    # -----------------------------
    # 1. PARAMETERS
    # -----------------------------
    st.sidebar.subheader("Model Parameters")
    
    # Diffusion coefficients
    D_PREY = st.sidebar.slider("Diff Prey", 0.0, 0.1, 0.02, format="%.3f")
    D_PRED = st.sidebar.slider("Diff Predator", 0.0, 0.1, 0.03, format="%.3f")
    
    # Lotka-Volterra coefficients
    mu = st.sidebar.slider("Prey Growth (μ)", 0.0, 0.1, 0.05, format="%.3f")
    alpha = st.sidebar.slider("Nutrient Consump (α)", 0.0, 0.1, 0.05, format="%.3f")
    beta = st.sidebar.slider("Predation Rate (β)", 0.0, 0.1, 0.03, format="%.3f")
    gamma = st.sidebar.slider("Predator Eff (γ)", 0.0, 1.0, 0.8, format="%.2f")
    delta = st.sidebar.slider("Predator Death (δ)", 0.0, 0.01, 0.002, format="%.4f")

    st.sidebar.subheader("System Settings")
    GRID_SIZE = 200
    STEPS_PER_FRAME = st.sidebar.slider("Simulation Speed", 1, 50, 5)
    
    # Optimization to prevent crashing
    GRAPH_UPDATE_FREQ = 10 

    # -----------------------------
    # 2. HELPER FUNCTIONS
    # -----------------------------
    def laplacian(arr):
        lap = np.zeros_like(arr)
        lap[1:-1,1:-1] = (arr[:-2,1:-1] + arr[2:,1:-1] +
                          arr[1:-1,:-2] + arr[1:-1,2:] -
                          4 * arr[1:-1,1:-1])
        return lap

    def create_colonies(mask, num_colonies, radius, intensity):
        arr = np.zeros_like(mask, dtype=float)
        ys, xs = np.where(mask)
        for _ in range(num_colonies):
            if len(ys) > 0:
                idx = np.random.randint(len(ys))
                cy, cx = ys[idx], xs[idx]
                yy, xx = np.ogrid[:GRID_SIZE, :GRID_SIZE]
                dist_sq = (yy - cy)**2 + (xx - cx)**2
                arr[dist_sq <= radius**2] = intensity
        return arr

    # -----------------------------
    # 3. INITIALIZATION
    # -----------------------------
    if 'lv_prey' not in st.session_state:
        st.session_state.lv_initialized = False

    def reset_simulation():
        y, x = np.ogrid[-GRID_SIZE/2:GRID_SIZE/2, -GRID_SIZE/2:GRID_SIZE/2]
        mask = x**2 + y**2 <= (GRID_SIZE/2 - 2)**2
        
        np.random.seed(42)
        prey = create_colonies(mask, 20, 5, 0.5)
        predator = create_colonies(mask, 10, 4, 0.3)
        nutrient = np.ones((GRID_SIZE, GRID_SIZE))
        nutrient[~mask] = 0

        st.session_state.lv_prey = prey
        st.session_state.lv_predator = predator
        st.session_state.lv_nutrient = nutrient
        st.session_state.lv_mask = mask
        st.session_state.lv_time = 0
        st.session_state.lv_frame_count = 0
        
        # History lists
        st.session_state.lv_hist_time = []
        st.session_state.lv_hist_prey = []
        st.session_state.lv_hist_pred = []
        st.session_state.lv_hist_nutr = []
        st.session_state.lv_hist_ratio = []
        
        st.session_state.lv_initialized = True

    if not st.session_state.lv_initialized:
        reset_simulation()

    if st.sidebar.button("Reset Simulation"):
        reset_simulation()
        st.rerun()

    # -----------------------------
    # 4. LAYOUT
    # -----------------------------
    col_main, col_graphs = st.columns([1, 1])
    
    with col_main:
        st.write("### Petri Dish View")
        petri_placeholder = st.empty()
    
    with col_graphs:
        st.write("### Real-time Dynamics")
        pop_chart = st.empty()
        nutr_chart = st.empty()
        ratio_chart = st.empty()

    run_sim = st.toggle("Run Simulation", value=False)

    # -----------------------------
    # 5. PHYSICS LOOP
    # -----------------------------
    if run_sim:
        prey = st.session_state.lv_prey
        predator = st.session_state.lv_predator
        nutrient = st.session_state.lv_nutrient
        mask = st.session_state.lv_mask
        
        for _ in range(STEPS_PER_FRAME):
            # Diffusion
            prey += D_PREY * laplacian(prey)
            predator += D_PRED * laplacian(predator)
            
            # Reactions
            delta_prey = mu * prey * nutrient - beta * prey * predator
            delta_pred = gamma * beta * prey * predator - delta * predator
            delta_nutrient = -alpha * prey * nutrient
            
            prey += delta_prey
            predator += delta_pred
            nutrient += delta_nutrient
            
            # Clamping
            prey = np.clip(prey, 0, 1)
            predator = np.clip(predator, 0, 1)
            nutrient = np.clip(nutrient, 0, 1)
            
            prey[~mask] = 0
            predator[~mask] = 0
            nutrient[~mask] = 0
            
            st.session_state.lv_time += 1
            
            # Record History (Every 5 steps)
            if st.session_state.lv_time % 5 == 0:
                s_prey = np.sum(prey)
                s_pred = np.sum(predator)
                st.session_state.lv_hist_time.append(st.session_state.lv_time)
                st.session_state.lv_hist_prey.append(s_prey)
                st.session_state.lv_hist_pred.append(s_pred)
                st.session_state.lv_hist_nutr.append(np.sum(nutrient))
                st.session_state.lv_hist_ratio.append(s_pred / s_prey if s_prey > 0 else 0)

        st.session_state.lv_frame_count += 1
        st.session_state.lv_prey = prey
        st.session_state.lv_predator = predator
        st.session_state.lv_nutrient = nutrient

    # -----------------------------
    # 6. VISUALIZATION
    # -----------------------------
    # A. Image Update (Every Frame)
    prey = st.session_state.lv_prey
    predator = st.session_state.lv_predator
    nutrient = st.session_state.lv_nutrient
    mask = st.session_state.lv_mask

    img = np.zeros((GRID_SIZE, GRID_SIZE, 3))
    img[..., 0] = np.clip(predator * 4, 0, 1) # Red (Predator)
    img[..., 1] = np.clip(nutrient * 4, 0, 1) # Green (Nutrient)
    img[..., 2] = np.clip(prey * 4, 0, 1)     # Blue (Prey)
    img[~mask] = 0
    
    petri_placeholder.image(img, caption=f"Time: {st.session_state.lv_time} mins", use_column_width=True, clamp=True)

    # B. Graph Update (Throttled & Altair for Labels)
    if len(st.session_state.lv_hist_time) > 0 and (st.session_state.lv_frame_count % GRAPH_UPDATE_FREQ == 0):
        
        # Downsample to keep it fast
        step_size = max(1, len(st.session_state.lv_hist_time) // 300)
        
        sliced_time = st.session_state.lv_hist_time[::step_size]
        sliced_prey = st.session_state.lv_hist_prey[::step_size]
        sliced_pred = st.session_state.lv_hist_pred[::step_size]
        sliced_nutr = st.session_state.lv_hist_nutr[::step_size]
        sliced_ratio = st.session_state.lv_hist_ratio[::step_size]

        # 1. Biomass Chart
        df_pop = pd.DataFrame({
            "Time": sliced_time,
            "Prey": sliced_prey,
            "Predator": sliced_pred
        })
        df_pop_melt = df_pop.melt('Time', var_name='Species', value_name='Biomass')
        
        chart_pop = alt.Chart(df_pop_melt).mark_line().encode(
            x=alt.X('Time', title='Time (minutes)'),
            y=alt.Y('Biomass', title='Total Biomass'),
            color=alt.Color('Species', scale=alt.Scale(domain=['Prey', 'Predator'], range=['blue', 'red'])),
            tooltip=['Time', 'Species', 'Biomass']
        ).properties(height=200, title="Population Dynamics")
        
        pop_chart.altair_chart(chart_pop, use_container_width=True)

        # 2. Nutrient Chart
        df_nut = pd.DataFrame({"Time": sliced_time, "Nutrient": sliced_nutr})
        chart_nut = alt.Chart(df_nut).mark_line(color='green').encode(
            x=alt.X('Time', title='Time (minutes)'),
            y=alt.Y('Nutrient', title='Nutrient Level'),
            tooltip=['Time', 'Nutrient']
        ).properties(height=150, title="Nutrient Availability")
        
        nutr_chart.altair_chart(chart_nut, use_container_width=True)
        
        # 3. Ratio Chart
        df_ratio = pd.DataFrame({"Time": sliced_time, "Ratio": sliced_ratio})
        chart_ratio = alt.Chart(df_ratio).mark_line(color='purple').encode(
            x=alt.X('Time', title='Time (minutes)'),
            y=alt.Y('Ratio', title='Predator/Prey Ratio'),
            tooltip=['Time', 'Ratio']
        ).properties(height=150, title="Predator-Prey Ratio")
        
        ratio_chart.altair_chart(chart_ratio, use_container_width=True)

    # -----------------------------
    # 7. ANIMATION TRIGGER
    # -----------------------------
    if run_sim:
        time.sleep(0.01) # Prevent CPU hogging
        st.rerun()
