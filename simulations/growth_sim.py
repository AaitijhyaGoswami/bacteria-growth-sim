import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from scipy.ndimage import gaussian_filter
import time

def app():
    st.title("Stochastic Bacterial Colony Growth")
    st.markdown("""
    **Simulation Details:**
    * **Model:** Reaction-Diffusion system with stochastic noise.
    * **Dynamics:** Nutrient consumption, metabolic diffusion, and exclusion principles.
    * **Visualization:** Real-time colony morphology and nutrient depletion fields.
    """)

    # -----------------------------
    # 1. PARAMETERS
    # -----------------------------
    st.sidebar.subheader("Physics Parameters")
    FOOD_DIFF = st.sidebar.slider("Food Diffusion", 0.0, 0.02, 0.008, format="%.4f")
    BACT_DIFF = st.sidebar.slider("Bacteria Diffusion", 0.0, 0.05, 0.02, format="%.4f")
    GROWTH_RATE = st.sidebar.slider("Growth Rate", 0.0, 0.1, 0.05, format="%.4f")
    SELF_GROWTH = st.sidebar.slider("Self Growth", 0.0, 0.05, 0.012, format="%.4f")
    FOOD_CONSUMPTION = st.sidebar.slider("Consumption Rate", 0.0, 0.02, 0.006, format="%.4f")
    
    NOISE_STRENGTH = st.sidebar.slider("Stochastic Noise", 0.0, 1.0, 0.65)
    TIP_GROWTH_FACTOR = st.sidebar.slider("Tip Growth Factor", 0.5, 2.0, 1.0)
    
    st.sidebar.subheader("System Settings")
    GRID_SIZE = 300 
    NUM_SEEDS = st.sidebar.slider("Number of Colonies", 1, 12, 12)
    SEED_INTENSITY = 0.03
    STEPS_PER_FRAME = st.sidebar.slider("Simulation Speed", 1, 100, 40)
    
    # PERFORMANCE OPTIMIZATION
    GRAPH_UPDATE_FREQ = 10 # Only update graphs every 10 frames to prevent crash

    # -----------------------------
    # 2. HELPER FUNCTIONS
    # -----------------------------
    def laplacian_interior(arr):
        lap = np.zeros_like(arr)
        lap[1:-1,1:-1] = (
            arr[:-2,1:-1] + arr[2:,1:-1] +
            arr[1:-1,:-2] + arr[1:-1,2:] -
            4 * arr[1:-1,1:-1]
        )
        return lap

    def rgb_to_hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

    # -----------------------------
    # 3. INITIALIZATION
    # -----------------------------
    if 'bg_bacteria' not in st.session_state:
        st.session_state.bg_initialized = False

    def reset_simulation():
        y, x = np.ogrid[-GRID_SIZE/2:GRID_SIZE/2, -GRID_SIZE/2:GRID_SIZE/2]
        mask = x**2 + y**2 <= (GRID_SIZE/2 - 2)**2
        
        bacteria = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
        food = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
        food[mask] = 1.0
        seed_ids = np.zeros_like(bacteria, dtype=int)
        
        np.random.seed(42)
        for seed_id in range(1, NUM_SEEDS+1):
            attempts = 0
            while True:
                r = np.random.randint(10, GRID_SIZE-10)
                c = np.random.randint(10, GRID_SIZE-10)
                attempts += 1
                if mask[r, c] and bacteria[r, c] == 0:
                    bacteria[r, c] = SEED_INTENSITY
                    seed_ids[r, c] = seed_id
                    break
                if attempts > 5000:
                    ys, xs = np.where(mask & (bacteria == 0))
                    if len(ys) == 0: break
                    idx = np.random.randint(len(ys))
                    r, c = ys[idx], xs[idx]
                    bacteria[r, c] = SEED_INTENSITY
                    seed_ids[r, c] = seed_id
                    break

        st.session_state.bg_bacteria = bacteria
        st.session_state.bg_food = food
        st.session_state.bg_seed_ids = seed_ids
        st.session_state.bg_mask = mask
        st.session_state.bg_time = 0
        st.session_state.bg_frame_count = 0
        
        st.session_state.bg_hist_time = []      
        st.session_state.bg_pop_history = []    
        st.session_state.bg_nut_history = [] 
        st.session_state.bg_colony_history = {i: [] for i in range(1, 13)}
        
        st.session_state.bg_initialized = True

    if not st.session_state.bg_initialized:
        reset_simulation()

    if st.sidebar.button("Reset Simulation"):
        reset_simulation()
        st.rerun()

    # -----------------------------
    # 4. LAYOUT
    # -----------------------------
    col1, col2, col3 = st.columns(3)
    placeholder_colony = col1.empty()
    placeholder_nutrient = col2.empty()
    placeholder_biomass = col3.empty()
    
    st.markdown("---")
    col_graph_global, col_graph_local = st.columns(2)
    chart_global_placeholder = col_graph_global.empty()
    chart_local_placeholder = col_graph_local.empty()

    run_sim = st.toggle("Run Simulation", value=False)

    # -----------------------------
    # 5. PHYSICS LOOP
    # -----------------------------
    if run_sim:
        bacteria = st.session_state.bg_bacteria
        food = st.session_state.bg_food
        seed_ids = st.session_state.bg_seed_ids
        mask = st.session_state.bg_mask
        
        for _ in range(STEPS_PER_FRAME):
            # Diffusion
            food += FOOD_DIFF * laplacian_interior(food)
            bacteria += BACT_DIFF * laplacian_interior(bacteria)
            
            # Clamp/Mask
            food = np.clip(food, 0.0, 1.0)
            bacteria = np.clip(bacteria, 0.0, 1.0)
            bacteria[~mask] = 0.0
            
            # Consumption
            consumption = FOOD_CONSUMPTION * bacteria
            food -= consumption
            food = np.clip(food, 0.0, 1.0)
            
            # Growth
            neighbor_sum = (
                np.roll(bacteria, 1, axis=0) + np.roll(bacteria, -1, axis=0) +
                np.roll(bacteria, 1, axis=1) + np.roll(bacteria, -1, axis=1)
            )
            neighbor = neighbor_sum / 4.0
            
            tip_driver = neighbor * (1 - bacteria) * TIP_GROWTH_FACTOR
            noise = np.random.random(bacteria.shape)
            noisy_factor = np.clip(neighbor - NOISE_STRENGTH * (noise - 0.5) + tip_driver, 0.0, 1.0)
            
            local_driver = SELF_GROWTH + (1 - SELF_GROWTH) * noisy_factor
            growth = GROWTH_RATE * bacteria * (1 - bacteria) * local_driver * food
            bacteria += growth
            bacteria = np.clip(bacteria, 0.0, 1.0)
            bacteria[~mask] = 0.0
            
            # Seed Propagation
            for i in range(1, NUM_SEEDS+1):
                neighbors_mask = (
                    np.roll(seed_ids==i, 1, 0) | np.roll(seed_ids==i, -1, 0) |
                    np.roll(seed_ids==i, 1, 1) | np.roll(seed_ids==i, -1, 1)
                )
                seed_ids[(neighbors_mask & (seed_ids==0) & (bacteria>0))] = i

        # Update History
        st.session_state.bg_time += STEPS_PER_FRAME
        st.session_state.bg_frame_count += 1
        
        st.session_state.bg_hist_time.append(st.session_state.bg_time)
        st.session_state.bg_pop_history.append(np.sum(bacteria))
        st.session_state.bg_nut_history.append(np.sum(food))
        
        for i in range(1, NUM_SEEDS+1):
            colony_mass = np.sum(bacteria[seed_ids == i])
            st.session_state.bg_colony_history[i].append(colony_mass)

        # Save State
        st.session_state.bg_bacteria = bacteria
        st.session_state.bg_food = food
        st.session_state.bg_seed_ids = seed_ids

    # -----------------------------
    # 6. VISUALIZATION
    # -----------------------------
    # Retrieve colors
    base_colors = np.array([
        [0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], 
        [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5], [1,0.5,0], [0.5,1,0], [1,0,0.5]
    ])
    
    # 1. Image Update (Every Frame - Fast)
    medium = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=float)
    bacteria = st.session_state.bg_bacteria
    seed_ids = st.session_state.bg_seed_ids
    mask = st.session_state.bg_mask
    food = st.session_state.bg_food
    
    for i in range(1, NUM_SEEDS+1):
        mask_i = (seed_ids == i)
        for c in range(3):
            medium[..., c] += mask_i * bacteria * base_colors[i, c]

    neighbor_sum_vis = (
        np.roll(bacteria, 1, axis=0) + np.roll(bacteria, -1, axis=0) +
        np.roll(bacteria, 1, axis=1) + np.roll(bacteria, -1, axis=1)
    ) / 4.0
    branch_tips = (bacteria > 0) & (neighbor_sum_vis < 0.3)
    halo = gaussian_filter(branch_tips.astype(float), sigma=1.2)
    if halo.max() > 0: halo /= halo.max()
    medium += (halo[..., None] * 0.6) 
    medium = np.clip(medium, 0, 1)
    medium[~mask] = 0.0

    nutr_img = np.zeros((GRID_SIZE, GRID_SIZE, 3))
    nutr_img[..., 1] = food  
    nutr_img[~mask] = 0.0

    bio_img = np.zeros((GRID_SIZE, GRID_SIZE, 3))
    bio_img[..., 0] = bacteria 
    bio_img[..., 2] = bacteria * 0.5 
    bio_img[~mask] = 0.0
    
    placeholder_colony.image(medium, caption=f"Colony Morphology (t={st.session_state.bg_time})", clamp=True, use_column_width=True)
    placeholder_nutrient.image(nutr_img, caption="Nutrient Concentration", clamp=True, use_column_width=True)
    placeholder_biomass.image(bio_img, caption="Biomass Density", clamp=True, use_column_width=True)

    # 2. Graph Update (THROTTLED - Only every N frames to stop crashes)
    if len(st.session_state.bg_pop_history) > 0 and (st.session_state.bg_frame_count % GRAPH_UPDATE_FREQ == 0):
        
        # OPTIMIZATION: Downsample data if too large (Max ~500 points for Altair)
        step_size = max(1, len(st.session_state.bg_hist_time) // 500)
        
        sliced_time = st.session_state.bg_hist_time[::step_size]
        sliced_pop = st.session_state.bg_pop_history[::step_size]
        sliced_nut = st.session_state.bg_nut_history[::step_size]

        # Global Graph
        df_global = pd.DataFrame({
            "Time (mins)": sliced_time,
            "Total Biomass": sliced_pop,
            "Total Nutrient": sliced_nut
        })
        df_global_melt = df_global.melt('Time (mins)', var_name='Metric', value_name='Value')
        
        chart_global = alt.Chart(df_global_melt).mark_line().encode(
            x=alt.X('Time (mins)', title='Time (minutes)'),
            y=alt.Y('Value', title='Quantity (a.u.)'),
            color=alt.Color('Metric', legend=alt.Legend(title="Global Metrics")),
            tooltip=['Time (mins)', 'Metric', 'Value']
        ).properties(title="Global Dynamics").interactive()
        
        chart_global_placeholder.altair_chart(chart_global, use_container_width=True)

        # Local Colony Graph
        data_colony = {"Time (mins)": sliced_time}
        for i in range(1, NUM_SEEDS+1):
            # Slice colony history same way
            data_colony[f"Colony {i}"] = st.session_state.bg_colony_history[i][:len(st.session_state.bg_hist_time)][::step_size]

        df_colony = pd.DataFrame(data_colony)
        df_colony_melt = df_colony.melt('Time (mins)', var_name='Colony', value_name='Biomass')

        color_domain = [f"Colony {i}" for i in range(1, NUM_SEEDS+1)]
        color_range = [rgb_to_hex(base_colors[i]) for i in range(1, NUM_SEEDS+1)]

        chart_local = alt.Chart(df_colony_melt).mark_line().encode(
            x=alt.X('Time (mins)', title='Time (minutes)'),
            y=alt.Y('Biomass', title='Biomass'),
            color=alt.Color('Colony', scale=alt.Scale(domain=color_domain, range=color_range)),
            tooltip=['Time (mins)', 'Colony', 'Biomass']
        ).properties(title="Growth per Colony").interactive()

        chart_local_placeholder.altair_chart(chart_local, use_container_width=True)

    # -----------------------------
    # 7. ANIMATION LOOP
    # -----------------------------
    if run_sim:
        time.sleep(0.01) # Small breather for the CPU
        st.rerun()
