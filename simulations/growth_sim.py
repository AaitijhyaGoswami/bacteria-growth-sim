import streamlit as st
import numpy as np
from scipy.ndimage import gaussian_filter

def app():
    # -----------------------------
    # 1. PARAMETERS (Matching Original)
    # -----------------------------
    st.sidebar.subheader("Physics Parameters")
    
    # Defaults set exactly to your original script values
    FOOD_DIFF = st.sidebar.slider("Food Diffusion", 0.0, 0.02, 0.008, format="%.4f")
    BACT_DIFF = st.sidebar.slider("Bacteria Diffusion", 0.0, 0.05, 0.02, format="%.4f")
    GROWTH_RATE = st.sidebar.slider("Growth Rate", 0.0, 0.1, 0.05, format="%.4f")
    SELF_GROWTH = st.sidebar.slider("Self Growth", 0.0, 0.05, 0.012, format="%.4f")
    FOOD_CONSUMPTION = st.sidebar.slider("Consumption Rate", 0.0, 0.02, 0.006, format="%.4f")
    
    NOISE_STRENGTH = st.sidebar.slider("Stochastic Noise", 0.0, 1.0, 0.65)
    TIP_GROWTH_FACTOR = st.sidebar.slider("Tip Growth Factor", 0.5, 2.0, 1.0)
    
    st.sidebar.subheader("System Settings")
    # Original Grid Size preserved
    GRID_SIZE = 300 
    NUM_SEEDS = st.sidebar.slider("Number of Colonies", 1, 12, 12)
    SEED_INTENSITY = 0.03
    
    # Speed control for the web loop
    STEPS_PER_FRAME = st.sidebar.slider("Simulation Speed (Steps/Frame)", 1, 100, 40)

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

    # -----------------------------
    # 3. INITIALIZATION (Session State)
    # -----------------------------
    if 'bg_bacteria' not in st.session_state:
        st.session_state.bg_initialized = False

    def reset_simulation():
        y, x = np.ogrid[-GRID_SIZE/2:GRID_SIZE/2, -GRID_SIZE/2:GRID_SIZE/2]
        # Circular mask matching original logic
        mask = x**2 + y**2 <= (GRID_SIZE/2 - 2)**2
        
        bacteria = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
        food = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
        food[mask] = 1.0
        seed_ids = np.zeros_like(bacteria, dtype=int)
        
        # Seeding Logic (Exact Copy)
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
                # Fallback logic if random placement fails
                if attempts > 5000:
                    ys, xs = np.where(mask & (bacteria == 0))
                    if len(ys) == 0: break
                    idx = np.random.randint(len(ys))
                    r, c = ys[idx], xs[idx]
                    bacteria[r, c] = SEED_INTENSITY
                    seed_ids[r, c] = seed_id
                    break

        # Store in session state
        st.session_state.bg_bacteria = bacteria
        st.session_state.bg_food = food
        st.session_state.bg_seed_ids = seed_ids
        st.session_state.bg_mask = mask
        st.session_state.bg_time = 0
        st.session_state.bg_pop_history = []
        st.session_state.bg_nut_history = []
        st.session_state.bg_initialized = True

    # Initialize on first load
    if not st.session_state.bg_initialized:
        reset_simulation()

    # Reset Button
    if st.sidebar.button("Reset Simulation"):
        reset_simulation()
        st.rerun()

    # -----------------------------
    # 4. MAIN SIMULATION LOOP
    # -----------------------------
    st.title("Stochastic Bacterial Colony Growth")
    
    col1, col2, col3 = st.columns(3)
    placeholder_colony = col1.empty()
    placeholder_nutrient = col2.empty()
    placeholder_biomass = col3.empty()
    
    chart_placeholder = st.empty()

    run_sim = st.toggle("Run Simulation", value=False)
    
    if run_sim:
        # Load State
        bacteria = st.session_state.bg_bacteria
        food = st.session_state.bg_food
        seed_ids = st.session_state.bg_seed_ids
        mask = st.session_state.bg_mask
        
        # Run physics steps
        for _ in range(STEPS_PER_FRAME):
            # 1. Diffusion
            food += FOOD_DIFF * laplacian_interior(food)
            bacteria += BACT_DIFF * laplacian_interior(bacteria)
            
            # 2. Clamp and Mask
            food = np.clip(food, 0.0, 1.0)
            bacteria = np.clip(bacteria, 0.0, 1.0)
            bacteria[~mask] = 0.0
            
            # 3. Consumption
            consumption = FOOD_CONSUMPTION * bacteria
            food -= consumption
            food = np.clip(food, 0.0, 1.0)
            
            # 4. Neighbor/Tip Logic
            neighbor_sum = (
                np.roll(bacteria, 1, axis=0) + np.roll(bacteria, -1, axis=0) +
                np.roll(bacteria, 1, axis=1) + np.roll(bacteria, -1, axis=1)
            )
            neighbor = neighbor_sum / 4.0
            
            tip_driver = neighbor * (1 - bacteria) * TIP_GROWTH_FACTOR
            noise = np.random.random(bacteria.shape)
            noisy_factor = np.clip(neighbor - NOISE_STRENGTH * (noise - 0.5) + tip_driver, 0.0, 1.0)
            
            # 5. Growth
            local_driver = SELF_GROWTH + (1 - SELF_GROWTH) * noisy_factor
            growth = GROWTH_RATE * bacteria * (1 - bacteria) * local_driver * food
            bacteria += growth
            bacteria = np.clip(bacteria, 0.0, 1.0)
            bacteria[~mask] = 0.0
            
            # 6. Seed ID Propagation
            for i in range(1, NUM_SEEDS+1):
                neighbors_mask = (
                    np.roll(seed_ids==i, 1, 0) | np.roll(seed_ids==i, -1, 0) |
                    np.roll(seed_ids==i, 1, 1) | np.roll(seed_ids==i, -1, 1)
                )
                seed_ids[(neighbors_mask & (seed_ids==0) & (bacteria>0))] = i

        # Update History
        st.session_state.bg_pop_history.append(np.sum(bacteria))
        st.session_state.bg_nut_history.append(np.sum(food))
        st.session_state.bg_time += STEPS_PER_FRAME

        # Save back to state
        st.session_state.bg_bacteria = bacteria
        st.session_state.bg_food = food
        st.session_state.bg_seed_ids = seed_ids
        
        # Force Rerun for Animation
        st.rerun()

    # -----------------------------
    # 5. RENDERING (Display State)
    # -----------------------------
    # Retrieve current state for rendering
    bacteria = st.session_state.bg_bacteria
    seed_ids = st.session_state.bg_seed_ids
    mask = st.session_state.bg_mask
    food = st.session_state.bg_food

    # A. Colony Color Map (Matching Original logic)
    base_colors = np.array([
        [0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0],
        [1,0,1], [0,1,1], [0.5,0.5,0], [0.5,0,0.5],
        [0,0.5,0.5], [0.8,0.4,0], [0.4,0.8,0], [0.8,0,0.4]
    ])
    
    medium = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=float)
    for i in range(1, NUM_SEEDS+1):
        mask_i = (seed_ids == i)
        for c in range(3):
            medium[..., c] += mask_i * bacteria * base_colors[i, c]

    # Halo calculation (Visual Only)
    neighbor_sum_vis = (
        np.roll(bacteria, 1, axis=0) + np.roll(bacteria, -1, axis=0) +
        np.roll(bacteria, 1, axis=1) + np.roll(bacteria, -1, axis=1)
    ) / 4.0
    branch_tips = (bacteria > 0) & (neighbor_sum_vis < 0.3)
    halo = gaussian_filter(branch_tips.astype(float), sigma=1.2)
    if halo.max() > 0: halo /= halo.max()
    medium += (halo[..., None] * 0.6) # Halo strength matches original

    medium = np.clip(medium, 0, 1)
    medium[~mask] = 0.0

    # B. Nutrient Map
    nutr_img = np.zeros((GRID_SIZE, GRID_SIZE, 3))
    nutr_img[..., 1] = food  # Green channel
    nutr_img[~mask] = 0.0

    # C. Biomass Map (Jet-like logic simplified for RGB)
    bio_img = np.zeros((GRID_SIZE, GRID_SIZE, 3))
    bio_img[..., 0] = bacteria # Red
    bio_img[..., 2] = bacteria * 0.5 
    bio_img[~mask] = 0.0
    
    # Update Placeholders
    placeholder_colony.image(medium, caption=f"Colony (t={st.session_state.bg_time})", clamp=True, use_column_width=True)
    placeholder_nutrient.image(nutr_img, caption="Nutrient", clamp=True, use_column_width=True)
    placeholder_biomass.image(bio_img, caption="Biomass", clamp=True, use_column_width=True)

    # Update Charts
    if len(st.session_state.bg_pop_history) > 0:
        with chart_placeholder:
            st.line_chart({
                "Biomass": st.session_state.bg_pop_history,
                "Nutrient": st.session_state.bg_nut_history
            })
