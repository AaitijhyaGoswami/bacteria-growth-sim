import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def app():
    st.title("The MEGA Plate Experiment")
    st.markdown("""
    **Simulation Details:**
    * **Model:** Spatial evolution of bacteria across antibiotic gradients.
    * **Mechanism:** Stochastic mutation and selection.
    * **Zones:** Bacteria must mutate to higher resistance levels (Cyan → Lime → Red) to survive in inner rings.
    """)

    # -----------------------------
    # 1. PARAMETERS (Preserving Constants)
    # -----------------------------
    st.sidebar.subheader("Evolution Parameters")
    
    # Defaults match your script exactly
    MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.0, 0.1, 0.01, format="%.3f")
    REGROW_PROB = st.sidebar.slider("Growth Prob", 0.0, 1.0, 0.2)
    
    st.sidebar.subheader("System Settings")
    SIZE = 100
    CENTER = SIZE // 2
    RADIUS = 45
    MAX_RES_LEVEL = 3
    
    # Speed control
    STEPS_PER_FRAME = st.sidebar.slider("Speed (Iterations/Frame)", 1, 10, 1)

    # -----------------------------
    # 2. INITIALIZATION (Session State)
    # -----------------------------
    if 'mp_grid' not in st.session_state:
        st.session_state.mp_initialized = False

    def reset_simulation():
        # A. Build Antibiotic Map (Exact Logic)
        ab_map = np.ones((SIZE, SIZE)) * 99
        for r in range(SIZE):
            for c in range(SIZE):
                dist = np.sqrt((r-CENTER)**2 + (c-CENTER)**2)
                if dist > RADIUS:
                    ab_map[r,c] = 99 # Wall
                elif dist < RADIUS/3:
                    ab_map[r,c] = 0  # No antibiotic (Center? No, logic says < R/3 is 0)
                    # WAIT: In your script:
                    # dist < R/3 -> 0
                    # dist < 2R/3 -> 1
                    # else -> 2
                    # Actually, usually MegaPlate has 0 on edges and high in center.
                    # Your script puts 0 in CENTER (< R/3). 
                    # So bacteria start in center (0 antibiotic) and move OUT?
                    # Let's check init: bacteria_grid[CENTER-1:...] = 1
                    # So bacteria start in center (zone 0). 
                    # If they move out to Zone 1, they need Level 2 resistance?
                    # Let's stick to YOUR code's logic exactly.
                elif dist < RADIUS/3: 
                    # (This elif is unreachable in your code logic order, 
                    # but let's follow the chain strictly)
                    ab_map[r,c] = 0
                elif dist < 2*RADIUS/3:
                     # Since < R/3 is caught above, this handles [R/3, 2R/3]
                    ab_map[r,c] = 1 
                else:
                    # [2R/3, R]
                    ab_map[r,c] = 2 

        # B. Bacteria Grid
        bac_grid = np.zeros((SIZE, SIZE), dtype=int)
        # Initialize small cluster in center
        bac_grid[CENTER-1:CENTER+2, CENTER-1:CENTER+2] = 1

        # Save to State
        st.session_state.mp_ab_map = ab_map
        st.session_state.mp_grid = bac_grid
        st.session_state.mp_time = 0.0
        
        # History
        st.session_state.mp_hist_time = []
        st.session_state.mp_hist_total = []
        st.session_state.mp_hist_res = [] # List of [n1, n2, n3]
        
        st.session_state.mp_initialized = True

    if not st.session_state.mp_initialized:
        reset_simulation()

    if st.sidebar.button("Reset Simulation"):
        reset_simulation()
        st.rerun()

    # -----------------------------
    # 3. MAIN SIMULATION LOOP
    # -----------------------------
    col_vis, col_stats = st.columns([1, 1])
    
    with col_vis:
        st.write("### Plate View")
        plate_placeholder = st.empty()
        st.caption("Cyan: Lvl 1 | Lime: Lvl 2 | Red: Lvl 3")

    with col_stats:
        st.write("### Population Dynamics")
        chart_total = st.empty()
        chart_frac = st.empty()

    run_sim = st.toggle("Run Simulation", value=False)

    if run_sim:
        bacteria_grid = st.session_state.mp_grid
        antibiotic_map = st.session_state.mp_ab_map
        
        # Loop for speed
        for _ in range(STEPS_PER_FRAME):
            # We must use the Python loop logic to preserve your exact stochastic behavior
            # (Vectorizing this changes the order of operations)
            new_grid = bacteria_grid.copy()
            
            rows, cols = np.where(bacteria_grid > 0)
            if len(rows) > 0:
                idx = np.arange(len(rows))
                np.random.shuffle(idx) # Exact Logic: Randomize update order
                
                for i in idx:
                    r, c = rows[i], cols[i]
                    res_level = bacteria_grid[r,c]

                    # Death Check
                    if (res_level - 1) < antibiotic_map[r,c]:
                        new_grid[r,c] = 0
                        continue

                    # Reproduction Check
                    neighbors = [(r-1,c), (r+1,c), (r,c-1), (r,c+1)]
                    np.random.shuffle(neighbors) # Exact Logic: Random neighbor order
                    
                    for nr, nc in neighbors:
                        if 0 <= nr < SIZE and 0 <= nc < SIZE:
                            # If empty and not a wall
                            if bacteria_grid[nr,nc] == 0 and antibiotic_map[nr,nc] != 99:
                                if np.random.random() < REGROW_PROB:
                                    child = res_level
                                    # Mutation
                                    if np.random.random() < MUTATION_RATE:
                                        child = min(MAX_RES_LEVEL, child + 1)
                                    
                                    # Survival of child in new zone
                                    if (child - 1) >= antibiotic_map[nr,nc]:
                                        new_grid[nr,nc] = child
                                        # Break after one successful reproduction? 
                                        # Your original code loops through all neighbors but usually 
                                        # cellular automata allow one birth per step or fill all?
                                        # In your code: you iterate neighbors. If you place a child, 
                                        # you update 'new_grid'. 
                                        # BUT you don't 'break'. 
                                        # However, since you check 'bacteria_grid[nr,nc]==0',
                                        # and 'bacteria_grid' is static during the loop, 
                                        # a single bacterium CAN reproduce into multiple empty neighbors 
                                        # in one frame.
                                        pass 

            bacteria_grid[:] = new_grid
            
            # Update Stats
            st.session_state.mp_time += 0.1 # HOURS_PER_FRAME
            
            total = np.sum(bacteria_grid > 0)
            c1 = np.sum(bacteria_grid == 1)
            c2 = np.sum(bacteria_grid == 2)
            c3 = np.sum(bacteria_grid == 3)
            
            st.session_state.mp_hist_time.append(st.session_state.mp_time)
            st.session_state.mp_hist_total.append(total)
            
            # For fractions, avoid division by zero
            if total > 0:
                st.session_state.mp_hist_res.append([c1/total, c2/total, c3/total])
            else:
                st.session_state.mp_hist_res.append([0, 0, 0])

        # Save State
        st.session_state.mp_grid = bacteria_grid
        st.rerun()

    # -----------------------------
    # 4. RENDERING
    # -----------------------------
    # A. Build RGB Image for the Plate
    bac_grid = st.session_state.mp_grid
    ab_map = st.session_state.mp_ab_map
    
    # Base: Antibiotic Map (Greyscale)
    # Map 0->0.2, 1->0.5, 2->0.8, 99->0.0
    img = np.zeros((SIZE, SIZE, 3))
    
    # Background layers
    mask_0 = (ab_map == 0)
    mask_1 = (ab_map == 1)
    mask_2 = (ab_map == 2)
    
    img[mask_0] = [0.2, 0.2, 0.2] # Dark Grey
    img[mask_1] = [0.4, 0.4, 0.4] # Medium Grey
    img[mask_2] = [0.6, 0.6, 0.6] # Light Grey
    
    # Overlay Bacteria
    # 1: Cyan [0, 1, 1]
    # 2: Lime [0, 1, 0]
    # 3: Red  [1, 0, 0]
    
    mask_b1 = (bac_grid == 1)
    mask_b2 = (bac_grid == 2)
    mask_b3 = (bac_grid == 3)
    
    img[mask_b1] = [0, 1, 1]
    img[mask_b2] = [0, 1, 0]
    img[mask_b3] = [1, 0, 0]
    
    # Clip just in case
    img = np.clip(img, 0, 1)
    
    # Display Plate
    plate_placeholder.image(img, caption=f"Time: {st.session_state.mp_time:.1f} Hours", use_column_width=True, clamp=True)
    
    # B. Display Charts
    if len(st.session_state.mp_hist_time) > 0:
        # Total Population
        chart_total.line_chart({
            "Total Population": st.session_state.mp_hist_total
        }, height=200)
        
        # Fractions
        # Prepare data for streamlt line chart (requires dict or df)
        # We need to unzip the list of lists
        hist_res = np.array(st.session_state.mp_hist_res)
        if len(hist_res) > 0:
            chart_frac.line_chart({
                "Wildtype": hist_res[:, 0],
                "Medium Res": hist_res[:, 1],
                "Superbug": hist_res[:, 2]
            }, height=200)
