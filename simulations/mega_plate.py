import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

def app():
    st.title("The MEGA Plate Experiment")
    st.markdown("""
    **Simulation Details:**
    * Spatial evolution across antibiotic gradients  
    * Stochastic mutation and selection  
    * Bacteria expand outward and adapt to increasing drug concentrations  
    """)

    st.sidebar.subheader("Evolution Parameters")
    MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.0, 0.1, 0.01, format="%.3f")
    REGROW_PROB = st.sidebar.slider("Growth Prob", 0.0, 1.0, 0.2)

    st.sidebar.subheader("System Settings")
    SIZE = 100
    CENTER = SIZE // 2
    RADIUS = 45
    MAX_RES_LEVEL = 3
    STEPS_PER_FRAME = st.sidebar.slider("Speed (Iterations/Frame)", 1, 10, 2)

    if 'mp_grid' not in st.session_state:
        st.session_state.mp_initialized = False

    def reset_simulation():
        # Build concentric antibiotic zones: center drug-free, middle low, outer high
        ab_map = np.ones((SIZE, SIZE)) * 99
        for r in range(SIZE):
            for c in range(SIZE):
                dist = np.sqrt((r - CENTER) ** 2 + (c - CENTER) ** 2)
                if dist > RADIUS:
                    ab_map[r, c] = 99
                elif dist < RADIUS / 3:
                    ab_map[r, c] = 0
                elif dist < 2 * RADIUS / 3:
                    ab_map[r, c] = 1
                else:
                    ab_map[r, c] = 2

        bac_grid = np.zeros((SIZE, SIZE), dtype=int)
        bac_grid[CENTER - 1:CENTER + 2, CENTER - 1:CENTER + 2] = 1  # Wildtype seed

        st.session_state.mp_ab_map = ab_map
        st.session_state.mp_grid = bac_grid
        st.session_state.mp_time = 0.0

        st.session_state.mp_hist_time = []
        st.session_state.mp_hist_total = []
        st.session_state.mp_hist_res = []

        st.session_state.mp_initialized = True

    if not st.session_state.mp_initialized:
        reset_simulation()

    if st.sidebar.button("Reset Simulation"):
        reset_simulation()
        st.rerun()

    col_vis, col_stats = st.columns([1, 1])

    with col_vis:
        st.write("### Plate View")

        # Small HTML legend for antibiotic zones + genotype colors
        legend_html = """
        <style>
            .legend-container { font-family: sans-serif; font-size: 12px; margin-bottom: 10px; }
            .legend-item { display: inline-flex; align-items: center; margin-right: 15px; }
            .box { width: 12px; height: 12px; border: 1px solid #777; margin-right: 5px; display: inline-block; }
        </style>
        <div class="legend-container">
            <strong>Antibiotic Zones:</strong><br>
            <div class="legend-item"><span class="box" style="background-color: #333333;"></span>None</div>
            <div class="legend-item"><span class="box" style="background-color: #666666;"></span>Low</div>
            <div class="legend-item"><span class="box" style="background-color: #999999;"></span>High</div>
            <br>
            <strong>Bacteria:</strong><br>
            <div class="legend-item"><span class="box" style="background-color: cyan;"></span>Wildtype (1)</div>
            <div class="legend-item"><span class="box" style="background-color: #00ff00;"></span>Mutant (2)</div>
            <div class="legend-item"><span class="box" style="background-color: red;"></span>Superbug (3)</div>
        </div>
        """
        st.markdown(legend_html, unsafe_allow_html=True)
        plate_placeholder = st.empty()

    with col_stats:
        st.write("### Population Dynamics")
        chart_total = st.empty()
        chart_frac = st.empty()

    run_sim = st.toggle("Run Simulation", value=False)

    if run_sim:
        bacteria_grid = st.session_state.mp_grid
        antibiotic_map = st.session_state.mp_ab_map

        for _ in range(STEPS_PER_FRAME):
            new_grid = bacteria_grid.copy()

            rows, cols = np.where(bacteria_grid > 0)
            if len(rows) > 0:
                idx = np.arange(len(rows))
                np.random.shuffle(idx)

                for i in idx:
                    r, c = rows[i], cols[i]
                    res_level = bacteria_grid[r, c]

                    # Survival constraint: resistance must exceed local antibiotic level
                    if (res_level - 1) < antibiotic_map[r, c]:
                        new_grid[r, c] = 0
                        continue

                    # Reproduction into available neighbors
                    neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
                    np.random.shuffle(neighbors)

                    for nr, nc in neighbors:
                        if 0 <= nr < SIZE and 0 <= nc < SIZE:
                            if bacteria_grid[nr, nc] == 0 and antibiotic_map[nr, nc] != 99:
                                if np.random.random() < REGROW_PROB:
                                    child = res_level

                                    # Mutation: increases resistance level by 1
                                    if np.random.random() < MUTATION_RATE:
                                        child = min(MAX_RES_LEVEL, child + 1)

                                    # Child survives only if resistance sufficient
                                    if (child - 1) >= antibiotic_map[nr, nc]:
                                        new_grid[nr, nc] = child

            bacteria_grid[:] = new_grid
            st.session_state.mp_time += 0.1

            total = np.sum(bacteria_grid > 0)
            c1 = np.sum(bacteria_grid == 1)
            c2 = np.sum(bacteria_grid == 2)
            c3 = np.sum(bacteria_grid == 3)

            st.session_state.mp_hist_time.append(st.session_state.mp_time)
            st.session_state.mp_hist_total.append(total)

            if total > 0:
                st.session_state.mp_hist_res.append([c1 / total, c2 / total, c3 / total])
            else:
                st.session_state.mp_hist_res.append([0, 0, 0])

        st.session_state.mp_grid = bacteria_grid
        st.rerun()

    bac_grid = st.session_state.mp_grid
    ab_map = st.session_state.mp_ab_map

    # Visualize antibiotic regions + bacterial genotypes
    img = np.zeros((SIZE, SIZE, 3))

    img[ab_map == 0] = [0.2, 0.2, 0.2]
    img[ab_map == 1] = [0.4, 0.4, 0.4]
    img[ab_map == 2] = [0.6, 0.6, 0.6]

    img[bac_grid == 1] = [0, 1, 1]
    img[bac_grid == 2] = [0, 1, 0]
    img[bac_grid == 3] = [1, 0, 0]

    plate_placeholder.image(
        np.clip(img, 0, 1),
        caption=f"Time: {st.session_state.mp_time:.1f} Hours",
        use_column_width=True,
        clamp=True
    )

    if len(st.session_state.mp_hist_time) > 0:
        df_total = pd.DataFrame({
            'Time': st.session_state.mp_hist_time,
            'Population': st.session_state.mp_hist_total
        })

        c_total = alt.Chart(df_total).mark_line(color='white').encode(
            x=alt.X('Time', axis=alt.Axis(title='Time (Hours)')),
            y=alt.Y('Population', axis=alt.Axis(title='Total Colony Size'))
        ).properties(height=200)

        chart_total.altair_chart(c_total, use_container_width=True)

        hist_res = np.array(st.session_state.mp_hist_res)
        df_frac = pd.DataFrame({
            'Time': st.session_state.mp_hist_time,
            'Wildtype (1)': hist_res[:, 0],
            'Mutant (2)': hist_res[:, 1],
            'Superbug (3)': hist_res[:, 2]
        })

        df_frac_melt = df_frac.melt('Time', var_name='Genotype', value_name='Frequency')

        c_frac = alt.Chart(df_frac_melt).mark_line().encode(
            x=alt.X('Time', axis=alt.Axis(title='Time (Hours)')),
            y=alt.Y('Frequency', axis=alt.Axis(title='Genotype Frequency', format='%')),
            color=alt.Color(
                'Genotype',
                scale=alt.Scale(
                    domain=['Wildtype (1)', 'Mutant (2)', 'Superbug (3)'],
                    range=['cyan', '#00ff00', 'red']
                )
            )
        ).properties(height=200)

        chart_frac.altair_chart(c_frac, use_container_width=True)


if __name__ == "__main__":
    app()
