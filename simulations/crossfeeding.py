import streamlit as st
import numpy np
import pandas as pd
import altair as alt

def app():
    # CONSTANTS
    D_DIFFUSION = 0.15
    DECAY_RATE = 0.05

    def laplacian(field):
        # simple 2D laplacian, periodic BCs
        return (
            np.roll(field, 1, axis=0) +
            np.roll(field, -1, axis=0) +
            np.roll(field, 1, axis=1) +
            np.roll(field, -1, axis=1) -
            4 * field
        )

    st.title("Chemically Mediated Cross-freeding")
    st.markdown("""
    **The Metabolyte Cycle:**
    1. <span style='color:#FF4444'>**A**</span> grows and releases food **X**.
    2. <span style='color:#44FF44'>**B**</span> consumes **X** and releases poison **Y**.
    3. B tends to chase X, A avoids Y.
    """, unsafe_allow_html=True)


    # SIDEBAR
    with st.sidebar:
        st.header("Dynamics Controls")

        params = {}
        params['STEPS'] = st.slider("Simulation Speed", 1, 20, 5)
        params['GRID'] = 300

        st.subheader("Species A")
        params['growth_a'] = st.slider("A Growth Rate", 0.0, 1.0, 0.1, step=0.01)
        params['prod_x'] = st.slider("X Production", 0.0, 1.0, 0.5, step=0.01)

        st.subheader("Species B")
        params['growth_b'] = st.slider("B Efficiency", 0.0, 2.0, 0.8, step=0.01)
        params['death_b'] = st.slider("Starvation Rate", 0.0, 0.1, 0.02, step=0.001, format="%.3f")

        st.subheader("Chemical Interaction")
        params['prod_y'] = st.slider("Poison Y Production", 0.0, 1.0, 0.5, step=0.01)
        params['toxicity'] = st.slider("Y Toxicity", 0.0, 2.0, 0.8, step=0.01)

        if st.button("Reset Simulation"):
            st.session_state.cf_initialized = False
            st.rerun()


    # INIT
    if 'cf_initialized' not in st.session_state:
        st.session_state.cf_initialized = False

    def init_simulation():
        GRID_SIZE = 300
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)

        r = np.random.random((GRID_SIZE, GRID_SIZE))
        grid[r < 0.05] = 1
        grid[(r > 0.05) & (r < 0.1)] = 2

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


    # STEP UPDATE
    def step_simulation(grid, X, Y, p):
        mask_A = (grid == 1)
        mask_B = (grid == 2)

        # secretion
        if p['prod_x'] > 0: X += p['prod_x'] * mask_A
        if p['prod_y'] > 0: Y += p['prod_y'] * mask_B

        # diffusion + decay
        X += D_DIFFUSION * laplacian(X)
        Y += D_DIFFUSION * laplacian(Y)

        consumption_factor = 0.2 / (p['growth_b'] + 0.2)
        X -= (DECAY_RATE * X) + mask_B * X * consumption_factor
        Y -= DECAY_RATE * Y

        X = np.clip(X, 0, 10)
        Y = np.clip(Y, 0, 10)

        rand_birth = np.random.random(grid.shape)
        rand_death = np.random.random(grid.shape)

        # A death
        prob_death_A = p['toxicity'] * Y * 0.1
        grid[mask_A & (rand_death < prob_death_A)] = 0

        # B death
        grid[mask_B & (rand_death < p['death_b'])] = 0

        # pick neighbor
        sx, sy = [(0,1),(0,-1),(1,0),(-1,0)][np.random.randint(0,4)]
        nbr = np.roll(np.roll(grid, sx, axis=0), sy, axis=1)

        empty = (grid == 0)

        # A reproduction
        if p['growth_a'] > 0:
            birth_A = empty & (nbr == 1) & (rand_birth < p['growth_a'])
            grid[birth_A] = 1

        # B reproduction
        if p['growth_b'] > 1e-6:
            prob_B = p['growth_b'] * X
            birth_B = empty & (nbr == 2) & (rand_birth < prob_B)
            grid[birth_B] = 2

        return grid, X, Y


    # LAYOUT
    col_main, col_plots = st.columns([1.5, 1])

    with col_main:
        st.write("### Spatial Distribution")
        st.markdown("""
        <div style='display:flex;gap:15px;font-size:14px;margin-bottom:10px;'>
            <div><span style='color:#FF4444;font-size:20px'>■</span> A</div>
            <div><span style='color:#44FF44;font-size:20px'>■</span> B</div>
            <div><span style='color:#8888FF;font-size:20px'>☁</span> Y Field</div>
        </div>
        """, unsafe_allow_html=True)
        dish_container = st.empty()

    with col_plots:
        st.write("### Population Cycles")
        chart_container = st.empty()

    run_sim = st.toggle("Run Simulation", value=False)


    # MAIN LOOP
    if run_sim:
        grid = st.session_state.cf_grid
        X = st.session_state.cf_x
        Y = st.session_state.cf_y

        for _ in range(params['STEPS']):
            grid, X, Y = step_simulation(grid, X, Y, params)
            st.session_state.cf_time += 1

            if st.session_state.cf_time % 5 == 0:
                h = st.session_state.cf_hist
                h["time"].append(st.session_state.cf_time)
                h["pop_a"].append(int(np.sum(grid == 1)))
                h["pop_b"].append(int(np.sum(grid == 2)))

        st.session_state.cf_grid = grid
        st.session_state.cf_x = X
        st.session_state.cf_y = Y
        st.rerun()


    # RENDER
    grid = st.session_state.cf_grid
    Y = st.session_state.cf_y

    img = np.zeros((300, 300, 3))
    img[grid == 1] = [1.0, 0.2, 0.2]
    img[grid == 2] = [0.2, 1.0, 0.2]

    empty = (grid == 0)
    poison_intensity = np.clip(Y / 5.0, 0, 0.8)
    img[empty, 2] = poison_intensity[empty]
    img[empty, 0] = poison_intensity[empty] * 0.5

    dish_container.image(img, use_column_width=True, clamp=True)


    # POPULATION PLOT
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
