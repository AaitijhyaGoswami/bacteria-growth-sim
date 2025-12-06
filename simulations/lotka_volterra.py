import streamlit as st
import numpy as np
import pandas as pd
import altair as alt


def app():
    st.title("Spatial Lotka-Volterra (Predator-Prey)")
    st.markdown("""
    **Simulation Details:**
    * **Model:** Reaction-Diffusion on a 2D Petri dish.
    * **Dynamics:** Prey (Blue) consume Nutrient (Green); Predators (Red) consume Prey.
    * **Logic:** Finite Difference diffusion with coupled LV reactions.
    """)

    st.sidebar.subheader("Model Parameters")

    # Diffusion
    d_prey = st.sidebar.slider("Diff Prey", 0.0, 0.1, 0.02, format="%.3f")
    d_pred = st.sidebar.slider("Diff Predator", 0.0, 0.1, 0.03, format="%.3f")

    # Lotka–Volterra coefficients
    mu = st.sidebar.slider("Prey Growth (μ)", 0.0, 0.1, 0.05, format="%.3f")
    alpha = st.sidebar.slider("Nutrient Consump (α)", 0.0, 0.1, 0.05, format="%.3f")
    beta = st.sidebar.slider("Predation Rate (β)", 0.0, 0.1, 0.03, format="%.3f")
    gamma = st.sidebar.slider("Predator Eff (γ)", 0.0, 1.0, 0.8, format="%.2f")
    delta = st.sidebar.slider("Predator Death (δ)", 0.0, 0.01, 0.002, format="%.4f")

    st.sidebar.subheader("System Settings")
    grid = 200
    steps_per_frame = st.sidebar.slider("Simulation Speed", 1, 50, 5)

    def laplacian(arr):
        """Discrete 5-point Laplacian for isotropic diffusion."""
        lap = np.zeros_like(arr)
        lap[1:-1, 1:-1] = (
            arr[:-2, 1:-1] + arr[2:, 1:-1] +
            arr[1:-1, :-2] + arr[1:-1, 2:] -
            4 * arr[1:-1, 1:-1]
        )
        return lap

    def seed_colonies(mask, count, radius, intensity):
        """Random circular colonies seeded inside the dish."""
        arr = np.zeros_like(mask, float)
        ys, xs = np.where(mask)
        for _ in range(count):
            if len(ys) == 0:
                break
            idx = np.random.randint(len(ys))
            cy, cx = ys[idx], xs[idx]
            yy, xx = np.ogrid[:grid, :grid]
            dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
            arr[dist2 <= radius ** 2] = intensity
        return arr

    if "lv_prey" not in st.session_state:
        st.session_state.lv_initialized = False

    def reset():
        """Initialize circular petri dish with seeded prey, predator, nutrient."""
        y, x = np.ogrid[-grid / 2:grid / 2, -grid / 2:grid / 2]
        mask = x ** 2 + y ** 2 <= (grid / 2 - 2) ** 2

        np.random.seed(42)
        prey = seed_colonies(mask, 20, 5, 0.5)
        predator = seed_colonies(mask, 10, 4, 0.3)

        nutrient = np.ones((grid, grid))
        nutrient[~mask] = 0

        st.session_state.lv_prey = prey
        st.session_state.lv_predator = predator
        st.session_state.lv_nutrient = nutrient
        st.session_state.lv_mask = mask
        st.session_state.lv_time = 0

        st.session_state.lv_hist_time = []
        st.session_state.lv_hist_prey = []
        st.session_state.lv_hist_pred = []
        st.session_state.lv_hist_nutr = []
        st.session_state.lv_hist_ratio = []

        st.session_state.lv_initialized = True

    if not st.session_state.lv_initialized:
        reset()

    if st.sidebar.button("Reset Simulation"):
        reset()
        st.rerun()

    col_main, col_graph = st.columns([1, 1])

    with col_main:
        st.write("### Petri Dish")
        petri_view = st.empty()

    with col_graph:
        st.write("### Real-time Dynamics")
        chart_pop = st.empty()
        chart_nutr = st.empty()
        chart_ratio = st.empty()

    running = st.toggle("Run Simulation", value=False)

    if running:
        prey = st.session_state.lv_prey
        predator = st.session_state.lv_predator
        nutrient = st.session_state.lv_nutrient
        mask = st.session_state.lv_mask

        for _ in range(steps_per_frame):
            # Diffusion
            prey += d_prey * laplacian(prey)
            predator += d_pred * laplacian(predator)

            # Lotka–Volterra reactions
            dp = mu * prey * nutrient - beta * prey * predator
            dq = gamma * beta * prey * predator - delta * predator
            dnut = -alpha * prey * nutrient

            prey += dp
            predator += dq
            nutrient += dnut

            prey = np.clip(prey, 0, 1)
            predator = np.clip(predator, 0, 1)
            nutrient = np.clip(nutrient, 0, 1)

            prey[~mask] = 0
            predator[~mask] = 0
            nutrient[~mask] = 0

            st.session_state.lv_time += 1

            if st.session_state.lv_time % 5 == 0:
                t = st.session_state.lv_time
                sp, sq = np.sum(prey), np.sum(predator)
                st.session_state.lv_hist_time.append(t)
                st.session_state.lv_hist_prey.append(sp)
                st.session_state.lv_hist_pred.append(sq)
                st.session_state.lv_hist_nutr.append(np.sum(nutrient))
                st.session_state.lv_hist_ratio.append(sq / sp if sp > 0 else 0)

        st.session_state.lv_prey = prey
        st.session_state.lv_predator = predator
        st.session_state.lv_nutrient = nutrient

        st.rerun()

    prey = st.session_state.lv_prey
    predator = st.session_state.lv_predator
    nutrient = st.session_state.lv_nutrient
    mask = st.session_state.lv_mask

    img = np.zeros((grid, grid, 3))
    img[..., 0] = np.clip(predator * 4, 0, 1)
    img[..., 1] = np.clip(nutrient * 4, 0, 1)
    img[..., 2] = np.clip(prey * 4, 0, 1)
    img[~mask] = 0

    petri_view.image(
        img,
        caption=f"Time: {st.session_state.lv_time} mins",
        use_column_width=True,
        clamp=True
    )

    if len(st.session_state.lv_hist_time) > 0:
        df_pop = pd.DataFrame({
            "Time": st.session_state.lv_hist_time,
            "Prey": st.session_state.lv_hist_prey,
            "Predator": st.session_state.lv_hist_pred
        })

        df_pop_melt = df_pop.melt("Time", var_name="Species", value_name="Population")

        pop_plot = (
            alt.Chart(df_pop_melt)
            .mark_line()
            .encode(
                x=alt.X("Time", axis=alt.Axis(title="Time (minutes)")),
                y=alt.Y("Population", axis=alt.Axis(title="Population")),
                color=alt.Color(
                    "Species",
                    scale=alt.Scale(domain=["Prey", "Predator"], range=["blue", "red"])
                )
            )
            .properties(height=200)
        )
        chart_pop.altair_chart(pop_plot, use_container_width=True)

        df_nutr = pd.DataFrame({
            "Time": st.session_state.lv_hist_time,
            "Nutrient": st.session_state.lv_hist_nutr
        })

        nutr_plot = (
            alt.Chart(df_nutr)
            .mark_line(color="green")
            .encode(
                x=alt.X("Time", axis=alt.Axis(title="Time (minutes)")),
                y=alt.Y("Nutrient", axis=alt.Axis(title="Total Nutrient"))
            )
            .properties(height=150)
        )
        chart_nutr.altair_chart(nutr_plot, use_container_width=True)

        df_ratio = pd.DataFrame({
            "Time": st.session_state.lv_hist_time,
            "Ratio": st.session_state.lv_hist_ratio
        })

        ratio_plot = (
            alt.Chart(df_ratio)
            .mark_line(color="orange")
            .encode(
                x=alt.X("Time", axis=alt.Axis(title="Time (minutes)")),
                y=alt.Y("Ratio", axis=alt.Axis(title="Predator/Prey"))
            )
            .properties(height=150)
        )
        chart_ratio.altair_chart(ratio_plot, use_container_width=True)


if __name__ == "__main__":
    app()
