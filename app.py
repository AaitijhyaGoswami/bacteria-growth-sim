
# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, Normalize
from scipy.ndimage import gaussian_filter
import time

st.set_page_config(layout="wide", page_title="Live Simulation Gallery")
st.title("🧫 Live Simulation Gallery (Matplotlib → Streamlit)")

st.sidebar.header("Choose simulation")
sim_choice = st.sidebar.selectbox("Simulation", [
    "Growth (multi-colony)", "Lotka-Volterra (spatial)", "Mega-Plate", "RPS (rock-paper-scissors)"
])

st.sidebar.markdown("---")
st.sidebar.write("General controls")
run_btn = st.sidebar.button("Start / Restart")
stop_btn = st.sidebar.button("Stop")

# session state for control
if "running" not in st.session_state:
    st.session_state.running = False
if "stop" not in st.session_state:
    st.session_state.stop = False

if run_btn:
    st.session_state.running = True
    st.session_state.stop = False

if stop_btn:
    st.session_state.stop = True
    st.session_state.running = False

# Layout: left column: controls/info, right: visualization
left, right = st.columns([1, 3])

with left:
    st.markdown("### Simulation parameters")
    # dictionary of per-sim params
    if sim_choice == "Growth (multi-colony)":
        GRID_SIZE = st.number_input("Grid size", min_value=80, max_value=400, value=150, step=10)
        STEPS = st.number_input("Steps", min_value=200, max_value=10000, value=2000, step=100)
        VIS_INTERVAL = st.number_input("Frames per update (VIS_INTERVAL)", min_value=1, max_value=200, value=20)
        NUM_SEEDS = st.number_input("Num colonies", min_value=2, max_value=40, value=12)
    elif sim_choice == "Lotka-Volterra (spatial)":
        GRID_SIZE = st.number_input("Grid size", min_value=80, max_value=400, value=140, step=10)
        STEPS = st.number_input("Steps", min_value=200, max_value=5000, value=1200, step=100)
        VIS_INTERVAL = st.number_input("Frames per update (VIS_INTERVAL)", min_value=1, max_value=200, value=5)
    elif sim_choice == "Mega-Plate":
        SIZE = st.number_input("Grid SIZE", min_value=50, max_value=200, value=100, step=10)
        STEPS = st.number_input("Frames (approx)", min_value=50, max_value=2000, value=500, step=50)
        INTERVAL_MS = st.number_input("Interval ms (UI)", min_value=10, max_value=2000, value=100, step=10)
        MUTATION_RATE = st.slider("Mutation rate", 0.0, 0.2, 0.01)
    elif sim_choice == "RPS (rock-paper-scissors)":
        GRID = st.number_input("Grid size", min_value=100, max_value=300, value=160, step=10)
        STEPS = st.number_input("Steps", min_value=100, max_value=2000, value=600, step=50)
        VIS_INTERVAL = st.number_input("Frames per update", min_value=1, max_value=200, value=2)

    st.markdown("---")
    st.write("Tips:")
    st.write("- `Stop` will halt the running simulation.")
    st.write("- Increase VIS_INTERVAL or reduce grid size for faster updates in-browser.")
    st.write("- For heavy runs, run locally (not Streamlit Cloud) for best performance.")

with right:
    vis_placeholder = st.empty()
    info_placeholder = st.empty()

# Helper: check for stop
def should_stop():
    return st.session_state.stop

# -----------------------------
# Simulation functions
# -----------------------------
def sim_growth_streamlit(grid_size=150, steps=2000, vis_interval=20, num_seeds=12):
    # Adapted from growth-sim.py but with smaller defaults for web
    GRID_SIZE = grid_size
    STEPS = steps
    VIS_INTERVAL = vis_interval
    NUM_SEEDS = num_seeds

    # parameters (kept modest)
    FOOD_DIFF = 0.008
    BACT_DIFF = 0.02
    GROWTH_RATE = 0.05
    SELF_GROWTH = 0.012
    FOOD_CONSUMPTION = 0.006
    NOISE_STRENGTH = 0.65
    TIP_GROWTH_FACTOR = 1.0
    SEED_INTENSITY = 0.03

    y, x = np.ogrid[-GRID_SIZE/2:GRID_SIZE/2, -GRID_SIZE/2:GRID_SIZE/2]
    mask = x**2 + y**2 <= (GRID_SIZE/2 - 2)**2

    bacteria = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
    food = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
    food[mask] = 1.0

    np.random.seed(42)
    seed_ids = np.zeros_like(bacteria, dtype=int)
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

    base_colors = np.array([
        [0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0],
        [1,0,1], [0,1,1], [0.5,0.5,0], [0.5,0,0.5],
        [0,0.5,0.5], [0.8,0.4,0], [0.4,0.8,0], [0.8,0,0.4]
    ])

    def laplacian_interior(arr):
        lap = np.zeros_like(arr)
        lap[1:-1,1:-1] = (
            arr[:-2,1:-1] + arr[2:,1:-1] +
            arr[1:-1,:-2] + arr[1:-1,2:] -
            4 * arr[1:-1,1:-1]
        )
        return lap

    # histories
    pop_history = []
    nutrient_history = []
    per_colony_history = {i: [] for i in range(1, NUM_SEEDS+1)}
    per_colony_consumed = {i: [] for i in range(1, NUM_SEEDS+1)}
    time_history = []

    # prepare figure
    fig = plt.figure(figsize=(14, 7))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1,1])
    ax_medium = fig.add_subplot(gs[0,0]); ax_nutr = fig.add_subplot(gs[0,1]); ax_bio = fig.add_subplot(gs[0,2])
    ax_global = fig.add_subplot(gs[1,0]); ax_percol = fig.add_subplot(gs[1,1]); ax_cons = fig.add_subplot(gs[1,2])

    ax_medium.axis('off'); ax_nutr.axis('off'); ax_bio.axis('off')
    ax_global.set_title("Global"); ax_percol.set_title("Per-colony"); ax_cons.set_title("Consumption")
    line_biomass, = ax_global.plot([], [], 'k-'); line_nutr, = ax_global.plot([], [], 'g-')

    percol_lines = {}
    cons_lines = {}
    for i in range(1, NUM_SEEDS+1):
        (ln,) = ax_percol.plot([], [], lw=1, color=base_colors[i], label=f"C{i}")
        percol_lines[i] = ln
        (cn,) = ax_cons.plot([], [], lw=1, color=base_colors[i], label=f"C{i}")
        cons_lines[i] = cn
    ax_percol.legend(fontsize='xx-small', ncol=2)
    ax_cons.legend(fontsize='xx-small', ncol=2)

    # main loop
    for t in range(STEPS):
        if should_stop():
            return

        food_prev = food.copy()
        food += FOOD_DIFF * laplacian_interior(food)
        bacteria += BACT_DIFF * laplacian_interior(bacteria)
        food = np.clip(food, 0, 1)
        bacteria = np.clip(bacteria, 0, 1)
        bacteria[~mask] = 0

        consumption_by_cells = FOOD_CONSUMPTION * bacteria
        food -= consumption_by_cells
        food = np.clip(food, 0, 1)

        neighbor_sum = (np.roll(bacteria,1,0)+np.roll(bacteria,-1,0)+np.roll(bacteria,1,1)+np.roll(bacteria,-1,1))
        neighbor = neighbor_sum/4.0
        tip_driver = neighbor * (1 - bacteria) * TIP_GROWTH_FACTOR
        noise = np.random.random(bacteria.shape)
        noisy_factor = np.clip(neighbor - NOISE_STRENGTH * (noise - 0.5) + tip_driver, 0.0, 1.0)
        local_driver = SELF_GROWTH + (1 - SELF_GROWTH) * noisy_factor
        growth = GROWTH_RATE * bacteria * (1 - bacteria) * local_driver * food
        bacteria += growth
        bacteria = np.clip(bacteria, 0, 1)
        bacteria[~mask] = 0

        for i in range(1, NUM_SEEDS+1):
            neighbors = (
                np.roll(seed_ids==i,1,0) | np.roll(seed_ids==i,-1,0) |
                np.roll(seed_ids==i,1,1) | np.roll(seed_ids==i,-1,1)
            )
            seed_ids[(neighbors & (seed_ids==0) & (bacteria>0))] = i

        branch_tips = (bacteria > 0) & (neighbor < 0.3)

        total_biomass = np.sum(bacteria)
        total_nutrient = np.sum(food)
        pop_history.append(total_biomass); nutrient_history.append(total_nutrient); time_history.append(t)

        delta_food = np.clip(food_prev - food, 0.0, None)
        for i in range(1, NUM_SEEDS+1):
            mask_i = (seed_ids == i)
            per_colony_history[i].append(np.sum(bacteria[mask_i]))
            cons_i = np.sum(delta_food[mask_i])
            prev = per_colony_consumed[i][-1] if per_colony_consumed[i] else 0
            per_colony_consumed[i].append(prev + cons_i)

        if t % VIS_INTERVAL == 0:
            medium = np.zeros((GRID_SIZE, GRID_SIZE, 3))
            for i in range(1, NUM_SEEDS+1):
                mask_i = (seed_ids == i)
                for c in range(3):
                    medium[..., c] += mask_i * bacteria * base_colors[i, c]
            halo = gaussian_filter(branch_tips.astype(float), sigma=1.0)
            if halo.max() > 0: halo /= halo.max()
            medium += halo[..., None] * 0.55
            medium = np.clip(medium, 0, 1)
            medium[~mask] = 0

            ax_medium.clear(); ax_nutr.clear(); ax_bio.clear()
            ax_medium.imshow(medium); ax_medium.axis('off'); ax_medium.set_title("Bacterial Colonies")
            ax_nutr.imshow(food, cmap='Greens', vmin=0, vmax=1); ax_nutr.axis('off'); ax_nutr.set_title("Nutrient")
            ax_bio.imshow(bacteria, cmap='jet', vmin=0, vmax=1); ax_bio.axis('off'); ax_bio.set_title("Biomass")

            line_biomass.set_data(time_history, pop_history); line_nutr.set_data(time_history, nutrient_history)
            ax_global.relim(); ax_global.autoscale_view()

            for i in range(1, NUM_SEEDS+1):
                percol_lines[i].set_data(time_history, per_colony_history[i])
                cons_lines[i].set_data(time_history, per_colony_consumed[i])
            ax_percol.relim(); ax_percol.autoscale_view(); ax_cons.relim(); ax_cons.autoscale_view()

            vis_placeholder.pyplot(fig)
            plt.pause(0.001)

    st.success("Growth simulation finished.")

def sim_lv_streamlit(grid_size=140, steps=1200, vis_interval=5):
    GRID_SIZE = grid_size
    STEPS = steps
    VIS_INTERVAL = vis_interval

    # parameters (kept moderate)
    D_PREY = 0.02; D_PRED = 0.03
    mu = 0.05; alpha = 0.05; beta = 0.03; gamma = 0.8; delta = 0.002

    y, x = np.ogrid[-GRID_SIZE/2:GRID_SIZE/2, -GRID_SIZE/2:GRID_SIZE/2]
    mask = x**2 + y**2 <= (GRID_SIZE/2 - 2)**2

    def create_colonies(mask, num_colonies, radius, intensity):
        arr = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
        ys, xs = np.where(mask)
        for _ in range(num_colonies):
            idx = np.random.randint(len(ys))
            cy, cx = ys[idx], xs[idx]
            yy, xx = np.ogrid[:GRID_SIZE, :GRID_SIZE]
            arr[(yy - cy)**2 + (xx - cx)**2 <= radius**2] = intensity
        return arr

    np.random.seed(42)
    prey = create_colonies(mask, 18, 4, 0.5)
    predator = create_colonies(mask, 9, 3, 0.3)
    nutrient = np.ones((GRID_SIZE, GRID_SIZE)); nutrient[~mask] = 0

    time_hist = []; prey_hist = []; pred_hist = []; nutr_hist = []; ratio_hist = []

    def laplacian(arr):
        lap = np.zeros_like(arr)
        lap[1:-1,1:-1] = (arr[:-2,1:-1] + arr[2:,1:-1] + arr[1:-1,:-2] + arr[1:-1,2:] - 4 * arr[1:-1,1:-1])
        return lap

    fig = plt.figure(figsize=(12,8))
    gs = GridSpec(2,2, figure=fig)
    ax_species = fig.add_subplot(gs[0,0]); ax_pop = fig.add_subplot(gs[0,1])
    ax_nutr = fig.add_subplot(gs[1,0]); ax_ratio = fig.add_subplot(gs[1,1])
    ax_species.axis('off')

    l_prey, = ax_pop.plot([], [], 'b-'); l_pred, = ax_pop.plot([], [], 'r-')
    l_nutr, = ax_nutr.plot([], [], 'g-'); l_ratio, = ax_ratio.plot([], [], 'm-')

    for t in range(STEPS):
        if should_stop():
            return

        prey += D_PREY * laplacian(prey)
        predator += D_PRED * laplacian(predator)

        delta_prey = mu * prey * nutrient - beta * prey * predator
        delta_pred = gamma * beta * prey * predator - delta * predator
        delta_nutrient = -alpha * prey * nutrient

        prey += delta_prey; predator += delta_pred; nutrient += delta_nutrient

        prey = np.clip(prey, 0, 1); predator = np.clip(predator, 0, 1); nutrient = np.clip(nutrient, 0, 1)
        prey[~mask] = 0; predator[~mask] = 0; nutrient[~mask] = 0

        if t % 5 == 0:
            time_hist.append(t); prey_hist.append(np.sum(prey)); pred_hist.append(np.sum(predator))
            nutr_hist.append(np.sum(nutrient)); ratio_hist.append(np.sum(predator) / np.sum(prey) if np.sum(prey)>0 else 0)

        if t % VIS_INTERVAL == 0:
            img = np.zeros((GRID_SIZE, GRID_SIZE, 3))
            img[..., 2] = np.clip(prey * 4, 0, 1)
            img[..., 0] = np.clip(predator * 4, 0, 1)
            img[..., 1] = np.clip(nutrient * 4, 0, 1)
            img[~mask] = 0
            ax_species.clear(); ax_species.imshow(img); ax_species.set_title("Prey (blue), Pred (red), Nut (green)"); ax_species.axis('off')

            l_prey.set_data(time_hist, prey_hist); l_pred.set_data(time_hist, pred_hist)
            l_nutr.set_data(time_hist, nutr_hist); l_ratio.set_data(time_hist, ratio_hist)
            ax_pop.relim(); ax_pop.autoscale_view(); ax_nutr.relim(); ax_nutr.autoscale_view(); ax_ratio.relim(); ax_ratio.autoscale_view()

            vis_placeholder.pyplot(fig)
            plt.pause(0.001)

    st.success("Lotka–Volterra simulation finished.")

def sim_mega_streamlit(size=100, steps=500, interval_ms=100, mutation_rate=0.01):
    SIZE = size
    CENTER = SIZE // 2
    RADIUS = int(SIZE * 0.45)
    MUTATION_RATE = mutation_rate
    REGROW_PROB = 0.2
    MAX_RES_LEVEL = 3

    antibiotic_map = np.ones((SIZE,SIZE))*99
    for r in range(SIZE):
        for c in range(SIZE):
            dist = np.sqrt((r-CENTER)**2 + (c-CENTER)**2)
            if dist > RADIUS:
                antibiotic_map[r,c] = 99
            elif dist < RADIUS/3:
                antibiotic_map[r,c] = 0
            elif dist < 2*RADIUS/3:
                antibiotic_map[r,c] = 1
            else:
                antibiotic_map[r,c] = 2

    bacteria_grid = np.zeros((SIZE,SIZE), dtype=int)
    bacteria_grid[CENTER-1:CENTER+2, CENTER-1:CENTER+2] = 1

    time_points = []; total_counts = []; res_counts = []

    fig, axes = plt.subplots(2,2, figsize=(10,8))
    ax_main = axes[0,0]; ax_total = axes[0,1]; ax_res = axes[1,0]; ax_ab = axes[1,1]
    cmap = ListedColormap(['black','cyan','lime','red'])
    norm = Normalize(vmin=0, vmax=MAX_RES_LEVEL)
    im = ax_main.imshow(bacteria_grid, cmap=cmap, norm=norm, origin='lower'); ax_main.axis('off')

    from matplotlib.patches import Patch
    ax_main.legend(handles=[
        Patch(color='cyan', label='Wildtype'),
        Patch(color='lime', label='Medium'),
        Patch(color='red', label='Superbug')], loc='upper right', fontsize=8)

    circle = plt.Circle((CENTER,CENTER), RADIUS/3, color='white', fill=False, linestyle='--')
    ax_main.add_patch(circle)

    line_total, = ax_total.plot([], [])
    lines_res = [ax_res.plot([], [], color=c)[0] for c in ['cyan','lime','red']]

    ab_map = np.zeros((SIZE,SIZE)); ab_map[antibiotic_map==0]=0.2; ab_map[antibiotic_map==1]=0.5; ab_map[antibiotic_map==2]=0.8
    ax_ab.imshow(ab_map, cmap='Greys', origin='lower'); ax_ab.axis('off')

    def step(frame):
        nonlocal bacteria_grid
        new_grid = bacteria_grid.copy()
        rows, cols = np.where(bacteria_grid>0)
        idx = np.arange(len(rows)); np.random.shuffle(idx)
        for i in idx:
            r,c = rows[i], cols[i]
            res_level = bacteria_grid[r,c]
            if (res_level-1) < antibiotic_map[r,c]:
                new_grid[r,c] = 0
                continue
            neighbors = [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]
            np.random.shuffle(neighbors)
            for nr,nc in neighbors:
                if 0<=nr<SIZE and 0<=nc<SIZE:
                    if bacteria_grid[nr,nc]==0 and antibiotic_map[nr,nc]!=99:
                        if np.random.random() < REGROW_PROB:
                            child = res_level
                            if np.random.random() < MUTATION_RATE:
                                child = min(MAX_RES_LEVEL, child+1)
                            if (child-1) >= antibiotic_map[nr,nc]:
                                new_grid[nr,nc] = child
        bacteria_grid[:] = new_grid

    for t in range(steps):
        if should_stop(): return
        step(t)
        time_h = t * (interval_ms/1000.0)
        time_points.append(time_h)
        total_counts.append(np.sum(bacteria_grid>0))
        counts_level = [(bacteria_grid==lvl).sum() for lvl in range(1,MAX_RES_LEVEL+1)]
        res_counts.append(counts_level)

        if t % 1 == 0:
            im.set_data(bacteria_grid)
            res_array = np.array(res_counts) / np.array(total_counts)[:,None]
            ax_total.clear(); ax_res.clear()
            ax_total.plot(time_points, total_counts); ax_total.set_title("Total count")
            for i,l in enumerate(lines_res):
                ax_res.plot(time_points, res_array[:,i], color=['cyan','lime','red'][i])
            ax_res.set_ylim(0,1)
            vis_placeholder.pyplot(fig)
            plt.pause(0.001)

    st.success("Mega-plate simulation finished.")

def sim_rps_streamlit(grid=160, steps=600, vis_interval=2):
    GRID = grid
    STEPS = steps
    VIS_INTERVAL = vis_interval

    EMPTY = 0; RED = 1; BLUE = 2; GREEN = 3
    INIT_RED = 0.02; INIT_BLUE = 0.02; INIT_GREEN = 0.02
    SPREAD_RATE = 0.25; EAT_PROB = 0.45

    yy, xx = np.indices((GRID, GRID))
    center = GRID // 2
    radius = GRID // 2 - 2
    mask = (xx - center)**2 + (yy - center)**2 <= radius**2

    grid_arr = np.zeros((GRID, GRID), dtype=int)
    rand = np.random.rand(GRID, GRID)
    grid_arr[(rand < INIT_RED) & mask] = RED
    grid_arr[(rand >= INIT_RED) & (rand < INIT_RED + INIT_BLUE) & mask] = BLUE
    grid_arr[(rand >= INIT_RED + INIT_BLUE) & (rand < INIT_RED + INIT_BLUE + INIT_GREEN) & mask] = GREEN

    red_counts, blue_counts, green_counts = [], [], []

    cmap = ListedColormap(["black", "#FF3333", "#3366FF", "#33FF33"])

    fig = plt.figure(figsize=(12,8))
    gs = fig.add_gridspec(2, 3, height_ratios=[2,1])
    ax_dish = fig.add_subplot(gs[0,0]); ax_frac = fig.add_subplot(gs[0,1:]); ax_r = fig.add_subplot(gs[1,0]); ax_b = fig.add_subplot(gs[1,1]); ax_g = fig.add_subplot(gs[1,2])
    im = ax_dish.imshow(np.where(mask, grid_arr, 0), cmap=cmap, vmin=0, vmax=3)
    ax_dish.add_patch(plt.Circle((center, center), radius, fill=False, edgecolor='white', linewidth=2))
    ax_dish.set_title("Petri Dish"); ax_dish.set_xticks([]); ax_dish.set_yticks([])

    line_frac_red, = ax_frac.plot([], [], color="#FF3333"); line_frac_blue, = ax_frac.plot([], [], color="#3366FF"); line_frac_green, = ax_frac.plot([], [], color="#33FF33")
    line_r, = ax_r.plot([], [], color="#FF3333"); line_b, = ax_b.plot([], [], color="#3366FF"); line_g, = ax_g.plot([], [], color="#33FF33")

    def update_grid(grid_local):
        new = grid_local.copy()
        dx = np.random.randint(-1, 2, size=(GRID, GRID))
        dy = np.random.randint(-1, 2, size=(GRID, GRID))
        for x in range(GRID):
            for y in range(GRID):
                if not mask[x, y]:
                    new[x, y] = EMPTY
                    continue
                s = grid_local[x, y]
                if s == EMPTY: continue
                nx = (x + dx[x, y]) % GRID
                ny = (y + dy[x, y]) % GRID
                if not mask[nx, ny]: continue
                t = grid_local[nx, ny]
                if t == EMPTY and np.random.rand() < SPREAD_RATE:
                    new[nx, ny] = s
                if np.random.rand() < EAT_PROB:
                    if s == RED and t == GREEN: new[nx, ny] = RED
                    elif s == BLUE and t == RED: new[nx, ny] = BLUE
                    elif s == GREEN and t == BLUE: new[nx, ny] = GREEN
        return new

    for frame in range(STEPS):
        if should_stop(): return
        grid_arr = update_grid(grid_arr)
        count_red = np.sum((grid_arr == RED) & mask)
        count_blue = np.sum((grid_arr == BLUE) & mask)
        count_green = np.sum((grid_arr == GREEN) & mask)
        red_counts.append(count_red); blue_counts.append(count_blue); green_counts.append(count_green)

        total = count_red + count_blue + count_green
        frac_red = count_red/total if total>0 else 0
        frac_blue = count_blue/total if total>0 else 0
        frac_green = count_green/total if total>0 else 0

        if frame % VIS_INTERVAL == 0:
            im.set_data(np.where(mask, grid_arr, 0))
            line_r.set_data(range(len(red_counts)), red_counts)
            line_b.set_data(range(len(blue_counts)), blue_counts)
            line_g.set_data(range(len(green_counts)), green_counts)
            line_frac_red.set_data(range(len(red_counts)), [frac_red]*len(red_counts))
            line_frac_blue.set_data(range(len(blue_counts)), [frac_blue]*len(blue_counts))
            line_frac_green.set_data(range(len(green_counts)), [frac_green]*len(green_counts))
            ax_r.relim(); ax_r.autoscale_view(); ax_b.relim(); ax_b.autoscale_view(); ax_g.relim(); ax_g.autoscale_view()
            ax_frac.set_xlim(0, max(50, len(red_counts)))
            vis_placeholder.pyplot(fig)
            plt.pause(0.001)

    st.success("RPS simulation finished.")

# -----------------------------
# Launch chosen sim
# -----------------------------
if st.session_state.running:
    st.session_state.stop = False
    info_placeholder.info(f"Running: {sim_choice}")
    if sim_choice == "Growth (multi-colony)":
        sim_growth_streamlit(grid_size=GRID_SIZE, steps=STEPS, vis_interval=VIS_INTERVAL, num_seeds=NUM_SEEDS)
    elif sim_choice == "Lotka-Volterra (spatial)":
        sim_lv_streamlit(grid_size=GRID_SIZE, steps=STEPS, vis_interval=VIS_INTERVAL)
    elif sim_choice == "Mega-Plate":
        sim_mega_streamlit(size=SIZE, steps=STEPS, interval_ms=INTERVAL_MS, mutation_rate=MUTATION_RATE)
    elif sim_choice == "RPS (rock-paper-scissors)":
        sim_rps_streamlit(grid=GRID, steps=STEPS, vis_interval=VIS_INTERVAL)
else:
    info_placeholder.info("Press Start to run the selected simulation.")
