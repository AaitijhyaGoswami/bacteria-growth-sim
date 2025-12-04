import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.gridspec import GridSpec

# -----------------------------
# PARAMETERS (unchanged, tuned)
# -----------------------------
GRID_SIZE = 300
STEPS = 8000

FOOD_DIFF = 0.008
BACT_DIFF = 0.02
GROWTH_RATE = 0.05
SELF_GROWTH = 0.012
FOOD_CONSUMPTION = 0.006

NOISE_STRENGTH = 0.65
TIP_GROWTH_FACTOR = 1.0

NUM_SEEDS = 12
SEED_INTENSITY = 0.03

# Visual timing: smaller pause -> smoother, adjust to taste
ANIMATION_SPEED = 0.001
# How often to update the screen (frames)
VIS_INTERVAL = 40

# -----------------------------
# INITIALIZATION
# -----------------------------
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

# Data storage (histories)
pop_history = []
nutrient_history = []
per_colony_history = {i: [] for i in range(1, NUM_SEEDS+1)}
per_colony_consumed = {i: [] for i in range(1, NUM_SEEDS+1)}
time_history = []

# Colors (index 0 unused, keep consistent)
base_colors = np.array([
    [0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0],
    [1,0,1], [0,1,1], [0.5,0.5,0], [0.5,0,0.5],
    [0,0.5,0.5], [0.8,0.4,0], [0.4,0.8,0], [0.8,0,0.4]
])

# -----------------------------
# LAPLACIAN (interior)
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
# FIGURE & PERSISTENT ARTISTS
# -----------------------------
plt.ion()
fig = plt.figure(figsize=(18, 10))
gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], wspace=0.25, hspace=0.3)

# Top row visuals
ax_medium = fig.add_subplot(gs[0, 0])
ax_nutr = fig.add_subplot(gs[0, 1])
ax_bio = fig.add_subplot(gs[0, 2])

# Bottom row graphs
ax_global = fig.add_subplot(gs[1, 0])
ax_percol = fig.add_subplot(gs[1, 1])
ax_consumption = fig.add_subplot(gs[1, 2])

# Initialize images (persistent)
medium_img = ax_medium.imshow(np.zeros((GRID_SIZE, GRID_SIZE, 3)), interpolation='bilinear', vmin=0, vmax=1)
ax_medium.set_title("Bacterial Colonies")
ax_medium.axis('off')

nutr_img = ax_nutr.imshow(food, cmap='Greens', interpolation='bilinear', vmin=0, vmax=1)
ax_nutr.set_title("Nutrient Concentration")
ax_nutr.axis('off')

bio_img = ax_bio.imshow(bacteria, cmap='jet', interpolation='bilinear', vmin=0, vmax=1)
ax_bio.set_title("Biomass Density (Graduated)")
ax_bio.axis('off')

# Initialize lines for global dynamics
line_biomass, = ax_global.plot([], [], 'k-', lw=2, label="Biomass")
line_nutrient, = ax_global.plot([], [], 'g-', lw=1, label="Nutrient")
ax_global.set_title("Global Dynamics")
ax_global.set_xlabel("Time (mins)")
ax_global.set_ylabel("Quantity (a.u.)")
ax_global.grid(True)
ax_global.legend()

# Per-colony lines
percol_lines = {}
for i in range(1, NUM_SEEDS+1):
    (ln,) = ax_percol.plot([], [], lw=1, color=base_colors[i], label=f"Colony {i}")
    percol_lines[i] = ln
ax_percol.set_title("Per-Colony Biomass")
ax_percol.set_xlabel("Time (mins)")
ax_percol.set_ylabel("Biomass (a.u.)")
ax_percol.grid(True)
ax_percol.legend(loc='upper left', fontsize='xx-small', ncol=2)

# Consumption lines
cons_lines = {}
for i in range(1, NUM_SEEDS+1):
    (ln,) = ax_consumption.plot([], [], lw=1, color=base_colors[i], label=f"Colony {i}")
    cons_lines[i] = ln
ax_consumption.set_title("Cumulative Consumption")
ax_consumption.set_xlabel("Time (mins)")
ax_consumption.set_ylabel("Nutrient Units Consumed")
ax_consumption.grid(True)
ax_consumption.legend(loc='upper left', fontsize='xx-small', ncol=2)

fig.canvas.draw()
plt.pause(0.1)

# -----------------------------
# MAIN SIMULATION LOOP
# -----------------------------
for t in range(STEPS):
    food_prev = food.copy()

    # Diffusion (interior laplacian)
    food += FOOD_DIFF * laplacian_interior(food)
    bacteria += BACT_DIFF * laplacian_interior(bacteria)

    # Clamp and mask
    food = np.clip(food, 0.0, 1.0)
    bacteria = np.clip(bacteria, 0.0, 1.0)
    bacteria[~mask] = 0.0

    # Consumption by cells
    consumption_by_cells = FOOD_CONSUMPTION * bacteria
    food -= consumption_by_cells
    food = np.clip(food, 0.0, 1.0)

    # Neighbor-driven growth with noise and tip factor
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

    # Seed propagation to assign colony ownership
    for i in range(1, NUM_SEEDS+1):
        neighbors = (
            np.roll(seed_ids==i, 1, 0) | np.roll(seed_ids==i, -1, 0) |
            np.roll(seed_ids==i, 1, 1) | np.roll(seed_ids==i, -1, 1)
        )
        seed_ids[(neighbors & (seed_ids==0) & (bacteria>0))] = i

    # Branch tips: where bacteria exists but neighbors low
    branch_tips = (bacteria > 0) & (neighbor < 0.3)

    # Metrics: global sums and per-colony
    total_biomass = np.sum(bacteria)
    total_nutrient = np.sum(food)
    pop_history.append(total_biomass)
    nutrient_history.append(total_nutrient)
    time_history.append(t)

    delta_food = np.clip(food_prev - food, 0.0, None)
    for i in range(1, NUM_SEEDS+1):
        mask_i = (seed_ids == i)
        biomass_i = np.sum(bacteria[mask_i])
        per_colony_history[i].append(biomass_i)

        consumed_i = np.sum(delta_food[mask_i])
        if per_colony_consumed[i]:
            per_colony_consumed[i].append(per_colony_consumed[i][-1] + consumed_i)
        else:
            per_colony_consumed[i].append(consumed_i)

    # -----------------------------
    # VISUAL UPDATE (persistent artists)
    # -----------------------------
    if t % VIS_INTERVAL == 0:
        # Compose medium image using colony colors and biomass
        medium = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=float)
        for i in range(1, NUM_SEEDS+1):
            mask_i = (seed_ids == i)
            # add colored biomass contribution per channel
            for c in range(3):
                medium[..., c] += mask_i * bacteria * base_colors[i, c]

        # Smooth white halo for branch tips (Option 2)
        halo = gaussian_filter(branch_tips.astype(float), sigma=1.2)
        # Normalize halo to 0..1
        if halo.max() > 0:
            halo = halo / halo.max()
        # Add soft white halo (scaled)
        halo_strength = 0.6  # adjust halo intensity (0..1)
        medium += (halo[..., None] * halo_strength)

        # Clip and apply mask
        medium = np.clip(medium, 0, 1)
        medium[~mask] = 0.0

        # Update images (no re-creation)
        medium_img.set_data(medium)
        nutr_img.set_data(food)
        bio_img.set_data(bacteria)

        # Update global dynamics lines
        line_biomass.set_data(time_history, pop_history)
        line_nutrient.set_data(time_history, nutrient_history)
        # Keep x limits reasonable
        ax_global.set_xlim(max(0, t - 1000), t + 10)
        # Recompute y-limits based on recent window
        recent_pop = pop_history[-500:] if len(pop_history) > 0 else [1]
        recent_nut = nutrient_history[-500:] if len(nutrient_history) > 0 else [1]
        ymax = max(max(recent_pop, default=1), max(recent_nut, default=1))
        ax_global.set_ylim(0, ymax * 1.05)

        # Update per-colony lines and autoscale
        for i in range(1, NUM_SEEDS+1):
            percol_lines[i].set_data(time_history, per_colony_history[i])
        ax_percol.set_xlim(max(0, t - 1000), t + 10)
        # autoscale Y based on recent maxima
        max_percol = 1e-6
        for i in range(1, NUM_SEEDS+1):
            if per_colony_history[i]:
                max_percol = max(max_percol, max(per_colony_history[i][-500:], default=0))
        ax_percol.set_ylim(0, max_percol * 1.05 if max_percol > 0 else 1)

        # Update consumption plots
        for i in range(1, NUM_SEEDS+1):
            cons_lines[i].set_data(time_history, per_colony_consumed[i])
        ax_consumption.set_xlim(max(0, t - 1000), t + 10)
        # autoscale Y for consumption
        max_cons = 1e-6
        for i in range(1, NUM_SEEDS+1):
            if per_colony_consumed[i]:
                max_cons = max(max_cons, max(per_colony_consumed[i][-500:], default=0))
        ax_consumption.set_ylim(0, max_cons * 1.05 if max_cons > 0 else 1)

        # Draw once per VIS_INTERVAL
        fig.canvas.draw_idle()
        plt.pause(ANIMATION_SPEED)

# Finalize
plt.ioff()
plt.show()
print("Simulation complete.")
