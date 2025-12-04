import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec    

# -----------------------------
# PARAMETERS
# -----------------------------
GRID_SIZE = 200
STEPS = 2000
VIS_INTERVAL = 5
ANIMATION_SPEED = 0.005

# Diffusion coefficients
D_PREY = 0.02
D_PRED = 0.03
D_NUTRIENT = 0.0  # nutrient is static

# Lotka-Volterra parameters
mu = 0.05        # prey growth per nutrient
alpha = 0.05     # nutrient consumption by prey
beta = 0.03      # predation rate
gamma = 0.8      # predator efficiency
delta = 0.002    # predator death rate

# -----------------------------
# PETRI DISK
# -----------------------------
y, x = np.ogrid[-GRID_SIZE/2:GRID_SIZE/2, -GRID_SIZE/2:GRID_SIZE/2]
mask = x**2 + y**2 <= (GRID_SIZE/2 - 2)**2

# -----------------------------
# INITIAL CONDITIONS
# -----------------------------
def create_colonies(mask, num_colonies, radius, intensity):
    arr = np.zeros_like(mask, dtype=float)
    ys, xs = np.where(mask)
    for _ in range(num_colonies):
        idx = np.random.randint(len(ys))
        cy, cx = ys[idx], xs[idx]
        yy, xx = np.ogrid[:GRID_SIZE, :GRID_SIZE]
        arr[(yy - cy)**2 + (xx - cx)**2 <= radius**2] = intensity
    return arr

np.random.seed(42)
prey = create_colonies(mask, 20, 5, 0.5)
predator = create_colonies(mask, 10, 4, 0.3)
nutrient = np.ones((GRID_SIZE, GRID_SIZE))
nutrient[~mask] = 0

# -----------------------------
# HISTORY
# -----------------------------
time_hist = []
prey_hist = []
pred_hist = []
nutrient_hist = []
ratio_hist = []

# -----------------------------
# FIGURE (2×2 GRID)
# -----------------------------
plt.ion()
fig = plt.figure(figsize=(14, 10))

gs = GridSpec(
    2, 2, 
    figure=fig,
    hspace=0.35,
    wspace=0.30
)

# Petri dish visualization
ax_species = fig.add_subplot(gs[0, 0])
species_img = ax_species.imshow(np.zeros((GRID_SIZE, GRID_SIZE, 3)))
ax_species.set_title("Petri Dish: Prey (Blue), Predator (Red), Nutrient (Green)")
ax_species.axis('off')

# Population dynamics
ax_pop = fig.add_subplot(gs[0, 1])
l_prey, = ax_pop.plot([], [], 'b-', lw=2, label="Prey")
l_pred, = ax_pop.plot([], [], 'r-', lw=2, label="Predator")
ax_pop.set_xlim(0, STEPS)
ax_pop.set_ylim(0, GRID_SIZE*GRID_SIZE*0.2)
ax_pop.set_xlabel("Time")
ax_pop.set_ylabel("Total Biomass")
ax_pop.set_title("Population Dynamics")
ax_pop.legend()
ax_pop.grid(True)

# Nutrient graph
ax_nutr = fig.add_subplot(gs[1, 0])
l_nutrient, = ax_nutr.plot([], [], 'g-', lw=2, label="Remaining Nutrient")
ax_nutr.set_xlim(0, STEPS)
ax_nutr.set_ylim(0, GRID_SIZE*GRID_SIZE)
ax_nutr.set_xlabel("Time")
ax_nutr.set_ylabel("Total Nutrient")
ax_nutr.set_title("Nutrient Dynamics")
ax_nutr.legend()
ax_nutr.grid(True)

# Ratio graph
ax_ratio = fig.add_subplot(gs[1, 1])
l_ratio, = ax_ratio.plot([], [], 'm-', lw=2, label="Predator / Prey Ratio")
ax_ratio.set_xlim(0, STEPS)
ax_ratio.set_ylim(0, 2)
ax_ratio.set_xlabel("Time")
ax_ratio.set_ylabel("Ratio")
ax_ratio.set_title("Predator–Prey Ratio")
ax_ratio.legend()
ax_ratio.grid(True)

# -----------------------------
# LAPLACIAN
# -----------------------------
def laplacian(arr):
    lap = np.zeros_like(arr)
    lap[1:-1,1:-1] = (arr[:-2,1:-1] + arr[2:,1:-1] +
                      arr[1:-1,:-2] + arr[1:-1,2:] -
                      4 * arr[1:-1,1:-1])
    return lap

# -----------------------------
# SIMULATION LOOP
# -----------------------------
ZOOM_WINDOW = 50

for t in range(STEPS):
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
    
    # Clamp to physical bounds
    prey = np.clip(prey, 0, 1)
    predator = np.clip(predator, 0, 1)
    nutrient = np.clip(nutrient, 0, 1)
    
    prey[~mask] = 0
    predator[~mask] = 0
    nutrient[~mask] = 0
    
    # Save history
    if t % 5 == 0:
        time_hist.append(t)
        prey_hist.append(np.sum(prey))
        pred_hist.append(np.sum(predator))
        nutrient_hist.append(np.sum(nutrient))
        ratio_hist.append(np.sum(predator) / np.sum(prey) if np.sum(prey) > 0 else 0)
    
    # Update visualization
    if t % VIS_INTERVAL == 0:
        # Petri dish image
        img = np.zeros((GRID_SIZE, GRID_SIZE, 3))
        img[..., 2] = np.clip(prey * 4, 0, 1)     # blue
        img[..., 0] = np.clip(predator * 4, 0, 1) # red
        img[..., 1] = np.clip(nutrient * 4, 0, 1) # green
        img[~mask] = 0
        species_img.set_data(img)
        
        # Graph updates
        l_prey.set_data(time_hist, prey_hist)
        l_pred.set_data(time_hist, pred_hist)
        ax_pop.set_ylim(0, max(prey_hist[-ZOOM_WINDOW:] + pred_hist[-ZOOM_WINDOW:]) * 1.2)
        
        l_nutrient.set_data(time_hist, nutrient_hist)
        ax_nutr.set_ylim(0, max(nutrient_hist[-ZOOM_WINDOW:]) * 1.2)
        
        l_ratio.set_data(time_hist, ratio_hist)
        ax_ratio.set_ylim(0, max(ratio_hist[-ZOOM_WINDOW:]) * 1.2)
        
        plt.draw()
        plt.pause(ANIMATION_SPEED)

plt.ioff()
plt.show()
print("Simulation complete.")
