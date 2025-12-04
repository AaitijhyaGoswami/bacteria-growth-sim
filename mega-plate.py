import matplotlib
matplotlib.use('TkAgg')  # Force GUI backend for Windows

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap, Normalize

# =============================
# PARAMETERS
# =============================
SIZE = 100
CENTER = SIZE // 2
RADIUS = 45

MUTATION_RATE = 0.01
STEPS_PER_FRAME = 1
INTERVAL_MS = 200  # Every frame = 0.2 seconds real-time
HOURS_PER_FRAME = 0.1  # label plots in HOURS

MAX_RES_LEVEL = 3
REGROW_PROB = 0.2

# =============================
# ANTIBIOTIC MAP
# =============================
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

# =============================
# BACTERIA GRID
# =============================
bacteria_grid = np.zeros((SIZE,SIZE), dtype=int)
bacteria_grid[CENTER-1:CENTER+2, CENTER-1:CENTER+2] = 1

# =============================
# DATA TRACKING
# =============================
time_points = []
total_counts = []
res_counts = []

# =============================
# FIGURE SETUP
# =============================
fig, axes = plt.subplots(2,2, figsize=(12,9))
ax_main = axes[0,0]
ax_total = axes[0,1]
ax_res = axes[1,0]
ax_ab = axes[1,1]

# Petri visualization
cmap = ListedColormap(['black','cyan','lime','red'])
norm = Normalize(vmin=0,vmax=MAX_RES_LEVEL)
im = ax_main.imshow(bacteria_grid, cmap=cmap, norm=norm, origin='lower')
ax_main.set_title("Pixelated Mega-Plate Growth", fontsize=12)
ax_main.axis('off')

# Legends for resistance levels
legend_elements = [
    plt.Line2D([0],[0], color='cyan', lw=4, label='Wildtype (Level 1)'),
    plt.Line2D([0],[0], color='lime', lw=4, label='Medium Resistance (Level 2)'),
    plt.Line2D([0],[0], color='red', lw=4, label='Superbug (Level 3)')
]
ax_main.legend(handles=legend_elements, loc='upper right', fontsize=9)

# Circles
circle1 = plt.Circle((CENTER,CENTER), RADIUS/3, color='white', fill=False, linestyle='--')
circle2 = plt.Circle((CENTER,CENTER), 2*RADIUS/3, color='white', fill=False, linestyle='--')
circle3 = plt.Circle((CENTER,CENTER), RADIUS, color='gray', fill=False, linewidth=2)
ax_main.add_patch(circle1)
ax_main.add_patch(circle2)
ax_main.add_patch(circle3)

# Total bacteria
line_total, = ax_total.plot([],[], color='blue')
ax_total.set_title("Total Bacterial Population", fontsize=12)
ax_total.set_xlabel("Time (hours)")
ax_total.set_ylabel("Number of Cells")
ax_total.grid(True)
ax_total.set_xlim(0,10)
ax_total.set_ylim(0, SIZE*SIZE)

# Resistance fractions
colors_res = ['cyan','lime','red']
lines_res = []
for color in colors_res:
    l, = ax_res.plot([],[], color=color, lw=2)
    lines_res.append(l)

ax_res.set_title("Fractions of Resistance Types", fontsize=12)
ax_res.set_xlabel("Time (hours)")
ax_res.set_ylabel("Fraction")
ax_res.set_ylim(0,1)
ax_res.set_xlim(0,10)
ax_res.grid(True)
ax_res.legend(["Wildtype", "Medium", "Superbug"], loc='upper right', fontsize=9)

# Antibiotic map
ab_map = np.zeros((SIZE,SIZE))
ab_map[antibiotic_map==0]=0.2
ab_map[antibiotic_map==1]=0.5
ab_map[antibiotic_map==2]=0.8

img2 = ax_ab.imshow(ab_map, cmap='Greys', origin='lower')
ax_ab.set_title("Antibiotic Concentration Zones", fontsize=12)
ax_ab.axis('off')

from matplotlib.patches import Patch
ax_ab.legend(
    handles=[
        Patch(color='lightgray', label='Low Antibiotic'),
        Patch(color='gray', label='Medium Antibiotic'),
        Patch(color='black', label='High Antibiotic')
    ],
    loc='lower right',
    fontsize=8
)

# =============================
# UPDATE FUNCTION
# =============================
def update(frame):
    global bacteria_grid
    new_grid = bacteria_grid.copy()
    
    rows, cols = np.where(bacteria_grid>0)
    idx = np.arange(len(rows))
    np.random.shuffle(idx)
    
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
    im.set_array(bacteria_grid)

    # --- Update stats ---
    time_h = frame * HOURS_PER_FRAME
    time_points.append(time_h)

    total_counts.append(np.sum(bacteria_grid>0))
    counts_level = [(bacteria_grid==lvl).sum() for lvl in range(1,MAX_RES_LEVEL+1)]
    res_counts.append(counts_level)

    line_total.set_data(time_points, total_counts)

    res_array = np.array(res_counts)/np.array(total_counts)[:,None]
    for i,l in enumerate(lines_res):
        l.set_data(time_points, res_array[:,i])

    ax_total.set_xlim(0, max(10, time_h+1))
    ax_res.set_xlim(0, max(10, time_h+1))

    return [im, line_total] + lines_res

# =============================
# ANIMATION
# =============================
ani = animation.FuncAnimation(fig, update, interval=INTERVAL_MS, blit=False)
plt.tight_layout()
plt.show(block=True)
