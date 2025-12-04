import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation

# -----------------------------
# PARAMETERS
# -----------------------------
GRID = 200
STEPS = 600

EMPTY = 0
RED = 1
BLUE = 2
GREEN = 3

INIT_RED = 0.02
INIT_BLUE = 0.02
INIT_GREEN = 0.02

SPREAD_RATE = 0.25
EAT_PROB = 0.45

# -----------------------------
# CIRCULAR MASK
# -----------------------------
yy, xx = np.indices((GRID, GRID))
center = GRID // 2
radius = GRID // 2 - 2
mask = (xx - center)**2 + (yy - center)**2 <= radius**2

# -----------------------------
# INITIAL GRID
# -----------------------------
grid = np.zeros((GRID, GRID), dtype=int)
rand = np.random.rand(GRID, GRID)
grid[(rand < INIT_RED) & mask] = RED
grid[(rand >= INIT_RED) & (rand < INIT_RED + INIT_BLUE) & mask] = BLUE
grid[(rand >= INIT_RED + INIT_BLUE) & 
     (rand < INIT_RED + INIT_BLUE + INIT_GREEN) & mask] = GREEN

# -----------------------------
# DATA STORAGE
# -----------------------------
red_counts, blue_counts, green_counts = [], [], []

# -----------------------------
# CUSTOM COLOR MAP
# -----------------------------
cmap = ListedColormap([
    "black",      # empty
    "#FF3333",    # red
    "#3366FF",    # blue
    "#33FF33"     # green
])

# -----------------------------
# UPDATE FUNCTION
# -----------------------------
def update_grid(grid):
    new = grid.copy()
    
    dx = np.random.randint(-1, 2, size=(GRID, GRID))
    dy = np.random.randint(-1, 2, size=(GRID, GRID))
    
    for x in range(GRID):
        for y in range(GRID):
            if not mask[x, y]:
                new[x, y] = EMPTY
                continue
            
            s = grid[x, y]
            if s == EMPTY:
                continue
            
            nx = (x + dx[x, y]) % GRID
            ny = (y + dy[x, y]) % GRID
            
            if not mask[nx, ny]:
                continue
            
            t = grid[nx, ny]
            
            # Reproduction
            if t == EMPTY and np.random.rand() < SPREAD_RATE:
                new[nx, ny] = s
            
            # RPS predation
            if np.random.rand() < EAT_PROB:
                if s == RED and t == GREEN:
                    new[nx, ny] = RED
                elif s == BLUE and t == RED:
                    new[nx, ny] = BLUE
                elif s == GREEN and t == BLUE:
                    new[nx, ny] = GREEN
    return new

# -----------------------------
# FIGURE SETUP WITH GRIDSPEC
# -----------------------------
fig = plt.figure(figsize=(14,10))
gs = fig.add_gridspec(2, 3, height_ratios=[2, 1])

ax_dish = fig.add_subplot(gs[0, 0])
ax_frac = fig.add_subplot(gs[0, 1:])
ax_red = fig.add_subplot(gs[1, 0])
ax_blue = fig.add_subplot(gs[1, 1])
ax_green = fig.add_subplot(gs[1, 2])

# -----------------------------
# PETRI DISH
# -----------------------------
im = ax_dish.imshow(np.where(mask, grid, 0), cmap=cmap, vmin=0, vmax=3)
circle = plt.Circle((center, center), radius, fill=False, edgecolor='white', linewidth=2)
ax_dish.add_patch(circle)
ax_dish.set_title("Petri Dish")
ax_dish.set_xticks([]); ax_dish.set_yticks([])

# -----------------------------
# FRACTION GRAPH
# -----------------------------
ax_frac.set_xlim(0, STEPS)
ax_frac.set_ylim(0, 1.05)
ax_frac.set_title("Fraction of Species")
ax_frac.set_xlabel("Time")
ax_frac.set_ylabel("Fraction")
ax_frac.grid(True)
line_frac_red, = ax_frac.plot([], [], color="#FF3333", label="Red")
line_frac_blue, = ax_frac.plot([], [], color="#3366FF", label="Blue")
line_frac_green, = ax_frac.plot([], [], color="#33FF33", label="Green")
ax_frac.legend()

# -----------------------------
# POPULATION GRAPHS
# -----------------------------
for ax, color, label in zip([ax_red, ax_blue, ax_green], ["#FF3333","#3366FF","#33FF33"], ["Red","Blue","Green"]):
    ax.set_xlim(0, STEPS)
    ax.set_ylim(0, GRID*GRID*0.05)
    ax.set_title(f"{label} Population")
    ax.set_xlabel("Time")
    ax.set_ylabel("Count")
    ax.grid(True)

line_red, = ax_red.plot([], [], color="#FF3333")
line_blue, = ax_blue.plot([], [], color="#3366FF")
line_green, = ax_green.plot([], [], color="#33FF33")

# -----------------------------
# ANIMATION FUNCTION
# -----------------------------
def animate(frame):
    global grid
    grid = update_grid(grid)
    
    # Population counts
    count_red = np.sum((grid == RED) & mask)
    count_blue = np.sum((grid == BLUE) & mask)
    count_green = np.sum((grid == GREEN) & mask)
    red_counts.append(count_red)
    blue_counts.append(count_blue)
    green_counts.append(count_green)
    
    # Fractions
    total = count_red + count_blue + count_green
    frac_red = count_red/total if total>0 else 0
    frac_blue = count_blue/total if total>0 else 0
    frac_green = count_green/total if total>0 else 0
    
    # Update Petri dish
    im.set_data(np.where(mask, grid, 0))
    
    # Update population graphs
    line_red.set_data(range(len(red_counts)), red_counts)
    line_blue.set_data(range(len(blue_counts)), blue_counts)
    line_green.set_data(range(len(green_counts)), green_counts)
    
    # Auto adjust population y-limits
    for ax, data in zip([ax_red, ax_blue, ax_green], [red_counts, blue_counts, green_counts]):
        ax.set_ylim(0, max(data)*1.1 + 1)
    
    # Update fraction graph
    line_frac_red.set_data(range(len(red_counts)), [np.sum((grid == RED) & mask)/(count_red+count_blue+count_green) if (count_red+count_blue+count_green)>0 else 0 for _ in range(len(red_counts))])
    line_frac_blue.set_data(range(len(blue_counts)), [np.sum((grid == BLUE) & mask)/(count_red+count_blue+count_green) if (count_red+count_blue+count_green)>0 else 0 for _ in range(len(blue_counts))])
    line_frac_green.set_data(range(len(green_counts)), [np.sum((grid == GREEN) & mask)/(count_red+count_blue+count_green) if (count_red+count_blue+count_green)>0 else 0 for _ in range(len(green_counts))])
    
    ax_frac.set_xlim(0, max(50, len(red_counts)))
    
    return im, line_red, line_blue, line_green, line_frac_red, line_frac_blue, line_frac_green

# -----------------------------
# RUN ANIMATION
# -----------------------------
anim = FuncAnimation(fig, animate, frames=STEPS, interval=100, blit=False)
plt.tight_layout()
plt.show()
