import pyproj
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 1. Inputs (Mock GPS Data)
ref_lat, ref_lon = 22.300000, 73.100000 # Start point (Reference)
mines_gps = [
    (22.300045, 73.100048), (22.300090, 73.100097), 
    (22.300135, 73.100048), (22.300108, 73.100145), (22.300060, 73.100120)
]
target_gps = (22.300180, 73.100190)

# 2. PyProj Conversion (Spherical GPS to Flat Meters)
proj = pyproj.Proj(proj='aeqd', lat_0=ref_lat, lon_0=ref_lon, datum='WGS84')

mines_xy = [proj(lon, lat) for lat, lon in mines_gps] # returns local x, y
target_x, target_y = proj(target_gps[1], target_gps[0])
start_x, start_y = proj(ref_lon, ref_lat) # Center (0, 0)

# 3. Digital Map & Obstacle Inflation Configuration
resolution = 0.5 # 0.5 meters per grid cell
safe_radius = 1.2 # 1 meter clearance + 0.2m human error buffer

grid_width = int(np.ceil((target_x + 5) / resolution))
grid_height = int(np.ceil((target_y + 5) / resolution))

# 4. Generate Safe Path (NetworkX)
G = nx.grid_2d_graph(grid_width, grid_height)

# Add diagonal edges to graph for smoother movement
for x in range(grid_width - 1):
    for y in range(grid_height - 1):
        G.add_edge((x, y), (x+1, y+1))
        G.add_edge((x+1, y), (x, y+1))

# Remove unsafe nodes (Obstacle Inflation)
for x in range(grid_width):
    for y in range(grid_height):
        real_x, real_y = x * resolution, y * resolution
        
        for mx, my in mines_xy:
            dist = np.sqrt((real_x - mx)**2 + (real_y - my)**2)
            if dist <= safe_radius: # If inside danger zone
                if (x, y) in G:
                    G.remove_node((x, y))

start_node = (int(start_x / resolution), int(start_y / resolution))
target_node = (int(target_x / resolution), int(target_y / resolution))

# A* Heuristic: straight-line distance to target
def dist_heuristic(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

try:
    path = nx.astar_path(G, start_node, target_node, heuristic=dist_heuristic)
    physical_path = [(p[0]*resolution, p[1]*resolution) for p in path]
except nx.NetworkXNoPath:
    print("FAILED: No valid safe path found! The area is completely blocked.")
    physical_path = []

# 5. Visualization (Matplotlib)
fig, ax = plt.subplots(figsize=(10, 8))

# Draw Grid Map
ax.set_xlim(-2, target_x + 3)
ax.set_ylim(-2, target_y + 3)
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.4)

# Plot Mines & 1.2m Danger Zones
for i, (mx, my) in enumerate(mines_xy):
    ax.plot(mx, my, 'kx', markersize=8)
    circle = patches.Circle((mx, my), radius=safe_radius, color='red', alpha=0.3)
    ax.add_patch(circle)

# Plot calculated Safe Path
if physical_path:
    path_x, path_y = zip(*physical_path)
    ax.plot(path_x, path_y, 'g-', linewidth=3, label='A* Safe Route')

# Plot Start / End
ax.plot(start_x, start_y, 'bo', markersize=10, label='Start Point')
ax.plot(target_x, target_y, 'mo', markersize=10, label='Destination')

ax.set_xlabel('Local Distance East (meters)')
ax.set_ylabel('Local Distance North (meters)')
ax.set_title('Minefield Safe Path Generation (1.0m Safe Radius + 0.2m Buffer)')
ax.legend(loc='lower right', framealpha=0.9)

plt.tight_layout()
plt.savefig('safe_path_map.png', dpi=300)