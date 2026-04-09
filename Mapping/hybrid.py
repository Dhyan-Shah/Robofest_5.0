# import pyproj
# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import random



# # 1. Map Dimensions & Local Start Point
# map_width = 20.0   # meters wide
# map_length = 100.0 # updated to 100 meters long

# start_local_x = map_width / 2 
# start_local_y = 0.0

# # 2. GPS Anchor & Mock Mine Generation
# anchor_gps_lat = 22.300000 
# anchor_gps_lon = 73.100000
# proj = pyproj.Proj(proj='aeqd', lat_0=anchor_gps_lat, lon_0=anchor_gps_lon, datum='WGS84')

# # Generate 30 mock GPS coordinates (15 on left, 15 on right)
# random.seed(42) # Ensures you get the same random mines every time you run the script
# mines_gps = []

# for _ in range(15):
#     # Left side: X between 1m and 8m, Y spread across 5m to 95m
#     local_x = random.uniform(1.0, 8.0)
#     local_y = random.uniform(5.0, 95.0)
#     offset_x = local_x - start_local_x
#     offset_y = local_y - start_local_y
#     lon, lat = proj(offset_x, offset_y, inverse=True) # Convert back to GPS
#     mines_gps.append((lat, lon))

# for _ in range(15):
#     # Right side: X between 12m and 19m, Y spread across 5m to 95m
#     local_x = random.uniform(12.0, 19.0)
#     local_y = random.uniform(5.0, 95.0)
#     offset_x = local_x - start_local_x
#     offset_y = local_y - start_local_y
#     lon, lat = proj(offset_x, offset_y, inverse=True)
#     mines_gps.append((lat, lon))


# # 3. Convert GPS back to Local Map Coordinates (The actual processing pipeline)
# mines_local_xy = []
# for lat, lon in mines_gps:
#     offset_x, offset_y = proj(lon, lat)
#     mine_x = offset_x + start_local_x
#     mine_y = offset_y + start_local_y
#     mines_local_xy.append((mine_x, mine_y))

# # 4. Map & Obstacle Inflation Configuration
# resolution = 0.5 # 0.5 meters per grid cell
# safe_radius = 1.2 # 1.0m clearance + 0.2m buffer

# grid_width = int(map_width / resolution)
# grid_height = int(map_length / resolution)

# # 5. Generate Safe Graph
# G = nx.Graph()

# for x in range(grid_width):
#     for y in range(grid_height):
#         G.add_node((x, y))

# edges_to_add = []
# for x in range(grid_width):
#     for y in range(grid_height):
#         if x < grid_width - 1:
#             edges_to_add.append(((x, y), (x+1, y), {'weight': 1.0}))
#         if y < grid_height - 1:
#             edges_to_add.append(((x, y), (x, y+1), {'weight': 1.0}))
#         # Diagonals
#         if x < grid_width - 1 and y < grid_height - 1:
#             edges_to_add.append(((x, y), (x+1, y+1), {'weight': 1.414}))
#             edges_to_add.append(((x+1, y), (x, y+1), {'weight': 1.414}))

# # Batch add edges (efficient for large grids)
# G.add_edges_from(edges_to_add)

# # 6. Obstacle Inflation (Remove Danger Zones)
# for x in range(grid_width):
#     for y in range(grid_height):
#         real_x = x * resolution
#         real_y = y * resolution
        
#         for mx, my in mines_local_xy:
#             dist = np.sqrt((real_x - mx)**2 + (real_y - my)**2)
#             if dist <= safe_radius:
#                 if (x, y) in G:
#                     G.remove_node((x, y))

# # 7. Dynamic Exit Pathfinding
# start_node = (int(start_local_x / resolution), int(start_local_y / resolution))

# if start_node not in G:
#     print("FATAL: Start point is inside a danger zone!")
#     exit()

# G.add_node("FINISH_LINE")
# top_edge_nodes = [(x, grid_height - 1) for x in range(grid_width) if (x, grid_height - 1) in G]
# finish_edges = [(node, "FINISH_LINE", {'weight': 0.0}) for node in top_edge_nodes]

# if top_edge_nodes:
#     G.add_edges_from(finish_edges)

# physical_path = []
# try:
#     path = nx.shortest_path(G, source=start_node, target="FINISH_LINE", weight='weight')
#     path.pop() # Remove virtual node
#     physical_path = [(p[0]*resolution, p[1]*resolution) for p in path]
#     target_x, target_y = physical_path[-1]
#     print(f"SUCCESS: Safe exit found at X={target_x:.1f}m, Y={target_y:.1f}m")
# except nx.NetworkXNoPath:
#     print("FAILED: No safe path to the other side.")
# # ==============================================================================
# # 8. Visualization (Updated for Off-Axis Legend)
# # ==============================================================================
# # Increased figure height to accommodate the 100m length without squishing
# # We use a slightly wider figure (7 instead of 6) to hold the off-axis legend.
# fig, ax = plt.subplots(figsize=(7, 15)) 

# ax.set_xlim(0, map_width)
# ax.set_ylim(0, map_length)
# ax.set_aspect('equal')
# ax.grid(True, linestyle='--', alpha=0.4)

# # Draw Mines & Danger Zones
# for mx, my in mines_local_xy:
#     ax.plot(mx, my, 'kx', markersize=6)
#     circle = patches.Circle((mx, my), radius=safe_radius, color='red', alpha=0.3)
#     ax.add_patch(circle)

# # Draw Safe Path elements (assigning labels for the legend)
# if physical_path:
#     path_x, path_y = zip(*physical_path)
#     ax.plot(path_x, path_y, 'g-', linewidth=2.5, label='Shortest Safe Route')
#     ax.plot(target_x, target_y, 'mo', markersize=8, label='Optimal Exit Point')

# # Draw Start Point and Target Boundary
# ax.plot(start_local_x, start_local_y, 'bo', markersize=8, label='Start Point')
# ax.axhline(y=map_length, color='purple', linestyle='--', alpha=0.5, label='Target Boundary')

# ax.set_xlabel('Width (meters)')
# ax.set_ylabel('Length (meters)')
# ax.set_title('100m Drone-Surveyed Minefield Test')

# # ------------------------------------------------------------------------------
# # UPDATED: Legend Placement (Moved outside to top right)
# # ------------------------------------------------------------------------------
# # bbox_to_anchor places the legend relative to the plot axes (0.0 to 1.0).
# # (1.02, 1.0) means 2% past the right edge, aligned with the very top edge.
# ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), framealpha=1.0, facecolor='white', edgecolor='black')

# # Use bbox_inches='tight' during saving ensures the off-axis legend is not cut off
# plt.savefig('100m_test_path_cleaned.png', dpi=300, bbox_inches='tight')
# print("Visual map saved as '100m_test_path_cleaned.png'.")



import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==============================================================================
# 1. Load Mine Coordinates from mines.json (output of pipeline.py)
# ==============================================================================
with open("mines.json") as f:
    data = json.load(f)

mines_local_xy = [(m[0], m[1]) for m in data["mines_meters"]]
print(f"Loaded {len(mines_local_xy)} mines from mines.json")

# ==============================================================================
# 2. Map Dimensions
# ==============================================================================
map_width  = 20.0   # meters wide
map_length = 100.0  # meters long

start_local_x = map_width / 2
start_local_y = 0.0

# ==============================================================================
# 3. Map & Obstacle Inflation Configuration
# ==============================================================================
resolution  = 0.5  # 0.5 meters per grid cell
safe_radius = 1.2  # 1.0m clearance + 0.2m buffer

grid_width  = int(map_width  / resolution)
grid_height = int(map_length / resolution)

# ==============================================================================
# 4. Generate Safe Graph
# ==============================================================================
G = nx.Graph()

for x in range(grid_width):
    for y in range(grid_height):
        G.add_node((x, y))

edges_to_add = []
for x in range(grid_width):
    for y in range(grid_height):
        if x < grid_width - 1:
            edges_to_add.append(((x, y), (x+1, y), {'weight': 1.0}))
        if y < grid_height - 1:
            edges_to_add.append(((x, y), (x, y+1), {'weight': 1.0}))
        if x < grid_width - 1 and y < grid_height - 1:
            edges_to_add.append(((x, y), (x+1, y+1), {'weight': 1.414}))
            edges_to_add.append(((x+1, y), (x, y+1), {'weight': 1.414}))

G.add_edges_from(edges_to_add)

# ==============================================================================
# 5. Obstacle Inflation (Remove Danger Zones)
# ==============================================================================
for x in range(grid_width):
    for y in range(grid_height):
        real_x = x * resolution
        real_y = y * resolution
        for mx, my in mines_local_xy:
            dist = np.sqrt((real_x - mx)**2 + (real_y - my)**2)
            if dist <= safe_radius:
                if (x, y) in G:
                    G.remove_node((x, y))
                    break  # no need to check other mines once removed

# ==============================================================================
# 6. Dynamic Exit Pathfinding (Dijkstra)
# ==============================================================================
start_node = (int(start_local_x / resolution), int(start_local_y / resolution))

if start_node not in G:
    print("FATAL: Start point is inside a danger zone!")
    exit()

G.add_node("FINISH_LINE")
top_edge_nodes = [(x, grid_height - 1) for x in range(grid_width) if (x, grid_height - 1) in G]

if not top_edge_nodes:
    print("FATAL: Entire exit edge is blocked!")
    exit()

G.add_edges_from([(node, "FINISH_LINE", {'weight': 0.0}) for node in top_edge_nodes])

physical_path = []
try:
    path = nx.shortest_path(G, source=start_node, target="FINISH_LINE", weight='weight')
    path.pop()  # remove virtual FINISH_LINE node
    physical_path = [(p[0]*resolution, p[1]*resolution) for p in path]
    target_x, target_y = physical_path[-1]
    print(f"SUCCESS: Safe exit found at X={target_x:.1f}m, Y={target_y:.1f}m")
except nx.NetworkXNoPath:
    print("FAILED: No safe path to the other side.")

# ==============================================================================
# 7. Visualization
# ==============================================================================
fig, ax = plt.subplots(figsize=(7, 15))

ax.set_xlim(0, map_width)
ax.set_ylim(0, map_length)
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.4)

# Draw Mines & Danger Zones
for mx, my in mines_local_xy:
    ax.plot(mx, my, 'kx', markersize=6)
    circle = patches.Circle((mx, my), radius=safe_radius, color='red', alpha=0.3)
    ax.add_patch(circle)

# Draw Safe Path
if physical_path:
    path_x, path_y = zip(*physical_path)
    ax.plot(path_x, path_y, 'g-', linewidth=2.5, label='Shortest Safe Route')
    ax.plot(target_x, target_y, 'mo', markersize=8, label='Optimal Exit Point')

# Draw Start Point and Target Boundary
ax.plot(start_local_x, start_local_y, 'bo', markersize=8, label='Start Point')
ax.axhline(y=map_length, color='purple', linestyle='--', alpha=0.5, label='Target Boundary')

ax.set_xlabel('Width (meters)')
ax.set_ylabel('Length (meters)')
ax.set_title('ARCNET — Drone-Surveyed Minefield Safe Path')
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), framealpha=1.0, facecolor='white', edgecolor='black')

plt.savefig('safe_path_output.png', dpi=300, bbox_inches='tight')
print("Saved: safe_path_output.png")