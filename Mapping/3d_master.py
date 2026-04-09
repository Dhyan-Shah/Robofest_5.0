import pyproj
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import random

# ==============================================================================
# 1. Map Dimensions & Takeoff Anchor
# ==============================================================================
map_width = 20.0   # meters wide
map_length = 100.0 # meters long
resolution = 0.5   # 0.5 meters per grid cell

# The bottom-center of our local map
start_local_x = map_width / 2.0 
start_local_y = 0.0

# The Drone's real-world GPS takeoff location (Anchor)
anchor_gps_lat = 22.300000 
anchor_gps_lon = 73.100000

# Set up the coordinate projection anchored to the takeoff point
proj = pyproj.Proj(proj='aeqd', lat_0=anchor_gps_lat, lon_0=anchor_gps_lon, datum='WGS84')

# ==============================================================================
# 2. SIMULATE DRONE FLIGHT: Generate Mock GPS Mine Data
# (In production, load your Raspberry Pi JSON file here instead)
# ==============================================================================
random.seed(42) 
mines_gps_raw = []

# Generate 15 mines on the left, 15 on the right, convert them to raw GPS
for _ in range(15):
    lon, lat = proj(random.uniform(1.0, 8.0) - start_local_x, random.uniform(5.0, 95.0) - start_local_y, inverse=True)
    mines_gps_raw.append((lat, lon))

for _ in range(15):
    lon, lat = proj(random.uniform(12.0, 19.0) - start_local_x, random.uniform(5.0, 95.0) - start_local_y, inverse=True)
    mines_gps_raw.append((lat, lon))

print(f"Loaded {len(mines_gps_raw)} GPS coordinates from drone telemetry.")

# ==============================================================================
# 3. GPS to Local Cartesian Conversion (pyproj)
# ==============================================================================
mines_local_xy = []
for lat, lon in mines_gps_raw:
    # Convert GPS back to meters relative to anchor
    offset_x, offset_y = proj(lon, lat)
    # Shift to match our 0-20m width and 0-100m length boundaries
    mine_x = offset_x + start_local_x
    mine_y = offset_y + start_local_y
    mines_local_xy.append((mine_x, mine_y))

# ==============================================================================
# 4. Gaussian Risk Heatmap Generation
# ==============================================================================
grid_width = int(map_width / resolution)
grid_height = int(map_length / resolution)

sigma = 1.0           # Spread of the danger zone
max_risk = 1000.0     # Peak penalty at exact mine center
safe_threshold = 50.0 # Maximum acceptable risk for a grid node

# Create the mathematical meshgrid for the risk surface
X_grid = np.linspace(0, map_width, grid_width)
Y_grid = np.linspace(0, map_length, grid_height)
X_mesh, Y_mesh = np.meshgrid(X_grid, Y_grid)
Z_risk = np.zeros_like(X_mesh)

# Apply Gaussian risk decay for every detected mine
for mx, my in mines_local_xy:
    dist_sq = (X_mesh - mx)**2 + (Y_mesh - my)**2
    Z_risk += max_risk * np.exp(-dist_sq / (2 * sigma**2))

# ==============================================================================
# 5. NetworkX Graph & Pathfinding
# ==============================================================================
G = nx.Graph()

# Add safe nodes
for x in range(grid_width):
    for y in range(grid_height):
        if Z_risk[y, x] < safe_threshold:
            G.add_node((x, y))

# Add weighted edges (Distance Cost + Risk Cost)
edges_to_add = []
for x in range(grid_width):
    for y in range(grid_height):
        if (x, y) not in G: continue
            
        if x < grid_width - 1 and (x+1, y) in G:
            edges_to_add.append(((x, y), (x+1, y), {'weight': 1.0 + Z_risk[y, x+1]}))
            
        if y < grid_height - 1 and (x, y+1) in G:
            edges_to_add.append(((x, y), (x, y+1), {'weight': 1.0 + Z_risk[y+1, x]}))
            
        if x < grid_width - 1 and y < grid_height - 1 and (x+1, y+1) in G:
            edges_to_add.append(((x, y), (x+1, y+1), {'weight': 1.414 + Z_risk[y+1, x+1]}))
            
        if x > 0 and y < grid_height - 1 and (x-1, y+1) in G:
            edges_to_add.append(((x, y), (x-1, y+1), {'weight': 1.414 + Z_risk[y+1, x-1]}))

G.add_edges_from(edges_to_add)

# Dynamic Exit Pathfinding
start_node = (int(start_local_x / resolution), int(start_local_y / resolution))
if start_node not in G:
    raise ValueError("Start position is too dangerous! Move the anchor.")

G.add_node("FINISH_LINE")
top_edge_nodes = [(x, grid_height - 1) for x in range(grid_width) if (x, grid_height - 1) in G]
G.add_edges_from([(node, "FINISH_LINE", {'weight': 0.0}) for node in top_edge_nodes])

try:
    path = nx.shortest_path(G, source=start_node, target="FINISH_LINE", weight='weight')
    path.pop() # Remove virtual node
    
    path_x = [p[0] * resolution for p in path]
    path_y = [p[1] * resolution for p in path]
    path_z = [Z_risk[p[1], p[0]] + 20 for p in path] # Elevate line for 3D visibility
    print(f"SUCCESS: 3D Safe path calculated. Exit at X={path_x[-1]:.1f}m, Y={path_y[-1]:.1f}m")
except nx.NetworkXNoPath:
    print("FAILED: No safe path exists.")
    path_x, path_y, path_z = [], [], []

# ==============================================================================
# 6. Interactive 3D Plotly Visualization
# ==============================================================================
fig = go.Figure()

# Plot Gaussian Risk Surface
fig.add_trace(go.Surface(
    z=Z_risk, x=X_mesh, y=Y_mesh, 
    colorscale='YlOrRd', opacity=0.8,
    name='Risk Surface', showscale=True,
    colorbar=dict(title="Risk Level")
))

# Plot the 3D Safe Path
if path_x:
    fig.add_trace(go.Scatter3d(
        x=path_x, y=path_y, z=path_z,
        mode='lines', line=dict(color='#00FF00', width=10),
        name='Optimal Safe Path'
    ))

# Plot Mines
mine_x = [m[0] for m in mines_local_xy]
mine_y = [m[1] for m in mines_local_xy]
mine_z = [max_risk] * len(mines_local_xy)

fig.add_trace(go.Scatter3d(
    x=mine_x, y=mine_y, z=mine_z,
    mode='markers', marker=dict(size=5, color='black', symbol='diamond'),
    name='Detected Mines'
))

# Layout Configuration
fig.update_layout(
    title='Robofest GCS: 3D Minefield Risk Map & Path Generation',
    scene=dict(
        xaxis_title='Width (m)', yaxis_title='Length (m)', zaxis_title='Danger Penalty',
        aspectmode='manual', aspectratio=dict(x=1, y=3, z=0.5), 
        camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2))
    ),
    template='plotly_dark'
)

html_filename = "robofest_3d_master_dashboard.html"
fig.write_html(html_filename)
print(f"Dashboard saved to {html_filename}. Open this file in your web browser!")