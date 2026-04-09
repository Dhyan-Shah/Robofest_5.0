import numpy as np
import networkx as nx
import plotly.graph_objects as go
import random

# 1. Map Dimensions & Local Start Point
map_width = 20.0
map_length = 100.0
resolution = 0.5 

start_local_x = map_width / 2.0 
start_local_y = 0.0

grid_width = int(map_width / resolution)
grid_height = int(map_length / resolution)

# 2. Generate Mock Mines (Same layout as the 100m test)
random.seed(42) 
mines_local_xy = []
for _ in range(15):
    mines_local_xy.append((random.uniform(1.0, 8.0), random.uniform(5.0, 95.0)))
for _ in range(15):
    mines_local_xy.append((random.uniform(12.0, 19.0), random.uniform(5.0, 95.0)))

# 3. Gaussian Risk Parameters
sigma = 1.0          # How wide the danger spreads (higher = wider risk zone)
max_risk = 1000.0    # Massive penalty for walking directly over a mine
safe_threshold = 50.0 # If risk is above this, do not even consider the node

# Pre-calculate the risk for every grid coordinate to build the 3D surface
X_grid = np.linspace(0, map_width, grid_width)
Y_grid = np.linspace(0, map_length, grid_height)
X_mesh, Y_mesh = np.meshgrid(X_grid, Y_grid)
Z_risk = np.zeros_like(X_mesh)

for mx, my in mines_local_xy:
    dist_sq = (X_mesh - mx)**2 + (Y_mesh - my)**2
    # Add Gaussian risk for this mine to the total map
    Z_risk += max_risk * np.exp(-dist_sq / (2 * sigma**2))

# 4. Generate the Risk-Weighted Graph
G = nx.Graph()

for x in range(grid_width):
    for y in range(grid_height):
        # Only add the node if it's below the absolute kill threshold
        if Z_risk[y, x] < safe_threshold:
            G.add_node((x, y))

edges_to_add = []
for x in range(grid_width):
    for y in range(grid_height):
        if (x, y) not in G: continue
            
        # Calculate edge weights: Base distance + the risk of the destination node
        if x < grid_width - 1 and (x+1, y) in G:
            weight = 1.0 + Z_risk[y, x+1]
            edges_to_add.append(((x, y), (x+1, y), {'weight': weight}))
            
        if y < grid_height - 1 and (x, y+1) in G:
            weight = 1.0 + Z_risk[y+1, x]
            edges_to_add.append(((x, y), (x, y+1), {'weight': weight}))
            
        if x < grid_width - 1 and y < grid_height - 1 and (x+1, y+1) in G:
            weight = 1.414 + Z_risk[y+1, x+1]
            edges_to_add.append(((x, y), (x+1, y+1), {'weight': weight}))
            
        if x > 0 and y < grid_height - 1 and (x-1, y+1) in G:
            weight = 1.414 + Z_risk[y+1, x-1]
            edges_to_add.append(((x, y), (x-1, y+1), {'weight': weight}))

G.add_edges_from(edges_to_add)

# 5. Dynamic Exit Pathfinding
start_node = (int(start_local_x / resolution), int(start_local_y / resolution))
if start_node not in G:
    print("FATAL: Start point is too dangerous!")
    exit()

G.add_node("FINISH_LINE")
top_edge_nodes = [(x, grid_height - 1) for x in range(grid_width) if (x, grid_height - 1) in G]
finish_edges = [(node, "FINISH_LINE", {'weight': 0.0}) for node in top_edge_nodes]
G.add_edges_from(finish_edges)

try:
    path = nx.shortest_path(G, source=start_node, target="FINISH_LINE", weight='weight')
    path.pop() 
    
    # Extract coordinates for plotting
    path_x = [p[0] * resolution for p in path]
    path_y = [p[1] * resolution for p in path]
    # Elevate the path slightly above the risk surface so it's visible in 3D
    path_z = [Z_risk[p[1], p[0]] + 20 for p in path] 
    
    print("SUCCESS: 3D Safe path found!")
except nx.NetworkXNoPath:
    print("FAILED: No safe path exists.")
    path_x, path_y, path_z = [], [], []

# 6. 3D Interactive Visualization with Plotly
fig = go.Figure()

# Plot the Risk Heatmap Surface
fig.add_trace(go.Surface(
    z=Z_risk, x=X_mesh, y=Y_mesh, 
    colorscale='YlOrRd', 
    opacity=0.8,
    name='Risk Surface',
    showscale=True,
    colorbar=dict(title="Risk Level")
))

# Plot the Safe Path as a thick green 3D line
if path_x:
    fig.add_trace(go.Scatter3d(
        x=path_x, y=path_y, z=path_z,
        mode='lines',
        line=dict(color='#00FF00', width=10),
        name='Optimal Safe Path'
    ))

# Plot the Mines as black markers directly on the grid
mine_x = [m[0] for m in mines_local_xy]
mine_y = [m[1] for m in mines_local_xy]
mine_z = [max_risk] * len(mines_local_xy) # Float them at the peak of the risk

fig.add_trace(go.Scatter3d(
    x=mine_x, y=mine_y, z=mine_z,
    mode='markers',
    marker=dict(size=5, color='black', symbol='diamond'),
    name='Detected Mines'
))

# Configure the 3D Layout
fig.update_layout(
    title='Interactive 3D Minefield Risk Map & Path Generation',
    scene=dict(
        xaxis_title='Width (m)',
        yaxis_title='Length (m)',
        zaxis_title='Danger Penalty',
        aspectmode='manual',
        aspectratio=dict(x=1, y=3, z=0.5), # Stretches the Y axis so the 100m map looks right
        camera=dict(
            eye=dict(x=1.5, y=-1.5, z=1.2) # Default viewing angle
        )
    ),
    template='plotly_dark' # Sleek dark mode theme for presentation
)

# Export to an interactive HTML file
html_filename = "robofest_3d_risk_map.html"
fig.write_html(html_filename)
print(f"Interactive 3D map saved to {html_filename}. Double click the file to open it in your browser!")