import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 1. Map Dimensions & Inputs (in meters)
map_width = 20.0  # 20 meters wide (X-axis)
map_length = 100.0 # 30 meters long (Y-axis)

# Starting position: Center of the bottom edge
start_x = map_width / 2.0
start_y = 0.0

# Detected Mines: Exact local coordinates (X, Y) in meters
# (e.g., as calculated by the drone's position within the grid)
mines_xy = [
    (10.0, 5.0),
    (8.0, 12.0),
    (11.0, 15.0),
    (10.5, 22.0),
    (15.0, 25.0),
    (5.0, 20.0)
]

# 2. Digital Map & Obstacle Inflation Configuration
resolution = 0.5 # 0.5 meters per grid cell
safe_radius = 1.2 # 1.0m clearance + 0.2m human error buffer

grid_width = int(map_width / resolution)
grid_height = int(map_length / resolution)

# 3. Generate Graph
G = nx.Graph()

# Create all nodes
for x in range(grid_width):
    for y in range(grid_height):
        G.add_node((x, y))

# Prepare edges with weights (1 for straight, 1.414 for diagonal)
edges_to_add = []
for x in range(grid_width):
    for y in range(grid_height):
        if x < grid_width - 1:
            edges_to_add.append(((x, y), (x+1, y), {'weight': 1.0}))
        if y < grid_height - 1:
            edges_to_add.append(((x, y), (x, y+1), {'weight': 1.0}))
        # Diagonals
        if x < grid_width - 1 and y < grid_height - 1:
            edges_to_add.append(((x, y), (x+1, y+1), {'weight': 1.414}))
            edges_to_add.append(((x+1, y), (x, y+1), {'weight': 1.414}))

# Batch add all edges to the graph
G.add_edges_from(edges_to_add)

# 4. Obstacle Inflation (Remove Danger Zones)
for x in range(grid_width):
    for y in range(grid_height):
        real_x = x * resolution
        real_y = y * resolution
        
        for mx, my in mines_xy:
            # Calculate Euclidean distance from cell center to the mine
            dist = np.sqrt((real_x - mx)**2 + (real_y - my)**2)
            if dist <= safe_radius:
                if (x, y) in G:
                    G.remove_node((x, y))

# 5. Pathfinding to ANY point on the opposite edge
start_node = (int(start_x / resolution), int(start_y / resolution))

# Check if the start node itself was placed on a mine
if start_node not in G:
    raise ValueError("Start position is inside a danger zone!")

# Create a Virtual Target ("Finish Line")
G.add_node("FINISH_LINE")

# Connect all safe nodes on the top edge (y = grid_height - 1) to the Finish Line
top_edge_nodes = [(x, grid_height - 1) for x in range(grid_width) if (x, grid_height - 1) in G]
finish_edges = [(node, "FINISH_LINE", {'weight': 0.0}) for node in top_edge_nodes]

if not top_edge_nodes:
    print("WARNING: The entire top edge is blocked by mines!")
else:
    G.add_edges_from(finish_edges)

# Calculate the shortest path using Dijkstra's algorithm
try:
    # This automatically finds the best exit point on the top edge
    path = nx.shortest_path(G, source=start_node, target="FINISH_LINE", weight='weight')
    
    # Remove the virtual node from the end of the list to get actual coordinates
    path.pop() 
    
    # Convert grid path back to physical (X, Y) meters
    physical_path = [(p[0]*resolution, p[1]*resolution) for p in path]
    target_x, target_y = physical_path[-1] # The algorithm's chosen exit point
    
    print(f"SUCCESS: Safe path found!")
    print(f"Chosen Exit Point: X={target_x:.1f}m, Y={target_y:.1f}m")
    
except nx.NetworkXNoPath:
    print("FAILED: No valid safe path exists to the other side.")
    physical_path = []

# 6. Visualization (Matplotlib)
fig, ax = plt.subplots(figsize=(8, 10))

# Map Boundaries
ax.set_xlim(0, map_width)
ax.set_ylim(0, map_length)
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.4)

# Draw Mines & Danger Zones
for i, (mx, my) in enumerate(mines_xy):
    ax.plot(mx, my, 'kx', markersize=8)
    circle = patches.Circle((mx, my), radius=safe_radius, color='red', alpha=0.3)
    ax.add_patch(circle)

# Draw Safe Path
if physical_path:
    path_x, path_y = zip(*physical_path)
    ax.plot(path_x, path_y, 'g-', linewidth=3, label='Shortest Safe Route')
    # Mark the dynamically chosen exit point
    ax.plot(target_x, target_y, 'mo', markersize=8, label='Optimal Exit Point')

# Draw Start Point
ax.plot(start_x, start_y, 'bo', markersize=8, label='Start Point')

# Highlight the "Finish Line" boundary
ax.axhline(y=map_length, color='purple', linestyle='--', alpha=0.5, label='Target Boundary')

ax.set_xlabel('Width (meters)')
ax.set_ylabel('Length (meters)')
ax.set_title('Dynamic Exit Safe Path Generation')
ax.legend(loc='lower left', framealpha=0.9)

plt.tight_layout()
plt.savefig('dynamic_safe_path.png', dpi=300)
print("Visual map saved as 'dynamic_safe_path.png'.")