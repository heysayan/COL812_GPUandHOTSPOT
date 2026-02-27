import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation

# File containing block temperature data
filename = "full_temperature_mem.trace"

# Lists to hold parsed data: each block is defined by (layer, row, col) and its temperature
temps = []   # will store corresponding temperature values
names = []
# Read and parse the file
with open(filename, 'r') as f:
    names = f.readline().split()
    for line in f.readlines():
        temps.append([float(temp) for temp in line.split()])

# Determine how many layers are present (assuming layers are numbered starting at 0)
num_layers = 9

# We assume each layer is a 4x4 grid.
nx, ny, nz = 4, 4, num_layers  # x: columns, y: rows, z: layers

# Create a boolean occupancy array for voxels and an array to store the face colors.
voxels = np.zeros((nx, ny, nz), dtype=bool)
facecolors = np.empty((nx, ny, nz), dtype=object)

# Normalize temperatures for colormap mapping
t = [r for temp in temps for r in temp]
temp_min = min(t)
temp_max = max(t)
norm = plt.Normalize(temp_min, temp_max)
cmap = plt.cm.viridis

# Populate the voxel grid.
# Here we map:
#   x coordinate ← column index,
#   y coordinate ← row index,
#   z coordinate ← layer index.



for i in range(len(temps[0])):
    layer = i//16 ; col = (i%16)//4 ; row = (i%16)%4
    voxels[col, row, layer] = True  # mark the cube as present
    facecolors[col, row, layer] = cmap(norm(temps[0][i]))


def update_voxels(time_val):

    for i in range(len(temps[time_val])):
        layer = i//16 ; col = (i%16)//4 ; row = (i%16)%4
        voxels[col, row, layer] = True  # mark the cube as present
        facecolors[col, row, layer] = cmap(norm(temps[time_val][i]))


# Create the 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
voxel_collection = ax.voxels(voxels, facecolors=facecolors, edgecolor='k')
# for i in range(len(names)):
#     # Compute the center of the voxel. Since each voxel has unit length,
#     # the center is at (col+0.5, row+0.5, layer+0.5)
#     layer = i//16 ; col = (i%16)//4 ; row = (i%16)%4
#     x = col
#     y = row
#     z = layer + 0.5
#     # Annotate with the temperature (formatted to one decimal place)
#     ax.text(x, y, z, names[i], color='black',
#             ha='center', va='center', fontsize=8)
# To add a colorbar, create a ScalarMappable
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Only needed for older versions of Matplotlib
cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Temperature')

# Label the axes
ax.set_xlabel('X (Column)')
ax.set_ylabel('Y (Row)')
ax.set_zlabel('Z (Layer)')
ax.set_title('when read/write access rate is 1M for alternating half times')


def animate(frame):
    update_voxels(frame)
    
    # Remove the previous voxel collection by clearing the axis
    ax.collections.clear()

    # Plot new voxels for the current time step.
    ax.voxels(voxels, facecolors=facecolors, edgecolor='k')
    for key, poly in voxel_collection.items():
        col, row, layer = key
        poly.set_facecolor(facecolors[col, row, layer])
    return voxel_collection.values()

ani = FuncAnimation(fig, animate, frames=len(temps), interval=200, repeat=True)

plt.show()
