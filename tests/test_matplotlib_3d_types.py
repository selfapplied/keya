"""Test file to verify matplotlib 3D API type checking works correctly."""

import matplotlib.pyplot as plt

# Create 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Test the 4 originally problematic methods:

# 1. set_zlabel (was not recognized)
ax.set_zlabel('Z Axis Label')

# 2. set_zlim (was not recognized)  
ax.set_zlim(-5, 5)

# 3. text2D (was not recognized)
ax.text2D(0.1, 0.9, 'Test text', transform=ax.transAxes)

# 4. axis pane properties (were not recognized)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Test scatter with size parameter (was causing conflicts)
x = [1, 2, 3]
y = [4, 5, 6] 
z = [7, 8, 9]
sizes = [20, 40, 60]
ax.scatter(x, y, z, s=sizes)

print("✅ All matplotlib 3D API methods type-checked successfully!") 