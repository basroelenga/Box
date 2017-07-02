from __future__ import division

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np

def plot(points, c, lbl):
	
	c_point = c + "."
	
	# Plot the points
	plt.plot(points[0], points[1], points[2], c_point, label=lbl)
	
	# Reorder
	indx = [0, 1, 3, 2, 0]
	
	# Split points
	fp = np.transpose(points)[:4].tolist()
	fp.append(fp[0])

	bp = np.transpose(points)[4:].tolist()
	bp.append(bp[0])
	
	# Transpose back
	fpt = np.transpose(np.asarray(fp)[indx])
	bpt = np.transpose(np.asarray(bp)[indx])
	
	plt.plot(fpt[0], fpt[1], fpt[2], c)
	plt.plot(bpt[0], bpt[1], bpt[2], c)
	
	# Connect the fields
	for i in range(0, 4):
		
		cpt = [(bpt[0][i], fpt[0][i]), (bpt[1][i], fpt[1][i]), (bpt[2][i], fpt[2][i])]
		plt.plot(cpt[0], cpt[1], cpt[2], c)
	
	plt.legend()

def multiply(m, v_list):
	
	box = []
	
	for v in np.transpose(v_list):
	
		x = m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2]
		y = m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2]
		z = m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2]
		
		box.append([x, y, z])

	return np.transpose(box)
	
def rotate_euler_z(angle):

	angle = np.radians(angle)

	# Create the rotation matrix
	m = np.asarray([np.cos(angle), -np.sin(angle), 0, np.sin(angle), np.cos(angle), 0, 0, 0, 1])
	m = np.reshape(m, (3, 3))
	
	return m

def rotate_euler_x(angle):

	angle = np.radians(angle)
	
	# Create the rotation matrix
	m = np.asarray([1, 0, 0, 0, np.cos(angle), -np.sin(angle), 0, np.sin(angle), np.cos(angle)])
	m = np.reshape(m, (3, 3))
	
	return m
	
def rotate_euler_y(angle):

	angle = np.radians(angle)
	
	# Create the rotation matrix
	m = np.asarray([np.cos(angle), 0, np.sin(angle), 0, 1, 0, -np.sin(angle), 0, np.cos(angle)])
	m = np.reshape(m, (3, 3))
	
	return m
	
def create_box(w, h, d):

	w2 = w / 2
	h2 = h / 2
	d2 = d / 2

	x = [w2, w2, w2, w2, -w2, -w2, -w2, -w2]
	y = [h2, h2, -h2, -h2, h2, h2, -h2, -h2]
	z = [-d2, d2, -d2, d2, -d2, d2, -d2, d2]
	
	return [x, y, z]

def create_frustum(fov, aspect, z_near, z_far):
	
	width_near = z_near * np.tan(np.deg2rad(fov / 2))
	height_near = width_near / aspect
	
	width_far = z_far * np.tan(np.deg2rad(fov / 2))
	height_far = width_far / aspect
	
	x = [width_near, -width_near, width_near, -width_near, width_far, -width_far, width_far, -width_far]
	y = [height_near, height_near, -height_near, -height_near, height_far, height_far, -height_far, -height_far]
	z = [z_near, z_near, z_near, z_near, z_far, z_far, z_far, z_far]
	
	return [x, y, z]
	
# Create a camera frustum
z_near = 1
z_far = 10

frustum = create_frustum(75, 16 / 9, 1, 10)

# Center the frustum around the zero point
for i in range(0, len(frustum[2])):
	frustum[2][i] -= (z_far + z_near) / 2

# Rotation of the light
theta_l = 0
phi_l = 0

# Rotation of the camera
theta_c = 0
phi_c = 0

# Rotate the frustum
m_c_theta = rotate_euler_y(theta_c)
m_c_phi = rotate_euler_x(phi_c)

frustum_r = multiply(m_c_theta, frustum) 
frustum_r = multiply(m_c_phi, frustum_r)

# Rotate to light frame
m_l_theta = rotate_euler_y(theta_l)
m_l_phi = rotate_euler_x(phi_l)

frustum_l = multiply(m_l_theta, frustum_r)
frustum_l = multiply(m_l_phi, frustum_l)

# Obtain the min and max
x_diff = np.max(frustum_l[0]) - np.min(frustum_l[0])
y_diff = np.max(frustum_l[1]) - np.min(frustum_l[1])
z_diff = np.max(frustum_l[2]) - np.min(frustum_l[2])

# Construct a box
box = create_box(x_diff, y_diff, z_diff)

# Rotate the box back to normal frame
m_n_theta = rotate_euler_y(-theta_l)
m_n_phi = rotate_euler_x(-phi_l)

box_l = multiply(m_n_theta, box)
box_l = multiply(m_n_phi, box_l)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#plot(frustum, "b", "Frustum")
plot(frustum_r, "r", "Rotated frustum")
#plot(frustum_l, "g", "Light rotated frustum")
plot(box_l, "c", "Box")

plt.title("Bounding box")

ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)
ax.set_zlim(-8, 8)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()