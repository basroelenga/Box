from __future__ import division

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np

def plot2d(points, c, lbl):
	
	c_point = c + "."
	
	plt.plot(points[0], points[1], c_point, label=lbl)
	
	# Reorder
	indx = [0, 1, 3, 2, 0]
	
	# Split points
	fp = np.transpose(points)[:4].tolist()
	fp.append(fp[0])
	
	fpt = np.transpose(np.asarray(fp)[indx])
	
	plt.plot(fpt[0], fpt[1], c)
	
	plt.legend()

def plot3d(points, c, lbl):
	
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

def multiply2d(m, v_list):
	
	box = []
	
	for v in np.transpose(v_list):
	
		x = m[0][0] * v[0] + m[0][1] * v[1]
		y = m[1][0] * v[0] + m[1][1] * v[1]
	
		box.append([x, y])

	return np.transpose(box) 

def multiply3d(m, v_list):
	
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

def transform2d(points, x, y):

	point_list = []
	points_t = np.transpose(points)

	for i in range(0, len(points_t)):
		
		xx = points_t[i][0] + x
		yy = points_t[i][1] + y
		
		point_list.append([xx, yy])
		
	return np.transpose(point_list)
	
def create_box2d(w, h, z_near):
	
	w2 = w / 2
	h2 = h / 2
	
	x = [w2, w2, -w2, -w2]
	y = [h2, -h2, h2, -h2]
	
	return [x, y]
	
def create_box3d(w, h, d):

	w2 = w / 2
	h2 = h / 2
	d2 = d / 2

	x = [w2, w2, w2, w2, -w2, -w2, -w2, -w2]
	y = [h2, h2, -h2, -h2, h2, h2, -h2, -h2]
	z = [-d2, d2, -d2, d2, -d2, d2, -d2, d2]
	
	return [x, y, z]

def create_frustum2d(fov, z_near, z_far):
	
	width_near = z_near * np.tan(np.deg2rad(fov / 2))
	width_far = z_far * np.tan(np.deg2rad(fov / 2))
	
	x = [width_near, -width_near, width_far, -width_far]
	y = [z_near, z_near, z_far, z_far]
		
	return [x, y]

def create_frustum3d(fov, aspect, z_near, z_far):
	
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

frustum = create_frustum2d(75, 1, 10)

# !Create the box not centered but at a distance of z_near!
# Rotation of the light
theta_l = 45

# Rotation of the camera
theta_c = 31

# Rotate the frustum
m_c_theta = rotate_euler_z(theta_c)
frustum_r = multiply2d(m_c_theta, frustum) 

# Rotate to light frame
m_l_theta = rotate_euler_z(theta_l)
frustum_l = multiply2d(m_l_theta, frustum_r)

# Find minimum and maximum
x_diff = np.max(frustum_l[0]) - np.min(frustum_l[0])
y_diff = np.max(frustum_l[1]) - np.min(frustum_l[1])

# Create a box
box = create_box2d(x_diff, y_diff, z_near)

# Rotate the box back to normal frame
m_n_theta = rotate_euler_z(-theta_l)
box_r = multiply2d(m_n_theta, box)

# Locate the rotated frustum
x_loc = ((np.max(frustum_r[0]) - np.min(frustum_r[0])) / 2) + np.min(frustum_r[0])
y_loc = ((np.max(frustum_r[1]) - np.min(frustum_r[1])) / 2) + np.min(frustum_r[1])

# Translation of box
print(x_loc, y_loc)

# Translate the box
box_tr = transform2d(box_r, x_loc, y_loc)

fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)

plot2d(frustum, "b", "Frustum")
plot2d(frustum_r, "r", "Camera rotated frustum")
plot2d(frustum_l, "g", "Both rotated frustum")
plot2d(box, "c", "Box")
plot2d(box_r, "y", "Box rotated")
plot2d(box_tr, "k", "Box rotated/transelated")

plt.title("Bounding box")

ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
#ax.set_zlim(-8, 8)

ax.set_xlabel("X")
ax.set_ylabel("Y")
#ax.set_zlabel("Z")

plt.show()