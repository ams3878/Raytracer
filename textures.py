'''
THIS DEFINES THE DIFFERENT TYPES OF TEXTURES THAT CAN BE APPLIED TO OBJECTS

'''

import classes as rt
import numpy as np
import png

from intersects import dot, vector, norm, distance, cross
def solid(color):
	return lambda x: np.array(color)

def bi_sphere(c1, c2):
	return lambda x: np.array(c1) if (x[0] > 0) else np.array(c2)

def checkerboard_sphere(c_1, c_2, ph, pw, corners, noise=lambda a, b: a):
	return lambda x: np.array(noise(c_1, x), dtype='float64') if \
		bool((x[0] // pw) % 2) ^ bool((x[1] // ph) % 2)\
		else np.array(noise(c_2, x), dtype='float64')

# corners : [top left, top right, bottom left, bottom right]
# center : world cords
# noise(a,b) where a is color and b is intersection point
def checkerboard_plane(c_1, c_2, ph, pw, corners, noise=lambda a, b: a):
	w_vec = vector(corners[0], corners[1])
	w_vec /= norm(w_vec)
	h_vec = vector(corners[0], corners[2])
	h_vec /= norm(h_vec)
	return lambda x, y: np.array(noise(c_1, x), dtype='float64') if \
		bool((dot(x - corners[0], w_vec) // pw) % 2) ^ bool((dot(x - corners[0], h_vec) // ph) % 2)\
		else np.array(noise(c_2, x), dtype='float64')

def image_plane(fn, corners, noise=lambda a, b: a):
	w_vec = vector(corners[0], corners[1])
	w_vec /= norm(w_vec)
	h_vec = vector(corners[0], corners[2])
	h_vec /= norm(h_vec)
	w_len = distance(corners[0], corners[1])
	h_len = distance(corners[0], corners[2])
	r = png.Reader(file=open(fn + ".png", 'rb')).read()
	color_map = np.vstack(map(np.float64, r[2]))
	color_map /= 256
	color_map = np.reshape(color_map, (r[1], r[0], 3))
	w_mod = r[0]/w_len
	h_mod = r[1]/h_len
	return lambda x: color_map[int(dot(x - corners[0], h_vec) * h_mod)][int(dot(x - corners[0], w_vec)*w_mod)]

def brick_plane(c_1, c_2, corners, ph=3, pw=3, space=.25, noise=lambda a, b: a):
	w_vec = vector(corners[0], corners[1])
	w_vec /= norm(w_vec)
	h_vec = vector(corners[2], corners[0])
	h_vec /= norm(h_vec)
	height_total = ph + space
	surface_normal = np.array(cross(w_vec, h_vec), dtype='float64')
	return [
		lambda x: np.array(noise(c_1, x), dtype='float64') if ((dot(x - corners[2], h_vec) % (ph+space)) < ph) and
		(((dot(x - corners[0], w_vec) + ((dot(x - corners[2], h_vec) // (ph+space)) % 2)*(pw+space)/2) % (pw+space)) < pw)
		else np.array(noise(c_2, x), dtype='float64'),
		lambda x: surface_normal if ((dot(x - corners[2], h_vec) % (ph+space)) < ph) and
		(((dot(x - corners[0], w_vec) + ((dot(x - corners[2], h_vec) // (ph+space)) % 2)*(pw+space)/2) % (pw+space)) < pw)
		else surface_normal * .85]
