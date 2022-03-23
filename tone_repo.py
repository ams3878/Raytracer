'''
Simple script to call tone reproduction on an already rendered pixel values
creating an instance of the Image class creates a .png
'''

import classes as rt
from numpy import array
objects = []
light_level = 'HIGH'
filename = f'0000_{light_level}_raw_900_600'
with open(f'{filename}', 'r') as f:
	x = f.readlines()
	p = x[1].split()
	n_rc = x[0].split()
	n_c = int(n_rc[1])
	n_r = int(n_rc[0])
	vp = []
	for r in range(n_r):
		t = []
		for c in range(n_c):
			t.append([float((p[(r * n_c*3) + 3*c])),
								float((p[(r * n_c*3) + 3*c + 1])),
								float((p[(r * n_c*3) + 3*c + 2]))])
		vp.append(t)
	vp = array(vp)
gam = 4
b = .55
rt.Image(save_name=f'{light_level}_log_b{b}g{gam}', tr_method='log_adaptive',
				MAX_DISPLAY_ILLUMINANCE=200, p={'b': b, 'pixels': vp}, gamma=2.2)

gam = 2.2
rt.Image(save_name=f'{light_level}_w_g{gam}', tr_method='ward',
				p={'pixels': vp}, gamma=gam)
a = .18
gam = 1.4
rt.Image(save_name=f'{light_level}_r_a{a}g{gam}', tr_method='reinhard',
				p={'a': a, 'pixels': vp}, gamma=gam)
