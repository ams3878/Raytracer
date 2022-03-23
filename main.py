'''
Main function that makes all the calls to the other files.
Biggest thing of note in here is how the world is defined.
 @line 128 : the dictionary created here defines the allowed shapes in the world
             All objects added must match one of these shapes
'''

import classes as rt
import intersects as inter
import textures as text
import numpy as np
import time
import random
from urllib.request import urlopen
setup_start = time.time()

reflection_default = {'blin': False, 'model': "phong"}
#reflection_default = {'blin': False, 'model': "strauss", 's': .5, 'm': .8}
a = rt.Vector((-3, -13, -4)),  rt.Vector((-19, 65, 2)), rt.Vector((-43.5, 48, 2)), rt.Vector((22.5, 4, -4))
a_points = [a[2].p, a[1].p, a[0].p, a[3].p]



textures = []
#0
textures.append(rt.Texture(cm=text.solid((.114, .8, .107)), rm=0, tm=.75, rfm=1))
#1
textures.append(rt.Texture(cm=text.solid((.657, .052, .8)), rm=.8, tm=0))
#2
textures.append(rt.Texture(cm=text.checkerboard_plane((.984, .921, .011), (.984, .289, .011), 5, 5, a_points)))
random_noise = lambda a, x: np.array(a)*(random.randint(85, 99)/100)
random_noise_2 = lambda a, x: (a[0]*random.random(), a[1]*random.random(), a[2]*random.random())
#3
textures.append(rt.Texture(cm=text.checkerboard_plane((.91, .91, .91), (1, 1, 1), 5, 5, a_points, random_noise_2)))
#4
textures.append(rt.Texture(cm=text.image_plane('whatmap', a_points)))
brickers = text.brick_plane((.3125, .09765, .1289), (.3359, .3125, .3164), a_points, noise=random_noise)
#5
textures.append(rt.Texture(cm=brickers[0], nm=brickers[1]))
#6
textures.append(rt.Texture(cm=text.solid((.657, .052, .8))))

objects = []
xxx = []
xxx += [rt.Object("Close Sphere", "Sphere", {"xyz": [4.69, 1.26, 4.05], "r": 4.25}
											, refl=reflection_default, text=textures[0])]
xxx += [rt.Object("Far Sphere", "Sphere", {"xyz": [5.93, 9.97, 2.36], "r": 3.73}
											,  refl=reflection_default, text=textures[1])]
xxx += [rt.Object("Closer Sphere", "Sphere", {"xyz": [-5, -5, 8.36], "r": 3.73}
											,  refl=reflection_default, text=textures[6])]


xxx += [rt.Object(n="floor", m="Plane", m_a={
	"xyz":  [-11, 26, -1],
	"left": rt.Object("Left floor", "Triangle",
										{"vertices": [a[0], a[1], a[2]],
										"vertices_points": [a[0].p, a[1].p, a[2].p],
										"xyz": [-11, 26, -1]}, refl=reflection_default, text=textures[2]),
	"right": rt.Object("Right floor", "Triangle",
										{"vertices": [a[0], a[3], a[1]],
										"vertices_points": [a[0].p, a[3].p, a[1].p],
										"xyz":  [-11, 26, -1]}, refl=reflection_default, text=textures[2])
}, refl=reflection_default, text=textures[2])]
a = [rt.Vector((.69, -3.26, 5.3)), rt.Vector((8.69, 5.26, 5.3)), rt.Vector((.69, 5.26, 5.3)),
		rt.Vector((8.69, -3.26, 5.3)), rt.Vector((4.69, 1.26, 15.3))]
objects += [
		rt.Object("Right floor", "Triangle", {
			"vertices": [a[0], a[1], a[2]], "vertices_points": [a[0].p, a[1].p, a[2].p],	"xyz": a[2].p},
							refl=reflection_default, text=textures[6]),
		rt.Object("Right floor", "Triangle", {
			"vertices": [a[0], a[3], a[4]],	"vertices_points": [a[0].p, a[3].p, a[4].p], "xyz": a[4].p},
							refl=reflection_default, text=textures[6]),
		rt.Object("Right floor", "Triangle", {
			"vertices": [a[3], a[1], a[4]],	"vertices_points": [a[3].p, a[1].p, a[4].p], "xyz": a[4].p},
							refl=reflection_default, text=textures[6]),
		rt.Object("Right floor", "Triangle", {
			"vertices": [a[0], a[3], a[1]], "vertices_points": [a[0].p, a[3].p, a[1].p],	"xyz": a[3].p},
							refl=reflection_default, text=textures[6]),
		rt.Object("Right floor", "Triangle", {
			"vertices": [a[0], a[2], a[4]],	"vertices_points": [a[0].p, a[2].p, a[4].p], "xyz": a[4].p},
							refl=reflection_default, text=textures[6]),
		rt.Object("Right floor", "Triangle", {
			"vertices": [a[2], a[1], a[4]],	"vertices_points": [a[2].p, a[1].p, a[4].p], "xyz": a[4].p},
							refl=reflection_default, text=textures[6])
	]
'''
objects += [rt.Object(n="hat", m="Pyramid", m_a={
	"xyz": [4.69, 1.26, 4.3],
	"faces": [
		rt.Object((.59, .02, .031), "Right floor", "Triangle", {
			"vertices": [a[0], a[1], a[2]], "vertices_points": [a[0].p, a[1].p, a[2].p],	"xyz": a[2].p}, refl=reflection_default),
		rt.Object((.59, .01, .031), "Right floor", "Triangle", {
			"vertices": [a[0], a[3], a[4]],	"vertices_points": [a[0].p, a[3].p, a[4].p], "xyz": a[4].p}, refl=reflection_default),
		rt.Object((.59, .01, .031), "Right floor", "Triangle", {
			"vertices": [a[3], a[1], a[4]],	"vertices_points": [a[3].p, a[1].p, a[4].p], "xyz": a[4].p}, refl=reflection_default),
		rt.Object((.59, .01, .031), "Right floor", "Triangle", {
			"vertices": [a[0], a[3], a[1]], "vertices_points": [a[0].p, a[3].p, a[1].p],	"xyz": a[3].p}, refl=reflection_default),
		rt.Object((.59, .01, .031), "Right floor", "Triangle", {
			"vertices": [a[0], a[2], a[4]],	"vertices_points": [a[0].p, a[2].p, a[4].p], "xyz": a[4].p}, refl=reflection_default),
		rt.Object((.59, .01, .031), "Right floor", "Triangle", {
			"vertices": [a[2], a[1], a[4]],	"vertices_points": [a[2].p, a[1].p, a[4].p], "xyz": a[4].p}, refl=reflection_default)
	]
}, refl=reflection_default)]
'''
base =100000
high_light = 200 * base
med_light = 20 * base
low_light = 5 * base
lights = [rt.Light(m_a={"xyz": (15, -15, 40), "intensity": high_light})]
objects += lights
#objects += [rt.Light(m_a={"xyz": (35, -25, 40), "intensity": 1})]
#objects += [rt.Light(m_a={"xyz": (0, 0, 0), "intensity": 10})]

cam_rotate = (85, 0, 42)
cam_up = rt.Vector((0, 1, 0)).rotate(cam_rotate, origin=(30, -25, 5))
cam_la = rt.Vector((30, -25, 2)).rotate(cam_rotate, origin=(30, -25, 5))
aspect = np.array((1.5, 1), dtype="float64")
samples_per_pixel = 2  # actually recursive limit
camera_list = [
	# 0 - camera_speeeeeeeeed
	rt.Camera(la=cam_la.p, n="Camera 1", m_a={"xyz": (30, -25, 5)}, res=aspect*50, vp_w=.03, vp_h=.02, f=.05,
						sampling=samples_per_pixel),
	# 1 - camera_fast
	rt.Camera(la=cam_la.p, n="Camera 1", m_a={"xyz": (30, -25, 5)}, res=aspect*100, vp_w=.03, vp_h=.02, f=.05,
						sampling=samples_per_pixel),
	# 2 - camera_norm
	rt.Camera(la=cam_la.p, n="Camera 1", m_a={"xyz": (30, -25, 5)}, res=aspect*200, vp_w=.03, vp_h=.02, f=.05,
						sampling=samples_per_pixel),
	# 3 - camera_final
	rt.Camera(la=cam_la.p, n="Camera 1", m_a={"xyz": (30, -25, 5)}, res=aspect*400, vp_w=.03, vp_h=.02, f=.05,
						sampling=samples_per_pixel),
	# 4 - camera_final
	rt.Camera(la=cam_la.p, n="Camera 1", m_a={"xyz": (30, -25, 5)}, res=aspect*600, vp_w=.03, vp_h=.02, f=.05,
						sampling=samples_per_pixel),
	# 5 - camera_opposite
	rt.Camera(la=[4.93, 9.97, 2.36], n="Camera 1", m_a={"xyz": rt.Vector((30, -25, 5)).rotate((180, 180, 0)).p},
						res=aspect*200, vp_w=.03, vp_h=.02, f=.05, sampling=samples_per_pixel),
	# 6 - camera_zoom_out
	rt.Camera(la=[4.93, 9.97, 2.36], n="Camera 1", m_a={"xyz": [30, -25, 10]}, res=aspect*150, vp_w=.03, vp_h=.02, f=.05,
						sampling=samples_per_pixel),
	# 7 - camera_uuuuge
	rt.Camera(la=cam_la.p, n="Camera 1", m_a={"xyz": (30, -25, 5)}, res=aspect * 1000, vp_w=.03, vp_h=.02, f=.05,
						sampling=samples_per_pixel),
	# 8 - camera_uuuugerrr
	rt.Camera(la=cam_la.p, n="Camera 1", m_a={"xyz": (30, -25, 5)}, res=aspect * 4000, vp_w=.03, vp_h=.02, f=.05,
						sampling=samples_per_pixel),
]
camera_1 = camera_list[2]
the_world = rt.World(s={"Point": rt.Model(['xyz'], inter.point, "Point"),
												"Sphere": rt.Model(['xyz', 'r'], inter.sphere, "Sphere"),
												"Triangle": rt.Model(['xyz', 'vertices'], inter.triangle, "Triangle"),
												"Pyramid": rt.Model(['xyz', 'faces'], inter.pyramid, "Pyramid"),
												"Plane": rt.Model(['xyz', 'left', 'right'], inter.plane, "plane")
												}, c=(.007, .023, .34), o=objects, ambiance=400000)
s = time.time()
setup_end = time.time()
print(f"World setup complete after {setup_end - setup_start}s\nBeginning render...", end='')
render_start = time.time()
camera_1.render(the_world)
render_end = time.time()
print(f"Render complete after {render_end - render_start}s\nBeginning synth...", end='')
img_name = "test"
rt.Image(save_name=f'no_tr_{img_name}_{camera_1.res[0]}_{camera_1.res[1]}', save=False, cam=camera_1, tr_method='NONE')
rt.Image(save_name=f'{img_name}_{camera_1.res[0]}_{camera_1.res[1]}', save=False, cam=camera_1,
				 MAX_DISPLAY_ILLUMINANCE=200, tr_method='log_adaptive',  p={'b': .85})


print(f"Image creating complete.\n\tTotal time: {time.time() - setup_start}s")
print(f"\tImage res: {camera_1.res}\n\tTotal initial rays:{camera_1.res[0]*camera_1.res[1]}")
print(f"\tTime per Initial ray:{(render_end-render_start)/(camera_1.res[0]*camera_1.res[1])}")
if_blin = "-blin" if reflection_default['blin'] else ""
print(f"\tIllumination Algo: {reflection_default['model']}{if_blin}")
print(f"\tSuper-Sampling: {samples_per_pixel != 1}")
print(f"\tNumber of Objects: {len(objects) - len(lights)}")
print(f"\tNumber of Lights: {len(lights)}")


