import classes as rt
import intersects as inter
import textures as text
import numpy as np
import time
import random
from urllib.request import urlopen
setup_start = time.time()
reflection_default = {'blin': True, 'model': "phong"}
#reflection_default = {'blin': False, 'model': "strauss", 's': .5, 'm': .8}
a = rt.Vector((-3, -13, -4)),  rt.Vector((-19, 65, 2)), rt.Vector((-43.5, 48, 2)), rt.Vector((22.5, 4, -4))
a = rt.Vector((-10, -10, -10)),  rt.Vector((10, -10, 10)), rt.Vector((-10, -10, 10)), rt.Vector((10, -10, -10))

a_points = [a[2].p, a[1].p, a[0].p, a[3].p]



textures = []
#0
textures.append(rt.Texture(cm=text.solid((.114, .8, .107)), rm=0, tm=.8, rfm=1))
#1
textures.append(rt.Texture(cm=text.solid((.657, .052, .8)), rm=0))
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
xxx += [rt.Object("Close Sphere", "Sphere", {"xyz": [0, 0, 0], "r": 4.25}
											, refl=reflection_default, text=textures[0])]
xxx += [rt.Object("Far Sphere", "Sphere", {"xyz": [-10, 0, 0], "r": 3.73}
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

with open('bun_zipper_res4.ply', 'r') as bn:
	x = bn.readline()
	count = 0
	while x:
		count += 1
		if x == 'end_header\n':
			break
		try:
			if x.split()[1] == 'vertex':
				num_vert = int(x.split()[2])
			if x.split()[1] == 'face':
				num_face = int(x.split()[2])
		except IndexError:
			x = bn.readline()
			continue
		x = bn.readline()
	verts = []
	for i in range(num_vert):
		verts += [[-100*float(t) for t in bn.readline().split()[0:3]]]
	# Assuming all triangles (BAD)
	for i in range(num_face):
		x = [int(t) for t in bn.readline().split()[0:4]]
		a = rt.Vector(verts[x[1]]), rt.Vector(verts[x[2]]), rt.Vector(verts[x[3]])
		objects += [rt.Object(f"t{i}", "Triangle", {
			"vertices": [a[0], a[1], a[2]],
			"vertices_points": [a[0].p, a[1].p, a[2].p],
			"xyz": a[0].p}, refl=reflection_default, text=textures[1])]
#objects += [rt.Object("Close Sphere", "Sphere", {"xyz": [0, 0, 0], "r": 2}
											#, refl=reflection_default, text=textures[0])]

lights = [rt.Light(m_a={"xyz": (0, 10, 0), "intensity": 10000})]
lights += [rt.Light(m_a={"xyz": (0, 0, 50), "intensity": 100})]
#lights += [rt.Light(m_a={"xyz": (0, 0, 50), "intensity": 10000})]

objects += lights

cam_up = rt.Vector((0, 0, 1))
cam_la = rt.Vector((0, 0, 0))
aspect = np.array((1, 1), dtype="float64")
samples_per_pixel = 2  # actually recursive limit
camera_list = [
	# 0 - camera_speeeeeeeeed
	rt.Camera(la=cam_la.p, n="Camera 1", m_a={"xyz": (0, 50, 0)}, res=aspect*50, vp_w=.02, vp_h=.02, f=.05,
						sampling=samples_per_pixel),
	# 1 - camera_fast
	rt.Camera(la=cam_la.p, n="Camera 1", m_a={"xyz": (30, -25, 5)}, res=aspect*100, vp_w=.02, vp_h=.02, f=.05,
						sampling=samples_per_pixel),
]
camera_1 = camera_list[0]
the_world = rt.World(s={"Point": rt.Model(['xyz'], inter.point, "Point"),
												"Sphere": rt.Model(['xyz', 'r'], inter.sphere, "Sphere"),
												"Triangle": rt.Model(['xyz', 'vertices'], inter.triangle, "Triangle"),
												"Pyramid": rt.Model(['xyz', 'faces'], inter.pyramid, "Pyramid"),
												"Plane": rt.Model(['xyz', 'left', 'right'], inter.plane, "plane")
												}, c=(.007, .023, .34), o=objects, bny=verts, ambiance=10000000)
print(f"NUMBER OF BUNNY TRIANGLES...{len(objects)-len(lights)}")
s = time.time()
setup_end = time.time()
print(f"World setup complete after {setup_end - setup_start}s\nBeginning render...", end='')
render_start = time.time()
camera_1.render(the_world)
render_end = time.time()
print(f"Render complete after {render_end - render_start}s\nBeginning synth...", end='')

rt.Image(save_name='BUNNY', world=the_world, cam=camera_1, tr_method='uniform_scale', MAX_DISPLAY_ILLUMINANCE=1)

print(f"Image creating complete.\n\tTotal time: {time.time() - setup_start}s")
print(f"\tImage res: {camera_1.res}\n\tTotal initial rays:{camera_1.res[0]*camera_1.res[1]}")
print(f"\tTime per Initial ray:{(render_end-render_start)/(camera_1.res[0]*camera_1.res[1])}")
if_blin = "-blin" if reflection_default['blin'] else ""
print(f"\tIllumination Algo: {reflection_default['model']}{if_blin}")
print(f"\tSuper-Sampling: {samples_per_pixel != 1}")
print(f"\tNumber of Objects: {len(objects) - len(lights)}")
print(f"\tNumber of Lights: {len(lights)}")

