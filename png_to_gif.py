from PIL import Image
import glob
def make_gif():
	# Create the frames
	frames = []
	imgs = glob.glob("Animation_0/*.png")
	print(f"Image Len = {len(imgs)}")
	print(f"frames Len = {len(frames)}")
	pop_list = []
	for i in imgs:
		if i[19] == '_':
			new_frame = Image.open(i)
			frames.append(new_frame)
			pop_list.append(i)

	for x in pop_list:
		imgs.remove(x)
	print(f"Image Len = {len(imgs)}")
	print(f"frames Len = {len(frames)}")

	pop_list = []
	for i in imgs:
		if i[20] == '_':
			new_frame = Image.open(i)
			frames.append(new_frame)
			pop_list.append(i)

	for x in pop_list:
		imgs.remove(x)
	print(f"Image Len = {len(imgs)}")
	print(f"frames Len = {len(frames)}")

	for i in imgs:
		new_frame = Image.open(i)
		frames.append(new_frame)

	# Save into a GIF file that loops forever
	frames[0].save('png_to_gif_test.gif', format='GIF',
								 append_images=frames,optimize=False,
								 save_all=True,
								 duration=33.3333, loop=0)