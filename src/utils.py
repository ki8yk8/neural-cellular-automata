import imageio
import os
from PIL import Image
import numpy as np

def create_video(image_folder, output_path, fps=4):
	images = sorted([
		f for f in os.listdir(image_folder) if f.endswith(".png")
	],
		key=lambda x: int(x.split(".")[0])
	)

	frames = []
	for img_names in images:
		img = Image.open(os.path.join(image_folder, img_names)).convert("RGB")
		frames.append(np.array(img))

	imageio.mimsave(output_path, frames, fps=fps)
