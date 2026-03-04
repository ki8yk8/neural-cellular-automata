import warnings
import imageio
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Resize

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

# IMAGE Utility Functions
def grid2img(grid, path):
	"""
	use to convert the cellular automata grid into image.
	"""
	if len(grid.shape) > 3:
		grid = grid[0]
		warnings.warn(f"Encountered grid with batch dimension, {grid.shape}, considers only the first sample", UserWarning)
	
	image = grid.detach()[:4, :, :]

	# transposing image back to h, w, c
	image = image.permute((1, 2, 0)).clip(0.0, 1.0)

	if path:
		plt.tight_layout()
		plt.axis("off")
		plt.imsave(path, image)
		plt.close()

	return image

def resize_image(image, max_size=128):
	"""
	returns the resized image for a given input image by shrinking the largest slide to max_size
	"""
	_, h, w = image.shape
	scale = max_size/max(h, w)
	h, w = int(h*scale), int(w*scale)

	return Resize((h, w))(image)
