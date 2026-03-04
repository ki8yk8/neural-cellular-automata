"""
creates utility function to work with grids like converting a given image to grid, visualizing it, etc.
"""
import torch
from torch.nn.functional import conv2d
from torchvision.io import decode_image
from torch.nn.functional import max_pool2d

from .utils import resize_image

def create_image_grid(path, n=1, channel=16, height=128, width=128):
	# grid shape (C, H, W)
	grid = torch.zeros((channel, height, width))

	image_with_alpha = (decode_image(path, mode="RGBA")/255.0).clip(0.0, 1.0)
	resized_image = resize_image(image_with_alpha)

	# centering the image into the grid
	_, grid_h, grid_w = grid.shape
	_, img_h, img_w = resized_image.shape

	x_offset = (grid_w-img_w)//2
	y_offset = (grid_h-img_h)//2

	# copies the rgba channels from resized image
	grid[:4, y_offset:y_offset+img_h, x_offset:x_offset+img_w] = resized_image
	return grid.repeat(n, 1, 1, 1)

def create_seed(n=1, channel=16, height=128, width=128):
	grid = torch.zeros((n, channel, height, width))

	# except rgb values all the channels is set to 1.0 for the center of grid
	grid[:, 3:, height//2, width//2] = 1.0

	return grid

def get_living_mask(grid, threshold=0.1):
	"""
	returns the mask for alive cells only. a cell is alive the cell or atleast one of the neirghbour hood cell has alpha > 0.1. The image is to be formed for only the alive cell other becomes dead instantly.
	"""
	if len(grid.shape) == 3:
		# batch dimension not present then
		grid = grid.unsqueeze(dim=0)
	
	pool = max_pool2d(grid[:, 3, :, :], kernel_size=3, stride=1, padding=1)
	return pool > threshold

def update(grid, delta, stochastic=0.5):
	"""
	updates the self.grid with new_state through stochastic update and alive cell masking
	"""
	stocastic_mask = torch.rand_like(delta) < stochastic
	delta = delta.squeeze(0) * stocastic_mask.squeeze(0)
	
	grid = grid+delta
	living_mask = get_living_mask(grid)
	grid = grid*living_mask

	return grid
