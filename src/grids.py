"""
creates utility function to work with grids like converting a given image to grid, visualizing it, etc.
"""
import torch
from torch.nn.functional import conv2d
from torchvision.io import decode_image
from torchvision.transforms import Resize
from torch.nn.functional import max_pool2d
import matplotlib.pyplot as plt

# constant represnting filters for the operators
SOVEL_X = torch.tensor([[
	[-1, 0, 1],
	[-2, 0, 2],
	[-1, 0, 1],
]], dtype=torch.float32).unsqueeze(dim=0).repeat(16, 1, 1, 1)
SOVEL_Y = SOVEL_X.transpose(2, 3)

class Grid:
	def __init__(self, height=128, width=128):
		"""
		grid = (channel, rows, cols) where, channels = (r, g, b, alpha, ....) and alpha = [0, 1] where a cell is living if it's or it's neighbourhood's alpha > 0.1 else it's dead
		"""
		self.height = height
		self.width = width
		self.grid = torch.zeros((16, height, width))

		self.stochasticity = 0.5

	def copy_image(self, path):
		image_with_alpha = (decode_image(path, mode="RGBA")/255.0).clip(0.0, 1.0)
		image_with_alpha[:3] = image_with_alpha[:3]*image_with_alpha[3]
		resized_image = self.resize_image(image_with_alpha)

		# centering the image into the grid
		_, grid_h, grid_w = self.grid.shape
		_, img_h, img_w = resized_image.shape

		x_offset = (grid_w-img_w)//2
		y_offset = (grid_h-img_h)//2

		# copies the 3 channels of resized image
		self.grid[:4, y_offset:y_offset+img_h, x_offset:x_offset+img_w] = resized_image

	def clear(self):
		self.grid = torch.zeros((16, self.height, self.width))
		self.grid[:4, :, :] = 1

	def copy_seed(self):
		self.grid[:, self.height//2, self.width//2] = 0.0
		self.grid[3:, self.height//2, self.width//2] = 1.0

	def grid2img(self, path, consider_dead=False):
		"""
		use to convert the cellular automata grid into image.
		"""
		alive_cells_mask = self.get_alive_mask()
		image = self.grid[:3, :, :].detach()

		if not consider_dead:
			image = image*alive_cells_mask
		
		# transposing image back to h, w, c
		image = image.permute((1, 2, 0)).clip(0.0, 1.0)

		if path:
			plt.tight_layout()
			plt.axis("off")
			plt.imsave(path, image)
			plt.close()

		return image

	def get_alive_mask(self, alive_threshold=0.1):
		"""
		returns the mask for alive cells only. a cell is alive the cell or atleast one of the neirghbour hood cell has alpha >= 0.1. The image is to be formed for only the alive cell other becomes dead instantly.
		"""
		pool = max_pool2d(self.grid[3, :, :].unsqueeze(0), kernel_size=3, stride=1, padding=1)
		return pool > alive_threshold

	def resize_image(self, image, max_size=128):
		"""
		returns the resized image for a given input image by shrinking the largest slide to max_size
		"""
		_, h, w = image.shape
		scale = max_size/max(h, w)
		h, w = int(h*scale), int(w*scale)

		return Resize((h, w))(image)

	def get_perception_vector(self):
		"""
		returns perception vector formed from the grid by concatenating sovel x, y, and identity
		"""
		grid = self.grid.unsqueeze(dim=0)

		sovel_x = conv2d(grid, SOVEL_X, padding=1, groups=16)
		sovel_y = conv2d(grid, SOVEL_Y, padding=1, groups=16)

		# return concatenated output
		return torch.cat((grid, sovel_x, sovel_y), dim=1)
	
	def update(self, new_state, use_mask=True, stochastic=True):
		"""
		updates the self.grid with new_state through stochastic update and alive cell masking
		"""
		stocastic_mask = torch.rand_like(new_state) < self.stochasticity

		if stochastic:
			new_state = new_state.squeeze(0) * stocastic_mask.squeeze(0)
		else:
			new_state = new_state.squeeze(0)
		
		self.grid = self.grid + new_state

		# only alive cell gets to the next step
		if use_mask:
			alive_mask = self.get_alive_mask()
			self.grid = self.grid * alive_mask
