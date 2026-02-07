"""
creates utility function to work with grids like converting a given image to grid, visualizing it, etc.
"""
import torch
from torchvision.io import decode_image
import matplotlib.pyplot as plt

class Grid:
	def __init__(self, height=128, width=128):
		"""
		grid = (channel, rows, cols) where, channels = (r, g, b, alpha, ....) and alpha = [0, 1] where a cell is living if it's or it's neighbourhood's alpha > 0.1 else it's dead
		"""
		self.grid = torch.zeros((16, height, width))

	def copy_image(self, path, verbose=False):
		image = decode_image(path, mode="RGB")
		# resized_image = self.resize_image(image)

		if verbose:
			plt.imshow(torch.permute(image, (1, 2, 0)))
			plt.tight_layout()
			plt.axis("off")
			plt.savefig("./outputs/hello.png", bbox_inches="tight")
			plt.close()

	def grid2img(self, path=None):
		"""
		use to convert the cellular automata grid into image.
		"""
		alive_cells_mask = self.get_alive_mask(self.grid)

	def get_alive_mask(self):
		"""
		returns the mask for alive cells only. a cell is alive the cell or atleast one of the neirghbour hood cell has alpha >= 0.1. The image is to be formed for only the alive cell other becomes dead instantly.
		"""
		pass

	def resize_image(image, max_size=125):
		"""
		returns the resized image for a given input image by shrinking the largest slide to max_size
		"""
		pass