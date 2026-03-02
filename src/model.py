import torch
from torch.nn import Module, Conv2d, ReLU
from torch.nn.functional import conv2d

# constant represnting filters for the operators
SOVEL_X = torch.tensor([[
	[-1, 0, 1],
	[-2, 0, 2],
	[-1, 0, 1],
]], dtype=torch.float32).unsqueeze(dim=0).repeat(16, 1, 1, 1)

SOVEL_Y = SOVEL_X.transpose(2, 3)
class CellularNeuralAutomata(Module):
	def __init__(self, channels=16, hidden_dim=128):
		super().__init__()
		self.conv1 = Conv2d(in_channels=channels*3, out_channels=hidden_dim, bias=True, kernel_size=1)
		self.conv2 = Conv2d(in_channels=hidden_dim, out_channels=16, bias=True, kernel_size=1)
		self.relu = ReLU()

	def forward(self, grids):
		perception_vectors = self.perceive(grids)

		output = self.conv1(perception_vectors)
		output = self.relu(output)
		output = self.conv2(output)

		return output
		
	def perceive(self, grid):
		"""
		returns perception vector formed from the grid by concatenating sovel x, y, and identity
		"""
		if len(grid.shape) == 3:
			grid = grid.unsqueeze(dim=0)

		sovel_x = conv2d(grid, SOVEL_X, padding=1, groups=16)
		sovel_y = conv2d(grid, SOVEL_Y, padding=1, groups=16)

		# return concatenated output
		return torch.cat((grid, sovel_x, sovel_y), dim=1)
