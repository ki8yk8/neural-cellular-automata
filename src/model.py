import torch
from torch.nn import Module, Conv2d, ReLU

class CellularNeuralAutomata(Module):
	def __init__(self, hidden_dim=128):
		super().__init__()
		self.conv1 = Conv2d(in_channels=48, out_channels=hidden_dim, bias=True, kernel_size=1)
		self.conv2 = Conv2d(in_channels=hidden_dim, out_channels=16, bias=True, kernel_size=1)
		self.relu = ReLU()

	def forward(self, percpetion_vector):
		output = self.conv1(percpetion_vector)
		output = self.relu(output)
		output = self.conv2(output)
		output = self.relu(output)

		return output
		