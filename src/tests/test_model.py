import torch
from ..model import CellularNeuralAutomata

def manual_depthwise_convolution(grid, filter):
	return False

def test_perception_vector():
	"""
	test case for the working of perception vectors. 
	"""
	model = CellularNeuralAutomata()
	random_grid = torch.randint(0, 5, (1, 16, 32, 32), dtype=torch.float32)
	perception_vectors = model.perceive(random_grid)

	# TEST 1: output perception vector should be 48 channel for each of the grid cell (first 16, identity, then sovel_x, then sovel_y)
	assert(perception_vectors.shape[1] == 16*3)
	identity, sovel_x, sovel_y = perception_vectors[:, :16], perception_vectors[:, 16:16+16], perception_vectors[:, 16+16:]

	# TEST 2: identity should be same
	assert(torch.equal(identity, random_grid))

	# TEST 3: sovel_x should be applied depthwise
	sovel_x_filter = torch.tensor([
		[-1, 0, 1],
		[-2, 0, 2],
		[-1, 0, 1],
	], dtype=torch.float32)
	sovel_y_filter = sovel_x_filter.transpose(0, 1)

	assert(torch.equal(sovel_x, manual_depthwise_convolution(random_grid, sovel_x_filter)))
	assert(torch.equal(sovel_y, manual_depthwise_convolution(random_grid, sovel_y_filter)))
