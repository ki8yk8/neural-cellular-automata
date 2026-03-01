"""
Script to train the neural cellular automata for creating neural rules that ultimately stores the image.

Model Training Steps
1. Load the image with a random pixel
2. Use the cnn operator to find the update for each pixel
3. Stochastic grid state update
4. Alive cell masking 
5. Repeat this for n number of times where n is a random number between 64, 96
6. Apply BPTT by averaging the l2 loss
"""
import numpy as np
import torch

from src.grids import Grid
from src.model import CellularNeuralAutomata

IMAGE_PATH = "./images/banana-no-bg.png"
EPOCHS = 100

# importing the image and creating the grid: training grid is used for training while ground grid acts as the ground truth for computing the losses
true_grid = Grid()
# copies the image to grid
true_grid.copy_image(IMAGE_PATH)

# initializing model, optimizer, and criterion for the training
model = CellularNeuralAutomata()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
for i in range(EPOCHS):
	training_grid = Grid()
	training_grid.copy_image(IMAGE_PATH)

	optimizer.zero_grad()
	# choose a random number of timesteps
	n_timesteps = np.random.randint(low=64, high=96)

	total_loss = 0.0
	for n in range(n_timesteps):
		perception_vector = training_grid.get_perception_vector()
		delta = model(perception_vector)
		training_grid.update(delta)

		current_image = training_grid.grid[:3,:,:]
		true_image = true_grid.grid[:3,:,:]
		
		loss = criterion(current_image, true_image)
		total_loss += loss

	average_loss = total_loss/n_timesteps
	average_loss.backward()
	optimizer.step()

	print(f"Epoch {i+1}, Loss: {average_loss.item()}")
