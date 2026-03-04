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

from src.model import CellularNeuralAutomata
from src.utils import create_video
from src.grids import create_image_grid, create_seed, update
from src.utils import grid2img

# Setting the seed values
SEED = 51
torch.manual_seed(SEED)
np.random.seed(SEED)

# setting up constants
IMAGE_PATH = "./images/banana-no-bg.png"
EPOCHS = 8000
LR = 2e-3
BATCH_SIZE = 8

def lr_lambda(step):
	return 0.1 if step>2000 else 1.0

# initializing model, optimizer, and criterion for the training
model = CellularNeuralAutomata()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

true_image = create_image_grid(IMAGE_PATH, BATCH_SIZE)

model.train()
for i in range(EPOCHS):
	training_grid = create_seed(BATCH_SIZE)

	optimizer.zero_grad()
	# choose a random number of timesteps
	n_timesteps = np.random.randint(low=64, high=96)

	for n in range(n_timesteps):
		delta = model(training_grid)

		training_grid = update(training_grid, delta)

	loss = criterion(training_grid[:,:4], true_image[:,:4])
	loss.backward()
	
	torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
	# for p in model.parameters():
	# 	p.grad /= (p.grad.norm() + 1e-8)

	optimizer.step()
	scheduler.step()

	print(f"Epoch {i+1}, Loss: {loss.item()}")
	torch.save(model.state_dict(), f"./outputs/logs/{i}.pt")

	# saving the end result of epoch on outputs for visualization
	grid2img(training_grid.detach(), f"./outputs/epochs/{i}.png")

create_video("./outputs/epochs/", output_path="./outputs/training.gif")
