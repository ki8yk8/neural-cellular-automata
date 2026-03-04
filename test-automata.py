"""
Uses the training.json obtained during the training phase and find the best checkpoint and iteration for that checkpoint or you can also manually define the best checkpoint visually. Then, recreates the gif for that checkpoint.
"""
import json
import torch
from src.model import CellularNeuralAutomata
from src.grids import create_seed, update
from src.utils import grid2img, create_video

# loading the metadata file
with open("./outputs/training.json", "r") as fp:
	metadata = json.load(fp)

# edit the constant to choose the best checkpoint
CHECKPOINT = -1
if CHECKPOINT == -1:
	lowest_loss, lowest_loss_index = 100.0, -1
	for i, m in metadata.items():
		i = int(i)
		if m["loss"] < lowest_loss:
			lowest_loss_index = i
			lowest_loss = m["loss"]

	CHECKPOINT = lowest_loss_index

# load the model with the given checkpoint
model = CellularNeuralAutomata()
model.load_state_dict(torch.load(f"./outputs/logs/{CHECKPOINT}.pt", weights_only=False))
model.eval()

# iterating for the n_iterations and creating the gif value
n_iter = metadata[str(CHECKPOINT)]["timesteps"]
grid = create_seed()

for i in range(n_iter):
	delta = model(grid)
	grid = update(grid, delta)

	grid2img(grid, f"./outputs/test/{i}.png")

create_video("./outputs/test/", output_path="./outputs/testing.gif")
