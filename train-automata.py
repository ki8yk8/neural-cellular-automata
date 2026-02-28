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
from src.grids import Grid
from src.model import CellularNeuralAutomata

IMAGE_PATH = "./images/banana-no-bg.png"

# importing the image and creating the grid
grid = Grid()

# copies the image to grid
grid.copy_image(IMAGE_PATH)

# showing the initial image
grid.grid2img("./outputs/frame.png")

perception_vector = grid.get_perception_vector()

# model = CellularNeuralAutomata()
breakpoint()