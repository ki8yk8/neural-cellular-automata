"""
Script to train the neural cellular automata for creating neural rules that ultimately stores the image.
"""
from src.grids import Grid

IMAGE_PATH = "./images/banana-no-bg.png"

# importing the image and creating the grid
grid = Grid()
grid.copy_image(IMAGE_PATH)
grid.grid2img("./outputs/frame.png")
