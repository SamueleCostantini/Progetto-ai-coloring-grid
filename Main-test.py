# %%
import sys
import os
py_file_location = "/content/aima-pyhton"
sys.path.append(os.path.abspath(py_file_location))


# %%
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import tempfile
import os

import numpy as np


# %%
model = tf.keras.models.load_model('letter_recognition_model.h5')

# %%
from AiTextExtractorService import AiTextExtractorService
from GridProblem import GridProblem
from utils import *
from search import *

# %%
estrattore = AiTextExtractorService(model, False)

#estrattore.analyzeImage('output_letters/letter_1_3.png')

letters, rows, cols = estrattore.runGridExtraction('costum-test/griglia1x.png')

print(letters)
print(rows)
print(cols)

"""Converte array 1D in griglia 2D."""
if len(letters) != rows * cols:
    raise ValueError(f"Array length {len(letters)} doesn't match grid size {rows}x{cols}")

grid = []
for i in range(rows):
    row = letters[i * cols:(i + 1) * cols]
    # Convert row (list of single-character strings) to a string
    grid.append("".join(row))

# Now grid is a list of strings, convert to tuple for hashability
initial_state = (tuple(grid), (0, 0))
gridProblem = GridProblem(
    initial=initial_state,
    goal_color='b',
    start_position=(0, 0),  # <-- also use tuple here
    color_costs=[1,1,1],
    rows=rows,
    cols=cols
)
    
# La tua griglia di esempio

initial_grid = grid

print(grid)

# Run depth-first search on the grid problem
solution = depth_first_graph_search(gridProblem)

# Print the solution path and steps
if solution:
    print("Solution found!")
    steps = []
    node = solution
    while node:
        steps.append(node.state)
        node = node.parent
    steps.reverse()
    print("Steps to solution:")
    for step_num, step in enumerate(steps):
        print(f"Step {step_num}: {step}")
else:
    print("No solution found.")

# Uninformed search: Depth-First Search
dfs_solution = depth_first_graph_search(gridProblem)
if dfs_solution:
    dfs_actions = dfs_solution.solution()
    print("DFS Solution (actions):", dfs_actions)
    print("DFS Solution length:", len(dfs_actions))
    print("DFS Solution cost:", dfs_solution.path_cost)
else:
    print("No DFS solution found.")

# Informed search: Uniform Cost Search
ucs_solution = uniform_cost_search(gridProblem)
if ucs_solution:
    ucs_actions = ucs_solution.solution()
    print("UCS Solution (actions):", ucs_actions)
    print("UCS Solution length:", len(ucs_actions))
    print("UCS Solution cost:", ucs_solution.path_cost)
else:
    print("No UCS solution found.")



# (Optional) Simulate execution of the best plan
def simulate_plan(initial_state, actions, nameGif):
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    import imageio
    import os

    state = initial_state
    grids = [state[0]]
    for action in actions:
        state = gridProblem.result(state, action)
        grids.append(state[0])

    images = []
    for idx, grid in enumerate(grids):
        fig, ax = plt.subplots(figsize=(len(grid[0]), len(grid)))
        ax.axis('off')
        # Draw grid as table
        table_data = [[cell for cell in row] for row in grid]
        ax.table(cellText=table_data, loc='center', cellLoc='center', edges='closed')
        plt.tight_layout()
        # Save to temporary PNG
        fname = f'_sim_grid_{idx}.png'
        plt.savefig(fname, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        images.append(imageio.imread(fname))
        #os.remove(fname)

    # Save as GIF
    imageio.mimsave(nameGif+'.gif', images, duration=0.8)
    print('GIF saved as simulation.gif')

# Example: simulate UCS plan
if ucs_solution:
    simulate_plan(gridProblem.initial, ucs_solution.solution(), 'ucs')

if dfs_solution:
   simulate_plan(gridProblem.initial, dfs_solution.solution(), 'dfs')

def goal_test(self, state):
    grid, position = state
    # All cells except the start must be goal_color, and only one 'T' at start
    is_at_start = position == self.start_position
    all_colored = all(
        (cell == self.goal_color or cell == 'T')
        for row in grid for cell in row
    )
    # Only one 'T' in the grid, and at the start position
    t_count = sum(cell == 'T' for row in grid for cell in row)
    t_at_start = grid[self.start_position[0]][self.start_position[1]] == 'T'
    return all_colored and is_at_start and t_count == 1 and t_at_start