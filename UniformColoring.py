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
import time



import numpy as np

output_dir = 'output'
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

if len(letters) != rows * cols:
    raise ValueError(f"Lunghezza array {len(letters)} non coincide con la dimensione della {rows}x{cols}")

grid = []
for i in range(rows):
    row = letters[i * cols:(i + 1) * cols]
    grid.append(row)
print(grid)
# Convert grid to tuple of tuples for hashability
tuple_grid = tuple(tuple(row) for row in grid)
initial_state = (tuple_grid, (0, 0))
gridProblem = GridProblem(
    initial=initial_state,
    goal_color='a',
    start_position=(0, 0),
    color_costs=[3,2,1], # costi colori ordine g, y, b
    rows=rows,
    cols=cols,
    letters=letters
)

initial_grid = grid

print(grid)

# Simula l'esecuzione del piano migliore
def simulate_plan(initial_state, solution, nameGif):
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    import imageio.v2 as imageio
    import os

    print("\n")
    if solution:
        actions = solution.solution()
        print("Soluzione (azioni):", actions)
        print("Lunghezza soluzione: ", len(actions))
        print("Costo soluzione:", solution.path_cost)
        print("Tempo di esecuzione:", solution.time if hasattr(solution, 'time') else "Non disponibile")
    else:
        print("Nessuna soluzione")
    print("\n__________________________________________________________\n\n")
    state = initial_state
    grids = [state[0]]
    for action in actions:
        state = gridProblem.result(state, action)
        grids.append(state[0])

    images = []
    for idx, (grid, action) in enumerate(zip(grids, [None] + actions)):
        fig, ax = plt.subplots(figsize=(len(grid[0]), len(grid)))
        ax.axis('off')
        # Disegna la tabella della griglia
        table_data = [[cell for cell in row] for row in grid]
        ax.table(cellText=table_data, loc='center', cellLoc='center', edges='closed')
        # Aggiungi il nome dell'azione in basso
        if action is not None:
            plt.figtext(0.5, 0.01, f"Azione: {action}", ha='center', fontsize=12)
        else:
            plt.figtext(0.5, 0.01, "Stato iniziale", ha='center', fontsize=12)
        plt.tight_layout()
        # Salva in PNG temporaneo
        fname = f'{output_dir}/_sim_grid_{idx}.png'
        plt.savefig(fname, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        images.append(imageio.imread(fname))
        os.remove(fname)

    # Salva la GIF
    imageio.mimsave(f'{output_dir}/{nameGif}.gif', images, format='GIF', fps=2)

print("\n\n\n")


dfs_solution = depth_first_graph_search(gridProblem)

astar_solution = astar_search(gridProblem)

ucs_solution = uniform_cost_search(gridProblem)

gbfs_solution = best_first_graph_search(gridProblem, f=gridProblem.h)

if ucs_solution:
   print("Soluzione trovata con UCS:")
   simulate_plan(gridProblem.initial, ucs_solution, 'ucs')

if dfs_solution:
   print("Soluzione trovata con DFS:")
   simulate_plan(gridProblem.initial, dfs_solution, 'dfs')

if gbfs_solution:
    print("Soluzione trovata con GBFS:")
    simulate_plan(gridProblem.initial, gbfs_solution, 'gbfs')

if astar_solution:
    print("Soluzione trovata con A*:")
    simulate_plan(gridProblem.initial, astar_solution, 'astar')


