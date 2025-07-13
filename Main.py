
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
from ResearchService import ColorGridProblem
from utils import *
from search import *

# %%
estrattore = AiTextExtractorService(model, False)


# %%
lettere, num_rows, num_columns = estrattore.runGridExtraction('costum-test/5x3.png')

# %%
print(lettere)
print(num_rows)
print(num_columns)



# %%
def array_to_grid(array, rows, cols):
        """Converte array 1D in griglia 2D."""
        if len(array) != rows * cols:
            raise ValueError(f"Array length {len(array)} doesn't match grid size {rows}x{cols}")
        
        grid = []
        for i in range(rows):
            row = array[i * cols:(i + 1) * cols]
            grid.append(row)
        return grid


# %%

gridProblem = array_to_grid(lettere, num_rows, num_columns)
print(gridProblem)

def print_solution_path(solution_node, problem):
    """Stampa il percorso della soluzione passo per passo"""
    if not solution_node:
        print("❌ Nessuna soluzione trovata!")
        return
    
    path = solution_node.path()
    print(f"✅ Soluzione trovata!")
    print(f"Numero di passi: {len(path) - 1}")
    print(f"Costo del percorso: {solution_node.path_cost}")
    print("\n" + "="*50)
    print("PERCORSO DELLA SOLUZIONE:")
    print("="*50)
    
    for i, node in enumerate(path):
        print(f"\nPasso {i}:")
        if node.action:
            pos, color = node.action
            print(f"Azione: Flood fill da posizione {pos} con colore '{color}'")
        else:
            print("Stato iniziale:")
        
        problem.print_grid(node.state)
        
        if i < len(path) - 1:  # Non stampare l'euristica per l'ultimo nodo
            print(f"Euristica h = {problem.h(node)}")
    
    print("🎯 GOAL RAGGIUNTO!")

def test_algorithm(algorithm_name, algorithm_func, problem):
    """Testa un algoritmo di ricerca e stampa i risultati"""
    print(f"\n{'='*60}")
    print(f"🔍 ALGORITMO: {algorithm_name}")
    print("="*60)
    
    try:
        # Esegui l'algoritmo
        solution_node = algorithm_func(problem)
        
        # Stampa i risultati
        print_solution_path(solution_node, problem)
        
        return solution_node
        
    except Exception as e:
        print(f"❌ Errore durante l'esecuzione di {algorithm_name}: {e}")
        return None


# %%
researchService = ColorGridProblem(gridProblem)
    
# La tua griglia di esempio

initial_grid = gridProblem

# Esempio più complesso per test aggiuntivi

print("🎨 COLOR GRID SEARCH PROBLEM")
print("="*60)
print("Obiettivo: Rendere tutta la griglia dello stesso colore")
print("Meccanismo: Flood fill da una posizione con un nuovo colore")
print("="*60)


choice = "1"

if choice == "1":
    selected_grid = initial_grid
    print("Hai scelto la griglia semplice")
    ("Greedy Best-First Search", lambda p: best_first_graph_search(p, p.h)),
# Crea il problema
problem = ColorGridProblem(selected_grid)

# Lista degli algoritmi da testare
algorithms = [
    #("Breadth-First Search", lambda p: breadth_first_search(p)),
    ("Depth-First Graph Search", lambda p: depth_first_graph_search(p)),
    ("Greedy Best-First Search", lambda p: greedy_best_first_graph_search(p, p.h)),
    ("A* Search", lambda p: astar_search(p, p.h))
]

print(f"\nVerranno testati {len(algorithms)} algoritmi di ricerca:")
for i, (name, _) in enumerate(algorithms, 1):
    print(f"{i}. {name}")

print("\n" + "="*60)

# Testa tutti gli algoritmi
results = {}
for name, algorithm in algorithms:
    solution = test_algorithm(name, algorithm, problem)
    results[name] = solution
    
    print(f"\n{'-'*60}")

# Riassunto finale
print(f"\n{'='*60}")
print("📊 RIASSUNTO RISULTATI")
print("="*60)

for name, solution in results.items():
    if solution:
        print(f"✅ {name:25} | Passi: {solution.depth:2} | Costo: {solution.path_cost}")
    else:
        print(f"❌ {name:25} | Nessuna soluzione trovata")

print(f"\n🎯 Test completato!")

