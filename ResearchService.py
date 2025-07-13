#!/usr/bin/env python3
"""
Progetto IA: Color Grid Problem usando AIMA-Python
Converte una griglia di colori in un problema di ricerca usando le funzioni AIMA
"""

import copy
from collections import deque
from typing import List, Tuple, Set

import sys
import os

from search import Problem
from search import breadth_first_tree_search as breadth_first_search
from search import depth_first_graph_search
from search import best_first_graph_search
from search import astar_search
from utils import *

class ColorGridProblem(Problem):
    """
    Problema della griglia colorata:
    - Stato: matrice di colori rappresentata come tupla di tuple
    - Azione: (posizione_start, nuovo_colore) - flood fill da posizione con nuovo colore
    - Goal: tutta la griglia dello stesso colore
    """
    
    def __init__(self, initial_grid):
        # Convertiamo la griglia in una tupla di tuple per renderla hashable
        initial_state = tuple(tuple(row) for row in initial_grid)
        super().__init__(initial_state)
        self.rows = len(initial_grid)
        self.cols = len(initial_grid[0])
        
        # Tutti i colori possibili nella griglia
        self.colors = set()
        for row in initial_grid:
            self.colors.update(row)
        self.colors = list(self.colors)
        
        print(f"🎨 Griglia iniziale ({self.rows}x{self.cols}):")
        self.print_grid(initial_state)
        print(f"Colori disponibili: {self.colors}")
        print(f"Goal: Rendere tutta la griglia dello stesso colore\n")
    
    def print_grid(self, state):
        """Stampa la griglia in modo leggibile con colori"""
        color_symbols = {
            'g': '🟢', 'b': '🔵', 'y': '🟡', 't': '🖌️'
        }
        
        for row in state:
            colored_row = []
            for cell in row:
                symbol = color_symbols.get(cell, cell)
                colored_row.append(f"{symbol}({cell})")
            print(' '.join(colored_row))
        print()
    
    def get_connected_component(self, grid, start_pos, original_color):
        """
        Trova tutte le posizioni connesse alla posizione di partenza
        che hanno lo stesso colore (flood fill component)
        """
        rows, cols = len(grid), len(grid[0])
        start_row, start_col = start_pos
        
        if (start_row < 0 or start_row >= rows or 
            start_col < 0 or start_col >= cols or
            grid[start_row][start_col] != original_color):
            return set()
        
        visited = set()
        queue = deque([start_pos])
        component = set()
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # destra, giù, sinistra, su
        
        while queue:
            row, col = queue.popleft()
            
            if (row, col) in visited:
                continue
                
            if (row < 0 or row >= rows or col < 0 or col >= cols or
                grid[row][col] != original_color):
                continue
            
            visited.add((row, col))
            component.add((row, col))
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                if (new_row, new_col) not in visited:
                    queue.append((new_row, new_col))
        
        return component
    
    def actions(self, state):
        """
        Restituisce tutte le azioni possibili:
        Ottimizzazione: considera solo le posizioni che sono al confine tra colori diversi
        """
        grid = [list(row) for row in state]
        actions = []
        
        # Trova posizioni interessanti (confini tra colori diversi o angoli)
        interesting_positions = set()
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for i in range(self.rows):
            for j in range(self.cols):
                current_color = grid[i][j]
                # Controlla se questa posizione è al confine con un colore diverso
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if (0 <= ni < self.rows and 0 <= nj < self.cols and 
                        grid[ni][nj] != current_color):
                        interesting_positions.add((i, j))
                        break
        
        # Aggiungi sempre gli angoli come posizioni interessanti
        corners = [(0, 0), (0, self.cols-1), (self.rows-1, 0), (self.rows-1, self.cols-1)]
        interesting_positions.update(corners)
        
        # Genera azioni per posizioni interessanti
        for pos in interesting_positions:
            current_color = grid[pos[0]][pos[1]]
            for color in self.colors:
                if color != current_color:
                    actions.append((pos, color))
        
        return actions
    
    def result(self, state, action):
        """
        Applica l'azione: sposta l'aspirapolvere (t) su una nuova posizione.
        La casella su cui si trova diventa del colore originale, la nuova posizione diventa 't'.
        """
        grid = [list(row) for row in state]
        start_pos, new_color = action

        # Trova la posizione corrente dell'aspirapolvere ('t')
        current_t_pos = None
        for i in range(self.rows):
            for j in range(self.cols):
                if grid[i][j] == 't':
                    current_t_pos = (i, j)
                    break
            if current_t_pos:
                break

        # Se non c'è 't', mettiamo 't' sulla posizione di partenza
        if not current_t_pos:
            original_color = grid[start_pos[0]][start_pos[1]]
            grid[start_pos[0]][start_pos[1]] = 't'
            return tuple(tuple(row) for row in grid)

        # Ripristina il colore originale dove era 't'
        grid[current_t_pos[0]][current_t_pos[1]] = new_color

        # Sposta 't' sulla nuova posizione
        grid[start_pos[0]][start_pos[1]] = 't'

        return tuple(tuple(row) for row in grid)
    
    def goal_test(self, state):
        """
        Testa se tutta la griglia ha lo stesso colore
        """
        first_color = state[0][0]
        for row in state:
            for cell in row:
                if cell != first_color:
                    return False
        return True
    
    def h(self, node):
        """
        Euristica per A*: numero di colori diversi nella griglia - 1
        Nel goal state c'è solo 1 colore, quindi h=0
        """
        state = node.state
        colors_present = set()
        for row in state:
            colors_present.update(row)
        return len(colors_present) - 1
    
    def path_cost(self, c, state1, action, state2):
        """Costo uniforme per ogni azione"""
        return c + 1


