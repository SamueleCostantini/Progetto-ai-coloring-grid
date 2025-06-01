
class GridProblem(Problem):
    """
    Problema di ricerca su griglia basato su array di colori.
    """
    
    def __init__(self, grid_array, rows=2, cols=4, initial_state=(0,0), goal_state=None):
        """
        Inizializza il problema della griglia.
        
        Args:
            grid_array: Lista di colori ['g', 't', 'g', 'b', 'g', 'y', 'g', 'b']
            rows: Numero di righe (default 2)
            cols: Numero di colonne (default 4)  
            initial_state: Posizione iniziale (riga, colonna)
            goal_state: Posizione obiettivo o None per trovare un colore specifico
        """
        # Converte l'array in griglia 2D
        self.grid = self._array_to_grid(grid_array, rows, cols)
        self.rows = rows
        self.cols = cols
        
        # Se non specificato, goal è trovare la prima 'y' (giallo)
        if goal_state is None:
            goal_state = self._find_color_position('y')
        
        super().__init__(initial_state, goal_state)
    
    def _array_to_grid(self, array, rows, cols):
        """Converte array 1D in griglia 2D."""
        if len(array) != rows * cols:
            raise ValueError(f"Array length {len(array)} doesn't match grid size {rows}x{cols}")
        
        grid = []
        for i in range(rows):
            row = array[i * cols:(i + 1) * cols]
            grid.append(row)
        return grid
    
    def actions(self, state):
        """Restituisce le azioni possibili da uno stato (su, giù, sinistra, destra)."""
        row, col = state
        possible_actions = []
        
        # Su
        if row > 0:
            possible_actions.append('UP')
        # Giù
        if row < self.rows - 1:
            possible_actions.append('DOWN')
        # Sinistra
        if col > 0:
            possible_actions.append('LEFT')
        # Destra
        if col < self.cols - 1:
            possible_actions.append('RIGHT')
            
        return possible_actions
    
    def result(self, state, action):
        """Restituisce il nuovo stato dopo aver eseguito un'azione."""
        row, col = state
        
        if action == 'UP':
            return (row - 1, col)
        elif action == 'DOWN':
            return (row + 1, col)
        elif action == 'LEFT':
            return (row, col - 1)
        elif action == 'RIGHT':
            return (row, col + 1)
        else:
            return state
    
    def goal_test(self, state):
        """Verifica se lo stato corrente è l'obiettivo."""
        return state == self.goal
    
    def path_cost(self, c, state1, action, state2):
        """Costo del percorso. Ogni movimento costa 1."""
        return c + 1
    
    