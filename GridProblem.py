from search import Problem
# Converte la griglia 2D in un formato utilizzabile per il problema di ricerca

class GridProblem(Problem):
    def __init__(self, initial, goal_color, start_position, color_costs, rows, cols):  # Inizializza il problema
        super().__init__(initial)
        self.goal_color = goal_color  # Colore obiettivo
        self.start_position = start_position  # Posizione iniziale
        self.color_costs = color_costs  # Costi dei colori
        self.rows = rows  # Numero di righe
        self.cols = cols  # Numero di colonne 

    def actions(self, state):  # Definisce le azioni possibili
        actions = []  # Lista delle azioni possibili
        grid, (x, y) = state  # Griglia e posizione corrente
        rows, cols = self.rows, self.cols  # Righe e colonne

        # Movimenti nella griglia
        if x > 0:
            actions.append('Up')
        if x < rows - 1:
            actions.append('Down')
        if y > 0:
            actions.append('Left')
        if y < cols - 1:
            actions.append('Right')

        # Azione di colorazione solo se la cella non è colorata e non è la posizione iniziale
        if grid[x][y] != self.goal_color and (x, y) != self.start_position:
            actions.append('Paint')  # Aggiungi l'azione di pittura

        return actions

    def result(self, state, action):
        grid, (x, y) = state
        new_grid = [list(row) for row in grid]  # Copia della griglia

        # Movimenti
        if action == 'Up':
            new_position = (x - 1, y)
        elif action == 'Down':
            new_position = (x + 1, y)
        elif action == 'Left':
            new_position = (x, y - 1)
        elif action == 'Right':
            new_position = (x, y + 1)
        elif action == 'Paint':
            new_grid[x][y] = self.goal_color
            new_position = (x, y)
        else:
            new_position = (x, y)

        # Rimuove la 'T' da tutta la griglia
        for i in range(self.rows):
            for j in range(self.cols):
                if new_grid[i][j] == 'T':
                    new_grid[i][j] = self.goal_color
        # Mette la 'T' nella nuova posizione
        new_grid[new_position[0]][new_position[0]] = 'T'
        return (tuple("".join(row) for row in new_grid), new_position)

    def goal_test(self, state):
        grid, position = state
        # Verifica che tutte le celle siano colorate e che la testina sia nella posizione iniziale
        all_colored = all(cell == self.goal_color for row in grid for cell in row if cell != 'T')  # Tutte le celle sono colorate
        is_at_start = position == self.start_position  # La testina è nella posizione iniziale
        return all_colored and is_at_start

    def path_cost(self, c, state1, action, state2):  # Calcola il costo del percorso
        # Ogni movimento e ogni pittura hanno un costo
        if action in ['Up', 'Down', 'Left', 'Right']:
            return c + 1
        elif action == 'Paint':
            # Map color string to index for color_costs
            color_to_index = {'g': 0, 'y': 1, 'b': 2}
            color_index = color_to_index.get(self.goal_color, 0)
            return c + self.color_costs[color_index]
        else:
            return c

    def h(self, node):   #Euristica per la ricerce
       # Calcola l'euristica come il numero di celle non colorate con il colore goal
       grid, _ = node.state
       uncolored_cells = 0
       # Itera su ogni cella della griglia
       for r in range(self.rows):
           for c in range(self.cols):
               # Conta la cella solo se non è la posizione iniziale e non ha il colore obiettivo
               if (r, c) != self.start_position and grid[r][c] != self.goal_color:
                   uncolored_cells += 1
       return uncolored_cells
    