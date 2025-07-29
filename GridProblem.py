from search import Problem
# Converte la griglia 2D in un formato utilizzabile per il problema di ricerca

class GridProblem(Problem):
    def __init__(self, initial, goal_color, start_position, color_costs, rows, cols, letters):  # Inizializza il problema
        super().__init__(initial)
        
        def most_common_color():
            colors = { 'b':0, 'g':0, 'y':0 }
            for letter in letters:
                    colors[letter] += 1
            return max(colors, key=colors.get)

        self.goal_color = goal_color  # Colore obiettivo
        if goal_color == 'a': # a sta per automatico, assegna il colore più comune senza considerare i costi
            self.goal_color = most_common_color() 
            
        print(f"Goal color assegnato: {self.goal_color}")
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
        actions.append('Paint green')  # Aggiunge l'azione di paint
        
        actions.append('Paint yellow')  # Aggiunge l'azione di paint

        actions.append('Paint blue')  # Aggiunge l'azione di paint



        return actions

    def result(self, state, action):
        grid, (x, y) = state
        new_grid = [list(row) for row in grid]  # Copia della griglia

        # Rimuovi 'T' da tutta la griglia
       
        """if new_grid[x][y].endswith('T'):
            new_grid[x][y].replace('T', '')"""

        new_position = (x, y)
        # Movimenti
        if action == 'Up':
            new_position = (x - 1, y)
            #new_grid[new_position[0]][new_position[1]] = 'T'
        elif action == 'Down':
            new_position = (x + 1, y)
            #new_grid[new_position[0]][new_position[1]] = 'T'
        elif action == 'Left':
            new_position = (x, y - 1)
            #new_grid[new_position[0]][new_position[1]] = 'T'
        elif action == 'Right':
            new_position = (x, y + 1)
            #new_grid[new_position[0]][new_position[1]] = 'T'
        elif action == 'Paint green':
            new_grid[x][y] = 'g'
            #new_position = (x, y)
        elif action == 'Paint blue':
            new_grid[x][y] = 'b'
            #new_position = (x, y)
        elif action == 'Paint yellow':
            new_grid[x][y] = 'y'
            #new_position = (x, y)
        else:
            new_position = (x, y)

        #print(f"Action: {action}, New Position: {new_position}, New Grid: {new_grid}")
        return (tuple(tuple(row) for row in new_grid), new_position)

    def goal_test(self, state):
        grid, position = state
        # Ottieni il colore della prima cella (escludendo 'T' se presente)
        first_color = None
        for row in grid:
            for cell in row:
                if cell != 'T':
                    first_color = cell
                    break
            if first_color:
                break
        # Verifica che tutte le celle (escludendo 'T') abbiano lo stesso colore
        all_same_color = all(cell == first_color for row in grid for cell in row )
        return all_same_color

    def path_cost(self, c, state1, action, state2):  # Calcola il costo del percorso
        # Ogni movimento e ogni pittura hanno un costo
        if action in ['Up', 'Down', 'Left', 'Right']:
            return c + 1
        elif action == 'Paint green':
            # Map color string to index for color_costs
            #color_to_index = {'g': 0, 'y': 1, 'b': 2}
            #color_index = color_to_index.get(self.goal_color, 0)
            return c + 3
        elif action == 'Paint yellow':
            # Map color string to index for color_costs
            #color_to_index = {'g': 0, 'y': 1, 'b': 2}
            #color_index = color_to_index.get(self.goal_color, 0)
            return c + 2
        elif action == 'Paint blue':
            # Map color string to index for color_costs
           # color_to_index = {'g': 0, 'y': 1, 'b': 2}
           # color_index = color_to_index.get(self.goal_color, 0)
            return c + 1
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
    