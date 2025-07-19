from PIL import Image # Libreria per la manipolazione delle immagini
import numpy as np # Libreria per il calcolo numerico
import tensorflow as tf # Libreria per il machine learning
import matplotlib.pyplot as plt # Libreria per la visualizzazione dei dati
import cv2 # Libreria per computer vision
import tempfile # Libreria per la gestione di file temporanei
import os # Libreria per la gestione dei file e delle directory

# Classe service che contiene metodi per utili l'estrazione delle lettere dall'immagine
class AiTextExtractorService:
    
    verbose = False # Per debug
    model = None

    def __init__(self, model, verbose):
        self.model = model # Modello usato per le predizioni
        self.verbose = verbose
        print("Model loaded successfully")

    # Metodo per il mapping delle classi e le lettere
    def mapPredictedClassToLetter(self, classValue):
        if classValue == 15:
            classValue = 24
        predicted_letter = chr(classValue + ord('a') )
        # Converte la classe in lettera
        print(f"La classe predetta {classValue} corrisponde alla lettera: {predicted_letter}")
        return predicted_letter

    # Metodo per rimuovere un range di colori da un'immagine
    def removeColorRange(self, img_array, start_hex, end_hex):
        # Converto i valori esadecimali in RGB
        start_rgb = tuple(int(start_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) 
        end_rgb = tuple(int(end_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        # Crea una maschera per i pixel che rientrano nel range di colore
        mask = np.ones_like(img_array[..., 0], dtype=bool)
        
        for i in range(3):
            channel_mask = (img_array[..., i] >= min(start_rgb[i], end_rgb[i]))
            channel_mask &= (img_array[..., i] <= max(start_rgb[i], end_rgb[i]))
            mask &= channel_mask
        
        # Create a copy to avoid modifying the original array
        filtered_array = img_array.copy()
        filtered_array[mask] = 0
        
        return filtered_array

    def autoCropLetter(self, img_array):
       
        # Converte in scala di grigi se l'immagine è RGB
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array.copy()
        
        # Applica thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Trova i contorni del box della lettera
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if self.verbose:
            print(f"Found {len(contours)} contours in the image.")

        if not contours:
            return img_array
        
        # Trova il contorno più grande
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Trova il centro della lettera
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Calcola la grandezza del quadrato, contorno maggiore + un padding personalizzato
        square_size = max(w, h)
        padding = int(square_size * 0.2)  # 20% padding
        square_size += 2 * padding  # Aggiungi padding a entrambi i lati
        
        # Calcola i limiti del quadrato dal centro
        half_size = square_size // 2
        start_x = max(center_x - half_size, 0)
        start_y = max(center_y - half_size, 0)
        end_x = min(center_x + half_size, img_array.shape[1])
        end_y = min(center_y + half_size, img_array.shape[0])
        
        # Assicuro che il contorno sia equilatero per passare al modello un immagine quadrata
        width = end_x - start_x
        height = end_y - start_y
        if width > height:
            diff = width - height
            start_y = max(start_y - diff // 2, 0)
            end_y = min(start_y + width, img_array.shape[0])

        elif height > width:
            diff = height - width
            start_x = max(start_x - diff // 2, 0)
            end_x = min(start_x + height, img_array.shape[1])
        
        cropped = img_array[start_y:end_y, start_x:end_x]
            
        # Aumento il padding se necessario per avere un'immagine quadrata
        # Faccio un ulteriore controllo per assicurarmi che l'immagine sia quadrata
        if cropped.shape[0] != cropped.shape[1]: #se l'altezza è diversa dalla larghezza
            max_dim = max(cropped.shape[0], cropped.shape[1])
            
            square = np.zeros((max_dim, max_dim), dtype=cropped.dtype)

            cropped = square # ritaglia l'immagine

            cropped = img_array[start_y:end_y, start_x:end_x] #immagine quaddrata della lettera
        
        # Per debug
        if self.verbose:
            plt.figure(figsize=(10, 5))
            plt.subplot(121)
            plt.imshow(binary, cmap='gray')
            plt.title('Binary threshold')
            plt.subplot(122)
            plt.imshow(cropped, cmap='gray' if len(cropped.shape) == 2 else None)
            plt.title(f'Cropped square {cropped.shape[0]}x{cropped.shape[1]}')
            plt.show()
           
        
        return cropped
        
    def analyzeImage(self, image_path):
        # Metodo che raggruppa tutte i metodi scritti prima
        # Serve solo per analizzare 1 singola lettera
        # Carica l'immagine come RGB per mantenere tutti i dettagli
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img) # Convverto l'immagine in un array con numpy
        
        # Ritaglio la lettera
        cropped_array = self.autoCropLetter(img_array)
        
        # Converto in grayscale dopo il ritaglio
        if len(cropped_array.shape) == 3:
            gray_img = Image.fromarray(cropped_array).convert("L")
        else:
            gray_img = Image.fromarray(cropped_array)
        
        # Ridimensiono a 28x28 per il modello
        resized_img = gray_img.resize((28, 28))
        
        # Converto in array e normalizzo
        img_array = np.array(resized_img)
        # Normalizzazione dell'immagine
        img_array = img_array / 255.0
        img_array = 1 - img_array
        
        # Visualizzo l'immagine processata (opzionale)
        if self.verbose:
            plt.imshow(img_array, cmap="gray")
            plt.show()
        
        # Reshape per il modello CNN
        img_array = img_array.reshape(1, 28, 28, 1)
        
        # Predizione
        prediction = self.model.predict(img_array)
        predicted_class = np.argmax(prediction)
        
        # Ritorna la lettera predetta
        return self.mapPredictedClassToLetter(predicted_class)
    
    def isolateLettersFromGrid(self, image_path):
        
        # Metodo per isolare tutte le lettere da una griglia
        # L'idea è isolare le singole lettere analizzando i contorni con OpenCV 
        # e di ritornare un array di tuple (x, y, letter_image) da passare alla 
        # funzione predictGridLetters per le predizioni che scandalgia l'array e da in pasto
        # l'immagine ad analyzeImage che esegue il preprocessing e la predizione

        output_dir = "output"
        if self.verbose and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if self.verbose:
            cv2.imwrite(os.path.join(output_dir, "original_image_gray.png"), img)

        if img is None:
            raise FileNotFoundError(f"L'immagine '{image_path}' non è stata trovata.")

        # Threshold
        _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

        if self.verbose:
            plt.savefig(os.path.join(output_dir, "binary_image.png"))

        # Individua linee orizzontali e verticali
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (binary.shape[1]//10, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, binary.shape[0]//10))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        grid = cv2.add(horizontal_lines, vertical_lines)
        if self.verbose:
            plt.imshow(grid, cmap='gray')
            plt.savefig(os.path.join(output_dir, "grid.png"))
        
        # Trova intersezioni
        intersections = cv2.bitwise_and(horizontal_lines, vertical_lines)
        coords = cv2.findNonZero(intersections)
        if coords is None:
            raise ValueError("Griglia non individuata")

        # Individua le coordinate uniche delle intersezioni
        coords = coords[:, 0, :]
        x_coords = sorted(set([x for x, y in coords]))
        y_coords = sorted(set([y for x, y in coords]))

        # Funzione per raggruppare le posizioni in cluster
        # per evitare di avere posizioni troppo vicine tra loro
        def cluster_positions(positions, threshold=10):
            clustered = []
            for p in positions:
                if not clustered or abs(p - clustered[-1]) > threshold:
                    clustered.append(p)
            return clustered

        x_coords = cluster_positions(x_coords)
        y_coords = cluster_positions(y_coords)

        # Con le coordinate salvate estraggo le lettere
        letters = []
        for row in range(len(y_coords) - 1):
            for col in range(len(x_coords) - 1):
                # Estrae la posizione del box dalle coordinate estrapolate prima
                x1, x2 = x_coords[col], x_coords[col + 1]
                y1, y2 = y_coords[row], y_coords[row + 1]
                cell = img[y1:y2, x1:x2]
                # Scontorna l'immagine con autoCropLetter
                cell = self.autoCropLetter(cell)
                letters.append((x1, y1, cell))
                if self.verbose:
                    cell_path = os.path.join(output_dir, f"cell_{row}_{col}.png")
                    cv2.imwrite(cell_path, cell)

        # Restituisce le lettere e le dimensioni della griglia, righe e colonne che serviranno
        # al problema per individuare le azioni possibili 
        return letters, len(y_coords) -1 , len(x_coords) - 1


    def predictGridLetters(self, isolated_letters):
        predictions = []
        for letter in isolated_letters:
            # la lettera è una tupla (x, y, letter_image)
            temp_img = Image.fromarray(letter[2])
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                temp_img.save(tmp.name)
                result = self.analyzeImage(tmp.name)
                predictions.append(result)
            if self.verbose == False:
                os.unlink(tmp.name)  

        print("Lettere predette:", predictions)
        if self.verbose:
            print("Array coordinate delle lettere:", letter)

        return predictions
                
    def runGridExtraction(self, image_path):
        # Unisce tutti i metodi
        isolated_letters, row, col = self.isolateLettersFromGrid(image_path)
        
        # Esegue le predizioni
        predictions = self.predictGridLetters(isolated_letters)
        
        #restituisce le lettere predette e le dimensioni della griglia
        return predictions, row, col
