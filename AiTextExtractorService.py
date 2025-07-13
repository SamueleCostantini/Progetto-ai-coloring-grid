from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import tempfile
import os


class AiTextExtractorService:
    
    verbose = False
    model = None

    def __init__(self, model, verbose):
        self.model = model
        self.verbose = verbose
        print("Model loaded successfully")

    def mapPredictedClassToLetter(self, classValue):
        if classValue == 15:
            classValue = 24
        predicted_letter = chr(classValue + ord('a') )
        # Converte la classe in lettera
        print(f"La classe predetta {classValue} corrisponde alla lettera: {predicted_letter}")
        return predicted_letter

    def removeColorRange(self, img_array, start_hex, end_hex):
        # Convert hex to RGB
        start_rgb = tuple(int(start_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        end_rgb = tuple(int(end_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        # Create mask for pixels in range
        mask = np.ones_like(img_array[..., 0], dtype=bool)
        
        for i in range(3):  # For each RGB channel
            channel_mask = (img_array[..., i] >= min(start_rgb[i], end_rgb[i]))
            channel_mask &= (img_array[..., i] <= max(start_rgb[i], end_rgb[i]))
            mask &= channel_mask
        
        # Create a copy to avoid modifying the original array
        filtered_array = img_array.copy()
        filtered_array[mask] = 0
        
        return filtered_array

    def autoCropLetter(self, img_array):
        """
        Automatically crops the image array to focus on a single letter in a square shape
        Args:
            img_array: numpy array of the image
        Returns:
            cropped square numpy array
        """
        # Convert to grayscale if image is RGB
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array.copy()
        
        # Apply thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"Found {len(contours)} contours in the image.")

        if not contours:
            return img_array
        
        # Find largest contour (assumed to be the letter)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Find center of the letter
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Calculate square size (use the larger dimension plus padding)
        square_size = max(w, h)
        padding = int(square_size * 0.2)  # 20% padding
        square_size += 2 * padding  # Add padding to both sides
        
        # Calculate square bounds from center
        half_size = square_size // 2
        start_x = max(center_x - half_size, 0)
        start_y = max(center_y - half_size, 0)
        end_x = min(center_x + half_size, img_array.shape[1])
        end_y = min(center_y + half_size, img_array.shape[0])
        
        # Ensure square dimensions by adjusting bounds if near image edges
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
        if len(img_array.shape) == 3:
            cropped = img_array[start_y:end_y, start_x:end_x, :]
        else:
            cropped = img_array[start_y:end_y, start_x:end_x]
            
        # Force square shape by padding if necessary
        if cropped.shape[0] != cropped.shape[1]:
            max_dim = max(cropped.shape[0], cropped.shape[1])
            if len(img_array.shape) == 3:
                square = np.zeros((max_dim, max_dim, 3), dtype=cropped.dtype)
            else:
                square = np.zeros((max_dim, max_dim), dtype=cropped.dtype)
            
            y_offset = (max_dim - cropped.shape[0]) // 2
            x_offset = (max_dim - cropped.shape[1]) // 2
            
            if len(img_array.shape) == 3:
                square[y_offset:y_offset+cropped.shape[0], 
                      x_offset:x_offset+cropped.shape[1], :] = cropped
            else:
                square[y_offset:y_offset+cropped.shape[0], 
                      x_offset:x_offset+cropped.shape[1]] = cropped
            cropped = square # Crop the image
        if len(img_array.shape) == 3:
            cropped = img_array[start_y:end_y, start_x:end_x, :]
        else:
            cropped = img_array[start_y:end_y, start_x:end_x]
        
        # self.verbose: visualization
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
    # Carica l'immagine come RGB per mantenere tutti i dettagli
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        
        # Prima fase: ritaglio la lettera
        cropped_array = self.autoCropLetter(img_array)
        
        # Seconda fase: converto in grayscale dopo il ritaglio
        if len(cropped_array.shape) == 3:
            gray_img = Image.fromarray(cropped_array).convert("L")
        else:
            gray_img = Image.fromarray(cropped_array)
        
        # Terza fase: ridimensiono a 28x28
        resized_img = gray_img.resize((28, 28))
        
        # Converto in array e normalizzo
        img_array = np.array(resized_img)
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
        
        return self.mapPredictedClassToLetter(predicted_class)
    
    def isolateLettersFromGrid(self, image_path):
        output_dir = "output"
        if self.verbose and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"L'immagine '{image_path}' non è stata trovata.")

        # Threshold to get binary image
        _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (binary.shape[1]//10, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, binary.shape[0]//10))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        grid = cv2.add(horizontal_lines, vertical_lines)

        # Find intersections
        intersections = cv2.bitwise_and(horizontal_lines, vertical_lines)
        coords = cv2.findNonZero(intersections)
        if coords is None:
            raise ValueError("No grid intersections found.")

        # Cluster intersection points to get unique row/col positions
        coords = coords[:, 0, :]
        x_coords = sorted(set([x for x, y in coords]))
        y_coords = sorted(set([y for x, y in coords]))

        def cluster_positions(positions, threshold=10):
            clustered = []
            for p in positions:
                if not clustered or abs(p - clustered[-1]) > threshold:
                    clustered.append(p)
            return clustered

        x_coords = cluster_positions(x_coords)
        y_coords = cluster_positions(y_coords)

        letters = []
        for row in range(len(y_coords) - 1):
            for col in range(len(x_coords) - 1):
                x1, x2 = x_coords[col], x_coords[col + 1]
                y1, y2 = y_coords[row], y_coords[row + 1]
                cell = img[y1:y2, x1:x2]
                # Optional: further crop to letter using your autoCropLetter
                cell = self.autoCropLetter(cell)
                letters.append((x1, y1, cell))
                if self.verbose:
                    cell_path = os.path.join(output_dir, f"cell_{row}_{col}.png")
                    cv2.imwrite(cell_path, cell)

        # Sort by row and column (already in order)
        return letters, len(y_coords) -1 , len(x_coords) - 1


    def predictGridLetters(self, isolated_letters):
        predictions = []
        for letter in isolated_letters:
            # letter is a tuple: (x, y, image)
            temp_img = Image.fromarray(letter[2])
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                temp_img.save(tmp.name)
                result = self.analyzeImage(tmp.name)
                predictions.append(result)
            os.unlink(tmp.name)  # Clean up the temporary file

        print("Predicted letters:", predictions)
        return predictions
                
    def runGridExtraction(self, image_path):
        # Step 1: Isolate letters from the grid
        isolated_letters, row, col = self.isolateLettersFromGrid(image_path)
        
        # Step 2: Predict letters from isolated images
        predictions = self.predictGridLetters(isolated_letters)
        
        # Optionally, compute rows/cols here if you add that logic
        # For now, just return predictions and isolated_letters
        return predictions, row, col
