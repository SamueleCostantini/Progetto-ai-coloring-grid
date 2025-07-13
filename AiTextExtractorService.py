


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
        
        # Debug visualization
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
        
        # Visualizzo l'immagine processata
        plt.imshow(img_array, cmap="gray")

        
        # Reshape per il modello CNN
        img_array = img_array.reshape(1, 28, 28, 1)
        
        # Predizione
        prediction = self.model.predict(img_array)
        predicted_class = np.argmax(prediction)
        
        return self.mapPredictedClassToLetter(predicted_class)
    
    def isolateLettersFromGrid(self, image_path):
        print("Isolating letters from grid by detecting letters...")
        # Read image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Adaptive threshold for better separation in grids
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8
        )
        plt.imshow(binary, cmap='gray')
        # Morphology to clean up noise and connect letter parts
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours (letters)
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area and reasonable size
        min_area = img.shape[0] * img.shape[1] / 3000  # Lower threshold for small letters
        letter_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            if area > min_area and w > 8 and h > 8:
                letter_contours.append((x, y, w, h))

        # Sort contours top-to-bottom, then left-to-right
        letter_contours = sorted(letter_contours, key=lambda b: (b[1], b[0]))

        # Group letters by row (using y coordinate)
        rows = []
        row_tolerance = max(10, img.shape[0] // 40)
        for box in letter_contours:
            x, y, w, h = box
            placed = False
            for row in rows:
                if abs(row[0][1] - y) < row_tolerance:
                    row.append(box)
                    placed = True
                    break
            if not placed:
                rows.append([box])

        num_rows = len(rows)
        num_columns = max(len(row) for row in rows) if rows else 0

        # If grid detection failed, try to split the image into grid cells
        if num_rows < 2 or num_columns < 2:
            # Estimate grid size from image dimensions
            estimated_rows = 3
            estimated_cols = 5
            cell_h = img.shape[0] // estimated_rows
            cell_w = img.shape[1] // estimated_cols
            letter_images = []
            for r in range(estimated_rows):
                for c in range(estimated_cols):
                    y1 = r * cell_h
                    y2 = (r + 1) * cell_h
                    x1 = c * cell_w
                    x2 = (c + 1) * cell_w
                    cell_img = img_rgb[y1:y2, x1:x2]
                    cropped_letter = self.autoCropLetter(cell_img)
                    letter_images.append(cropped_letter)
            num_rows = estimated_rows
            num_columns = estimated_cols
        else:
            # Extract letter images row by row
            letter_images = []
            for row in rows:
                row = sorted(row, key=lambda b: b[0])  # left-to-right
                for x, y, w, h in row:
                    letter_img = img_rgb[y:y+h, x:x+w]
                    cropped_letter = self.autoCropLetter(letter_img)
                    letter_images.append(cropped_letter)
                    # Draw rectangle for debug
                    cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Show the detected letters
        if self.verbose:
            plt.figure(figsize=(15, 5))
            plt.subplot(121)
            plt.imshow(img_rgb)
            plt.title(f'Detected Letters ({num_rows} rows, max {num_columns} columns)')
            plt.subplot(122)
            if letter_images:
                fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 8))
                idx = 0
                for row in range(num_rows):
                    for col in range(num_columns):
                        if idx < len(letter_images):
                            if num_rows == 1:
                                axes[col].imshow(letter_images[idx])
                                axes[col].axis('off')
                                axes[col].set_title(f'Letter {idx+1}')
                            else:
                                axes[row, col].imshow(letter_images[idx])
                                axes[row, col].axis('off')
                                axes[row, col].set_title(f'Letter {idx+1}')
                            idx += 1
                plt.tight_layout()

        return letter_images, num_rows, num_columns

    def predictGridLetters(self, isolated_letters):
        predictions = []
        for letter in isolated_letters[1:]:
            temp_img = Image.fromarray(letter)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                temp_img.save(tmp.name)
                result = self.analyzeImage(tmp.name)
                predictions.append(result)
            os.unlink(tmp.name)  # Clean up the temporary file
        
        print("Predicted letters:", predictions)
        return predictions
                
    def runGridExtraction(self, image_path):
        # Step 1: Isolate letters from the grid
        isolated_letters, num_rows, num_columns = self.isolateLettersFromGrid(image_path)
        
        # Step 2: Predict letters from isolated images
        predictions = self.predictGridLetters(isolated_letters)
        
        return predictions, num_rows, num_columns
    