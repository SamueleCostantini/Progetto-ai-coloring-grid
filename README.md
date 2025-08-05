<h2>🔧 Required Libraries for Execution</h2>
<pre>
pip install scipy
pip install matplotlib
pip install numpy
pip install tensorflow
pip install PIL
pip install opencv-python-headless
pip install imageio
</pre>

<h2>📁 Project Structure</h2>
<p>
The project is structured with modularity in mind, allowing each system component to be used independently of the main execution flow.
</p>

<h3>📚 Libraries Used</h3>
<ul>
  <li>Aima: <code>search.py</code> and <code>utils.py</code></li>
  <li>PIL: image manipulation</li>
  <li>NumPy: numerical computation</li>
  <li>TensorFlow: machine learning</li>
  <li>Matplotlib: data visualization</li>
  <li>OpenCV (cv2): computer vision</li>
  <li>Tempfile: temporary file handling</li>
</ul>

<h3>🧩 Project Breakdown</h3>
<h4>1. Model Training</h4>
<ul>
  <li><code>training-modello.ipynb</code>: Jupyter Notebook for tracking training progress</li>
  <li><code>letter_reconition_model.h5</code>: trained model file</li>
</ul>

<h4>2. Image Preprocessing and Model Implementation</h4>
<ul>
  <li><code>AiTextExtractorService.py</code>: Implements preprocessing, mapping, and model execution</li>
  <li>Main functions include:
    <ul>
      <li><code>mapPredictedClassToLetter(classValue)</code>: maps class to letter</li>
      <li><code>removeColorRange(img_array, start_hex, end_hex)</code>: removes specific color range (unused)</li>
      <li><code>autoCropLetter(img_array)</code>: crops and centers the letter using OpenCV</li>
      <li><code>analyzeImage(imagePath)</code>: preprocesses image and applies classification model</li>
      <li><code>isolateLettersFromGrid(image_path)</code>: detects letters in a grid</li>
      <li><code>predictGridLetters(isolatedLetters)</code>: classifies grid letters using temporary files</li>
      <li><code>runGridExtraction(image_path)</code>: combines all above functions</li>
    </ul>
  </li>
</ul>

<h4>3. Converting Extracted Grid into Aima Search Problem</h4>
<ul>
  <li><code>GridProblem.py</code>: Defines the search problem structure</li>
  <li>Key attributes and methods:
    <ul>
      <li><strong>initial</strong>: tuple with grid and starting coordinates</li>
      <li><strong>goal_color</strong>: target color (specific or most frequent)</li>
      <li><strong>start_position</strong>: starting coordinates</li>
      <li><strong>color_cost</strong>: costs for colors (green, yellow, blue)</li>
      <li><strong>rows / cols</strong>: grid dimensions</li>
      <li><strong>letters</strong>: raw letter array</li>
      <li><strong>actions(state)</strong>: available actions</li>
      <li><strong>result(state, action)</strong>: returns new state</li>
      <li><strong>goal_test(state)</strong>: goal state evaluation</li>
      <li><strong>path_cost(c, action)</strong>: cost calculation based on color</li>
      <li><strong>h(node)</strong>: heuristic definition</li>
    </ul>
  </li>
</ul>

<h3>🧠 Main Class</h3>
<p>
The main class initializes all modules, imports Aima components, runs search simulations, and outputs results for each approach.
</p>

<h3>🖼️ Image Directory</h3>
<p>
Contains all intermediate and final image assets generated during processing.
</p>


🖼️ Image Directory
Contains all image files generated or used during the workflow.
<br><br>
<h2>🚀 Using AiTextExtractorService from This Repository</h2>
<p>
This project includes a modular class called <code>AiTextExtractorService.py</code> for preprocessing image data and extracting handwritten letters. Here's how to use it:
</p>

<h3>1. 📥 Clone the Repository</h3>
<p>Run this command in your terminal:</p>
<pre><code>git clone https://github.com/SamueleCostantini/cnn-for-handwritten-letter-reconitioning.git</code></pre>
<p>
Ensure the file <code>AiTextExtractorService.py</code> is located in the <code>/src</code> directory.
</p>

<h3>2. 📦 Import the Class into Your Python Project</h3>
<p>Use this code snippet to import the class:</p>
<pre><code>import sys
import os

# Add the repo path to Python's module search path
repo_path = os.path.join(os.getcwd(), "cnn-for-handwritten-letter-reconitioning", "src")
sys.path.append(repo_path)

# Import the class
from AiTextExtractorService import AiTextExtractorService

# Initialize the service
extractor = AiTextExtractorService()</code></pre>

<h3>✅ Optional: Clone Programmatically with GitPython</h3>
<p>
You can also clone the repo from within your Python script using <code>GitPython</code>:
</p>
<pre><code>pip install gitpython</code></pre>

<p>Then use this code:</p>
<pre><code>from git import Repo
import sys
import os

# Clone the repository
repo_url = "https://github.com/SamueleCostantini/cnn-for-handwritten-letter-reconitioning.git"
local_dir = "cnn_temp"
Repo.clone_from(repo_url, local_dir)

# Add the module to the path
sys.path.append(os.path.join(local_dir, "src"))
from AiTextExtractorService import AiTextExtractorService

extractor = AiTextExtractorService()</code></pre>

<p>Now you're ready to start extracting letters from images using your trained CNN model and preprocessing pipeline! 🧠📸</p>

