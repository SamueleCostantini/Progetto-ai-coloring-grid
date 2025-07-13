<h1>Il progetto è formato da:</h1>
<ol>
  <li> Script training del modello : <b>training-modello.ipynb</b> (a posto, testato e funziona con un accuracy del 93%)</li>
  <li> Modello: <b>letter_reconition_model.h5</b></li>
  <li> Microservizio che ha il compito di estrarre lettere dalla griglia: <b>AiTextExtractorService.ph</b> (Il problema si trova qui, nella funzione isolateLettersFromGrid(), non riconosce correttamente la dimensione della griglia, sto usando OpenCv per il
      processing e la gestione dell'immagine, il suo compito è quello di individuare il numero di righe e colonne e isolare le immagini delle lettere mettendole in un array e dandole in pasto al modello che riconosce la lettera, per ora conta male righe e colonne)</li>
  <li> Microservizio per la ricerca: <b>ResearchService.py</b>: contiene le funzioni utile per impostare il problema di ricerca (da terminare)</li>
  <li> Core del progetto: <b>Main.py</b>b> racchiude tutte le chiamate e il flusso generale del progetto</li>
  <li> Per il testing uso le immagini che sono nella directory costum-test </li>
</ol>

<h2> Istruzioni </h2>
<p> Per il testing uso due vie, o quella di testare le funzioni in riga di comando pero c'è bisogno di importare tutte le librerie. Per facilitare il lavoro basta incollare questo script nel terminale della cartella del progetto: </p>
<h3> mandare il comando "python" poi copiare e incollare nel terminale questo script </h3>
----------------------------------------------------------<br>
from PIL import Image <br>
import numpy as np <br>
import tensorflow as tf <br>
import matplotlib.pyplot as plt<br>
import cv2<br>
import tempfile<br>
import os<br>
<br>
import numpy as np<br>
<br>
<br>
# %%<br>
model = tf.keras.models.load_model('letter_recognition_model.h5')<br>
<br>
# %%<br>
from AiTextExtractorService import AiTextExtractorService<br>
from ResearchService import ColorGridProblem<br>
from utils import *<br>
from search import *<br>
<br>
# %%<br>
estrattore = AiTextExtractorService(model, False)  <br>
--------------------------------------------------------<br>

<p>Dopo di che puoi utilizzare tutte le funzioni di AiTextExtractorService e quindi testare le modifiche che fai alla funzione isolateLettersFromGrid cosi:</p>

<br>
lettere, num_rows, num_columns = estrattore.runGridExtraction('costum-test/5x3.png')
<br>

<p> runGridExtraction esegue isolateLettersFromGrid </p>
