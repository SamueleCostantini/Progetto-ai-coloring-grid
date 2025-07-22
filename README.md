<p> Per l'esecuzione sono richieste le seguenti librerie <br>

pip install scipy<br>
pip install matplotlib<br>
pip install numpy<br>
pip install tensorflow<br>
pip install PIL<br>
pip install opencv-python-headless<br>
pip install image io<br>

</p>

<ul>
<li><strong>Struttura del progetto</strong><br>
Il progetto è stato strutturato orientato alla modularità quindi ogni funzione del sistema è richiamabile ed usabile separatamente dal flusso principale di esecuzione del progetto.
</li>

<li><strong>Librerie utilizzate:</strong>
<ul>
<li>Aima: search.py e utils.py</li>
<li>PIL: manipolazione immagini</li>
<li>numpy: calcolo numerico</li>
<li>tensorflow: machine learning</li>
<li>matplotlib: visualizzazione dei dati</li>
<li>OpenCv (cv2): computer vision</li>
<li>tempfile: file temporanei</li>
</ul>
</li>

<li>Il progetto è suddiviso principalmente in tre parti:
<ul>
<li><strong>Training del modello:</strong>
<ul>
<li>training-modello.ipynb: file jupiter nootebook usato per tenere traccia del training</li>
<li>letter_reconition_model.h5: modello vero e proprio</li>
</ul>
</li>

<li><strong>Preprocessing immagine e implementazione modello:</strong>
<ul>
<li>AiTextExtractorService.py: classe che implementa tutte le funzioni di preprocessing, di mapping e dell’implementazione vera e propria del modello<br>
Formata da:
<ul>
<li>mapPredictedClassToLetter( classValue ): utilizzata per mappare classe a lettera effettiva</li>
<li>removeColorRange( img_array, start_hex, end_hex ): rimuove un certo range di colori ( non utilizzata )</li>
<li>autoCropLetter(img_array): sfruttando la Computer Vision di OpenCv ritaglia l’immagine mantenendo le proporzioni della lettera e centrandola in un box quadrato, così da convertire successivamente in 28x28 per il modello</li>
<li>analyzeImage( imagePath ): data un'immagine fisica, la trasforma in array di bit e con autoCropLetter ed altre operazioni di preprocessing che approfondiremo dopo, implementa il modello di classificazione per estrapolare la classe di una sola lettera</li>
<li>isolateLettersFromGrid( image_path): partendo da un'immagine di una griglia, con OpenCv, riconosce i contorni delle lettere e riesce a restituire un array lettere con numero di righe e numero di colonne individuate</li>
<li>predictGridLetters( isolatedLetters ): prendendo l’array di lettere, usando file temporanei, ritorna l’array di classi (lettere estratte) utilizzando analyzeImage</li>
<li>runGridExtraction( image_path ): riassume tutte le funzioni della classe</li>
</ul>
</li>
</ul>
</li>

<li><strong>Conversione griglia estratta in problema di ricerca Aima</strong>
<ul>
<li>GridProblem.py: classe che definisce tutte gli attributi e le funzioni utili per la creazione di un search problem Aima<br>
initial: contiene l’initial state ed è una tupla formata da una tupla con la griglia e una tupla con le coordinate<br>
goal_color: il colore con cui colorare tutta la griglia che può essere un colore della griglia oppure ‘a’ che prende il colore con più occorrenze, a prescindere dal costo<br>
start_position: una tupla con le coordinate<br>
color_cost: impostazione dei costi dei colori nell’ordine g, y, b<br>
rows: numero righe<br>
cols: numero colonne<br>
letters: array raw delle lettere<br>
actions( state ): definizione delle azioni<br>
result ( state, action ): ritorna il nuovo stato data un azione<br>
goal_test ( state ): riconosce se lo stato è quello goal<br>
path_cost (c, action): ritorna il costo in base all’attributo color_cost, assegna un costo ad ogni paint di un colore specifico<br>
h (node): definizione di un euristica
</li>
</ul>
</li>
</ul>
</li>

<li><strong>Classe principale che contiene l’esecuzione del programma:</strong><br>
La classe main contiene l’istanza delle varie classi descritte prima, l’import delle classi di Aima per la ricerca, simulazioni delle varie ricerche e stampe sull’andamento di ogni singola ricerca.
</li>

<li>Directory con tutte le varie immagini ricavate durante i processi</li>
</ul>

