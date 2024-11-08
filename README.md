### Guida Step-by-Step per il Fine-Tuning di YOLOv8 per Rilevamento di Bounding Box e Segmentazione

Questa guida descrive il processo di **fine-tuning di YOLOv8** per due task principali: **rilevamento di oggetti (bounding box)** e **segmentazione di oggetti** utilizzando dataset personalizzati. Utilizzerai modelli YOLOv8 pre-addestrati (`yolov8s.pt` per il rilevamento e `yolov8s-seg.pt` per la segmentazione) per effettuare il fine-tuning sui tuoi dataset.

---

### **1. Struttura della Directory del Progetto**

Assicurati che il tuo progetto sia organizzato come segue:

```
/tuo_progetto
  ├── /bbox_dataset
  │   ├── /images
  │   │   ├── /train  # Immagini di training per i bounding box
  │   │   └── /val    # Immagini di validazione per i bounding box
  │   ├── /labels
  │   │   ├── /train  # Annotazioni dei bounding box per il training
  │   │   └── /val    # Annotazioni dei bounding box per la validazione
  │   └── bbox_data.yaml  # File di configurazione del dataset di bounding box
  ├── /seg_dataset
  │   ├── /images
  │   │   ├── /train  # Immagini di training per la segmentazione
  │   │   └── /val    # Immagini di validazione per la segmentazione
  │   ├── /labels
  │   │   ├── /train  # Maschere di segmentazione per il training
  │   │   └── /val    # Maschere di segmentazione per la validazione
  │   └── seg_data.yaml  # File di configurazione del dataset di segmentazione
  ├── train_yolov8.py   # Script Python per addestrare YOLOv8 (bounding box e segmentazione)
```

### **2. Preparare i File di Configurazione del Dataset**

Sono necessari due file di configurazione `.yaml` che definiscono la struttura del tuo dataset: uno per l'addestramento del rilevamento di bounding box e uno per la segmentazione.

#### **`bbox_data.yaml`** (Config del Dataset di Bounding Box):
```yaml
train: path/to/bbox_dataset/images/train  # Percorso delle immagini di training
val: path/to/bbox_dataset/images/val      # Percorso delle immagini di validazione
nc: 1                                     # Numero di classi
names: ['oggetto']                        # Nome della/e classe/i
```

#### **`seg_data.yaml`** (Config del Dataset di Segmentazione):
```yaml
train: path/to/seg_dataset/images/train   # Percorso delle immagini di training
val: path/to/seg_dataset/images/val       # Percorso delle immagini di validazione
nc: 1                                     # Numero di classi
names: ['oggetto']                        # Nome della/e classe/i
```

### **3. Installare YOLOv8**

Prima di eseguire lo script di fine-tuning, installa la versione necessaria di YOLOv8 utilizzando il seguente comando:

```bash
pip install ultralytics==8.0.28
```

### **4. Script Python per il Fine-Tuning (train_yolov8.py)**

Ecco lo script completo per effettuare il fine-tuning di YOLOv8 per entrambi i task: **rilevamento di bounding box** e **segmentazione**.

```python
# Installare la libreria YOLOv8
# pip install ultralytics==8.0.28

# Importare il pacchetto YOLO da Ultralytics
from ultralytics import YOLO

# Definire i percorsi ai dataset di bounding box e segmentazione
BBOX_DATA_YAML_PATH = 'path/to/bbox_data.yaml'  # Sostituisci con il percorso reale di bbox_data.yaml
SEG_DATA_YAML_PATH = 'path/to/seg_data.yaml'    # Sostituisci con il percorso reale di seg_data.yaml

# Percorso del modello per Bounding Box
MODEL_PATH_DETECT = 'yolov8s.pt'  # Modello YOLOv8 pre-addestrato per il rilevamento

# Inizializzare il modello YOLOv8 per il rilevamento di bounding box
model_detect = YOLO(MODEL_PATH_DETECT)

# Addestrare il modello per il rilevamento di oggetti (bounding box)
model_detect.train(
    data=BBOX_DATA_YAML_PATH,   # Percorso al file bbox_data.yaml
    epochs=50,                  # Numero di epoche di addestramento
    imgsz=640,                  # Dimensione delle immagini (default 640x640)
    batch=16,                   # Batch size
    name='yolov8_bounding_box_experiment',  # Nome dell'esperimento
    task='detect'               # Specifica che è un task di rilevamento
)

# Percorso del modello per la Segmentazione
MODEL_PATH_SEG = 'yolov8s-seg.pt'  # Modello YOLOv8 pre-addestrato per la segmentazione

# Inizializzare il modello YOLOv8 per la segmentazione
model_seg = YOLO(MODEL_PATH_SEG)

# Addestrare il modello per la segmentazione
model_seg.train(
    data=SEG_DATA_YAML_PATH,    # Percorso al file seg_data.yaml
    epochs=50,                  # Numero di epoche di addestramento
    imgsz=640,                  # Dimensione delle immagini
    batch=16,                   # Batch size
    name='yolov8_segmentation_experiment',  # Nome dell'esperimento
    task='segment'              # Specifica che è un task di segmentazione
)

# Validare il modello addestrato per i bounding box
model_detect.val(
    data=BBOX_DATA_YAML_PATH,   # Percorso al file bbox_data.yaml
    task='detect'               # Validazione per il rilevamento di oggetti
)

# Validare il modello addestrato per la segmentazione
model_seg.val(
    data=SEG_DATA_YAML_PATH,    # Percorso al file seg_data.yaml
    task='segment'              # Validazione per la segmentazione
)

# Effettuare previsioni su nuove immagini utilizzando il modello di bounding box
model_detect.predict(
    source='path/to/test/images',  # Percorso alle immagini di test
    save=True,                     # Salva le previsioni come output
    task='detect'                  # Previsione dei bounding box
)

# Effettuare previsioni su nuove immagini utilizzando il modello di segmentazione
model_seg.predict(
    source='path/to/test/images',  # Percorso alle immagini di test
    save=True,                     # Salva le previsioni come output
    task='segment'                 # Previsione delle maschere di segmentazione
)
```

### **5. Procedura di Fine-Tuning**

#### **Passo 1: Organizza il tuo Dataset**
- Assicurati che il dataset sia strutturato correttamente come mostrato sopra.
- Posiziona le immagini e i file di annotazione (bounding box o maschere di segmentazione) nelle rispettive cartelle.

#### **Passo 2: Aggiorna i File di Configurazione del Dataset**
- Modifica i file `bbox_data.yaml` e `seg_data.yaml` per riflettere i percorsi reali del tuo dataset.

#### **Passo 3: Addestramento di YOLOv8**
- Esegui lo script Python (`train_yolov8.py`) per effettuare il fine-tuning di YOLOv8 per **bounding box** e **segmentazione**:
   - Addestramento per **bounding box**:
     ```bash
     python train_yolov8.py
     ```
   - Addestramento per **segmentazione**:
     ```bash
     python train_yolov8.py
     ```

#### **Passo 4: Monitoraggio dell'Addestramento**
- YOLOv8 mostrerà i log di addestramento con metriche come perdita, precisione (mAP), e altre. Puoi visualizzare i progressi di addestramento usando TensorBoard.

#### **Passo 5: Validazione dei Modelli**
- Dopo l'addestramento, valida i modelli per bounding box e segmentazione:
   - Per i bounding box:
     ```bash
     python train_yolov8.py
     ```
   - Per la segmentazione:
     ```bash
     python train_yolov8.py
     ```

#### **Passo 6: Effettua Previsioni**
- Usa i modelli addestrati per fare previsioni su nuove immagini e salva l'output con bounding box o maschere di segmentazione.

### **6. Distribuzione o Uso dei Modelli Addestrati**
Dopo aver completato l'addestramento e la validazione, puoi usare i modelli per l'inferenza o distribuirli in diversi ambienti. YOLOv8 supporta l'esportazione in formati come ONNX, CoreML, e altri per una facile integrazione.

---
