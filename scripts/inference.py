import json
import os
import cv2
import torch
from torchvision import transforms
import pandas as pd
import glob

from scripts.train_model import DepthOrderingModel

PROJECT_PATH= './' #put your path
# geth the last model trained
directory = PROJECT_PATH + 'models/'
list_of_files = sorted(filter(os.path.isfile, glob.glob(directory + '*')), key=os.path.getmtime) 
latest_file = list_of_files[-1]

#paths
model_path = latest_file  # Path dell'ultimo modello PyTorch
test_data_dir = PROJECT_PATH + 'data/images/test/'              #Caricamento dei dati di validazione
test_segments_path = PROJECT_PATH + 'data/annotations/depth_TEST_segments.json' 
results_dir = PROJECT_PATH + 'results'           #risultati della inferenza


# Carica le annotazioni ground truth
with open(test_segments_path, 'r') as f:
    test_segments = json.load(f)

# Trasforma i dati per un lookup più semplice
test_segments_dict = {}
for ann in test_segments['annotations']:
    img_id = str(ann['image_id'])
    category_id = str(ann['category_id'])
    intradepth = ann['attributes'].get('Intradepth', 0)
    interdepth = ann['attributes'].get('Interdepth', 0)
    if img_id not in test_segments_dict:
        test_segments_dict[img_id] = {}
    test_segments_dict[img_id][category_id] = {
        'Intradepth': intradepth,
        'Interdepth': interdepth
    }

# Definizione del dispositivo (CPU o GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carica il modello PyTorch
model = DepthOrderingModel()  # Usa la stessa architettura del modello usata in fase di training
if torch.cuda.is_available():
    model.load_state_dict(torch.load(model_path, weights_only= True)) # Carica lo stato del modello su gpu
else:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only= True)) # Carica lo stato del modello e mappa i pesi sulla CPU se la GPU non è disponibile
model.to(device)
model.eval()


# Trasformazione delle immagini in tensori PyTorch
transform = transforms.Compose([
    transforms.ToTensor(),  # Converte le immagini in tensor
    transforms.Resize((128, 128)),  # Ridimensiona le immagini
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizza le immagini
])

# Carica e processa le immagini
X_test = []
img_ids = []
for img_info in test_segments['images']:
    img_id = img_info['id']
    img_name = img_info['file_name']
    img_path = os.path.join(test_data_dir, img_name)
    image = cv2.imread(img_path)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converti in RGB
        image = transform(image).to(device)  # Applica la trasformazione
        X_test.append(image)
        img_ids.append(img_id)

# Converti la lista di immagini in un batch di tensori
X_test = torch.stack(X_test)

# Ottieni le previsioni dal modello PyTorch
with torch.no_grad():
    pred_intra_depth, pred_inter_depth = model(X_test)

# Trasforma le previsioni in array 1D
pred_intra_depth = pred_intra_depth.cpu().numpy().flatten()
pred_inter_depth = pred_inter_depth.cpu().numpy().flatten()

# Assicurati che tutte le liste abbiano la stessa lunghezza
category_ids = [cat_id for img in test_segments_dict.values() for cat_id in img.keys()]
min_length = min(len(pred_intra_depth), len(pred_inter_depth), len(img_ids), len(category_ids))

# Taglia gli array alla lunghezza minima
pred_intra_depth = pred_intra_depth[:min_length]
pred_inter_depth = pred_inter_depth[:min_length]
img_ids = img_ids[:min_length]
category_ids = category_ids[:min_length]

# Crea il DataFrame delle previsioni
predictions_df = pd.DataFrame({
    'img_id': img_ids,
    'category_id': category_ids,
    'pred_Intradepth': pred_intra_depth,
    'pred_Interdepth': pred_inter_depth
})

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Save the DataFrame to a CSV file
pred_csv = predictions_df.to_csv(os.path.join(results_dir, 'test-predictions.csv'), index=False)
print(predictions_df.to_string())