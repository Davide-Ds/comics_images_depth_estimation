from datetime import datetime
import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from torchvision import transforms
import os
import pandas as pd
import json
from train_model import DepthOrderingModel

# Funzione per valutare il modello
def evaluate_model(predictions, ground_truth):
    pred_inter_depths = []
    pred_intra_depths = []
    true_inter_depths = []
    true_intra_depths = []

    # Processa ogni previsione
    for index, row in predictions.iterrows():
        img_id = str(row['img_id'])
        category_id = str(row['category_id'])

        pred_inter_depth = row['pred_Interdepth']
        pred_intra_depth = row['pred_Intradepth']

        # Verifica se l'immagine e la categoria esistono nel ground truth
        if img_id in ground_truth and category_id in ground_truth[img_id]:
            true_inter_depth = ground_truth[img_id][category_id]['Interdepth']
            true_intra_depth = ground_truth[img_id][category_id]['Intradepth']

            # Aggiungi previsioni e ground truth
            pred_inter_depths.append(pred_inter_depth)
            pred_intra_depths.append(pred_intra_depth)
            true_inter_depths.append(true_inter_depth)
            true_intra_depths.append(true_intra_depth)
        else:
            print(f"Warning: Image ID {img_id} or Category ID {category_id} not found in ground truth")

    # Calcolo del MSE
    if len(true_inter_depths) > 0 and len(pred_inter_depths) > 0:
        mse_inter = mean_squared_error(true_inter_depths, pred_inter_depths)
        mse_intra = mean_squared_error(true_intra_depths, pred_intra_depths)
        mse_overall = (mse_inter + mse_intra) / 2
    else:
        mse_inter, mse_intra, mse_overall = None, None, None
        print("No valid data for MSE calculation.")

    return mse_inter, mse_intra, mse_overall


# Caricamento dei dati di validazione
val_data_dir = './data/images/val/'
gt_path = './annotations/val-annotations.json'  # Path agli annotations del ground truth

# Carica le annotazioni ground truth
with open(gt_path, 'r') as f:
    ground_truth_data = json.load(f)

# Trasforma i dati del ground truth per un lookup più semplice
ground_truth = {}
for ann in ground_truth_data['annotations']:
    img_id = str(ann['image_id'])
    category_id = str(ann['category_id'])
    intradepth = ann['attributes'].get('Intradepth', 0)
    interdepth = ann['attributes'].get('Interdepth', 0)
    if img_id not in ground_truth:
        ground_truth[img_id] = {}
    ground_truth[img_id][category_id] = {
        'Intradepth': intradepth,
        'Interdepth': interdepth
    }

# Definizione del dispositivo (CPU o GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carica il modello PyTorch
model_path = './models/best_model.pth'
model = DepthOrderingModel()  # Usa la stessa architettura del modello usata in fase di training
model.load_state_dict(torch.load(model_path)) # Carica lo stato del modello
model.to(device)
model.eval()


# Trasformazione delle immagini in tensori PyTorch
transform = transforms.Compose([
    transforms.ToTensor(),  # Converte le immagini in tensor
    transforms.Resize((128, 128)),  # Ridimensiona le immagini
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizza le immagini
])

# Carica e processa le immagini
X_val = []
img_ids = []
for img_info in ground_truth_data['images']:
    img_id = img_info['id']
    img_name = img_info['file_name']
    img_path = os.path.join(val_data_dir, img_name)
    image = cv2.imread(img_path)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converti in RGB
        image = transform(image).to(device)  # Applica la trasformazione
        X_val.append(image)
        img_ids.append(img_id)

# Converti la lista di immagini in un batch di tensori
X_val = torch.stack(X_val)

# Ottieni le previsioni dal modello PyTorch
with torch.no_grad():
    pred_intra_depth, pred_inter_depth = model(X_val)

# Trasforma le previsioni in array 1D
pred_intra_depth = pred_intra_depth.cpu().numpy().flatten()
pred_inter_depth = pred_inter_depth.cpu().numpy().flatten()

# Assicurati che tutte le liste abbiano la stessa lunghezza
category_ids = [cat_id for img in ground_truth.values() for cat_id in img.keys()]
min_length = min(len(pred_intra_depth), len(pred_inter_depth), len(img_ids), len(category_ids))

# Taglia gli array alla lunghezza minima
pred_intra_depth = pred_intra_depth[:min_length]
pred_inter_depth = pred_inter_depth[:min_length]
img_ids = img_ids[:min_length]
category_ids = category_ids[:min_length]

# Crea il DataFrame delle previsioni
predictions_df = pd.DataFrame({
    'pred_Intradepth': pred_intra_depth,
    'pred_Interdepth': pred_inter_depth,
    'img_id': img_ids,
    'category_id': category_ids
})

# Valuta il modello
mse_inter, mse_intra, mse_overall = evaluate_model(predictions_df, ground_truth)

# Salva i risultati
results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    
# Ottieni la data e l'ora corrente
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Scrivi i risultati nel file, includendo data e ora
with open(os.path.join(results_dir, 'evaluation_metrics.txt'), 'a') as file:
    file.write(f"\nEvaluation written on: {current_time}\n")
    file.write(f'Model: {model_path}\n')
    file.write(f'MSE of Inter-depth: {mse_inter}\n')
    file.write(f'MSE of Intra-depth: {mse_intra}\n')
    file.write(f'Overall MSE: {mse_overall}\n')

print(f'MSE of Inter-depth: {mse_inter}')
print(f'MSE of Intra-depth: {mse_intra}')
print(f'Overall MSE: {mse_overall}')
