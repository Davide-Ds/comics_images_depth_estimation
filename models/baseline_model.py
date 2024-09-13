# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.models import Model
import numpy as np
import os
import cv2
import json


def prepare_dataset(data_dir, annotations_file, img_size=(128, 128)):
    """
    Prepara il dataset di immagini e annotazioni per l'addestramento del modello di stima della profondità.
    
    Args:
    - data_dir (str): La directory dove si trovano le immagini.
    - annotations_file (str): Il file JSON contenente le annotazioni.
    - img_size (tuple): La dimensione a cui ridimensionare le immagini (default: (128, 128)).
    
    Returns:
    - X_train (np.array): Array contenente le immagini preprocessate.
    - y_train_intra (np.array): Array contenente le etichette di intradepth.
    - y_train_inter (np.array): Array contenente le etichette di interdepth.
    """
    
    # Carica le annotazioni
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)

    # Inizializza le liste per i dati di training
    X_train = []
    y_train_intra = []
    y_train_inter = []

    # Processa ogni immagine e le sue annotazioni
    for img_info in annotations['images']:
        img_id = img_info['id']
        img_name = img_info['file_name']
        img_path = os.path.join(data_dir, img_name)
        
        # Legge e preelabora l'immagine
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Warning: Image {img_name} not found.")
            continue
        
        # Ridimensiona e normalizza l'immagine
        image = cv2.resize(image, img_size) / 255.0
        X_train.append(image)
        
        # Trova le annotazioni per questa immagine
        for annotation in annotations['annotations']:
            if annotation['image_id'] == img_id:
                intradepth = annotation['attributes'].get('Intradepth', 0)  # Valore di default 0 se non presente
                interdepth = annotation['attributes'].get('Interdepth', 0)  # Valore di default 0 se non presente
                y_train_intra.append(intradepth)
                y_train_inter.append(interdepth)

    # Converte le liste in array NumPy
    X_train = np.array(X_train)
    y_train_intra = np.array(y_train_intra)
    y_train_inter = np.array(y_train_inter)

    return X_train, y_train_intra, y_train_inter

# a simple baseline model for depth ordering
def create_baseline_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # Convolutional layers for feature extraction
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Flatten and fully connected layers for prediction
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    inter_depth = Dense(1, name='inter_depth')(x)
    intra_depth = Dense(1, name='intra_depth')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=[inter_depth, intra_depth])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae','mae'])
    return model

# Esempio di utilizzo della funzione
data_dir = './data/images/train/'
annotations_file = './annotations/train-annotations.json'
X_train, y_train_intra, y_train_inter = prepare_dataset(data_dir, annotations_file)

# Verifica delle shape degli array
print(X_train.shape)  # Dovrebbe essere (num_images, 128, 128, 3)
print(y_train_intra.shape)  # Dovrebbe essere (num_images,)
print(y_train_inter.shape)  # Dovrebbe essere (num_images,)

# Example of model training
model = create_baseline_model((128, 128, 3))
y_train = {'inter_depth': y_train_inter, 'intra_depth': y_train_intra}
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Save the model
model.save('baseline_model.keras')

