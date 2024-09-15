import json
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# Dataset class for custom dataset loading
class DepthDataset(Dataset):
    def __init__(self, data_dir, annotations_file, img_size=(128, 128), transform=None):
        """
        Args:
        - data_dir (str): La directory dove si trovano le immagini.
        - annotations_file (str): Il file JSON contenente le annotazioni.
        - img_size (tuple): La dimensione a cui ridimensionare le immagini (default: (128, 128)).
        - transform: Trasformazioni da applicare alle immagini.
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.transform = transform

        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)

        self.image_paths = []
        self.intradepth = []
        self.interdepth = []

        # Processa ogni immagine e le sue annotazioni
        for img_info in self.annotations['images']:
            img_id = img_info['id']
            img_name = img_info['file_name']
            img_path = os.path.join(data_dir, img_name)
            
            # Verifica che l'immagine esista
            if not os.path.exists(img_path):
                print(f"Warning: Image {img_name} not found.")
                continue
            
            self.image_paths.append(img_path)

            # Trova le annotazioni per questa immagine
            for annotation in self.annotations['annotations']:
                if annotation['image_id'] == img_id:
                    self.intradepth.append(annotation['attributes'].get('Intradepth', 0))
                    self.interdepth.append(annotation['attributes'].get('Interdepth', 0))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.resize(image, self.img_size)
        image = image / 255.0  # Normalizza
        
        if self.transform:
            image = self.transform(image)
        
        # Converte l'immagine in torch.float32
        image = image.float() 
        
        intradepth = torch.tensor(self.intradepth[idx], dtype=torch.float32)
        interdepth = torch.tensor(self.interdepth[idx], dtype=torch.float32)

        return image, intradepth, interdepth

# Definisco il modello
class DepthOrderingModel(nn.Module):
    def __init__(self):
        super(DepthOrderingModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 128 * 128, 256)
        self.fc_inter = nn.Linear(256, 1)
        self.fc_intra = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.upsample(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        inter_depth = self.fc_inter(x)
        intra_depth = self.fc_intra(x)
        return inter_depth, intra_depth

#se lo script viene runnato e non importato
if __name__ == "__main__":
    # Definisco le trasformazioni per le immagini
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Dati di allenamento
    data_dir = './data/images/train/'
    annotations_file = './annotations/train-annotations.json'
    dataset = DepthDataset(data_dir, annotations_file, transform=transform)

    # Divido il dataset in training e validation set
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Istanzia il modello e lo sposta sulla GPU se disponibile
    model = DepthOrderingModel().to(device)

    # Definisco l'ottimizzatore e la funzione di loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # training del modello
    num_epochs = 15
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, y_intra, y_inter in train_loader:
            inputs, y_intra, y_inter = inputs.to(device), y_intra.to(device), y_inter.to(device)
            
            optimizer.zero_grad()
            
            outputs_inter, outputs_intra = model(inputs)
            loss_inter = criterion(outputs_inter.squeeze(), y_inter)
            loss_intra = criterion(outputs_intra.squeeze(), y_intra)
            loss = loss_inter + loss_intra
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}")
        
        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, y_intra, y_inter in val_loader:
                inputs, y_intra, y_inter = inputs.to(device), y_intra.to(device), y_inter.to(device)
                outputs_inter, outputs_intra = model(inputs)
                loss_inter = criterion(outputs_inter.squeeze(), y_inter)
                loss_intra = criterion(outputs_intra.squeeze(), y_intra)
                val_loss += (loss_inter.item() + loss_intra.item())
        
        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

    # Salvataggio del modello
    torch.save(model.state_dict(), './models/best_model.pth')
