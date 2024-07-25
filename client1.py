import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple
from u_net_model import UNet
from torchvision import transforms
from preprocess_data import ImageMaskDataset, read_data  
from dice_b_loss import DiceBCELoss
import csv
import os
from sklearn.model_selection import train_test_split

# Klijent sa `DataLoader`-om
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, criterion, optimizer):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.csv_file = 'training_data.csv'
        self._initialize_csv()

    def get_parameters(self):
        return [val.cpu().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        for param, val in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(val, dtype=param.data.dtype)

    def _initialize_csv(self):
        if not os.path.isfile(self.csv_file):
            with open(self.csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['epoch', 'train_loss', 'train_accuracy', 'test_loss', 'test_accuracy'])

    def _log_to_csv(self, epoch, train_loss, train_accuracy, test_loss, test_accuracy):
        with open(self.csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, train_loss, train_accuracy, test_loss, test_accuracy])

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(config["epochs"]):
            epoch_loss = 0.0
            for images, masks in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                pred = (torch.sigmoid(outputs) > 0.5).float()
                correct += (pred == masks).sum().item()
                total += masks.numel()
            
            train_loss = epoch_loss / len(self.train_loader.dataset)
            train_accuracy = correct / total
            
            # Evaluacija na kraju epohe
            test_loss, test_accuracy = self.evaluate(self.get_parameters(), config)
            
            # Zapišite rezultate u CSV
            self._log_to_csv(epoch, train_loss, train_accuracy, test_loss, test_accuracy)
        
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, masks in self.test_loader:
                outputs = self.model(images)
                loss += self.criterion(outputs, masks).item()
                pred = (torch.sigmoid(outputs) > 0.5).float()
                correct += (pred == masks).sum().item()
                total += masks.numel()
        
        loss /= len(self.test_loader.dataset)
        accuracy = correct / total
        return loss, accuracy

# Model, optimizer i Flower klijent inicijalizacija
model = UNet(in_channels=1, out_channels=1)
images, masks = read_data()
train_images, test_images, train_masks, test_masks = train_test_split(images, masks, test_size=0.2, random_state=42)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = ImageMaskDataset(train_images, train_masks, transform=transform)
test_dataset = ImageMaskDataset(test_images, test_masks, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Definišemo kriterijum za gubitak
criterion = DiceBCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
client = FlowerClient(model, train_loader, test_loader, criterion, optimizer)


fl.client.start_client(
    server_address='hostname:port',
    client=FlowerClient(model, train_loader, test_loader, criterion, optimizer).to_client() 
)
