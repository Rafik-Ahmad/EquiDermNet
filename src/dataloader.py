
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import numpy as np

class HAM10000Dataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Mapping
        self.label_map = {
            'mel': 0, 'nv': 1, 'bcc': 2, 'akiec': 3, 
            'bkl': 4, 'df': 5, 'vasc': 6
        }
        
        # Pre-compute labels for sampling
        self.labels = [self.label_map.get(dx, 1) for dx in self.df['dx']]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image_id'] + '.jpg'
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # Fallback for missing images
            image = Image.new('RGB', (224, 224))
        
        # Mock ITA logic if missing
        if 'ita' in row:
            ita = row['ita']
        else:
            ita = np.random.uniform(10, 50) 
            
        skin_label = 0 if ita > 28 else 1
        diagnosis = self.label_map.get(row['dx'], 1)
        
        if self.transform:
            image = self.transform(image)
            
        return image, diagnosis, skin_label

def get_loaders(batch_size, data_dir, metadata_path):
    # Heavy Augmentation for Training
    train_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_path = os.path.join(data_dir, 'images')
    full_dataset = HAM10000Dataset(metadata_path, img_path, transform=None)
    
    # Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Apply specific transforms
    train_ds.dataset.transform = train_tfms
    val_ds.dataset.transform = val_tfms
    
    # --- WEIGHTED SAMPLING LOGIC ---
    # 1. Get targets from the underlying dataset subset
    targets = [full_dataset.labels[i] for i in train_ds.indices]
    
    # 2. Count classes
    class_counts = np.bincount(targets)
    class_weights = 1. / class_counts
    
    # 3. Assign weight to each sample
    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Create Loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader
