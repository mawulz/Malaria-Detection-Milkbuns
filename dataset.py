import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive
from sklearn.model_selection import train_test_split

class MalariaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['Uninfected', 'Parasitized']
        self.image_paths = []
        self.labels = []
        
        for idx, cls in enumerate(self.classes):
            cls_dir = os.path.join(root_dir, 'cell_images', cls)
            if os.path.isdir(cls_dir):
                for img_name in sorted(os.listdir(cls_dir)):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(cls_dir, img_name))
                        self.labels.append(idx)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
                        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def prepare_data(data_dir, batch_size=32):
    import zipfile
    cell_images_path = os.path.join(data_dir, 'cell_images')
    zip_path = os.path.join(data_dir, 'cell_images.zip')
    url = "https://ceb.nlm.nih.gov/proj/malaria/cell_images.zip"
    
    if not os.path.exists(cell_images_path):
        print("Dataset directory not found. Attempting to download from NIH...")
        try:
            download_and_extract_archive(url, data_dir, extract_root=data_dir, filename="cell_images.zip")
            if not os.path.exists(cell_images_path):
                raise RuntimeError("Extraction finished, but the 'cell_images' folder is missing.")
        except Exception as e:
            # If the zip downloaded partially or is corrupted (like an HTML page), clean it up
            if os.path.exists(zip_path):
                try:
                    with zipfile.ZipFile(zip_path, 'r') as z:
                        pass
                except zipfile.BadZipFile:
                    print(f"Corrupted zip detected. Removing: {zip_path}")
                    os.remove(zip_path)
                    
            print(f"\n[ERROR] Auto-download failed: {e}")
            print("The NIH server may have blocked the request or the link is temporarily offline.")
            print("Please perform a MANUAL DOWNLOAD of the NIH Malaria dataset (e.g., from Kaggle).")
            print(f"Once downloaded, extract the files so that 'Parasitized' and 'Uninfected' are strictly located at:")
            print(f" 👉 {os.path.abspath(cell_images_path)}")
            raise SystemExit("\nExiting: The dataset must be present to proceed.")
    
    # Transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.03),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load full dataset to get structure
    full_dataset = MalariaDataset(root_dir=data_dir, transform=None)
    
    # Assert distribution is approximately equal (or exactly equal based on the dataset)
    total_imgs = len(full_dataset)
    num_uninfected = (full_dataset.labels == 0).sum().item()
    num_parasitized = (full_dataset.labels == 1).sum().item()
    print(f"Total Images: {total_imgs} | Uninfected: {num_uninfected} | Parasitized: {num_parasitized}")
    
    # Identical splits across seeds (80/10/10)
    # We use a fixed random state = 42 for splitting only to ensure consistency across any seed used in main loop
    targets = full_dataset.labels.tolist()
    train_idx, temp_idx = train_test_split(list(range(total_imgs)), test_size=0.2, stratify=targets, random_state=42)
    
    # Temp is 20%. Split it evenly to get 10% val, 10% test
    temp_targets = [targets[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=temp_targets, random_state=42)
    
    train_dataset = MalariaDataset(root_dir=data_dir, transform=train_transform)
    train_dataset.image_paths = [full_dataset.image_paths[i] for i in train_idx]
    train_dataset.labels = full_dataset.labels[train_idx]
    
    val_dataset = MalariaDataset(root_dir=data_dir, transform=val_test_transform)
    val_dataset.image_paths = [full_dataset.image_paths[i] for i in val_idx]
    val_dataset.labels = full_dataset.labels[val_idx]
    
    test_dataset = MalariaDataset(root_dir=data_dir, transform=val_test_transform)
    test_dataset.image_paths = [full_dataset.image_paths[i] for i in test_idx]
    test_dataset.labels = full_dataset.labels[test_idx]
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4, persistent_workers=True)
    
    return train_loader, val_loader, test_loader
