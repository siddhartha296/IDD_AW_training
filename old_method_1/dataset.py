import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

class IDDAWDataset(Dataset):
    """IDD-AW Dataset for RGB-NIR semantic segmentation"""
    
    def __init__(self, root_dir, split='train', weather_conditions=['FOG', 'RAIN', 'LOWLIGHT', 'SNOW'], 
                 transform=None, img_size=(512, 1024)):
        """
        Args:
            root_dir: Path to IDDAW directory
            split: 'train' or 'val'
            weather_conditions: List of weather types to include
            transform: Albumentations transforms
            img_size: (height, width) for resizing
        """
        self.root_dir = root_dir
        self.split = split
        self.weather_conditions = weather_conditions
        self.img_size = img_size
        
        # Build file list
        self.samples = []
        for weather in weather_conditions:
            weather_path = os.path.join(root_dir, split, weather)
            rgb_path = os.path.join(weather_path, 'rgb')
            nir_path = os.path.join(weather_path, 'nir')
            seg_path = os.path.join(weather_path, 'gtSeg')
            
            if os.path.exists(rgb_path):
                # List all sequence folders
                for seq_id in os.listdir(rgb_path):
                    seq_rgb = os.path.join(rgb_path, seq_id)
                    seq_nir = os.path.join(nir_path, seq_id)
                    seq_seg = os.path.join(seg_path, seq_id)
                    
                    if os.path.isdir(seq_rgb):
                        # List all images in sequence
                        rgb_files = sorted([f for f in os.listdir(seq_rgb) if f.endswith('.png')])
                        
                        for img_name in rgb_files:
                            # Try different naming conventions
                            base_name = img_name.replace('_rgb.png', '').replace('.png', '')
                            
                            # Possible NIR and GT naming patterns
                            nir_patterns = [
                                img_name.replace('_rgb.png', '_nir.png'),
                                img_name.replace('.png', '_nir.png'),
                                base_name + '_nir.png',
                                img_name  # Same name in different folder
                            ]
                            
                            seg_patterns = [
                                img_name.replace('_rgb.png', '_gt.png'),
                                base_name + '_gt.png',
                                img_name.replace('_rgb.png', '_gtFine_labelIds.png'),
                                img_name  # Same name in different folder
                            ]
                            
                            rgb_file = os.path.join(seq_rgb, img_name)
                            
                            # Find matching NIR file
                            nir_file = None
                            for nir_name in nir_patterns:
                                candidate = os.path.join(seq_nir, nir_name)
                                if os.path.exists(candidate):
                                    nir_file = candidate
                                    break
                            
                            # Find matching segmentation file
                            seg_file = None
                            for seg_name in seg_patterns:
                                candidate = os.path.join(seq_seg, seg_name)
                                if os.path.exists(candidate):
                                    seg_file = candidate
                                    break
                            
                            if nir_file and seg_file:
                                self.samples.append({
                                    'rgb': rgb_file,
                                    'nir': nir_file,
                                    'seg': seg_file,
                                    'weather': weather,
                                    'seq_id': seq_id
                                })
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
        
        # Default transform
        if transform is None:
            self.transform = A.Compose([
                A.Resize(height=img_size[0], width=img_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = transform
            
        # Weather encoding
        self.weather_to_idx = {w: i for i, w in enumerate(['FOG', 'RAIN', 'LOWLIGHT', 'SNOW'])}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load images
        rgb = np.array(Image.open(sample['rgb']).convert('RGB'))
        nir = np.array(Image.open(sample['nir']).convert('L'))  # Grayscale NIR
        seg = np.array(Image.open(sample['seg']))
        
        # Resize segmentation separately (nearest neighbor)
        seg_resized = np.array(Image.fromarray(seg).resize(
            (self.img_size[1], self.img_size[0]), 
            Image.NEAREST
        ))
        
        # Apply transforms to RGB
        transformed = self.transform(image=rgb)
        rgb_tensor = transformed['image']
        
        # Transform NIR separately (same normalization)
        nir_normalized = (nir.astype(np.float32) / 255.0 - 0.5) / 0.5  # Normalize to [-1, 1]
        nir_tensor = torch.from_numpy(nir_normalized).unsqueeze(0).float()
        
        # Resize NIR to match RGB
        nir_tensor = torch.nn.functional.interpolate(
            nir_tensor.unsqueeze(0), 
            size=self.img_size, 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        seg_tensor = torch.from_numpy(seg_resized).long()
        
        # Weather encoding
        weather_idx = self.weather_to_idx.get(sample['weather'], 0)
        weather_tensor = torch.tensor(weather_idx, dtype=torch.long)
        
        return {
            'rgb': rgb_tensor,
            'nir': nir_tensor,
            'seg': seg_tensor,
            'weather': weather_tensor,
            'weather_name': sample['weather'],
            'path': sample['rgb']
        }


def get_dataloaders(root_dir, batch_size=4, num_workers=4, img_size=(512, 1024)):
    """Create train and validation dataloaders"""
    
    # Training augmentations
    train_transform = A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Validation (no augmentation)
    val_transform = A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    train_dataset = IDDAWDataset(root_dir, split='train', transform=train_transform, img_size=img_size)
    val_dataset = IDDAWDataset(root_dir, split='val', transform=val_transform, img_size=img_size)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


# Test the dataloader
if __name__ == '__main__':
    root_dir = '/home/jupyter-228w1a12b8/dec_7/IDD_AW/IDD_AW/IDDAW'
    
    train_loader, val_loader = get_dataloaders(root_dir, batch_size=2, num_workers=2)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test loading one batch
    for batch in train_loader:
        print("\nBatch shapes:")
        print(f"RGB: {batch['rgb'].shape}")
        print(f"NIR: {batch['nir'].shape}")
        print(f"Segmentation: {batch['seg'].shape}")
        print(f"Weather: {batch['weather']}")
        print(f"Weather names: {batch['weather_name']}")
        break