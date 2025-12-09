#!/usr/bin/env python3
"""
dataset_LH.py (Fixed to match working dataset.py structure)
High-resolution dataset with advanced augmentations for IDD-AW
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class IDDAWDataset(Dataset):
    """IDD Adverse Weather Dataset with RGB, NIR, and segmentation"""
    
    def __init__(self, data_root, split='train', img_size=(512, 1024), 
                 transform=None, weather_conditions=['FOG', 'RAIN', 'LOWLIGHT', 'SNOW']):
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        self.transform = transform
        self.weather_conditions = weather_conditions
        
        # Weather mapping
        self.weather_map = {'FOG': 0, 'RAIN': 1, 'LOWLIGHT': 2, 'SNOW': 3}
        
        # Load sample list
        self.samples = self._load_samples()
        
        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found for {split} split in {data_root}")
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_samples(self):
        """Load all available samples using the same structure as dataset.py"""
        samples = []
        
        for weather in self.weather_conditions:
            weather_path = os.path.join(self.data_root, self.split, weather)
            rgb_path = os.path.join(weather_path, 'rgb')
            nir_path = os.path.join(weather_path, 'nir')
            seg_path = os.path.join(weather_path, 'gtSeg')
            
            if not os.path.exists(rgb_path):
                continue
            
            # List all sequence folders
            for seq_id in os.listdir(rgb_path):
                seq_rgb = os.path.join(rgb_path, seq_id)
                seq_nir = os.path.join(nir_path, seq_id)
                seq_seg = os.path.join(seg_path, seq_id)
                
                if not os.path.isdir(seq_rgb):
                    continue
                
                # List all images in sequence
                rgb_files = sorted([f for f in os.listdir(seq_rgb) if f.endswith('.png')])
                
                for img_name in rgb_files:
                    base_name = img_name.replace('_rgb.png', '').replace('.png', '')
                    
                    # Possible NIR and GT naming patterns
                    nir_patterns = [
                        img_name.replace('_rgb.png', '_nir.png'),
                        img_name.replace('.png', '_nir.png'),
                        base_name + '_nir.png',
                        img_name
                    ]
                    
                    seg_patterns = [
                        img_name.replace('_rgb.png', '_gt.png'),
                        base_name + '_gt.png',
                        img_name.replace('_rgb.png', '_gtFine_labelIds.png'),
                        img_name
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
                        samples.append({
                            'rgb': rgb_file,
                            'nir': nir_file,
                            'seg': seg_file,
                            'weather': self.weather_map[weather],
                            'weather_name': weather,
                            'seq_id': seq_id
                        })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load images
        rgb = np.array(Image.open(sample['rgb']).convert('RGB'))
        nir = np.array(Image.open(sample['nir']).convert('L'))
        seg = np.array(Image.open(sample['seg']))
        
        # Resize
        rgb = cv2.resize(rgb, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)
        nir = cv2.resize(nir, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)
        seg = cv2.resize(seg, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=rgb, mask=seg)
            rgb = transformed['image']
            seg = transformed['mask']
        
        # CRITICAL FIX: Check if seg is already a tensor after transforms
        if isinstance(seg, torch.Tensor):
            seg_tensor = seg.long()
        else:
            seg_tensor = torch.from_numpy(seg).long()
        
        # Handle NIR separately (not affected by color transforms)
        nir_tensor = torch.from_numpy(nir).float().unsqueeze(0) / 255.0
        
        return {
            'rgb': rgb,
            'nir': nir_tensor,
            'seg': seg_tensor,
            'weather': torch.tensor(sample['weather'], dtype=torch.long),
            'weather_name': sample['weather_name']
        }


def get_train_transforms(img_size=(512, 1024)):
    """Advanced augmentations for training"""
    return A.Compose([
        # Spatial transforms
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.8, 1.2),
            translate_percent=(-0.1, 0.1),
            rotate=(-15, 15),
            shear=(-5, 5),
            p=0.5
        ),
        
        # Weather simulation
        A.OneOf([
            A.RandomRain(p=1.0),
            A.RandomFog(p=1.0),
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_limit=(1, 2),
                shadow_dimension=5,
                p=1.0
            ),
        ], p=0.3),
        
        # Color transforms
        A.OneOf([
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0
            ),
            A.RGBShift(
                r_shift_limit=25,
                g_shift_limit=25,
                b_shift_limit=25,
                p=1.0
            ),
        ], p=0.5),
        
        # CLAHE for low-light enhancement
        A.CLAHE(
            clip_limit=4.0,
            tile_grid_size=(8, 8),
            p=0.3
        ),
        
        # Brightness/Contrast
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.5
        ),
        
        # Blur
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
        ], p=0.2),
        
        # Noise
        A.GaussNoise(p=0.2),
        
        # Normalization and tensor conversion
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


def get_val_transforms(img_size=(512, 1024)):
    """Validation transforms (no augmentation)"""
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


def get_dataloaders(data_root, batch_size=8, img_size=(512, 1024), num_workers=4):
    """Create train and validation dataloaders"""
    
    train_transform = get_train_transforms(img_size)
    val_transform = get_val_transforms(img_size)
    
    train_dataset = IDDAWDataset(
        data_root=data_root,
        split='train',
        img_size=img_size,
        transform=train_transform
    )
    
    val_dataset = IDDAWDataset(
        data_root=data_root,
        split='val',
        img_size=img_size,
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    # Test the dataset
    data_root = '/home/jupyter-228w1a12b8/dec_7/IDD_AW/IDD_AW/IDDAW'
    
    print("Creating dataloaders...")
    train_loader, val_loader = get_dataloaders(
        data_root,
        batch_size=2,
        img_size=(512, 1024),
        num_workers=0
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    print("\nTesting train loader...")
    batch = next(iter(train_loader))
    
    print(f"RGB shape: {batch['rgb'].shape}")
    print(f"NIR shape: {batch['nir'].shape}")
    print(f"Seg shape: {batch['seg'].shape}")
    print(f"Weather: {batch['weather']}")
    print(f"Weather names: {batch['weather_name']}")
    
    print("\n✓ Dataset test passed!")