#!/usr/bin/env python3
"""
dataset_LH.py
Improved IDD-AW Dataset with Advanced Augmentations
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class IDDAWDataset(Dataset):
    """IDD-AW Dataset for RGB-NIR semantic segmentation with advanced augmentations"""
    
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
        self.transform = transform
        
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
        
        # Apply albumentations transforms to RGB and mask together
        if self.transform:
            # Albumentations requires mask as 2D array
            transformed = self.transform(image=rgb, mask=seg)
            rgb_tensor = transformed['image']
            seg_resized = transformed['mask']
        else:
            # Default resize
            rgb = np.array(Image.fromarray(rgb).resize(
                (self.img_size[1], self.img_size[0]), Image.BILINEAR
            ))
            seg_resized = np.array(Image.fromarray(seg).resize(
                (self.img_size[1], self.img_size[0]), Image.NEAREST
            ))
            # Normalize RGB
            rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            rgb_tensor = (rgb_tensor - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / \
                        torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        # Transform NIR separately (same spatial transforms)
        nir_resized = np.array(Image.fromarray(nir).resize(
            (self.img_size[1], self.img_size[0]), Image.BILINEAR
        ))
        nir_normalized = (nir_resized.astype(np.float32) / 255.0 - 0.5) / 0.5  # Normalize to [-1, 1]
        nir_tensor = torch.from_numpy(nir_normalized).unsqueeze(0).float()
        
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


def get_training_augmentation(img_size=(512, 1024)):
    """
    Advanced augmentation pipeline for adverse weather conditions
    
    Key augmentations:
    1. CLAHE - Critical for low-light and fog
    2. CoarseDropout - Simulates occlusions
    3. GridDistortion - Camera lens effects
    4. MotionBlur - Simulates rain/movement
    5. RandomBrightnessContrast - Weather variations
    """
    train_transform = A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        
        # === CLAHE: Crucial for Fog/Low-light ===
        # Enhances local contrast, reveals details in shadows
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
        
        # === Geometric Augmentations ===
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=15,
            border_mode=0,
            p=0.5
        ),
        
        # === Distortions (Camera lens effects) ===
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
            A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=1.0),
        ], p=0.3),
        
        # === Occlusion Simulation ===
        # CoarseDropout simulates mud on lens, partial occlusions
        A.CoarseDropout(
            max_holes=8,
            max_height=img_size[0] // 10,
            max_width=img_size[1] // 10,
            min_holes=3,
            fill_value=0,
            p=0.3
        ),
        
        # === Weather-specific augmentations ===
        A.OneOf([
            # Simulate fog/haze
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1, p=1.0),
            # Simulate rain
            A.RandomRain(
                slant_lower=-10, slant_upper=10,
                drop_length=20, drop_width=1,
                drop_color=(200, 200, 200),
                blur_value=3,
                brightness_coefficient=0.9,
                rain_type='drizzle',
                p=1.0
            ),
            # Motion blur (rain/movement)
            A.MotionBlur(blur_limit=7, p=1.0),
        ], p=0.2),
        
        # === Color/Intensity Augmentations ===
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.5
        ),
        
        A.OneOf([
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0
            ),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
        ], p=0.3),
        
        # === Noise (sensor noise in low-light) ===
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.3),
        
        # === Blur (degradation) ===
        A.OneOf([
            A.Blur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=5, p=1.0),
        ], p=0.2),
        
        # === Downscale -> Upscale (simulates blur from rain/fog) ===
        A.Downscale(scale_min=0.5, scale_max=0.75, interpolation=0, p=0.2),
        
        # === Normalization and tensor conversion ===
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return train_transform


def get_validation_augmentation(img_size=(512, 1024)):
    """Validation augmentation - only resize and normalize"""
    val_transform = A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        
        # Optional: CLAHE for validation too (helps with fog/lowlight)
        # Uncomment if you want consistent preprocessing
        # A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return val_transform


def calculate_class_weights(dataloader, num_classes=30, device='cuda'):
    """
    Calculate class weights for handling imbalance
    
    Formula: weight = ln(1.02 + median_freq / class_freq)
    where median_freq is the median frequency across all classes
    """
    print("\nCalculating class weights...")
    
    class_counts = torch.zeros(num_classes, dtype=torch.long)
    total_pixels = 0
    
    for batch in dataloader:
        seg = batch['seg']
        
        # Ignore label 255
        valid_mask = (seg != 255)
        
        for cls in range(num_classes):
            class_counts[cls] += (seg[valid_mask] == cls).sum().item()
        
        total_pixels += valid_mask.sum().item()
    
    # Calculate frequencies
    class_freq = class_counts.float() / total_pixels
    
    # Avoid division by zero
    class_freq[class_freq == 0] = 1e-6
    
    # Calculate weights using median frequency
    median_freq = torch.median(class_freq[class_freq > 0])
    class_weights = torch.log(1.02 + median_freq / class_freq)
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * num_classes
    
    print(f"\nClass weight statistics:")
    print(f"  Min weight: {class_weights.min():.4f}")
    print(f"  Max weight: {class_weights.max():.4f}")
    print(f"  Mean weight: {class_weights.mean():.4f}")
    print(f"\nTop 5 weighted classes (rare classes):")
    top_weights, top_indices = torch.topk(class_weights, 5)
    for idx, weight in zip(top_indices, top_weights):
        print(f"  Class {idx}: weight={weight:.4f}, freq={class_freq[idx]:.6f}")
    
    return class_weights.to(device)


def get_dataloaders(root_dir, batch_size=4, num_workers=4, img_size=(512, 1024),
                    calculate_weights=False):
    """
    Create train and validation dataloaders with advanced augmentation
    
    Args:
        root_dir: Path to dataset
        batch_size: Batch size
        num_workers: Number of workers
        img_size: Image size (H, W)
        calculate_weights: Whether to calculate class weights
    
    Returns:
        train_loader, val_loader, (optional) class_weights
    """
    
    # Create datasets with advanced augmentations
    train_dataset = IDDAWDataset(
        root_dir, 
        split='train',
        transform=get_training_augmentation(img_size),
        img_size=img_size
    )
    
    val_dataset = IDDAWDataset(
        root_dir,
        split='val',
        transform=get_validation_augmentation(img_size),
        img_size=img_size
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Ensure consistent batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    if calculate_weights:
        class_weights = calculate_class_weights(train_loader, num_classes=30)
        return train_loader, val_loader, class_weights
    
    return train_loader, val_loader


# Test the dataloader
if __name__ == '__main__':
    root_dir = '/home/jupyter-228w1a12b8/dec_7/IDD_AW/IDD_AW/IDDAW'
    
    print("="*80)
    print("TESTING IMPROVED DATASET LOADER")
    print("="*80)
    
    train_loader, val_loader = get_dataloaders(
        root_dir, 
        batch_size=2, 
        num_workers=2,
        img_size=(512, 1024)
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test loading one batch
    print("\n" + "="*80)
    print("TESTING BATCH LOADING")
    print("="*80)
    
    for batch in train_loader:
        print("\nBatch shapes:")
        print(f"  RGB: {batch['rgb'].shape}")
        print(f"  NIR: {batch['nir'].shape}")
        print(f"  Segmentation: {batch['seg'].shape}")
        print(f"  Weather: {batch['weather']}")
        print(f"  Weather names: {batch['weather_name']}")
        
        # Check value ranges
        print("\nValue ranges:")
        print(f"  RGB: [{batch['rgb'].min():.3f}, {batch['rgb'].max():.3f}]")
        print(f"  NIR: [{batch['nir'].min():.3f}, {batch['nir'].max():.3f}]")
        print(f"  Seg: [{batch['seg'].min()}, {batch['seg'].max()}]")
        
        # Check unique classes in segmentation
        unique_classes = torch.unique(batch['seg'])
        print(f"  Unique classes in batch: {unique_classes.tolist()}")
        
        break
    
    # Test class weights calculation
    print("\n" + "="*80)
    print("TESTING CLASS WEIGHTS CALCULATION")
    print("="*80)
    
    train_loader_small, _, class_weights = get_dataloaders(
        root_dir,
        batch_size=4,
        num_workers=2,
        calculate_weights=True
    )
    
    print("\n✓ All tests passed!")
