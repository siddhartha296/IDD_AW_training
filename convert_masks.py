#!/usr/bin/env python3
"""
Convert IDD-AW JSON mask annotations to PNG segmentation masks
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import cv2

# IDD-AW class mapping (from the dataset paper)
# This is a simplified version - you may need to adjust based on actual JSON structure
CLASS_MAPPING = {
    'road': 0,
    'drivable': 0,
    'non-drivable': 1,
    'parking': 1,
    'sidewalk': 2,
    'rail track': 3,
    'building': 4,
    'wall': 5,
    'fence': 6,
    'guard rail': 7,
    'bridge': 8,
    'tunnel': 9,
    'pole': 10,
    'polegroup': 11,
    'traffic light': 12,
    'traffic sign': 13,
    'vegetation': 14,
    'terrain': 15,
    'sky': 16,
    'person': 17,
    'rider': 18,
    'car': 19,
    'truck': 20,
    'bus': 21,
    'caravan': 22,
    'trailer': 23,
    'train': 24,
    'motorcycle': 25,
    'bicycle': 26,
    'autorickshaw': 27,
    'animal': 28,
    'fallback': 29,
    'void': 255,
    'unlabeled': 255,
}


def parse_json_mask(json_path, img_height=1080, img_width=1920):
    """Parse JSON annotation and create segmentation mask"""
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create empty mask
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    mask.fill(255)  # Start with unlabeled
    
    # Check if it's the IDD format with 'objects' key
    if 'objects' in data:
        objects = data['objects']
    elif 'shapes' in data:  # Alternative format
        objects = data['shapes']
    else:
        print(f"Unknown JSON format in {json_path}")
        return mask
    
    # Parse each object
    for obj in objects:
        try:
            # Get label
            label = obj.get('label', 'unlabeled').lower()
            
            # Map to class ID
            class_id = CLASS_MAPPING.get(label, 255)
            
            # Get polygon points
            if 'polygon' in obj:
                points = obj['polygon']
            elif 'points' in obj:
                points = obj['points']
            else:
                continue
            
            # Convert points to numpy array
            if isinstance(points, list) and len(points) > 0:
                if isinstance(points[0], list):
                    poly = np.array(points, dtype=np.int32)
                elif isinstance(points[0], dict):
                    poly = np.array([[p['x'], p['y']] for p in points], dtype=np.int32)
                else:
                    continue
                
                # Fill polygon
                cv2.fillPoly(mask, [poly], class_id)
        
        except Exception as e:
            print(f"Error parsing object in {json_path}: {e}")
            continue
    
    return mask


def get_image_size(rgb_path):
    """Get image dimensions from RGB image"""
    try:
        img = Image.open(rgb_path)
        return img.height, img.width
    except:
        return 1080, 1920  # Default IDD size


def convert_json_to_png(root_dir, dry_run=False):
    """Convert all JSON masks to PNG format"""
    
    print("="*80)
    print("CONVERTING JSON MASKS TO PNG")
    print("="*80)
    
    stats = {'total': 0, 'converted': 0, 'skipped': 0, 'errors': 0}
    
    for split in ['train', 'val']:
        split_path = os.path.join(root_dir, split)
        
        if not os.path.exists(split_path):
            continue
        
        print(f"\n{split.upper()} Split:")
        
        for weather in ['FOG', 'RAIN', 'LOWLIGHT', 'SNOW']:
            weather_path = os.path.join(split_path, weather)
            
            if not os.path.exists(weather_path):
                continue
            
            seg_path = os.path.join(weather_path, 'gtSeg')
            rgb_path = os.path.join(weather_path, 'rgb')
            
            if not os.path.exists(seg_path):
                continue
            
            # Process each sequence
            sequences = [d for d in os.listdir(seg_path) if os.path.isdir(os.path.join(seg_path, d))]
            
            print(f"  {weather}: {len(sequences)} sequences")
            
            for seq in tqdm(sequences, desc=f"    {weather}"):
                seq_seg = os.path.join(seg_path, seq)
                seq_rgb = os.path.join(rgb_path, seq)
                
                # Find all JSON files
                json_files = [f for f in os.listdir(seq_seg) if f.endswith('_mask.json')]
                
                for json_file in json_files:
                    json_path = os.path.join(seq_seg, json_file)
                    
                    # Output PNG name
                    png_file = json_file.replace('_mask.json', '_gt.png')
                    png_path = os.path.join(seq_seg, png_file)
                    
                    stats['total'] += 1
                    
                    # Skip if already exists
                    if os.path.exists(png_path):
                        stats['skipped'] += 1
                        continue
                    
                    if dry_run:
                        print(f"Would convert: {json_path} -> {png_path}")
                        continue
                    
                    try:
                        # Get image size from corresponding RGB
                        rgb_file = json_file.replace('_mask.json', '_rgb.png')
                        rgb_full_path = os.path.join(seq_rgb, rgb_file)
                        
                        if os.path.exists(rgb_full_path):
                            img_h, img_w = get_image_size(rgb_full_path)
                        else:
                            img_h, img_w = 1080, 1920
                        
                        # Convert JSON to mask
                        mask = parse_json_mask(json_path, img_h, img_w)
                        
                        # Save as PNG
                        Image.fromarray(mask).save(png_path)
                        
                        stats['converted'] += 1
                    
                    except Exception as e:
                        print(f"      ✗ Error converting {json_file}: {e}")
                        stats['errors'] += 1
    
    print("\n" + "="*80)
    print("CONVERSION SUMMARY")
    print("="*80)
    print(f"Total JSON files: {stats['total']}")
    print(f"Converted: {stats['converted']}")
    print(f"Skipped (already exist): {stats['skipped']}")
    print(f"Errors: {stats['errors']}")
    
    return stats


def verify_conversion(root_dir, num_samples=3):
    """Verify that conversion worked correctly"""
    
    print("\n" + "="*80)
    print("VERIFYING CONVERSION")
    print("="*80)
    
    split_path = os.path.join(root_dir, 'train', 'FOG')
    seg_path = os.path.join(split_path, 'gtSeg')
    
    sequences = [d for d in os.listdir(seg_path) if os.path.isdir(os.path.join(seg_path, d))]
    
    if sequences:
        seq = sequences[0]
        seq_seg = os.path.join(seg_path, seq)
        
        png_files = [f for f in os.listdir(seq_seg) if f.endswith('_gt.png')]
        
        print(f"\nChecking sequence '{seq}':")
        print(f"  Found {len(png_files)} PNG masks")
        
        if png_files:
            # Check first mask
            mask_path = os.path.join(seq_seg, png_files[0])
            mask = np.array(Image.open(mask_path))
            
            unique_classes = np.unique(mask)
            
            print(f"\n  Sample mask: {png_files[0]}")
            print(f"  Shape: {mask.shape}")
            print(f"  Unique classes: {unique_classes}")
            print(f"  Class distribution:")
            for cls in unique_classes:
                count = (mask == cls).sum()
                percent = count / mask.size * 100
                print(f"    Class {cls}: {count} pixels ({percent:.2f}%)")
            
            return True
    
    return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert IDD-AW JSON masks to PNG')
    parser.add_argument('--root_dir', type=str, 
                       default='/home/jupyter-228w1a12b8/dec_7/IDD_AW/IDD_AW/IDDAW',
                       help='Root directory of IDD-AW dataset')
    parser.add_argument('--dry_run', action='store_true',
                       help='Just show what would be converted without actually converting')
    
    args = parser.parse_args()
    
    # Convert
    stats = convert_json_to_png(args.root_dir, dry_run=args.dry_run)
    
    # Verify
    if not args.dry_run and stats['converted'] > 0:
        verify_conversion(args.root_dir)
    
    print("\n✓ Done! Now you can run the training script.")
