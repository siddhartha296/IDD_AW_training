#!/usr/bin/env python3
"""
Debug script to check IDD-AW dataset structure
"""

import os
from collections import defaultdict

def check_dataset_structure(root_dir):
    """Analyze the actual dataset structure"""
    
    print("="*80)
    print("ANALYZING DATASET STRUCTURE")
    print("="*80)
    
    for split in ['train', 'val']:
        print(f"\n{'='*80}")
        print(f"{split.upper()} SPLIT")
        print(f"{'='*80}")
        
        split_path = os.path.join(root_dir, split)
        if not os.path.exists(split_path):
            print(f"  ✗ {split_path} does not exist!")
            continue
        
        for weather in ['FOG', 'RAIN', 'LOWLIGHT', 'SNOW']:
            weather_path = os.path.join(split_path, weather)
            
            if not os.path.exists(weather_path):
                print(f"\n  ✗ {weather} not found")
                continue
            
            print(f"\n  {weather}:")
            
            rgb_path = os.path.join(weather_path, 'rgb')
            nir_path = os.path.join(weather_path, 'nir')
            seg_path = os.path.join(weather_path, 'gtSeg')
            
            # Check folders exist
            for folder_name, folder_path in [('RGB', rgb_path), ('NIR', nir_path), ('GTSeg', seg_path)]:
                exists = os.path.exists(folder_path)
                print(f"    {folder_name}: {'✓' if exists else '✗'}")
            
            if not os.path.exists(rgb_path):
                continue
            
            # Check sequences
            sequences = [d for d in os.listdir(rgb_path) if os.path.isdir(os.path.join(rgb_path, d))]
            print(f"    Sequences: {len(sequences)}")
            
            if sequences:
                # Check first sequence in detail
                first_seq = sequences[0]
                print(f"\n    Checking sequence '{first_seq}':")
                
                seq_rgb = os.path.join(rgb_path, first_seq)
                seq_nir = os.path.join(nir_path, first_seq)
                seq_seg = os.path.join(seg_path, first_seq)
                
                # List files
                rgb_files = sorted([f for f in os.listdir(seq_rgb) if f.endswith('.png')])[:3]
                
                print(f"      RGB files (first 3): {rgb_files}")
                
                if os.path.exists(seq_nir):
                    nir_files = sorted([f for f in os.listdir(seq_nir) if f.endswith('.png')])[:3]
                    print(f"      NIR files (first 3): {nir_files}")
                else:
                    print(f"      NIR folder: ✗ Not found")
                
                if os.path.exists(seq_seg):
                    all_seg_files = os.listdir(seq_seg)
                    seg_files = sorted([f for f in all_seg_files if f.endswith('.png')])[:3]
                    print(f"      GT files (first 3): {seg_files}")
                    print(f"      GT folder contents (all files): {all_seg_files[:5]}")  # Show first 5 files
                else:
                    print(f"      GT folder: ✗ Not found")
                
                # Sample full paths
                if rgb_files:
                    print(f"\n      Sample RGB path:")
                    print(f"        {os.path.join(seq_rgb, rgb_files[0])}")
                
                # Count total images
                total_rgb = len([f for f in os.listdir(seq_rgb) if f.endswith('.png')])
                print(f"\n      Total images in sequence '{first_seq}': {total_rgb}")
                
                # Check for matching triplets
                print(f"\n      Checking file matching...")
                matches = 0
                for rgb_file in rgb_files:
                    base_name = rgb_file.replace('_rgb.png', '').replace('.png', '')
                    
                    # Try different naming patterns
                    nir_candidates = [
                        rgb_file.replace('_rgb.png', '_nir.png'),
                        rgb_file.replace('.png', '_nir.png'),
                        f"{base_name}_nir.png",
                        rgb_file
                    ]
                    
                    seg_candidates = [
                        rgb_file.replace('_rgb.png', '_gt.png'),
                        rgb_file.replace('.png', '_gt.png'),
                        f"{base_name}_gt.png",
                        rgb_file
                    ]
                    
                    nir_found = None
                    seg_found = None
                    
                    if os.path.exists(seq_nir):
                        for nir_name in nir_candidates:
                            if os.path.exists(os.path.join(seq_nir, nir_name)):
                                nir_found = nir_name
                                break
                    
                    if os.path.exists(seq_seg):
                        for seg_name in seg_candidates:
                            if os.path.exists(os.path.join(seq_seg, seg_name)):
                                seg_found = seg_name
                                break
                    
                    if nir_found and seg_found:
                        matches += 1
                        print(f"        ✓ {rgb_file} -> NIR: {nir_found}, GT: {seg_found}")
                    else:
                        print(f"        ✗ {rgb_file} -> NIR: {nir_found or 'NOT FOUND'}, GT: {seg_found or 'NOT FOUND'}")
                
                print(f"\n      Matched triplets (RGB+NIR+GT): {matches}/{len(rgb_files)} (sample)")


def suggest_fix(root_dir):
    """Suggest how to fix the dataset loader"""
    
    print("\n" + "="*80)
    print("SUGGESTED FIX")
    print("="*80)
    
    # Check one example in detail
    example_path = os.path.join(root_dir, 'train', 'FOG', 'rgb')
    
    if not os.path.exists(example_path):
        print("Cannot find training data!")
        return
    
    sequences = [d for d in os.listdir(example_path) if os.path.isdir(os.path.join(example_path, d))]
    
    if not sequences:
        print("No sequences found!")
        return
    
    seq = sequences[0]
    seq_rgb = os.path.join(example_path, seq)
    rgb_files = [f for f in os.listdir(seq_rgb) if f.endswith('.png')]
    
    if rgb_files:
        sample_file = rgb_files[0]
        print(f"\nSample RGB filename: {sample_file}")
        
        # Detect naming pattern
        if '_rgb.png' in sample_file:
            print("✓ Detected pattern: *_rgb.png")
            print("  Expected NIR: *_nir.png")
            print("  Expected GT: *_gt.png")
        elif '_nir.png' not in sample_file:
            print("✓ Detected pattern: *.png (no suffix)")
            print("  This means RGB, NIR, and GT have the same filename in different folders")
        else:
            print("? Unknown pattern")
            print(f"  Please check the naming convention")


if __name__ == '__main__':
    root_dir = '/home/jupyter-228w1a12b8/dec_7/IDD_AW/IDD_AW/IDDAW'
    
    check_dataset_structure(root_dir)
    suggest_fix(root_dir)
    
    print("\n" + "="*80)
    print("Next steps:")
    print("  1. Check the output above")
    print("  2. Verify file naming patterns")
    print("  3. Run this script output to me so I can fix the dataset.py")
    print("="*80)
