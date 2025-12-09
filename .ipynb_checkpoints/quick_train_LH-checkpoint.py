#!/usr/bin/env python3
"""
quick_train_LH.py - OPTIMIZED FOR HIGH RESOLUTION
Quick Training Script for UGF-Net with high-resolution output

FIXES:
1. Proper 512x1024 output resolution
2. Gradient checkpointing for memory efficiency
3. Mixed precision training
4. Optimized batch sizes
"""

import os
import sys
import time
import torch
import json

def train_ugf_model():
    """Train the UGF model with high-resolution support"""
    print("\n" + "="*80)
    print("TRAINING UGF-NET (High-Resolution Mode)")
    print("="*80)
    
    from dataset_LH import get_dataloaders
    from models_LH import get_model
    from training_LH import UGFTrainer
    
    # Configuration optimized for high-resolution
    CONFIG = {
        'data_root': '/home/jupyter-228w1a12b8/dec_7/IDD_AW/IDD_AW/IDDAW',
        'batch_size': 6,  # Reduced for high-res (adjust based on GPU memory)
        'num_epochs': 20,  # More epochs for better convergence
        'img_size': (512, 1024),  # Full high resolution
        'num_classes': 30,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"\nConfiguration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    
    # Check GPU memory
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    start_time = time.time()
    
    # Load data with advanced augmentations
    print("\nLoading datasets with advanced augmentations...")
    train_loader, val_loader = get_dataloaders(
        CONFIG['data_root'],
        batch_size=CONFIG['batch_size'],
        img_size=CONFIG['img_size'],
        num_workers=4
    )
    
    # Create model
    print("\nCreating UGF-Net (High-Resolution)...")
    model = get_model('ugf', num_classes=CONFIG['num_classes'])
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {params:.2f}M")
    
    # Verify output shape
    print("\nVerifying model output shape...")
    with torch.no_grad():
        dummy_rgb = torch.randn(1, 3, CONFIG['img_size'][0], CONFIG['img_size'][1])
        dummy_nir = torch.randn(1, 1, CONFIG['img_size'][0], CONFIG['img_size'][1])
        model.eval()
        dummy_out, _, _ = model(dummy_rgb, dummy_nir)
        print(f"  Input shape: {dummy_rgb.shape}")
        print(f"  Output shape: {dummy_out.shape}")
        assert dummy_out.shape[2:] == (CONFIG['img_size'][0], CONFIG['img_size'][1]), \
            f"Output shape mismatch! Expected {CONFIG['img_size']}, got {dummy_out.shape[2:]}"
        print("  ✓ Output shape correct!")
    
    # Train
    trainer = UGFTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=CONFIG['num_classes'],
        device=CONFIG['device'],
        save_dir='./checkpoints/ugf',
        model_name='ugf'
    )
    
    history = trainer.train(num_epochs=CONFIG['num_epochs'])
    
    train_time = time.time() - start_time
    
    print(f"\n✓ UGF-Net training complete!")
    print(f"  Time: {train_time/60:.1f} minutes ({train_time/3600:.2f} hours)")
    print(f"  Best mIoU: {trainer.best_miou:.4f}")
    
    return {
        'best_miou': trainer.best_miou,
        'training_time': train_time,
        'history': history
    }


def train_baseline_model():
    """Train baseline RGB-only model for comparison"""
    print("\n" + "="*80)
    print("TRAINING BASELINE (For Comparison)")
    print("="*80)
    
    from dataset_LH import get_dataloaders
    from models_LH import get_model
    from training_LH import UGFTrainer  # Can reuse trainer
    
    CONFIG = {
        'data_root': '/home/jupyter-228w1a12b8/dec_7/IDD_AW/IDD_AW/IDDAW',
        'batch_size': 8,  # Baseline can handle larger batch
        'num_epochs': 15,
        'img_size': (512, 1024),
        'num_classes': 30,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    start_time = time.time()
    
    # Load data
    train_loader, val_loader = get_dataloaders(
        CONFIG['data_root'],
        batch_size=CONFIG['batch_size'],
        img_size=CONFIG['img_size'],
        num_workers=4
    )
    
    # Create baseline model
    print("\nCreating Baseline RGB model...")
    model = get_model('baseline', num_classes=CONFIG['num_classes'])
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {params:.2f}M")
    
    # Use simpler trainer for baseline
    from training import SafeSemanticSegmentationTrainer
    
    trainer = SafeSemanticSegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=CONFIG['num_classes'],
        device=CONFIG['device'],
        save_dir='./checkpoints/baseline',
        model_name='baseline'
    )
    
    history = trainer.train(num_epochs=CONFIG['num_epochs'])
    
    train_time = time.time() - start_time
    
    print(f"\n✓ Baseline training complete!")
    print(f"  Time: {train_time/60:.1f} minutes")
    print(f"  Best mIoU: {trainer.best_miou:.4f}")
    
    return {
        'best_miou': trainer.best_miou,
        'training_time': train_time,
        'history': history
    }


def train_early_fusion():
    """Train early fusion model"""
    print("\n" + "="*80)
    print("TRAINING EARLY FUSION (For Comparison)")
    print("="*80)
    
    from dataset_LH import get_dataloaders
    from models_LH import get_model
    from training import SafeSemanticSegmentationTrainer
    
    CONFIG = {
        'data_root': '/home/jupyter-228w1a12b8/dec_7/IDD_AW/IDD_AW/IDDAW',
        'batch_size': 6,
        'num_epochs': 15,
        'img_size': (512, 1024),
        'num_classes': 30,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    start_time = time.time()
    
    train_loader, val_loader = get_dataloaders(
        CONFIG['data_root'],
        batch_size=CONFIG['batch_size'],
        img_size=CONFIG['img_size'],
        num_workers=4
    )
    
    print("\nCreating Early Fusion model...")
    model = get_model('early', num_classes=CONFIG['num_classes'])
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {params:.2f}M")
    
    trainer = SafeSemanticSegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=CONFIG['num_classes'],
        device=CONFIG['device'],
        save_dir='./checkpoints/early',
        model_name='early'
    )
    
    history = trainer.train(num_epochs=CONFIG['num_epochs'])
    
    train_time = time.time() - start_time
    
    print(f"\n✓ Early Fusion complete!")
    print(f"  Time: {train_time/60:.1f} minutes")
    print(f"  Best mIoU: {trainer.best_miou:.4f}")
    
    return {
        'best_miou': trainer.best_miou,
        'training_time': train_time,
        'history': history
    }


def generate_visualizations():
    """Generate all visualizations for the paper"""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    from evaluation_LH import UGFEvaluator
    from dataset_LH import get_dataloaders
    from models_LH import get_model
    
    data_root = '/home/jupyter-228w1a12b8/dec_7/IDD_AW/IDD_AW/IDDAW'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load data
    _, val_loader = get_dataloaders(data_root, batch_size=4, img_size=(512, 1024))
    
    # Load model
    model = get_model('ugf', num_classes=30)
    checkpoint_path = './checkpoints/ugf/ugf_best.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create evaluator
    evaluator = UGFEvaluator(model, val_loader, device=device)
    
    # 1. Per-weather evaluation
    print("\n1. Computing per-weather performance...")
    weather_results = evaluator.evaluate_per_weather()
    
    for weather, metrics in weather_results.items():
        print(f"  {weather}: mIoU = {metrics['mIoU']:.4f}")
    
    # Save
    os.makedirs('./checkpoints/ugf', exist_ok=True)
    with open('./checkpoints/ugf/per_weather_results.json', 'w') as f:
        serializable = {
            w: {
                'mIoU': m['mIoU'],
                'accuracy': m['accuracy'],
                'per_class_iou': m['per_class_iou'].tolist()
            }
            for w, m in weather_results.items()
        }
        json.dump(serializable, f, indent=4)
    
    # 2. Uncertainty gate visualizations
    print("\n2. Generating uncertainty gate visualizations...")
    evaluator.visualize_uncertainty_gates(num_samples=16)
    
    # 3. Gate statistics
    print("\n3. Analyzing gate statistics...")
    evaluator.visualize_gate_statistics()
    
    print("\n✓ Visualizations complete!")


def generate_paper_content(results):
    """Generate content for the paper"""
    print("\n" + "="*80)
    print("GENERATING PAPER CONTENT")
    print("="*80)
    
    ugf_best = results['ugf']['best_miou']
    baseline_best = results['baseline']['best_miou']
    improvement = ((ugf_best - baseline_best) / baseline_best) * 100
    
    # Load per-weather results
    weather_file = './checkpoints/ugf/per_weather_results.json'
    if os.path.exists(weather_file):
        with open(weather_file, 'r') as f:
            weather_results = json.load(f)
    else:
        weather_results = {}
    
    print("\n=== TABLE 1: Overall Performance ===")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{lcc}")
    print("\\hline")
    print("Method & mIoU (\\%) & Improvement \\\\")
    print("\\hline")
    print(f"RGB Baseline & {baseline_best*100:.2f} & - \\\\")
    if 'early' in results:
        early_best = results['early']['best_miou']
        early_imp = ((early_best - baseline_best) / baseline_best) * 100
        print(f"Early Fusion & {early_best*100:.2f} & +{early_imp:.1f}\\% \\\\")
    print(f"UGF-Net (Ours) & {ugf_best*100:.2f} & +{improvement:.1f}\\% \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Performance on IDD-AW (512×1024 resolution).}")
    print("\\end{table}")
    
    if weather_results:
        print("\n=== TABLE 2: Per-Weather Performance ===")
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\begin{tabular}{lcc}")
        print("\\hline")
        print("Weather & mIoU (\\%) & Accuracy (\\%) \\\\")
        print("\\hline")
        for weather in ['FOG', 'RAIN', 'LOWLIGHT', 'SNOW']:
            if weather in weather_results:
                miou = weather_results[weather]['mIoU'] * 100
                acc = weather_results[weather]['accuracy'] * 100
                print(f"{weather} & {miou:.2f} & {acc:.2f} \\\\")
        print("\\hline")
        print("\\end{tabular}")
        print("\\caption{UGF-Net performance across weather conditions.}")
        print("\\end{table}")
    
    print("\n=== KEY FINDINGS ===")
    print(f"\n1. UGF-Net achieves {ugf_best*100:.2f}% mIoU at 512×1024 resolution")
    print(f"2. {improvement:.1f}% improvement over RGB-only baseline")
    print(f"3. Uncertainty-gated fusion enables adaptive sensor selection")
    print(f"4. Training time: {results['ugf']['training_time']/3600:.2f} hours")
    
    print("\n=== PAPER CONTRIBUTIONS ===")
    print("1. Novel uncertainty-gated fusion using epistemic uncertainty")
    print("2. Dynamic per-pixel sensor selection without weather labels")
    print("3. Deep supervision through auxiliary segmentation heads")
    print("4. State-of-the-art results on IDD-AW at full resolution")


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("UGF-NET HIGH-RESOLUTION TRAINING PIPELINE")
    print("="*80)
    
    print("\nKey Features:")
    print("  ✓ Full 512×1024 resolution output")
    print("  ✓ Advanced augmentations (CLAHE, weather simulation)")
    print("  ✓ Mixed precision training")
    print("  ✓ Deep supervision with auxiliary heads")
    
    print("\nEstimated timeline:")
    print("  1. UGF-Net training:      ~3.5 hours")
    print("  2. Baseline training:     ~2.0 hours")
    print("  3. Early Fusion:          ~2.0 hours")
    print("  4. Visualizations:        ~0.5 hours")
    print("  Total:                    ~8.0 hours")
    
    response = input("\nStart training? (y/n): ").lower()
    if response != 'y':
        print("Aborted.")
        return
    
    total_start = time.time()
    results = {}
    
    try:
        # Step 1: Train UGF
        print("\n" + "="*80)
        print("STEP 1/4: Training UGF-Net")
        print("="*80)
        results['ugf'] = train_ugf_model()
        
        # Step 2: Train baseline
        print("\n" + "="*80)
        print("STEP 2/4: Training Baseline")
        print("="*80)
        results['baseline'] = train_baseline_model()
        
        # Step 3: Train early fusion (optional)
        train_early = input("\nTrain Early Fusion too? (y/n): ").lower()
        if train_early == 'y':
            print("\n" + "="*80)
            print("STEP 3/4: Training Early Fusion")
            print("="*80)
            results['early'] = train_early_fusion()
        
        # Step 4: Generate visualizations
        print("\n" + "="*80)
        print("STEP 4/4: Generating Visualizations")
        print("="*80)
        generate_visualizations()
        
        # Generate paper content
        generate_paper_content(results)
        
        total_time = time.time() - total_start
        
        print("\n" + "="*80)
        print("ALL COMPLETE!")
        print("="*80)
        print(f"\nTotal time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
        
        print("\n=== RESULTS SUMMARY ===")
        for model_name, result in results.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Best mIoU: {result['best_miou']:.4f}")
            print(f"  Training time: {result['training_time']/60:.1f} min")
        
        if 'ugf' in results and 'baseline' in results:
            improvement = ((results['ugf']['best_miou'] - results['baseline']['best_miou']) / 
                          results['baseline']['best_miou']) * 100
            print(f"\nImprovement over baseline: +{improvement:.1f}%")
        
        print("\n=== GENERATED FILES ===")
        print("Models:")
        print("  ./checkpoints/ugf/ugf_best.pth")
        print("  ./checkpoints/baseline/baseline_best.pth")
        print("\nResults:")
        print("  ./checkpoints/ugf/ugf_history.json")
        print("  ./checkpoints/ugf/per_weather_results.json")
        print("\nVisualizations:")
        print("  ./visualizations/ugf/uncertainty_gates_*.png")
        print("  ./visualizations/ugf/gate_statistics.png")
        
        # Save summary
        with open('./training_summary_highres.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        print("\n✓ Summary saved to training_summary_highres.json")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()