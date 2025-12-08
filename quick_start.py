#!/usr/bin/env python3
"""
Quick Start Script for IDD-AW RGB-NIR Fusion Project
Run this to train all models and generate results in 5 hours
"""

import os
import sys
import time
import torch
import subprocess

def check_environment():
    """Check if environment is ready"""
    print("="*80)
    print("CHECKING ENVIRONMENT")
    print("="*80)
    
    # Check CUDA
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Check required packages
    required_packages = [
        'torch', 'torchvision', 'segmentation_models_pytorch',
        'albumentations', 'opencv-python', 'tqdm', 'pandas', 
        'matplotlib', 'seaborn', 'PIL'
    ]
    
    print("\nChecking packages...")
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_').split('[')[0])
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (MISSING)")
            missing.append(package)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    return True


def install_dependencies():
    """Install required packages"""
    print("\n" + "="*80)
    print("INSTALLING DEPENDENCIES")
    print("="*80)
    
    packages = [
        'torch', 'torchvision', 
        'segmentation-models-pytorch',
        'albumentations',
        'opencv-python',
        'tqdm',
        'pandas',
        'matplotlib',
        'seaborn'
    ]
    
    print("\nInstalling packages...")
    for package in packages:
        print(f"  Installing {package}...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', package, '-q'])
    
    print("\n✓ All dependencies installed!")


def run_quick_experiment():
    """Run a quick 2-epoch experiment to verify everything works"""
    print("\n" + "="*80)
    print("QUICK VERIFICATION EXPERIMENT (2 epochs)")
    print("="*80)
    
    from dataset import get_dataloaders
    from models import get_model
    from training import SafeSemanticSegmentationTrainer
    
    # Configuration
    data_root = '/home/jupyter-228w1a12b8/dec_7/IDD_AW/IDD_AW/IDDAW'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load small subset
    print("\nLoading data (small batch for testing)...")
    train_loader, val_loader = get_dataloaders(
        data_root, 
        batch_size=2,
        num_workers=2,
        img_size=(256, 512)  # Smaller for speed
    )
    
    # Test one model
    print("\nTesting adaptive fusion model...")
    model = get_model('adaptive', num_classes=30)
    
    trainer = SafeSemanticSegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=30,
        device=device,
        save_dir='./test_checkpoint',
        model_name='adaptive_test'
    )
    
    # Train for 2 epochs
    history = trainer.train(num_epochs=2, early_stopping_patience=10)
    
    print("\n✓ Verification successful! Ready for full training.")
    return True


def run_full_training():
    """Run full training on all models"""
    print("\n" + "="*80)
    print("STARTING FULL TRAINING")
    print("="*80)
    
    start_time = time.time()
    
    from dataset import get_dataloaders
    from models import get_model
    from training import SafeSemanticSegmentationTrainer
    
    # Configuration
    CONFIG = {
        'data_root': '/home/jupyter-228w1a12b8/dec_7/IDD_AW/IDD_AW/IDDAW',
        'batch_size': 8,  # Adjust based on GPU memory
        'num_epochs': 15,
        'img_size': (512, 1024),
        'num_classes': 30,
        'models_to_train': ['baseline', 'early', 'late', 'adaptive'],
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # Load data
    print("\nLoading datasets...")
    train_loader, val_loader = get_dataloaders(
        CONFIG['data_root'],
        batch_size=CONFIG['batch_size'],
        img_size=CONFIG['img_size'],
        num_workers=4
    )
    
    results = {}
    
    # Train each model
    for i, model_type in enumerate(CONFIG['models_to_train'], 1):
        print(f"\n{'='*80}")
        print(f"MODEL {i}/{len(CONFIG['models_to_train'])}: {model_type.upper()}")
        print(f"{'='*80}")
        
        model_start = time.time()
        
        # Create model
        model = get_model(model_type, num_classes=CONFIG['num_classes'])
        
        # Create trainer
        trainer = SafeSemanticSegmentationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=CONFIG['num_classes'],
            device=CONFIG['device'],
            save_dir=f'./checkpoints/{model_type}',
            model_name=model_type
        )
        
        # Train
        history = trainer.train(num_epochs=CONFIG['num_epochs'])
        
        model_time = time.time() - model_start
        
        results[model_type] = {
            'best_miou': trainer.best_miou,
            'best_safe_miou': trainer.best_safe_miou,
            'training_time': model_time,
            'history': history
        }
        
        print(f"\n✓ {model_type} completed in {model_time/60:.1f} minutes")
    
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nTotal time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print("\nResults:")
    for model_type, result in results.items():
        print(f"\n{model_type.upper()}:")
        print(f"  Best mIoU: {result['best_miou']:.4f}")
        print(f"  Best Safe mIoU: {result['best_safe_miou']:.4f}")
        print(f"  Training time: {result['training_time']/60:.1f} minutes")
    
    # Save summary
    import json
    with open('./results_summary.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    return results


def run_evaluation():
    """Run evaluation and generate visualizations"""
    print("\n" + "="*80)
    print("EVALUATION AND VISUALIZATION")
    print("="*80)
    
    from evaluation import compare_models, ModelEvaluator
    from dataset import get_dataloaders
    from models import get_model
    
    data_root = '/home/jupyter-228w1a12b8/dec_7/IDD_AW/IDD_AW/IDDAW'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load validation data
    _, val_loader = get_dataloaders(data_root, batch_size=4, img_size=(512, 1024))
    
    # Evaluate each model
    for model_type in ['baseline', 'early', 'late', 'adaptive']:
        print(f"\nEvaluating {model_type}...")
        
        model = get_model(model_type, num_classes=30)
        checkpoint_path = f'./checkpoints/{model_type}/{model_type}_best.pth'
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            evaluator = ModelEvaluator(model, val_loader, device=device)
            
            # Per-weather evaluation
            weather_results = evaluator.evaluate_per_weather()
            
            # Visualizations
            evaluator.visualize_predictions(
                num_samples=6, 
                save_dir=f'./visualizations/{model_type}'
            )
            evaluator.plot_confusion_matrix(
                save_path=f'./visualizations/{model_type}/confusion_matrix.png'
            )
    
    # Compare all models
    compare_models()
    
    print("\n✓ Evaluation complete! Check ./visualizations/ and ./comparisons/")


def generate_paper_tables():
    """Generate LaTeX tables for paper"""
    print("\n" + "="*80)
    print("GENERATING PAPER TABLES")
    print("="*80)
    
    import json
    
    # Load results
    with open('./results_summary.json', 'r') as f:
        results = json.load(f)
    
    # Main results table
    print("\n=== Table 1: Overall Performance ===")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{lcc}")
    print("\\hline")
    print("Method & mIoU & Safe mIoU \\\\")
    print("\\hline")
    
    for model_type in ['baseline', 'early', 'late', 'adaptive']:
        if model_type in results:
            miou = results[model_type]['best_miou']
            safe_miou = results[model_type]['best_safe_miou']
            print(f"{model_type.capitalize()} & {miou:.2f} & {safe_miou:.2f} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Performance comparison on IDD-AW validation set}")
    print("\\end{table}")
    
    # Per-weather table
    print("\n=== Table 2: Per-Weather Performance (Adaptive Model) ===")
    weather_file = './checkpoints/adaptive/per_weather_results.json'
    if os.path.exists(weather_file):
        with open(weather_file, 'r') as f:
            weather_results = json.load(f)
        
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\begin{tabular}{lcc}")
        print("\\hline")
        print("Weather & mIoU & Accuracy \\\\")
        print("\\hline")
        
        for weather in ['FOG', 'RAIN', 'LOWLIGHT', 'SNOW']:
            if weather in weather_results:
                miou = weather_results[weather]['mIoU']
                acc = weather_results[weather]['accuracy']
                print(f"{weather} & {miou:.2f} & {acc:.2f} \\\\")
        
        print("\\hline")
        print("\\end{tabular}")
        print("\\caption{Adaptive fusion performance across weather conditions}")
        print("\\end{table}")
    
    print("\n✓ Tables generated! Copy to your LaTeX paper.")


def main():
    """Main execution flow"""
    print("\n" + "="*80)
    print("IDD-AW RGB-NIR FUSION PROJECT - QUICK START")
    print("="*80)
    print("\nThis will:")
    print("  1. Check environment")
    print("  2. Train 4 models (baseline, early, late, adaptive)")
    print("  3. Evaluate and visualize results")
    print("  4. Generate paper tables")
    print("\nEstimated time: 3-4 hours on 2x V100 GPUs")
    
    # Get user confirmation
    response = input("\nProceed? (y/n): ").lower()
    if response != 'y':
        print("Aborted.")
        return
    
    # Step 1: Check environment
    if not check_environment():
        install = input("\nInstall missing packages? (y/n): ").lower()
        if install == 'y':
            install_dependencies()
        else:
            print("Please install missing packages and try again.")
            return
    
    # Step 2: Quick verification (optional)
    verify = input("\nRun quick verification? (recommended, ~5 min) (y/n): ").lower()
    if verify == 'y':
        try:
            run_quick_experiment()
        except Exception as e:
            print(f"\n✗ Verification failed: {e}")
            print("Fix the error and try again.")
            return
    
    # Step 3: Full training
    try:
        results = run_full_training()
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Evaluation
    try:
        run_evaluation()
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 5: Generate paper content
    try:
        generate_paper_tables()
    except Exception as e:
        print(f"\n✗ Table generation failed: {e}")
    
    print("\n" + "="*80)
    print("ALL DONE!")
    print("="*80)
    print("\nGenerated files:")
    print("  ./checkpoints/          - Model weights")
    print("  ./visualizations/       - Prediction visualizations")
    print("  ./comparisons/          - Model comparison plots")
    print("  ./results_summary.json  - Numerical results")
    print("\nNext steps:")
    print("  1. Review visualizations in ./visualizations/")
    print("  2. Check comparison plots in ./comparisons/")
    print("  3. Use tables printed above in your paper")
    print("  4. Write discussion based on results")


if __name__ == '__main__':
    main()
