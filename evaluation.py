import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import json
import pandas as pd
from PIL import Image

class ModelEvaluator:
    """Comprehensive evaluation and visualization"""
    
    def __init__(self, model, val_loader, device='cuda', num_classes=30):
        self.model = model.to(device)
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes
        
        # Weather types
        self.weather_types = ['FOG', 'RAIN', 'LOWLIGHT', 'SNOW']
        
    def evaluate_per_weather(self):
        """Evaluate model performance per weather condition"""
        self.model.eval()
        
        # Initialize metrics per weather
        weather_metrics = {w: {'intersection': torch.zeros(self.num_classes).to(self.device),
                               'union': torch.zeros(self.num_classes).to(self.device),
                               'correct': 0,
                               'total': 0}
                          for w in range(4)}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Evaluating per weather'):
                rgb = batch['rgb'].to(self.device)
                nir = batch['nir'].to(self.device)
                seg = batch['seg'].to(self.device)
                weather = batch['weather'].to(self.device)
                
                output = self.model(rgb, nir, weather)
                pred = output.argmax(dim=1)
                
                # Per weather metrics
                for w_idx in range(4):
                    mask = (weather == w_idx)
                    if mask.sum() == 0:
                        continue
                    
                    pred_w = pred[mask]
                    seg_w = seg[mask]
                    
                    for cls in range(self.num_classes):
                        pred_mask = (pred_w == cls)
                        true_mask = (seg_w == cls)
                        
                        weather_metrics[w_idx]['intersection'][cls] += (pred_mask & true_mask).sum().float()
                        weather_metrics[w_idx]['union'][cls] += (pred_mask | true_mask).sum().float()
                    
                    weather_metrics[w_idx]['correct'] += (pred_w == seg_w).sum().item()
                    weather_metrics[w_idx]['total'] += seg_w.numel()
        
        # Compute mIoU per weather
        results = {}
        for w_idx in range(4):
            iou = weather_metrics[w_idx]['intersection'] / (weather_metrics[w_idx]['union'] + 1e-10)
            miou = iou[weather_metrics[w_idx]['union'] > 0].mean().item()
            accuracy = weather_metrics[w_idx]['correct'] / (weather_metrics[w_idx]['total'] + 1e-10)
            
            results[self.weather_types[w_idx]] = {
                'mIoU': miou,
                'accuracy': accuracy,
                'per_class_iou': iou.cpu().numpy()
            }
        
        return results
    
    def visualize_predictions(self, num_samples=8, save_dir='./visualizations'):
        """Visualize model predictions"""
        os.makedirs(save_dir, exist_ok=True)
        self.model.eval()
        
        # Color map for segmentation (simplified)
        colors = plt.cm.tab20(np.linspace(0, 1, self.num_classes))[:, :3] * 255
        
        sample_count = 0
        with torch.no_grad():
            for batch in self.val_loader:
                if sample_count >= num_samples:
                    break
                
                rgb = batch['rgb'].to(self.device)
                nir = batch['nir'].to(self.device)
                seg = batch['seg'].to(self.device)
                weather = batch['weather'].to(self.device)
                weather_names = batch['weather_name']
                
                output = self.model(rgb, nir, weather)
                pred = output.argmax(dim=1)
                
                # Process each image in batch
                batch_size = rgb.shape[0]
                for i in range(min(batch_size, num_samples - sample_count)):
                    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                    
                    # RGB image
                    rgb_img = rgb[i].cpu().permute(1, 2, 0).numpy()
                    rgb_img = (rgb_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
                    rgb_img = np.clip(rgb_img, 0, 1)
                    axes[0, 0].imshow(rgb_img)
                    axes[0, 0].set_title(f'RGB ({weather_names[i]})')
                    axes[0, 0].axis('off')
                    
                    # NIR image
                    nir_img = nir[i, 0].cpu().numpy()
                    axes[0, 1].imshow(nir_img, cmap='gray')
                    axes[0, 1].set_title('NIR')
                    axes[0, 1].axis('off')
                    
                    # Ground truth
                    gt = seg[i].cpu().numpy()
                    gt_colored = colors[gt.astype(int)]
                    axes[0, 2].imshow(gt_colored.astype(np.uint8))
                    axes[0, 2].set_title('Ground Truth')
                    axes[0, 2].axis('off')
                    
                    # Prediction
                    pred_img = pred[i].cpu().numpy()
                    pred_colored = colors[pred_img.astype(int)]
                    axes[1, 0].imshow(pred_colored.astype(np.uint8))
                    axes[1, 0].set_title('Prediction')
                    axes[1, 0].axis('off')
                    
                    # Overlay
                    overlay = rgb_img.copy()
                    overlay = (overlay * 0.6 + pred_colored.astype(float) / 255 * 0.4)
                    axes[1, 1].imshow(overlay)
                    axes[1, 1].set_title('Prediction Overlay')
                    axes[1, 1].axis('off')
                    
                    # Error map
                    errors = (pred_img != gt).astype(float)
                    axes[1, 2].imshow(errors, cmap='Reds', vmin=0, vmax=1)
                    axes[1, 2].set_title(f'Errors ({errors.sum() / errors.size * 100:.1f}%)')
                    axes[1, 2].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f'sample_{sample_count:03d}.png'), 
                               dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    sample_count += 1
                    if sample_count >= num_samples:
                        break
        
        print(f"Saved {sample_count} visualizations to {save_dir}")
    
    def plot_confusion_matrix(self, save_path='./confusion_matrix.png', max_classes=10):
        """Plot confusion matrix for top classes"""
        self.model.eval()
        
        confusion = torch.zeros(self.num_classes, self.num_classes).to(self.device)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Computing confusion matrix'):
                rgb = batch['rgb'].to(self.device)
                nir = batch['nir'].to(self.device)
                seg = batch['seg'].to(self.device)
                weather = batch['weather'].to(self.device)
                
                output = self.model(rgb, nir, weather)
                pred = output.argmax(dim=1)
                
                # Update confusion matrix
                for t in range(self.num_classes):
                    for p in range(self.num_classes):
                        confusion[t, p] += ((seg == t) & (pred == p)).sum()
        
        # Normalize by row (true class)
        confusion_normalized = confusion / (confusion.sum(dim=1, keepdim=True) + 1e-10)
        confusion_np = confusion_normalized.cpu().numpy()
        
        # Plot only top classes by frequency
        class_frequency = confusion.sum(dim=1).cpu().numpy()
        top_classes = np.argsort(class_frequency)[-max_classes:][::-1]
        
        confusion_subset = confusion_np[np.ix_(top_classes, top_classes)]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(confusion_subset, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=top_classes, yticklabels=top_classes)
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.title(f'Confusion Matrix (Top {max_classes} Classes)')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {save_path}")


def compare_models(checkpoint_dir='./checkpoints', save_dir='./comparisons'):
    """Compare all trained models"""
    os.makedirs(save_dir, exist_ok=True)
    
    model_types = ['baseline', 'early', 'late', 'adaptive']
    results = {}
    
    # Load results
    for model_type in model_types:
        history_path = os.path.join(checkpoint_dir, model_type, f'{model_type}_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                results[model_type] = json.load(f)
    
    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss
    for model_type, history in results.items():
        axes[0, 0].plot(history['train_loss'], label=model_type, linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation loss
    for model_type, history in results.items():
        axes[0, 1].plot(history['val_loss'], label=model_type, linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Validation mIoU
    for model_type, history in results.items():
        axes[1, 0].plot(history['val_miou'], label=model_type, linewidth=2, marker='o')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('mIoU')
    axes[1, 0].set_title('Validation mIoU')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Validation Safe mIoU
    for model_type, history in results.items():
        axes[1, 1].plot(history['val_safe_miou'], label=model_type, linewidth=2, marker='o')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Safe mIoU')
    axes[1, 1].set_title('Validation Safe mIoU')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Bar chart comparison
    best_scores = {}
    for model_type, history in results.items():
        best_scores[model_type] = {
            'mIoU': max(history['val_miou']),
            'Safe mIoU': max(history['val_safe_miou'])
        }
    
    df = pd.DataFrame(best_scores).T
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    df.plot(kind='bar', ax=ax, rot=45, width=0.7)
    ax.set_xlabel('Model Type')
    ax.set_ylabel('Score')
    ax.set_title('Best Performance Comparison')
    ax.legend(title='Metric')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'best_scores.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save comparison table
    df.to_csv(os.path.join(save_dir, 'comparison_table.csv'))
    
    print(f"\nModel comparison saved to {save_dir}")
    print("\nBest Scores:")
    print(df.to_string())


# Main evaluation script
if __name__ == '__main__':
    from dataset import get_dataloaders
    from models import get_model
    
    # Configuration
    data_root = '/home/jupyter-228w1a12b8/dec_7/IDD_AW/IDD_AW/IDDAW'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load validation data
    _, val_loader = get_dataloaders(data_root, batch_size=4, img_size=(512, 1024))
    
    # Evaluate each model
    for model_type in ['baseline', 'early', 'late', 'adaptive']:
        print(f"\n{'='*80}")
        print(f"Evaluating {model_type.upper()} model")
        print(f"{'='*80}")
        
        # Load model
        model = get_model(model_type, num_classes=30)
        checkpoint_path = f'./checkpoints/{model_type}/{model_type}_best.pth'
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Evaluate
            evaluator = ModelEvaluator(model, val_loader, device=device)
            
            # Per-weather evaluation
            weather_results = evaluator.evaluate_per_weather()
            print("\nPer-Weather Results:")
            for weather, metrics in weather_results.items():
                print(f"  {weather}: mIoU = {metrics['mIoU']:.4f}, Acc = {metrics['accuracy']:.4f}")
            
            # Save results
            with open(f'./checkpoints/{model_type}/per_weather_results.json', 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_results = {}
                for weather, metrics in weather_results.items():
                    serializable_results[weather] = {
                        'mIoU': metrics['mIoU'],
                        'accuracy': metrics['accuracy'],
                        'per_class_iou': metrics['per_class_iou'].tolist()
                    }
                json.dump(serializable_results, f, indent=4)
            
            # Visualizations
            evaluator.visualize_predictions(num_samples=8, 
                                          save_dir=f'./visualizations/{model_type}')
            evaluator.plot_confusion_matrix(
                save_path=f'./visualizations/{model_type}/confusion_matrix.png'
            )
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
    
    # Compare all models
    print(f"\n{'='*80}")
    print("Comparing all models")
    print(f"{'='*80}")
    compare_models()
