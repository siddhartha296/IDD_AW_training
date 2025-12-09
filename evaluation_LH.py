#!/usr/bin/env python3
"""
evaluation_LH.py
Evaluation and visualization for UGF-Net
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import json
from PIL import Image

class UGFEvaluator:
    """Evaluator specifically for UGF-Net with gate visualization"""
    
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
                
                # Model returns (output, rgb_gate, nir_gate) during inference
                output, rgb_gate, nir_gate = self.model(rgb, nir, weather)
                pred = output.argmax(dim=1)
                
                # Per weather metrics
                for w_idx in range(4):
                    mask = (weather == w_idx)
                    if mask.sum() == 0:
                        continue
                    
                    pred_w = pred[mask]
                    seg_w = seg[mask]
                    
                    # Ignore label 255
                    valid_mask = (seg_w != 255)
                    pred_w = pred_w[valid_mask]
                    seg_w = seg_w[valid_mask]
                    
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
    
    def visualize_uncertainty_gates(self, num_samples=12, save_dir='./visualizations/ugf'):
        """Visualize RGB/NIR uncertainty gates"""
        os.makedirs(save_dir, exist_ok=True)
        self.model.eval()
        
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
                
                # Get predictions and gates
                output, rgb_gate, nir_gate = self.model(rgb, nir, weather)
                pred = output.argmax(dim=1)
                
                # Process each image in batch
                batch_size = rgb.shape[0]
                for i in range(min(batch_size, num_samples - sample_count)):
                    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
                    
                    # RGB image
                    rgb_img = rgb[i].cpu().permute(1, 2, 0).numpy()
                    rgb_img = (rgb_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
                    rgb_img = np.clip(rgb_img, 0, 1)
                    axes[0, 0].imshow(rgb_img)
                    axes[0, 0].set_title(f'RGB ({weather_names[i]})', fontsize=12)
                    axes[0, 0].axis('off')
                    
                    # NIR image
                    nir_img = nir[i, 0].cpu().numpy()
                    axes[0, 1].imshow(nir_img, cmap='gray')
                    axes[0, 1].set_title('NIR', fontsize=12)
                    axes[0, 1].axis('off')
                    
                    # RGB Gate (higher = more trusted)
                    rgb_gate_img = rgb_gate[i, 0].cpu().numpy()
                    rgb_gate_resized = torch.nn.functional.interpolate(
                        rgb_gate[i:i+1], size=rgb.shape[2:], mode='bilinear', align_corners=False
                    )[0, 0].cpu().numpy()
                    im = axes[0, 2].imshow(rgb_gate_resized, cmap='RdYlGn', vmin=0, vmax=1)
                    axes[0, 2].set_title(f'RGB Gate (mean: {rgb_gate_img.mean():.2f})', fontsize=12)
                    axes[0, 2].axis('off')
                    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
                    
                    # NIR Gate (higher = more trusted)
                    nir_gate_img = nir_gate[i, 0].cpu().numpy()
                    nir_gate_resized = torch.nn.functional.interpolate(
                        nir_gate[i:i+1], size=rgb.shape[2:], mode='bilinear', align_corners=False
                    )[0, 0].cpu().numpy()
                    im = axes[0, 3].imshow(nir_gate_resized, cmap='RdYlGn', vmin=0, vmax=1)
                    axes[0, 3].set_title(f'NIR Gate (mean: {nir_gate_img.mean():.2f})', fontsize=12)
                    axes[0, 3].axis('off')
                    plt.colorbar(im, ax=axes[0, 3], fraction=0.046)
                    
                    # Ground truth
                    gt = seg[i].cpu().numpy()
                    colors = plt.cm.tab20(np.linspace(0, 1, self.num_classes))[:, :3] * 255
                    gt_colored = np.zeros((*gt.shape, 3), dtype=np.uint8)
                    for cls in range(self.num_classes):
                        gt_colored[gt == cls] = colors[cls]
                    gt_colored[gt == 255] = [0, 0, 0]
                    axes[1, 0].imshow(gt_colored)
                    axes[1, 0].set_title('Ground Truth', fontsize=12)
                    axes[1, 0].axis('off')
                    
                    # Prediction
                    pred_img = pred[i].cpu().numpy()
                    pred_colored = np.zeros((*pred_img.shape, 3), dtype=np.uint8)
                    for cls in range(self.num_classes):
                        pred_colored[pred_img == cls] = colors[cls]
                    axes[1, 1].imshow(pred_colored)
                    axes[1, 1].set_title('Prediction', fontsize=12)
                    axes[1, 1].axis('off')
                    
                    # RGB overlay with gate
                    overlay_rgb = rgb_img.copy()
                    overlay_rgb = (overlay_rgb * 0.5 + 
                                  plt.cm.RdYlGn(rgb_gate_resized)[:, :, :3] * 0.5)
                    axes[1, 2].imshow(overlay_rgb)
                    axes[1, 2].set_title('RGB + Gate Overlay', fontsize=12)
                    axes[1, 2].axis('off')
                    
                    # Error map
                    valid_mask = (gt != 255)
                    errors = np.zeros_like(gt, dtype=float)
                    errors[valid_mask] = (pred_img[valid_mask] != gt[valid_mask]).astype(float)
                    error_rate = errors[valid_mask].sum() / valid_mask.sum() * 100 if valid_mask.sum() > 0 else 0
                    axes[1, 3].imshow(errors, cmap='Reds', vmin=0, vmax=1)
                    axes[1, 3].set_title(f'Errors ({error_rate:.1f}%)', fontsize=12)
                    axes[1, 3].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f'uncertainty_gates_{sample_count:03d}.png'), 
                               dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    sample_count += 1
                    if sample_count >= num_samples:
                        break
        
        print(f"Saved {sample_count} uncertainty gate visualizations to {save_dir}")
    
    def visualize_gate_statistics(self, save_dir='./visualizations/ugf'):
        """Analyze and plot gate statistics per weather"""
        os.makedirs(save_dir, exist_ok=True)
        self.model.eval()
        
        # Collect gate statistics
        stats = {w: {'rgb_gates': [], 'nir_gates': []} for w in self.weather_types}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Collecting gate statistics'):
                rgb = batch['rgb'].to(self.device)
                nir = batch['nir'].to(self.device)
                weather = batch['weather'].to(self.device)
                weather_names = batch['weather_name']
                
                # Get gates
                _, rgb_gate, nir_gate = self.model(rgb, nir, weather)
                
                # Store statistics per weather
                for i in range(rgb.shape[0]):
                    w_name = weather_names[i]
                    stats[w_name]['rgb_gates'].append(rgb_gate[i].mean().item())
                    stats[w_name]['nir_gates'].append(nir_gate[i].mean().item())
        
        # Plot statistics
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Box plot
        data_rgb = [stats[w]['rgb_gates'] for w in self.weather_types]
        data_nir = [stats[w]['nir_gates'] for w in self.weather_types]
        
        positions = np.arange(len(self.weather_types))
        width = 0.35
        
        bp1 = axes[0].boxplot(data_rgb, positions=positions - width/2, widths=width, 
                              patch_artist=True, labels=self.weather_types)
        bp2 = axes[0].boxplot(data_nir, positions=positions + width/2, widths=width,
                              patch_artist=True, labels=[''] * len(self.weather_types))
        
        for patch in bp1['boxes']:
            patch.set_facecolor('skyblue')
        for patch in bp2['boxes']:
            patch.set_facecolor('lightcoral')
        
        axes[0].set_ylabel('Gate Weight', fontsize=12)
        axes[0].set_title('Gate Weights by Weather Condition', fontsize=14)
        axes[0].legend([bp1["boxes"][0], bp2["boxes"][0]], ['RGB Gate', 'NIR Gate'])
        axes[0].grid(True, alpha=0.3)
        
        # Bar plot with means
        means_rgb = [np.mean(stats[w]['rgb_gates']) for w in self.weather_types]
        means_nir = [np.mean(stats[w]['nir_gates']) for w in self.weather_types]
        
        x = np.arange(len(self.weather_types))
        axes[1].bar(x - width/2, means_rgb, width, label='RGB Gate', color='skyblue')
        axes[1].bar(x + width/2, means_nir, width, label='NIR Gate', color='lightcoral')
        
        axes[1].set_ylabel('Mean Gate Weight', fontsize=12)
        axes[1].set_title('Average Gate Weights per Weather', fontsize=14)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(self.weather_types)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'gate_statistics.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print statistics
        print("\n=== Gate Statistics ===")
        for w in self.weather_types:
            rgb_mean = np.mean(stats[w]['rgb_gates'])
            nir_mean = np.mean(stats[w]['nir_gates'])
            print(f"{w}:")
            print(f"  RGB Gate: {rgb_mean:.3f} ± {np.std(stats[w]['rgb_gates']):.3f}")
            print(f"  NIR Gate: {nir_mean:.3f} ± {np.std(stats[w]['nir_gates']):.3f}")
            print(f"  Ratio (RGB/NIR): {rgb_mean/nir_mean:.2f}")
        
        print(f"\nGate statistics plot saved to {save_dir}/gate_statistics.png")


# Main evaluation
if __name__ == '__main__':
    from dataset import get_dataloaders
    from models_LH import get_model
    
    data_root = '/home/jupyter-228w1a12b8/dec_7/IDD_AW/IDD_AW/IDDAW'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load validation data
    _, val_loader = get_dataloaders(data_root, batch_size=4, img_size=(512, 1024))
    
    # Load model
    model = get_model('ugf', num_classes=30)
    checkpoint_path = './checkpoints/ugf/ugf_best.pth'
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create evaluator
        evaluator = UGFEvaluator(model, val_loader, device=device)
        
        # Per-weather evaluation
        print("Evaluating per weather...")
        weather_results = evaluator.evaluate_per_weather()
        
        print("\nPer-Weather Results:")
        for weather, metrics in weather_results.items():
            print(f"  {weather}: mIoU = {metrics['mIoU']:.4f}, Acc = {metrics['accuracy']:.4f}")
        
        # Save results
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
        
        # Visualizations
        print("\nGenerating uncertainty gate visualizations...")
        evaluator.visualize_uncertainty_gates(num_samples=12)
        
        print("\nAnalyzing gate statistics...")
        evaluator.visualize_gate_statistics()
        
        print("\n✓ Evaluation complete!")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")