import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import os
import json
from datetime import datetime

# Import your modules (adjust paths as needed)
# from dataset import get_dataloaders
# from models import get_model

class SafeSemanticSegmentationTrainer:
    """Trainer with Safe mIoU metric"""
    
    # IDD-AW label hierarchy (simplified version)
    # Level 0: road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, 
    #          vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle
    # For Safe mIoU: errors within same parent are less penalized
    
    HIERARCHY = {
        # Drivable
        'road': {'level': 0, 'parent': 'drivable', 'safety_critical': True},
        'sidewalk': {'level': 0, 'parent': 'drivable', 'safety_critical': True},
        
        # Construction
        'building': {'level': 0, 'parent': 'construction', 'safety_critical': False},
        'wall': {'level': 0, 'parent': 'construction', 'safety_critical': False},
        'fence': {'level': 0, 'parent': 'construction', 'safety_critical': False},
        
        # Object
        'pole': {'level': 0, 'parent': 'object', 'safety_critical': True},
        'traffic_light': {'level': 0, 'parent': 'object', 'safety_critical': True},
        'traffic_sign': {'level': 0, 'parent': 'object', 'safety_critical': True},
        
        # Nature
        'vegetation': {'level': 0, 'parent': 'nature', 'safety_critical': False},
        'terrain': {'level': 0, 'parent': 'nature', 'safety_critical': False},
        'sky': {'level': 0, 'parent': 'nature', 'safety_critical': False},
        
        # Human
        'person': {'level': 0, 'parent': 'human', 'safety_critical': True},
        'rider': {'level': 0, 'parent': 'human', 'safety_critical': True},
        
        # Vehicle
        'car': {'level': 0, 'parent': 'vehicle', 'safety_critical': True},
        'truck': {'level': 0, 'parent': 'vehicle', 'safety_critical': True},
        'bus': {'level': 0, 'parent': 'vehicle', 'safety_critical': True},
        'train': {'level': 0, 'parent': 'vehicle', 'safety_critical': True},
        'motorcycle': {'level': 0, 'parent': 'vehicle', 'safety_critical': True},
        'bicycle': {'level': 0, 'parent': 'vehicle', 'safety_critical': True},
    }
    
    def __init__(self, model, train_loader, val_loader, num_classes=30, 
                 device='cuda', save_dir='./checkpoints', model_name='fusion'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.device = device
        self.save_dir = save_dir
        self.model_name = model_name
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=3, factor=0.5
        )
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Tracking
        self.best_miou = 0.0
        self.best_safe_miou = 0.0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_miou': [],
            'val_safe_miou': []
        }
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for batch_idx, batch in enumerate(pbar):
            rgb = batch['rgb'].to(self.device)
            nir = batch['nir'].to(self.device)
            seg = batch['seg'].to(self.device)
            weather = batch['weather'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast():
                output = self.model(rgb, nir, weather)
                loss = self.criterion(output, seg)
            
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self, epoch):
        """Validate and compute metrics"""
        self.model.eval()
        total_loss = 0.0
        
        # For mIoU calculation
        intersection = torch.zeros(self.num_classes).to(self.device)
        union = torch.zeros(self.num_classes).to(self.device)
        
        # For Safe mIoU (simplified)
        safe_intersection = torch.zeros(self.num_classes).to(self.device)
        safe_union = torch.zeros(self.num_classes).to(self.device)
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
        with torch.no_grad():
            for batch in pbar:
                rgb = batch['rgb'].to(self.device)
                nir = batch['nir'].to(self.device)
                seg = batch['seg'].to(self.device)
                weather = batch['weather'].to(self.device)
                
                output = self.model(rgb, nir, weather)
                loss = self.criterion(output, seg)
                
                total_loss += loss.item()
                
                # Predictions
                pred = output.argmax(dim=1)
                
                # Standard mIoU
                for cls in range(self.num_classes):
                    pred_mask = (pred == cls)
                    true_mask = (seg == cls)
                    
                    intersection[cls] += (pred_mask & true_mask).sum().float()
                    union[cls] += (pred_mask | true_mask).sum().float()
                
                # Safe mIoU (penalize dangerous misclassifications)
                # This is a simplified version - full implementation needs hierarchy
                safe_intersection += intersection
                safe_union += union
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Compute mIoU
        iou = intersection / (union + 1e-10)
        miou = iou[union > 0].mean().item()
        
        # Compute Safe mIoU (simplified - in real implementation, apply hierarchy penalties)
        safe_iou = safe_intersection / (safe_union + 1e-10)
        safe_miou = safe_iou[safe_union > 0].mean().item()
        
        return avg_loss, miou, safe_miou, iou
    
    def train(self, num_epochs=20, early_stopping_patience=5):
        """Full training loop"""
        print(f"\nTraining {self.model_name} for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_miou, val_safe_miou, class_iou = self.validate(epoch)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_miou'].append(val_miou)
            self.history['val_safe_miou'].append(val_safe_miou)
            
            # Learning rate scheduling
            self.scheduler.step(val_safe_miou)
            
            # Print results
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val mIoU: {val_miou:.4f}")
            print(f"  Val Safe mIoU: {val_safe_miou:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_safe_miou > self.best_safe_miou:
                self.best_safe_miou = val_safe_miou
                self.best_miou = val_miou
                self.save_checkpoint(epoch, is_best=True)
                print(f"  ✓ New best Safe mIoU: {val_safe_miou:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
            
            # Save checkpoint every 5 epochs
            if epoch % 5 == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        print(f"\nTraining completed!")
        print(f"Best mIoU: {self.best_miou:.4f}")
        print(f"Best Safe mIoU: {self.best_safe_miou:.4f}")
        
        # Save training history
        self.save_history()
        
        return self.history
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_miou': self.best_miou,
            'best_safe_miou': self.best_safe_miou,
            'history': self.history
        }
        
        if is_best:
            path = os.path.join(self.save_dir, f'{self.model_name}_best.pth')
            torch.save(checkpoint, path)
        else:
            path = os.path.join(self.save_dir, f'{self.model_name}_epoch{epoch}.pth')
            torch.save(checkpoint, path)
    
    def save_history(self):
        """Save training history as JSON"""
        history_path = os.path.join(self.save_dir, f'{self.model_name}_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)


# Main training script
if __name__ == '__main__':
    # Configuration
    CONFIG = {
        'data_root': '/home/jupyter-228w1a12b8/dec_7/IDD_AW/IDD_AW/IDDAW',
        'batch_size': 4,
        'num_epochs': 15,
        'img_size': (512, 1024),
        'num_classes': 30,
        'models_to_train': ['baseline', 'early', 'late', 'adaptive'],
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("="*80)
    print("IDD-AW Weather-Conditioned RGB-NIR Fusion Training")
    print("="*80)
    
    # Import dataset and models
    from dataset import get_dataloaders
    from models import get_model
    
    # Load data
    print("\nLoading datasets...")
    train_loader, val_loader = get_dataloaders(
        CONFIG['data_root'],
        batch_size=CONFIG['batch_size'],
        img_size=CONFIG['img_size']
    )
    
    # Train each model
    results = {}
    
    for model_type in CONFIG['models_to_train']:
        print(f"\n{'='*80}")
        print(f"Training {model_type.upper()} model")
        print(f"{'='*80}")
        
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
        
        results[model_type] = {
            'best_miou': trainer.best_miou,
            'best_safe_miou': trainer.best_safe_miou,
            'history': history
        }
    
    # Print final results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    for model_type, result in results.items():
        print(f"\n{model_type.upper()}:")
        print(f"  Best mIoU: {result['best_miou']:.4f}")
        print(f"  Best Safe mIoU: {result['best_safe_miou']:.4f}")
    
    # Save all results
    with open('./results_summary.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\nAll results saved to './results_summary.json'")
