#training_LH.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import os
import json

class UGFTrainer:
    """
    Trainer for Uncertainty-Gated Fusion with Deep Supervision
    
    Key difference: Trains 3 heads simultaneously:
    - Main segmentation head
    - RGB auxiliary head (forces RGB encoder to learn semantic features)
    - NIR auxiliary head (forces NIR encoder to learn semantic features)
    """
    
    def __init__(self, model, train_loader, val_loader, num_classes=30,
                 device='cuda', save_dir='./checkpoints', model_name='ugf'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.device = device
        self.save_dir = save_dir
        self.model_name = model_name
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        
        # Optimizer with slightly higher LR for faster convergence
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=2e-4,  # Higher than baseline
            weight_decay=1e-4
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            patience=3,
            factor=0.5
        )
        
        # Mixed precision
        self.scaler = GradScaler()
        
        # Tracking
        self.best_miou = 0.0
        self.history = {
            'train_loss': [],
            'train_main_loss': [],
            'train_aux_loss': [],
            'val_loss': [],
            'val_miou': []
        }
    
    def train_epoch(self, epoch):
        """Train one epoch with deep supervision"""
        self.model.train()
        
        total_loss = 0.0
        total_main_loss = 0.0
        total_aux_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch in pbar:
            rgb = batch['rgb'].to(self.device)
            nir = batch['nir'].to(self.device)
            seg = batch['seg'].to(self.device)
            weather = batch['weather'].to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast():
                # Get all outputs
                output, rgb_aux, nir_aux, _, _ = self.model(rgb, nir, weather)
                
                # === DEEP SUPERVISION LOSS ===
                # Main loss: final prediction vs ground truth
                loss_main = self.criterion(output, seg)
                
                # Auxiliary losses: force encoders to learn semantic features
                loss_rgb_aux = self.criterion(rgb_aux, seg)
                loss_nir_aux = self.criterion(nir_aux, seg)
                
                # Combined loss with weights
                # 0.4 weight ensures auxiliary heads contribute but don't dominate
                loss = loss_main + 0.4 * loss_rgb_aux + 0.4 * loss_nir_aux
            
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            total_main_loss += loss_main.item()
            total_aux_loss += (loss_rgb_aux.item() + loss_nir_aux.item()) / 2
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'main': f'{loss_main.item():.4f}',
                'aux': f'{(loss_rgb_aux.item() + loss_nir_aux.item())/2:.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_main = total_main_loss / len(self.train_loader)
        avg_aux = total_aux_loss / len(self.train_loader)
        
        return avg_loss, avg_main, avg_aux
    
    def validate(self, epoch):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        
        # mIoU calculation
        intersection = torch.zeros(self.num_classes).to(self.device)
        union = torch.zeros(self.num_classes).to(self.device)
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
        
        with torch.no_grad():
            for batch in pbar:
                rgb = batch['rgb'].to(self.device)
                nir = batch['nir'].to(self.device)
                seg = batch['seg'].to(self.device)
                weather = batch['weather'].to(self.device)
                
                # During inference, model returns (output, rgb_gate, nir_gate)
                output, _, _ = self.model(rgb, nir, weather)
                
                loss = self.criterion(output, seg)
                total_loss += loss.item()
                
                # Predictions
                pred = output.argmax(dim=1)
                
                # Compute IoU per class
                for cls in range(self.num_classes):
                    pred_mask = (pred == cls)
                    true_mask = (seg == cls)
                    
                    intersection[cls] += (pred_mask & true_mask).sum().float()
                    union[cls] += (pred_mask | true_mask).sum().float()
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Compute mIoU
        iou = intersection / (union + 1e-10)
        valid_classes = union > 0
        miou = iou[valid_classes].mean().item()
        
        return avg_loss, miou, iou
    
    def train(self, num_epochs=15, early_stopping_patience=5):
        """Full training loop"""
        print(f"\nTraining UGF-Net for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_main, train_aux = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_miou, class_iou = self.validate(epoch)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_main_loss'].append(train_main)
            self.history['train_aux_loss'].append(train_aux)
            self.history['val_loss'].append(val_loss)
            self.history['val_miou'].append(val_miou)
            
            # Learning rate scheduling
            self.scheduler.step(val_miou)
            
            # Print results
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} (Main: {train_main:.4f}, Aux: {train_aux:.4f})")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val mIoU: {val_miou:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_miou > self.best_miou:
                self.best_miou = val_miou
                self.save_checkpoint(epoch, is_best=True)
                print(f"  ✓ New best mIoU: {val_miou:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        print(f"\nTraining completed!")
        print(f"Best mIoU: {self.best_miou:.4f}")
        
        # Save history
        self.save_history()
        
        return self.history
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_miou': self.best_miou,
            'history': self.history
        }
        
        if is_best:
            path = os.path.join(self.save_dir, f'{self.model_name}_best.pth')
        else:
            path = os.path.join(self.save_dir, f'{self.model_name}_epoch{epoch}.pth')
        
        torch.save(checkpoint, path)
    
    def save_history(self):
        """Save training history"""
        path = os.path.join(self.save_dir, f'{self.model_name}_history.json')
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=4)


if __name__ == '__main__':
    from dataset import get_dataloaders
    from models_LH import get_model
    
    # Configuration
    CONFIG = {
        'data_root': '/home/jupyter-228w1a12b8/dec_7/IDD_AW/IDD_AW/IDDAW',
        'batch_size': 8,
        'num_epochs': 15,
        'img_size': (512, 1024),
        'num_classes': 30,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("="*80)
    print("UGF-Net Training with Deep Supervision")
    print("="*80)
    
    # Load data
    print("\nLoading datasets...")
    train_loader, val_loader = get_dataloaders(
        CONFIG['data_root'],
        batch_size=CONFIG['batch_size'],
        img_size=CONFIG['img_size'],
        num_workers=4
    )
    
    # Create UGF model
    print("\nCreating UGF-Net...")
    model = get_model('ugf', num_classes=CONFIG['num_classes'])
    
    # Create trainer
    trainer = UGFTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=CONFIG['num_classes'],
        device=CONFIG['device'],
        save_dir='./checkpoints/ugf',
        model_name='ugf'
    )
    
    # Train
    history = trainer.train(num_epochs=CONFIG['num_epochs'])
    
    print("\n✓ Training complete!")
    print(f"Best mIoU: {trainer.best_miou:.4f}")
    print(f"Results saved to ./checkpoints/ugf/")
