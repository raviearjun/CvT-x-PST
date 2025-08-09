"""
Training Script for CvT-PST Classifier on Paddy Disease Dataset
==============================================================

This script demonstrates how to train the CvT-PST classifier for paddy disease
classification using the existing training infrastructure.

Key Features:
- Uses CvT-PST wrapper without modifying original CvT code
- Compatible with Google Colab environment
- Supports pretrained CvT backbone loading
- Implements progressive training (backbone frozen -> full training)
- Includes comprehensive evaluation metrics

Usage:
    python train_cvt_pst.py --config experiments/imagenet/cvt/cvt-21-224x224_paddy_pst.yaml

Author: CvT-x-PST Project
"""

import os
import sys
import logging
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# Add project root to path
sys.path.insert(0, '/content/CvT/tools')
sys.path.insert(0, '/content/CvT/lib')

from lib.models.cvt_pst_classifier import create_cvt_pst_classifier, PADDY_DISEASE_CLASSES
from lib.config import config, update_config
from lib.utils.utils import create_logger
from lib.core.function import train_epoch, validate


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train CvT-PST on Paddy Disease Dataset')
    
    parser.add_argument('--cfg', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to paddy disease dataset')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained CvT weights')
    parser.add_argument('--output-dir', type=str, default='/content/output',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--progressive', action='store_true',
                        help='Use progressive training (freeze backbone first)')
    parser.add_argument('--pst-scales', nargs='+', type=int, default=[1, 2, 4, 8],
                        help='PST pyramid scales')
    parser.add_argument('--pst-reduction', type=int, default=4,
                        help='PST channel reduction ratio')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def setup_device(device_arg):
    """Setup training device"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    logging.info(f"Using device: {device}")
    if device.type == 'cuda':
        logging.info(f"GPU: {torch.cuda.get_device_name()}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device


def create_data_transforms():
    """Create data transformations for training and validation"""
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms without augmentation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def create_data_loaders(data_path, batch_size, num_workers):
    """Create training and validation data loaders"""
    
    train_transform, val_transform = create_data_transforms()
    
    # Dataset paths
    train_dir = os.path.join(data_path, 'train')
    val_dir = os.path.join(data_path, 'val')
    
    # Check if directories exist
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")
    
    # Create datasets
    train_dataset = ImageFolder(train_dir, transform=train_transform)
    val_dataset = ImageFolder(val_dir, transform=val_transform)
    
    # Verify class mappings
    logging.info(f"Found {len(train_dataset.classes)} classes:")
    for i, class_name in enumerate(train_dataset.classes):
        logging.info(f"  {i}: {class_name}")
    
    # Create data loaders
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
    
    logging.info(f"Training samples: {len(train_dataset)}")
    logging.info(f"Validation samples: {len(val_dataset)}")
    logging.info(f"Batches per epoch: {len(train_loader)}")
    
    return train_loader, val_loader, len(train_dataset.classes)


def create_model(config_path, num_classes, pst_scales, pst_reduction, pretrained_path, device):
    """Create CvT-PST model"""
    
    # Update config
    update_config(config, config_path)
    
    # Create model
    model = create_cvt_pst_classifier(
        config=config,
        num_classes=num_classes,
        pst_scales=pst_scales,
        pst_reduction_ratio=pst_reduction,
        pst_dropout=0.1,
        pretrained_path=pretrained_path
    )
    
    # Move to device
    model = model.to(device)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"Model created:")
    logging.info(f"  Total parameters: {total_params:,}")
    logging.info(f"  Trainable parameters: {trainable_params:,}")
    logging.info(f"  PST scales: {pst_scales}")
    logging.info(f"  PST reduction ratio: {pst_reduction}")
    
    return model


def create_optimizer_scheduler(model, lr, epochs, train_loader_len):
    """Create optimizer and learning rate scheduler"""
    
    # Different learning rates for backbone vs PST+classifier
    backbone_params = []
    pst_classifier_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name.startswith('backbone.'):
                backbone_params.append(param)
            else:
                pst_classifier_params.append(param)
    
    # Optimizer with parameter groups
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': lr * 0.1},  # Lower LR for backbone
        {'params': pst_classifier_params, 'lr': lr}   # Higher LR for new modules
    ], weight_decay=0.05)
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )
    
    logging.info(f"Optimizer created:")
    logging.info(f"  Backbone parameters: {len(backbone_params)}")
    logging.info(f"  PST+Classifier parameters: {len(pst_classifier_params)}")
    logging.info(f"  Base learning rate: {lr}")
    
    return optimizer, scheduler


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Log progress
        if batch_idx % 50 == 0:
            logging.info(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                        f'Loss: {loss.item():.4f} '
                        f'Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate_one_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, output_dir, is_best=False):
    """Save training checkpoint"""
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': best_acc,
    }
    
    # Save latest checkpoint
    checkpoint_path = os.path.join(output_dir, 'checkpoint_latest.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(output_dir, 'checkpoint_best.pth')
        torch.save(checkpoint, best_path)
        logging.info(f"New best model saved with accuracy: {best_acc:.2f}%")


def main():
    """Main training function"""
    
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logger = create_logger(args.output_dir, 'train_cvt_pst')
    
    # Setup device
    device = setup_device(args.device)
    
    # Create data loaders
    train_loader, val_loader, num_classes = create_data_loaders(
        args.data_path, args.batch_size, args.num_workers
    )
    
    # Create model
    model = create_model(
        args.cfg, num_classes, args.pst_scales, 
        args.pst_reduction, args.pretrained, device
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Progressive training option
    if args.progressive:
        logging.info("=== Phase 1: Training PST and Classifier only ===")
        model.freeze_backbone()
        
        # Create optimizer for phase 1
        optimizer, scheduler = create_optimizer_scheduler(
            model, args.lr, args.epochs // 2, len(train_loader)
        )
        
        # Train phase 1
        for epoch in range(args.epochs // 2):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch + 1
            )
            val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
            scheduler.step()
            
            logging.info(f'Phase 1 Epoch {epoch+1}: '
                        f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                        f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        logging.info("=== Phase 2: Full model fine-tuning ===")
        model.unfreeze_backbone()
    
    # Create optimizer for full training
    optimizer, scheduler = create_optimizer_scheduler(
        model, args.lr * 0.1 if args.progressive else args.lr, 
        args.epochs if not args.progressive else args.epochs // 2, 
        len(train_loader)
    )
    
    # Training loop
    best_acc = 0.0
    start_epoch = 0
    
    # Resume from checkpoint if provided
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        logging.info(f"Resumed from epoch {start_epoch} with best acc {best_acc:.2f}%")
    
    total_epochs = args.epochs if not args.progressive else args.epochs // 2
    
    for epoch in range(start_epoch, total_epochs):
        epoch_start_time = time.time()
        
        # Training
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1
        )
        
        # Validation
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Log results
        epoch_time = time.time() - epoch_start_time
        logging.info(f'Epoch {epoch+1}/{total_epochs} ({epoch_time:.1f}s): '
                    f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                    f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save checkpoint
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
        
        save_checkpoint(model, optimizer, scheduler, epoch + 1, best_acc, 
                       args.output_dir, is_best)
    
    logging.info(f"Training completed! Best validation accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()
