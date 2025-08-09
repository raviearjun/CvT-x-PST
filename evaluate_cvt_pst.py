"""
Evaluation Script for CvT-PST Classifier
========================================

This script provides comprehensive evaluation of the trained CvT-PST model
including accuracy metrics, confusion matrix, feature visualization, and
per-class performance analysis.

Usage:
    python evaluate_cvt_pst.py --model-path checkpoint_best.pth --data-path /path/to/test/data

Features:
- Comprehensive classification metrics
- Confusion matrix visualization
- Feature map visualization
- Per-class accuracy analysis
- Model inference time benchmarking

Author: CvT-x-PST Project
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import time

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from lib.models.cvt_pst_classifier import create_cvt_pst_classifier, PADDY_DISEASE_CLASSES
from lib.config import config, update_config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate CvT-PST Model')
    
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to model config file')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to test dataset')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save individual predictions')
    parser.add_argument('--visualize-features', action='store_true',
                        help='Visualize feature maps')
    parser.add_argument('--benchmark', action='store_true',
                        help='Benchmark inference speed')
    
    return parser.parse_args()


def setup_device(device_arg):
    """Setup evaluation device"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    return device


def create_test_transforms():
    """Create test data transformations"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def load_model(model_path, config_path, device):
    """Load trained CvT-PST model"""
    
    # Update config
    update_config(config, config_path)
    
    # Create model
    model = create_cvt_pst_classifier(
        config=config,
        num_classes=10,
        pst_scales=[1, 2, 4, 8],
        pst_reduction_ratio=4,
        pst_dropout=0.1
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        best_acc = checkpoint.get('best_acc', 0.0)
        epoch = checkpoint.get('epoch', 0)
        print(f"Loaded model from epoch {epoch} with best accuracy {best_acc:.2f}%")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model weights")
    
    model = model.to(device)
    model.eval()
    
    return model


def evaluate_model(model, test_loader, device, class_names):
    """Evaluate model on test dataset"""
    
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    inference_times = []
    
    print("Evaluating model...")
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Measure inference time
            start_time = time.time()
            outputs = model(inputs)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Get predictions and probabilities
            probabilities = F.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            if batch_idx % 20 == 0:
                print(f"Processed {batch_idx}/{len(test_loader)} batches")
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_predictions, average=None
    )
    
    # Average inference time
    avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
    
    print(f"\nEvaluation Results:")
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Average Inference Time: {avg_inference_time:.2f} ms/batch")
    
    return {
        'predictions': all_predictions,
        'targets': all_targets,
        'probabilities': all_probabilities,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'inference_time': avg_inference_time,
        'class_names': class_names
    }


def plot_confusion_matrix(results, output_dir):
    """Plot and save confusion matrix"""
    
    cm = confusion_matrix(results['targets'], results['predictions'])
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=results['class_names'],
                yticklabels=results['class_names'])
    plt.title('Confusion Matrix - CvT-PST Classifier')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Confusion matrix saved to confusion_matrix.png")


def plot_per_class_metrics(results, output_dir):
    """Plot per-class performance metrics"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    class_names = results['class_names']
    metrics = ['precision', 'recall', 'f1', 'support']
    titles = ['Precision', 'Recall', 'F1-Score', 'Support (# samples)']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        if metric == 'support':
            values = results[metric]
            colors = 'skyblue'
        else:
            values = results[metric]
            # Color bars based on performance
            colors = ['green' if v > 0.8 else 'orange' if v > 0.6 else 'red' for v in values]
        
        bars = ax.bar(range(len(class_names)), values, color=colors, alpha=0.7)
        ax.set_xlabel('Disease Classes')
        ax.set_ylabel(title)
        ax.set_title(f'Per-Class {title}')
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            if metric == 'support':
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(values),
                       f'{int(value)}', ha='center', va='bottom', fontsize=9)
            else:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        if metric != 'support':
            ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Per-class metrics saved to per_class_metrics.png")


def save_classification_report(results, output_dir):
    """Save detailed classification report"""
    
    # Generate classification report
    report = classification_report(
        results['targets'], 
        results['predictions'],
        target_names=results['class_names'],
        digits=4
    )
    
    # Save to file
    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("CvT-PST Classifier - Detailed Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Overall Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Average Inference Time: {results['inference_time']:.2f} ms/batch\n\n")
        f.write("Per-Class Metrics:\n")
        f.write("-" * 30 + "\n")
        f.write(report)
    
    print(f"Classification report saved to {report_path}")


def visualize_predictions(model, test_loader, device, class_names, output_dir, num_samples=16):
    """Visualize sample predictions"""
    
    model.eval()
    
    # Get a batch of test images
    data_iter = iter(test_loader)
    images, targets = next(data_iter)
    images, targets = images.to(device), targets.to(device)
    
    # Select subset for visualization
    if len(images) > num_samples:
        indices = torch.randperm(len(images))[:num_samples]
        images = images[indices]
        targets = targets[indices]
    
    with torch.no_grad():
        outputs = model(images)
        probabilities = F.softmax(outputs, dim=1)
        _, predictions = torch.max(outputs, 1)
    
    # Plot predictions
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.ravel()
    
    # Denormalize images for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    for i in range(min(num_samples, len(images))):
        # Denormalize image
        img = images[i].cpu() * std + mean
        img = torch.clamp(img, 0, 1)
        
        # Convert to numpy and transpose
        img_np = img.permute(1, 2, 0).numpy()
        
        # Plot image
        axes[i].imshow(img_np)
        axes[i].axis('off')
        
        # Get prediction info
        true_class = class_names[targets[i].item()]
        pred_class = class_names[predictions[i].item()]
        confidence = probabilities[i, predictions[i]].item()
        
        # Set title with color coding
        is_correct = targets[i].item() == predictions[i].item()
        color = 'green' if is_correct else 'red'
        
        title = f"True: {true_class}\nPred: {pred_class}\nConf: {confidence:.3f}"
        axes[i].set_title(title, fontsize=10, color=color)
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_predictions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Sample predictions saved to sample_predictions.png")


def benchmark_inference(model, test_loader, device):
    """Benchmark model inference speed"""
    
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            if i >= 5:  # 5 warmup batches
                break
            inputs = inputs.to(device)
            _ = model(inputs)
    
    # Benchmark
    torch.cuda.synchronize() if device.type == 'cuda' else None
    
    times = []
    total_samples = 0
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            if i >= 50:  # Benchmark on 50 batches
                break
                
            inputs = inputs.to(device)
            
            start_time = time.time()
            _ = model(inputs)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.time()
            
            batch_time = end_time - start_time
            times.append(batch_time)
            total_samples += inputs.size(0)
    
    # Calculate statistics
    avg_batch_time = np.mean(times) * 1000  # ms
    std_batch_time = np.std(times) * 1000   # ms
    avg_sample_time = avg_batch_time / test_loader.batch_size  # ms per sample
    throughput = total_samples / sum(times)  # samples per second
    
    print(f"\nInference Benchmark Results:")
    print(f"Average batch time: {avg_batch_time:.2f} ± {std_batch_time:.2f} ms")
    print(f"Average per-sample time: {avg_sample_time:.2f} ms")
    print(f"Throughput: {throughput:.1f} samples/second")
    
    return {
        'avg_batch_time': avg_batch_time,
        'std_batch_time': std_batch_time,
        'avg_sample_time': avg_sample_time,
        'throughput': throughput
    }


def main():
    """Main evaluation function"""
    
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = setup_device(args.device)
    
    # Load model
    print("Loading model...")
    model = load_model(args.model_path, args.config, device)
    
    # Create test dataset
    print("Loading test data...")
    test_transform = create_test_transforms()
    test_dataset = ImageFolder(args.data_path, transform=test_transform)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Test dataset: {len(test_dataset)} samples")
    print(f"Classes: {test_dataset.classes}")
    
    # Evaluate model
    results = evaluate_model(model, test_loader, device, test_dataset.classes)
    
    # Generate reports and visualizations
    print("\nGenerating evaluation reports...")
    
    # Save classification report
    save_classification_report(results, args.output_dir)
    
    # Plot confusion matrix
    plot_confusion_matrix(results, args.output_dir)
    
    # Plot per-class metrics
    plot_per_class_metrics(results, args.output_dir)
    
    # Visualize predictions
    visualize_predictions(model, test_loader, device, test_dataset.classes, args.output_dir)
    
    # Benchmark inference if requested
    if args.benchmark:
        print("\nBenchmarking inference speed...")
        benchmark_results = benchmark_inference(model, test_loader, device)
        
        # Save benchmark results
        benchmark_path = os.path.join(args.output_dir, 'benchmark_results.txt')
        with open(benchmark_path, 'w') as f:
            f.write("CvT-PST Inference Benchmark Results\n")
            f.write("=" * 40 + "\n")
            f.write(f"Average batch time: {benchmark_results['avg_batch_time']:.2f} ± {benchmark_results['std_batch_time']:.2f} ms\n")
            f.write(f"Average per-sample time: {benchmark_results['avg_sample_time']:.2f} ms\n")
            f.write(f"Throughput: {benchmark_results['throughput']:.1f} samples/second\n")
    
    print(f"\nEvaluation completed! Results saved to {args.output_dir}")
    print(f"Final Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")


if __name__ == '__main__':
    main()
