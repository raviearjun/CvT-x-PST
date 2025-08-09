#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script untuk CvT Paddy Disease Classification
Menghasilkan classification report, confusion matrix, dan detailed analysis
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add CvT lib to path
sys.path.append('/content/CvT/tools')
sys.path.append('/content/CvT/lib')

import _init_paths
from config import config, update_config
from models import build_model
from dataset import build_dataloader
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Define class names for paddy disease
CLASS_NAMES = [
    'Bacterial_leaf_blight',
    'Brown_spot', 
    'Leaf_smut',
    'Normal',
    'Blast',
    'Dead_heart',
    'Downy_mildew',
    'Hispa', 
    'Tungro',
    'Rice_bug'
]

def load_model(config, model_path):
    """Load trained model"""
    logger.info(f"Loading model from: {model_path}")
    
    # Build model
    model = build_model(config)
    
    # Load weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Load state dict
    model.load_state_dict(state_dict, strict=True)
    model.cuda()
    model.eval()
    
    logger.info("✅ Model loaded successfully")
    return model

def predict_batch(model, data_loader, device='cuda'):
    """Get predictions for entire dataset"""
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    logger.info("🔍 Running inference on dataset...")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(data_loader, desc="Predicting")):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_probabilities), np.array(all_labels)

def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'{title}\n', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    return cm

def plot_classification_metrics(report_dict, class_names, title="Classification Metrics"):
    """Plot precision, recall, f1-score bar chart"""
    # Extract metrics for each class
    metrics = ['precision', 'recall', 'f1-score']
    class_metrics = {}
    
    for metric in metrics:
        class_metrics[metric] = [report_dict[str(i)][metric] for i in range(len(class_names))]
    
    # Create bar plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{title}', fontsize=16, fontweight='bold')
    
    x = np.arange(len(class_names))
    width = 0.6
    
    for idx, metric in enumerate(metrics):
        axes[idx].bar(x, class_metrics[metric], width, alpha=0.7, 
                     color=plt.cm.Set3(np.linspace(0, 1, len(class_names))))
        axes[idx].set_title(f'{metric.capitalize()}', fontweight='bold')
        axes[idx].set_ylabel(f'{metric.capitalize()} Score')
        axes[idx].set_xlabel('Disease Classes')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(class_names, rotation=45, ha='right')
        axes[idx].set_ylim(0, 1.1)
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(class_metrics[metric]):
            axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig

def analyze_misclassifications(y_true, y_pred, y_prob, class_names, top_k=5):
    """Analyze top misclassifications"""
    misclassified = []
    
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            confidence = y_prob[i][y_pred[i]]
            true_confidence = y_prob[i][y_true[i]]
            misclassified.append({
                'sample_idx': i,
                'true_class': class_names[y_true[i]],
                'pred_class': class_names[y_pred[i]],
                'confidence': confidence,
                'true_confidence': true_confidence,
                'confidence_diff': confidence - true_confidence
            })
    
    # Sort by confidence (most confident wrong predictions)
    misclassified.sort(key=lambda x: x['confidence'], reverse=True)
    
    print(f"\n📊 Top {top_k} Most Confident Misclassifications:")
    print("=" * 80)
    for i, mis in enumerate(misclassified[:top_k]):
        print(f"{i+1:2d}. Sample {mis['sample_idx']:4d}: "
              f"True: {mis['true_class']:20s} → "
              f"Pred: {mis['pred_class']:20s} "
              f"(Conf: {mis['confidence']:.3f})")
    
    return misclassified

def comprehensive_evaluation(config_path, model_path, dataset_type='test'):
    """Run comprehensive evaluation"""
    logger.info("🚀 Starting Comprehensive Model Evaluation")
    logger.info("=" * 60)
    
    # Setup config
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=config_path, type=str)
    args = parser.parse_args([])
    
    update_config(config, args)
    logger.info(f"✅ Config loaded from: {config_path}")
    logger.info(f"📊 Dataset type: {dataset_type}")
    logger.info(f"🎯 Number of classes: {config.MODEL.NUM_CLASSES}")
    
    # Load model
    model = load_model(config, model_path)
    
    # Load dataset
    logger.info(f"📂 Loading {dataset_type} dataset...")
    data_loader = build_dataloader(config, is_train=False, distributed=False, dataset_type=dataset_type)
    logger.info(f"✅ Dataset loaded: {len(data_loader)} batches")
    
    # Get predictions
    y_pred, y_prob, y_true = predict_batch(model, data_loader)
    
    logger.info(f"📊 Evaluation completed on {len(y_true)} samples")
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    logger.info(f"🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, 
                                 output_dict=True, zero_division=0)
    report_str = classification_report(y_true, y_pred, target_names=CLASS_NAMES, 
                                     zero_division=0)
    
    print("\n" + "="*80)
    print("📋 DETAILED CLASSIFICATION REPORT")
    print("="*80)
    print(report_str)
    
    # Class-wise accuracy
    print("\n📊 PER-CLASS ACCURACY:")
    print("-" * 50)
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    for i, (class_name, acc) in enumerate(zip(CLASS_NAMES, class_accuracies)):
        print(f"{i:2d}. {class_name:20s}: {acc:.4f} ({acc*100:6.2f}%)")
    
    # Macro averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0)
    
    print(f"\n🎯 MACRO AVERAGES:")
    print(f"   Precision: {macro_precision:.4f}")
    print(f"   Recall:    {macro_recall:.4f}")
    print(f"   F1-Score:  {macro_f1:.4f}")
    
    # Create visualizations
    logger.info("📈 Creating visualizations...")
    
    # 1. Confusion Matrix
    plt.figure(figsize=(15, 12))
    cm = plot_confusion_matrix(y_true, y_pred, CLASS_NAMES, 
                              f"Confusion Matrix - {dataset_type.title()} Set")
    plt.savefig(f'/content/confusion_matrix_{dataset_type}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Classification metrics bar chart
    fig = plot_classification_metrics(report, CLASS_NAMES, 
                                    f"Classification Metrics - {dataset_type.title()} Set")
    plt.savefig(f'/content/classification_metrics_{dataset_type}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Top predictions confidence analysis
    logger.info("🔍 Analyzing prediction confidence...")
    confidence_scores = np.max(y_prob, axis=1)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(confidence_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Prediction Confidence Distribution', fontweight='bold')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    correct_mask = y_true == y_pred
    plt.hist([confidence_scores[correct_mask], confidence_scores[~correct_mask]], 
             bins=30, alpha=0.7, label=['Correct', 'Incorrect'], 
             color=['green', 'red'], edgecolor='black')
    plt.title('Confidence: Correct vs Incorrect', fontweight='bold')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    avg_confidence_per_class = [np.mean(confidence_scores[y_true == i]) for i in range(len(CLASS_NAMES))]
    bars = plt.bar(range(len(CLASS_NAMES)), avg_confidence_per_class, 
                   color=plt.cm.Set3(np.linspace(0, 1, len(CLASS_NAMES))), alpha=0.7)
    plt.title('Average Confidence per Class', fontweight='bold')
    plt.xlabel('Class')
    plt.ylabel('Average Confidence')
    plt.xticks(range(len(CLASS_NAMES)), [name[:8] + '...' if len(name) > 8 else name 
                                        for name in CLASS_NAMES], rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, avg_confidence_per_class):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.subplot(2, 2, 4)
    error_rate_per_class = [1 - acc for acc in class_accuracies]
    bars = plt.bar(range(len(CLASS_NAMES)), error_rate_per_class, 
                   color='salmon', alpha=0.7)
    plt.title('Error Rate per Class', fontweight='bold')
    plt.xlabel('Class')
    plt.ylabel('Error Rate')
    plt.xticks(range(len(CLASS_NAMES)), [name[:8] + '...' if len(name) > 8 else name 
                                        for name in CLASS_NAMES], rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, error_rate_per_class):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'/content/confidence_analysis_{dataset_type}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Analyze misclassifications
    misclassified = analyze_misclassifications(y_true, y_pred, y_prob, CLASS_NAMES)
    
    # Save detailed results
    results = {
        'overall_accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'class_names': CLASS_NAMES,
        'total_samples': len(y_true),
        'misclassified_count': len(misclassified)
    }
    
    # Save to file
    import json
    with open(f'/content/evaluation_results_{dataset_type}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"💾 Results saved to: /content/evaluation_results_{dataset_type}.json")
    logger.info("🎉 Evaluation completed successfully!")
    
    return results

def main():
    """Main evaluation function"""
    
    # Configuration
    CONFIG_PATH = '/content/CvT/experiments/imagenet/cvt/cvt-21-224x224_paddy_dataset.yaml'
    MODEL_PATH = '/content/output/imagenet/cvt-21-224x224_paddy_dataset/model_best.pth'
    
    print("🌾 CvT Paddy Disease Classification - Model Evaluation")
    print("=" * 60)
    print(f"📝 Config: {CONFIG_PATH}")
    print(f"🤖 Model: {MODEL_PATH}")
    print("=" * 60)
    
    # Check if files exist
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ Config file not found: {CONFIG_PATH}")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("📂 Looking for alternative model files...")
        model_dir = os.path.dirname(MODEL_PATH)
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
        print(f"Available models: {model_files}")
        return
    
    # Evaluate on test set
    print("\n🧪 EVALUATING ON TEST SET")
    print("=" * 40)
    test_results = comprehensive_evaluation(CONFIG_PATH, MODEL_PATH, 'test')
    
    # Evaluate on validation set for comparison
    print("\n🔍 EVALUATING ON VALIDATION SET (for comparison)")
    print("=" * 50)
    val_results = comprehensive_evaluation(CONFIG_PATH, MODEL_PATH, 'val')
    
    # Compare results
    print("\n📊 TEST vs VALIDATION COMPARISON:")
    print("=" * 40)
    print(f"Test Accuracy:       {test_results['overall_accuracy']:.4f} ({test_results['overall_accuracy']*100:.2f}%)")
    print(f"Validation Accuracy: {val_results['overall_accuracy']:.4f} ({val_results['overall_accuracy']*100:.2f}%)")
    print(f"Difference:          {abs(test_results['overall_accuracy'] - val_results['overall_accuracy']):.4f}")
    
    if abs(test_results['overall_accuracy'] - val_results['overall_accuracy']) < 0.02:
        print("✅ Model generalizes well (< 2% difference)")
    else:
        print("⚠️  Potential overfitting detected (> 2% difference)")
    
    print("\n🎉 Complete evaluation finished!")
    print("📁 Files generated:")
    print("   - confusion_matrix_test.png")
    print("   - confusion_matrix_val.png") 
    print("   - classification_metrics_test.png")
    print("   - classification_metrics_val.png")
    print("   - confidence_analysis_test.png")
    print("   - confidence_analysis_val.png")
    print("   - evaluation_results_test.json")
    print("   - evaluation_results_val.json")

if __name__ == "__main__":
    main()
