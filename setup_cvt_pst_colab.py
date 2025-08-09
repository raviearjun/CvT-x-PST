"""
Google Colab Setup Script for CvT-PST Training
==============================================

This script sets up the environment and demonstrates how to train
CvT with Pyramid Sparse Transformer (PST) on Google Colab.

Run this script in Google Colab to:
1. Setup the environment
2. Download and prepare the dataset
3. Download pretrained weights
4. Train CvT-PST model
5. Evaluate the results

Author: CvT-x-PST Project
"""

import os
import sys
import subprocess
import zipfile
import urllib.request
from pathlib import Path
import shutil


def run_command(command, description=""):
    """Run shell command and handle errors"""
    if description:
        print(f"🔄 {description}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Error: {result.stderr}")
            return False
        else:
            if result.stdout.strip():
                print(f"✅ {result.stdout.strip()}")
            return True
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False


def setup_environment():
    """Setup Google Colab environment"""
    print("🚀 Setting up CvT-PST environment...")
    
    # Install additional packages if needed
    packages = [
        "timm",
        "einops", 
        "yacs",
        "tensorboardX"
    ]
    
    for package in packages:
        run_command(f"pip install {package}", f"Installing {package}")
    
    # Verify PyTorch installation
    try:
        import torch
        import torchvision
        print(f"✅ PyTorch {torch.__version__} installed")
        print(f"✅ TorchVision {torchvision.__version__} installed")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name()}")
    except ImportError as e:
        print(f"❌ PyTorch import error: {e}")
        return False
    
    return True


def clone_repository():
    """Clone CvT-x-PST repository"""
    print("📦 Cloning CvT-x-PST repository...")
    
    repo_url = "https://github.com/raviearjun/CvT-x-PST.git"
    target_dir = "/content/CvT"
    
    # Remove existing directory if it exists
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    
    success = run_command(
        f"git clone {repo_url} {target_dir}",
        "Cloning repository"
    )
    
    if success:
        os.chdir(target_dir)
        print(f"✅ Repository cloned to {target_dir}")
        return True
    else:
        print("❌ Failed to clone repository")
        return False


def download_pretrained_weights():
    """Download pretrained CvT weights"""
    print("⬇️ Downloading pretrained CvT weights...")
    
    # CvT-21 pretrained weights URL (you may need to update this)
    weight_url = "https://github.com/microsoft/CvT/releases/download/v1.0/CvT-21-224x224-IN-1k.pth"
    target_path = "/content/CvT/CvT-21-224x224-IN-1k.pth"
    
    try:
        urllib.request.urlretrieve(weight_url, target_path)
        print(f"✅ Downloaded pretrained weights to {target_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to download weights: {e}")
        print("ℹ️ Please manually download CvT-21-224x224-IN-1k.pth")
        return False


def prepare_dataset():
    """Prepare paddy disease dataset structure"""
    print("📁 Preparing dataset structure...")
    
    dataset_dir = "/content/CvT/paddy_disease_dataset"
    
    # Create dataset directories
    dirs_to_create = [
        f"{dataset_dir}/train",
        f"{dataset_dir}/val", 
        f"{dataset_dir}/test"
    ]
    
    for disease_class in [
        'Bacterial_leaf_blight', 'Brown_spot', 'Leaf_smut', 'Normal',
        'Blast', 'Dead_heart', 'Downy_mildew', 'Hispa', 'Tungro', 'Rice_bug'
    ]:
        for split in ['train', 'val', 'test']:
            dirs_to_create.append(f"{dataset_dir}/{split}/{disease_class}")
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"✅ Dataset directories created at {dataset_dir}")
    print("ℹ️ Please upload your paddy disease images to the appropriate directories")
    
    return True


def create_colab_training_notebook():
    """Create Jupyter notebook for Colab training"""
    print("📓 Creating training notebook...")
    
    notebook_content = '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# CvT-PST Training on Google Colab\\n",
    "\\n",
    "This notebook demonstrates training CvT with Pyramid Sparse Transformer on paddy disease dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Setup environment\\n",
    "import os\\n",
    "import sys\\n",
    "\\n",
    "# Add project to path\\n",
    "sys.path.insert(0, '/content/CvT')\\n",
    "os.chdir('/content/CvT')\\n",
    "\\n",
    "# Check GPU\\n",
    "import torch\\n",
    "print(f'CUDA available: {torch.cuda.is_available()}')\\n",
    "if torch.cuda.is_available():\\n",
    "    print(f'GPU: {torch.cuda.get_device_name()}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import CvT-PST modules\\n",
    "from lib.models.cvt_pst_classifier import create_cvt_pst_classifier, PADDY_DISEASE_CLASSES\\n",
    "from lib.config import config, update_config\\n",
    "\\n",
    "print('CvT-PST modules imported successfully!')\\n",
    "print(f'Paddy disease classes: {PADDY_DISEASE_CLASSES}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test model creation\\n",
    "from types import SimpleNamespace\\n",
    "\\n",
    "# Mock config for testing\\n",
    "mock_spec = {\\n",
    "    'NUM_STAGES': 3,\\n",
    "    'DIM_EMBED': [64, 192, 768],\\n",
    "    'PATCH_SIZE': [7, 3, 3],\\n",
    "    'PATCH_STRIDE': [4, 2, 2],\\n",
    "    'PATCH_PADDING': [2, 1, 1],\\n",
    "    'DEPTH': [1, 2, 10],\\n",
    "    'NUM_HEADS': [1, 3, 12],\\n",
    "    'MLP_RATIO': [4, 4, 4],\\n",
    "    'QKV_BIAS': [True, True, True],\\n",
    "    'DROP_RATE': [0.0, 0.0, 0.0],\\n",
    "    'ATTN_DROP_RATE': [0.0, 0.0, 0.0],\\n",
    "    'DROP_PATH_RATE': [0.0, 0.0, 0.1],\\n",
    "    'CLS_TOKEN': [False, False, False],\\n",
    "    'QKV_PROJ_METHOD': ['dw_bn', 'dw_bn', 'dw_bn'],\\n",
    "    'KERNEL_QKV': [3, 3, 3],\\n",
    "    'PADDING_Q': [1, 1, 1],\\n",
    "    'PADDING_KV': [1, 1, 1],\\n",
    "    'STRIDE_KV': [2, 2, 2],\\n",
    "    'STRIDE_Q': [1, 1, 1],\\n",
    "}\\n",
    "\\n",
    "mock_config = SimpleNamespace()\\n",
    "mock_config.MODEL = SimpleNamespace()\\n",
    "mock_config.MODEL.SPEC = mock_spec\\n",
    "\\n",
    "# Test model creation\\n",
    "model = create_cvt_pst_classifier(\\n",
    "    config=mock_config,\\n",
    "    num_classes=10,\\n",
    "    pst_scales=[1, 2, 4],\\n",
    "    pst_reduction_ratio=4\\n",
    ")\\n",
    "\\n",
    "# Test forward pass\\n",
    "x = torch.randn(2, 3, 224, 224)\\n",
    "with torch.no_grad():\\n",
    "    output = model(x)\\n",
    "\\n",
    "print(f'Input shape: {x.shape}')\\n",
    "print(f'Output shape: {output.shape}')\\n",
    "print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')\\n",
    "print('✅ CvT-PST model test successful!')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Training script (replace with your actual training code)\\n",
    "# You can use the train_cvt_pst.py script here\\n",
    "\\n",
    "print('Training setup complete!')\\n",
    "print('Next steps:')\\n",
    "print('1. Upload your paddy disease dataset to /content/CvT/paddy_disease_dataset/')\\n",
    "print('2. Download pretrained CvT weights to /content/CvT/')\\n",
    "print('3. Run training script or use the training functions above')\\n",
    "print('4. Evaluate the trained model')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}'''
    
    notebook_path = "/content/CvT/train_cvt_pst_colab.ipynb"
    with open(notebook_path, 'w') as f:
        f.write(notebook_content)
    
    print(f"✅ Training notebook created at {notebook_path}")
    return True


def display_usage_instructions():
    """Display usage instructions"""
    print("=" * 60)
    print("🎉 CvT-PST Setup Complete!")
    print("=" * 60)
    print()
    print("📋 Next Steps:")
    print()
    print("1. 📁 Upload Dataset:")
    print("   - Upload your paddy disease images to:")
    print("     /content/CvT/paddy_disease_dataset/train/[class_name]/")
    print("     /content/CvT/paddy_disease_dataset/val/[class_name]/")
    print("     /content/CvT/paddy_disease_dataset/test/[class_name]/")
    print()
    print("2. ⬇️ Download Pretrained Weights:")
    print("   - Download CvT-21-224x224-IN-1k.pth to /content/CvT/")
    print()
    print("3. 🚀 Start Training:")
    print("   - Option A: Use the Jupyter notebook:")
    print("     train_cvt_pst_colab.ipynb")
    print("   - Option B: Run training script:")
    print("     !python train_cvt_pst.py --cfg experiments/imagenet/cvt/cvt-21-224x224_paddy_pst.yaml --data-path /content/CvT/paddy_disease_dataset")
    print()
    print("4. 📊 Evaluate Results:")
    print("   !python evaluate_cvt_pst.py --model-path output/checkpoint_best.pth --config experiments/imagenet/cvt/cvt-21-224x224_paddy_pst.yaml --data-path /content/CvT/paddy_disease_dataset/test")
    print()
    print("🔗 Key Files:")
    print("   - CvT-PST Model: lib/models/cvt_pst_classifier.py")
    print("   - Config: experiments/imagenet/cvt/cvt-21-224x224_paddy_pst.yaml")
    print("   - Training: train_cvt_pst.py")
    print("   - Evaluation: evaluate_cvt_pst.py")
    print()
    print("📚 Classes: 10 paddy disease types")
    for i, class_name in enumerate([
        'Bacterial_leaf_blight', 'Brown_spot', 'Leaf_smut', 'Normal',
        'Blast', 'Dead_heart', 'Downy_mildew', 'Hispa', 'Tungro', 'Rice_bug'
    ]):
        print(f"   {i}: {class_name}")
    print()
    print("🎯 Expected Performance:")
    print("   - Training Time: ~2-3 hours on Colab GPU")
    print("   - Expected Accuracy: 85-95% (depending on dataset quality)")
    print("   - Model Size: ~32M parameters")
    print()
    print("⚠️ Memory Tips for Colab:")
    print("   - Reduce batch size if GPU memory error occurs")
    print("   - Use progressive training for better results")
    print("   - Enable gradient checkpointing if needed")
    print()


def main():
    """Main setup function"""
    print("🌾 CvT-PST Setup for Google Colab")
    print("=" * 50)
    
    # Setup steps
    steps = [
        ("Environment Setup", setup_environment),
        ("Repository Clone", clone_repository),
        ("Dataset Preparation", prepare_dataset),
        ("Pretrained Weights", download_pretrained_weights),
        ("Training Notebook", create_colab_training_notebook),
    ]
    
    success_count = 0
    for step_name, step_func in steps:
        print(f"\n📋 Step: {step_name}")
        if step_func():
            success_count += 1
        else:
            print(f"⚠️ {step_name} had issues, but continuing...")
    
    print(f"\n✅ Setup completed: {success_count}/{len(steps)} steps successful")
    
    # Display usage instructions
    display_usage_instructions()


if __name__ == "__main__":
    main()
