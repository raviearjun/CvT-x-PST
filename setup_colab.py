#!/usr/bin/env python3
"""
Setup script untuk membantu persiapan fine-tuning CvT di Google Colab
"""

import os
import shutil
import argparse
from pathlib import Path

def setup_directories():
    """Buat direktori yang diperlukan"""
    directories = [
        '/content/CvT/paddy_disease_dataset/train',
        '/content/CvT/paddy_disease_dataset/val',
        '/content/CvT/paddy_disease_dataset/test',
        '/content/output'
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")

def verify_dataset_exists():
    """Verifikasi apakah dataset sudah diisi"""
    dataset_root = '/content/CvT/paddy_disease_dataset'
    weights_file = '/content/CvT/CvT-21-224x224-IN-1k.pth'
    
    # Check dataset
    if not os.path.exists(dataset_root):
        print("❌ Dataset directory not found!")
        return False
        
    # Check if dataset is populated
    train_path = os.path.join(dataset_root, 'train')
    val_path = os.path.join(dataset_root, 'val')
    test_path = os.path.join(dataset_root, 'test')
    
    missing_dirs = []
    if not os.path.exists(train_path):
        missing_dirs.append('train')
    if not os.path.exists(val_path):
        missing_dirs.append('val')
    if not os.path.exists(test_path):
        missing_dirs.append('test')
        
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        print("   Harap isi direktori:")
        print("   - paddy_disease_dataset/train/ (80% data)")
        print("   - paddy_disease_dataset/val/ (10% data)")
        print("   - paddy_disease_dataset/test/ (10% data)")
        return False
    
    # Check if directories have content
    train_classes = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
    val_classes = [d for d in os.listdir(val_path) if os.path.isdir(os.path.join(val_path, d))]
    test_classes = [d for d in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, d))]
    
    if len(train_classes) == 0 or len(val_classes) == 0 or len(test_classes) == 0:
        print("❌ Dataset kosong! Harap isi dengan kelas-kelas penyakit padi.")
        print("   Expected: 10 classes (9 diseases + 1 normal)")
        print(f"   Found: train={len(train_classes)}, val={len(val_classes)}, test={len(test_classes)}")
        return False
    
    print(f"✓ Dataset found: {len(train_classes)} train, {len(val_classes)} val, {len(test_classes)} test classes")
    
    # Check weights
    if not os.path.exists(weights_file):
        print("❌ Pretrained weights not found!")
        print("   Please ensure: CvT-21-224x224-IN-1k.pth exists in /content/CvT/")
        return False
    
    print("✓ Pretrained weights found")
    return True

def verify_dataset_structure():
    """Verifikasi struktur dataset"""
    dataset_root = '/content/CvT/paddy_disease_dataset'
    
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(dataset_root, split)
        if not os.path.exists(split_path):
            print(f"❌ Missing {split} directory")
            continue
            
        classes = [d for d in os.listdir(split_path) 
                  if os.path.isdir(os.path.join(split_path, d))]
        
        print(f"\n{split.upper()} dataset:")
        print(f"  Classes found: {len(classes)}")
        
        total_images = 0
        for class_name in sorted(classes):
            class_path = os.path.join(split_path, class_name)
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"  {class_name}: {len(images)} images")
            total_images += len(images)
        
        print(f"  Total {split} images: {total_images}")

def check_requirements():
    """Check apakah semua requirements sudah terinstall"""
    import importlib
    
    required_packages = [
        'torch', 'torchvision', 'timm', 'yacs', 
        'tensorboard', 'cv2', 'PIL', 'numpy', 
        'matplotlib', 'sklearn', 'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                importlib.import_module('cv2')
            elif package == 'PIL':
                importlib.import_module('PIL')
            elif package == 'sklearn':
                importlib.import_module('sklearn')
            else:
                importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Run: !pip install -r requirements.txt")
    else:
        print("\n✓ All required packages are installed")

def main():
    parser = argparse.ArgumentParser(description='Setup CvT fine-tuning environment')
    parser.add_argument('--skip-check', action='store_true',
                       help='Skip dataset verification')
    
    args = parser.parse_args()
    
    print("🚀 Setting up CvT fine-tuning environment...")
    print("=" * 50)
    
    # Setup directories
    print("\n📁 Setting up directories...")
    setup_directories()
    
    # Verify dataset and weights
    if not args.skip_check:
        print("\n� Verifying dataset and weights...")
        success = verify_dataset_exists()
        if not success:
            print("\n❌ Setup verification failed!")
            print("\nPlease:")
            print("1. Download and place your dataset in: /content/CvT/paddy_disease_dataset/")
            print("2. Download pretrained weights: CvT-21-224x224-IN-1k.pth to /content/CvT/")
            print("3. Run this script again")
            return
    else:
        print("\n⚠️  Skipping dataset verification")
    
    # Verify dataset structure
    print("\n� Dataset structure:")
    verify_dataset_structure()
    
    # Check requirements
    print("\n📦 Checking requirements...")
    check_requirements()
    
    print("\n✅ Setup completed!")
    print("\nNext steps:")
    print("1. Verify dataset and weights are correctly placed")
    print("2. Run: python tools/train.py --cfg experiments/imagenet/cvt/cvt-21-224x224_paddy_dataset.yaml")

if __name__ == '__main__':
    main()
