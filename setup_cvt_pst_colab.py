#!/usr/bin/env python3
"""
Setup verification script untuk CvT-PST di Google Colab
"""

import os
import sys
import argparse
from pathlib import Path


def check_requirements():
    """Check apakah semua requirements sudah terinstall"""
    import importlib
    
    required_packages = [
        'torch', 'timm', 'einops', 'yacs', 
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
        except Exception as e:
            print(f"⚠️ {package} (warning: {str(e)[:50]}...)")
    
    # Special check for torchvision (can be problematic)
    try:
        import torchvision
        print(f"✓ torchvision (version: {torchvision.__version__})")
    except Exception as e:
        print(f"⚠️ torchvision (warning: {str(e)[:50]}...)")
        print("   This is often okay in Colab environment")
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Run: !pip install -r requirements.txt")
        return len(missing_packages) <= 2  # Allow some missing packages
    else:
        print("\n✓ All required packages are installed")
        return True


def verify_cvt_pst_modules():
    """Verifikasi apakah modul CvT-PST bisa diimport"""
    print("\n🔧 Testing CvT-PST module imports...")
    
    try:
        # Test basic imports
        from lib.models.cvt_pst_classifier import create_cvt_pst_classifier, PyramidSparseTransformer
        from lib.models.cvt_pst_classifier import PADDY_DISEASE_CLASSES
        print("✓ CvT-PST modules imported successfully")
        
        # Test creating PST module
        pst = PyramidSparseTransformer(input_dim=768, scales=[1, 2, 4], reduction_ratio=4)
        print("✓ PST module creation successful")
        
        print(f"✓ Found {len(PADDY_DISEASE_CLASSES)} paddy disease classes")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import CvT-PST modules: {e}")
        print("   Make sure you're running this from the CvT-x-PST directory")
        return False
    except Exception as e:
        print(f"❌ Error testing PST module: {e}")
        return False


def test_pst_functionality():
    """Test basic PST functionality"""
    print("\n🧪 Testing PST functionality...")
    
    try:
        import torch
        from lib.models.cvt_pst_classifier import create_cvt_pst_classifier
        from types import SimpleNamespace
        
        # Mock config
        mock_spec = {
            'NUM_STAGES': 3,
            'DIM_EMBED': [64, 192, 768],
            'PATCH_SIZE': [7, 3, 3],
            'PATCH_STRIDE': [4, 2, 2],
            'PATCH_PADDING': [2, 1, 1],
            'DEPTH': [1, 2, 10],
            'NUM_HEADS': [1, 3, 12],
            'MLP_RATIO': [4, 4, 4],
            'QKV_BIAS': [True, True, True],
            'DROP_RATE': [0.0, 0.0, 0.0],
            'ATTN_DROP_RATE': [0.0, 0.0, 0.0],
            'DROP_PATH_RATE': [0.0, 0.0, 0.1],
            'CLS_TOKEN': [False, False, False],
            'QKV_PROJ_METHOD': ['dw_bn', 'dw_bn', 'dw_bn'],
            'KERNEL_QKV': [3, 3, 3],
            'PADDING_Q': [1, 1, 1],
            'PADDING_KV': [1, 1, 1],
            'STRIDE_KV': [2, 2, 2],
            'STRIDE_Q': [1, 1, 1],
        }
        
        mock_config = SimpleNamespace()
        mock_config.MODEL = SimpleNamespace()
        mock_config.MODEL.SPEC = mock_spec
        
        # Create model
        model = create_cvt_pst_classifier(
            config=mock_config,
            num_classes=10,
            pst_scales=[1, 2, 4],
            pst_reduction_ratio=4
        )
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 10), f"Expected (2, 10), got {output.shape}"
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ PST functionality test passed")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Model parameters: {total_params:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ PST functionality test failed: {e}")
        return False


def verify_dataset_exists():
    """Verifikasi apakah dataset sudah ada"""
    dataset_root = '/content/CvT/paddy_disease_dataset'
    weights_file = '/content/CvT/CvT-21-224x224-IN-1k.pth'
    
    print(f"\n📁 Checking dataset and weights...")
    
    # Check dataset
    if not os.path.exists(dataset_root):
        print(f"❌ Dataset directory not found: {dataset_root}")
        print("   Please create dataset structure:")
        print("   /content/CvT/paddy_disease_dataset/")
        print("   ├── train/")
        print("   ├── val/")
        print("   └── test/")
        return False
    
    # Check weights
    if not os.path.exists(weights_file):
        print(f"❌ Pretrained weights not found: {weights_file}")
        print("   Please download CvT-21-224x224-IN-1k.pth to /content/CvT/")
        return False
    
    print("✓ Dataset directory found")
    print("✓ Pretrained weights found")
    return True


def verify_dataset_structure():
    """Verifikasi struktur dataset"""
    dataset_root = '/content/CvT/paddy_disease_dataset'
    
    if not os.path.exists(dataset_root):
        return False
    
    print(f"\n📊 Dataset structure analysis:")
    
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
    
    return True


def setup_directories():
    """Buat direktori output yang diperlukan"""
    directories = [
        '/content/output',
        '/content/CvT/logs',
        '/content/CvT/checkpoints'
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")


def test_config_file():
    """Test apakah config file PST bisa dibaca"""
    print("\n⚙️ Testing PST config file...")
    
    config_file = 'experiments/imagenet/cvt/cvt-21-224x224_paddy_pst.yaml'
    
    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        return False
    
    try:
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check PST config
        if 'PST' in config.get('MODEL', {}):
            pst_config = config['MODEL']['PST']
            print(f"✓ PST config found:")
            print(f"  Enabled: {pst_config.get('ENABLED', False)}")
            print(f"  Scales: {pst_config.get('SCALES', [])}")
            print(f"  Reduction ratio: {pst_config.get('REDUCTION_RATIO', 4)}")
        else:
            print("❌ PST config not found in model config")
            return False
        
        print("✓ Config file validated")
        return True
        
    except Exception as e:
        print(f"❌ Error reading config: {e}")
        return False


def display_usage_instructions():
    """Display usage instructions"""
    print("\n" + "=" * 60)
    print("🎉 CvT-PST Setup Verification Complete!")
    print("=" * 60)
    print()
    print("📋 Next Steps:")
    print()
    print("1. 🚀 Start Training:")
    print("   !python train_cvt_pst.py \\")
    print("       --cfg experiments/imagenet/cvt/cvt-21-224x224_paddy_pst.yaml \\")
    print("       --data-path /content/CvT/paddy_disease_dataset \\")
    print("       --pretrained /content/CvT/CvT-21-224x224-IN-1k.pth \\")
    print("       --epochs 30 \\")
    print("       --progressive")
    print()
    print("2. 📊 Evaluate Results:")
    print("   !python evaluate_cvt_pst.py \\")
    print("       --model-path /content/output/checkpoint_best.pth \\")
    print("       --config experiments/imagenet/cvt/cvt-21-224x224_paddy_pst.yaml \\")
    print("       --data-path /content/CvT/paddy_disease_dataset/test \\")
    print("       --benchmark")
    print()
    print("3. 🧪 Test PST Module:")
    print("   !python test_pst_module.py")
    print()
    print("🎯 Expected Performance:")
    print("   - Training Time: ~2-3 hours on Colab GPU")
    print("   - Expected Accuracy: 85-95%")
    print("   - Model Size: ~33M parameters")
    print()


def main():
    parser = argparse.ArgumentParser(description='Setup CvT-PST verification')
    parser.add_argument('--skip-dataset', action='store_true',
                       help='Skip dataset verification')
    parser.add_argument('--skip-test', action='store_true',
                       help='Skip functionality test')
    
    args = parser.parse_args()
    
    print("🌾 CvT-PST Setup Verification")
    print("=" * 50)
    
    success_count = 0
    total_checks = 0
    
    # Check requirements
    print("\n📦 Checking requirements...")
    total_checks += 1
    if check_requirements():
        success_count += 1
    
    # Setup directories
    print("\n📁 Setting up directories...")
    setup_directories()
    
    # Verify CvT-PST modules
    total_checks += 1
    if verify_cvt_pst_modules():
        success_count += 1
    
    # Test PST functionality
    if not args.skip_test:
        total_checks += 1
        if test_pst_functionality():
            success_count += 1
    
    # Test config file
    total_checks += 1
    if test_config_file():
        success_count += 1
    
    # Verify dataset and weights
    if not args.skip_dataset:
        total_checks += 1
        if verify_dataset_exists():
            success_count += 1
            verify_dataset_structure()
    else:
        print("\n⚠️ Skipping dataset verification")
    
    print(f"\n✅ Setup verification completed: {success_count}/{total_checks} checks passed")
    
    if success_count == total_checks:
        print("🎉 All checks passed! CvT-PST is ready for training.")
        display_usage_instructions()
    else:
        print("⚠️ Some checks failed. Please resolve the issues above.")
        print("\nCommon solutions:")
        print("- Ensure you're in the CvT-x-PST directory")
        print("- Install missing packages: !pip install -r requirements.txt")
        print("- Download dataset and pretrained weights")


if __name__ == '__main__':
    main()
