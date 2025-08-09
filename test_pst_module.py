"""
Simple Test Script for CvT-PST Module
=====================================

This script provides basic testing and demonstration of the PST module
functionality without requiring the full training setup.

Run this to verify PST implementation works correctly.

Author: CvT-x-PST Project
"""

import torch
import torch.nn as nn
import sys
import os
from types import SimpleNamespace

# Add lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

def test_pst_module():
    """Test PST module independently"""
    print("🧪 Testing PST Module...")
    
    try:
        from models.cvt_pst_classifier import PyramidSparseTransformer
        
        # Test parameters
        input_dim = 768
        batch_size = 2
        height, width = 14, 14
        
        # Create PST module
        pst = PyramidSparseTransformer(
            input_dim=input_dim,
            scales=[1, 2, 4, 8],
            reduction_ratio=4,
            dropout=0.1
        )
        
        # Test input
        x = torch.randn(batch_size, input_dim, height, width)
        
        # Forward pass
        with torch.no_grad():
            output = pst(x)
        
        # Check output shape
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
        
        # Check parameters
        total_params = sum(p.numel() for p in pst.parameters())
        
        print(f"✅ PST Module Test Passed!")
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Parameters: {total_params:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ PST Module Test Failed: {e}")
        return False


def test_cvt_pst_classifier():
    """Test full CvT-PST classifier"""
    print("\n🧪 Testing CvT-PST Classifier...")
    
    try:
        from models.cvt_pst_classifier import create_cvt_pst_classifier
        
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
            pst_scales=[1, 2, 4, 8],
            pst_reduction_ratio=4,
            pst_dropout=0.1
        )
        
        # Test forward pass
        batch_size = 2
        x = torch.randn(batch_size, 3, 224, 224)
        
        with torch.no_grad():
            output = model(x)
        
        # Check output shape
        expected_shape = (batch_size, 10)
        assert output.shape == expected_shape, f"Shape mismatch: {output.shape} vs {expected_shape}"
        
        # Check parameters
        total_params = sum(p.numel() for p in model.parameters())
        backbone_params = sum(p.numel() for p in model.backbone.parameters())
        pst_params = sum(p.numel() for p in model.pst_module.parameters())
        classifier_params = sum(p.numel() for p in model.classifier.parameters())
        
        print(f"✅ CvT-PST Classifier Test Passed!")
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Backbone parameters: {backbone_params:,}")
        print(f"   PST parameters: {pst_params:,}")
        print(f"   Classifier parameters: {classifier_params:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ CvT-PST Classifier Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_extraction():
    """Test feature extraction capability"""
    print("\n🧪 Testing Feature Extraction...")
    
    try:
        from models.cvt_pst_classifier import create_cvt_pst_classifier
        
        # Mock config (simplified)
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
            pst_scales=[1, 2, 4],  # Reduced for faster testing
            pst_reduction_ratio=4
        )
        
        # Test feature extraction
        x = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            # Test feature extraction
            features = model.forward_features(x)
            
            # Test feature maps
            feature_maps = model.get_feature_maps(x)
        
        print(f"✅ Feature Extraction Test Passed!")
        print(f"   Final features shape: {features.shape}")
        print(f"   Feature map stages: {list(feature_maps.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature Extraction Test Failed: {e}")
        return False


def test_freeze_unfreeze():
    """Test backbone freezing functionality"""
    print("\n🧪 Testing Freeze/Unfreeze...")
    
    try:
        from models.cvt_pst_classifier import create_cvt_pst_classifier
        
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
            num_classes=10
        )
        
        # Test freeze
        model.freeze_backbone()
        backbone_frozen = not any(p.requires_grad for p in model.backbone.parameters())
        pst_trainable = any(p.requires_grad for p in model.pst_module.parameters())
        classifier_trainable = any(p.requires_grad for p in model.classifier.parameters())
        
        assert backbone_frozen, "Backbone should be frozen"
        assert pst_trainable, "PST should be trainable"
        assert classifier_trainable, "Classifier should be trainable"
        
        # Test unfreeze
        model.unfreeze_backbone()
        backbone_trainable = any(p.requires_grad for p in model.backbone.parameters())
        
        assert backbone_trainable, "Backbone should be trainable after unfreeze"
        
        print(f"✅ Freeze/Unfreeze Test Passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Freeze/Unfreeze Test Failed: {e}")
        return False


def run_performance_benchmark():
    """Simple performance benchmark"""
    print("\n⏱️ Running Performance Benchmark...")
    
    try:
        from models.cvt_pst_classifier import create_cvt_pst_classifier
        import time
        
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
            pst_scales=[1, 2, 4]  # Reduced for benchmark
        )
        
        model.eval()
        
        # Benchmark parameters
        batch_size = 4
        num_runs = 20
        
        x = torch.randn(batch_size, 3, 224, 224)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(x)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(x)
                end_time = time.time()
                times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        avg_per_sample = avg_time / batch_size
        
        print(f"✅ Performance Benchmark Completed!")
        print(f"   Average batch time: {avg_time*1000:.2f} ms")
        print(f"   Average per-sample time: {avg_per_sample*1000:.2f} ms")
        print(f"   Throughput: {1/avg_per_sample:.1f} samples/second")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance Benchmark Failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 CvT-PST Module Testing")
    print("=" * 50)
    
    tests = [
        ("PST Module", test_pst_module),
        ("CvT-PST Classifier", test_cvt_pst_classifier),
        ("Feature Extraction", test_feature_extraction),
        ("Freeze/Unfreeze", test_freeze_unfreeze),
        ("Performance Benchmark", run_performance_benchmark),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} Test...")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! CvT-PST implementation is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == len(tests)


if __name__ == "__main__":
    main()
