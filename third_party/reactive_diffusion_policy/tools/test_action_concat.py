#!/usr/bin/env python3
"""
Test script to verify action concatenation with tactile embedding
"""
import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from omegaconf import OmegaConf
import hydra
from hydra import initialize, compose

OmegaConf.register_new_resolver("eval", eval, replace=True)

def test_action_shape():
    print("=" * 60)
    print("Testing Action Concatenation with Tactile Embedding")
    print("=" * 60)
    
    with initialize(config_path="reactive_diffusion_policy/config"):
        cfg = compose(config_name='train_diffusion_unet_real_image_workspace',
                     overrides=['task=kinedex'])
        OmegaConf.resolve(cfg)
        
        print("\n1. Config loaded successfully")
        print(f"   Action shape from config: {cfg.shape_meta.action.shape}")
        
        # Instantiate dataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        print("\n2. Dataset instantiated successfully")
        print(f"   Dataset length: {len(dataset)}")
        
        # Get first sample
        sample = dataset[0]
        print("\n3. Sample retrieved successfully")
        print(f"   Sample keys: {list(sample.keys())}")
        print(f"   Action shape: {sample['action'].shape}")
        print(f"   Expected shape: torch.Size([horizon, 25])")
        
        # Verify action shape
        expected_action_dim = 25  # 10 (base) + 15 (tactile)
        actual_action_dim = sample['action'].shape[-1]
        
        if actual_action_dim == expected_action_dim:
            print(f"\n✓ SUCCESS: Action dimension is {actual_action_dim} (as expected)")
        else:
            print(f"\n✗ FAILURE: Action dimension is {actual_action_dim}, expected {expected_action_dim}")
            return False
        
        # Check if quantization fields exist (if enabled)
        if 'action_quantized' in sample:
            print(f"\n4. Quantization enabled")
            print(f"   action_quantized shape: {sample['action_quantized'].shape}")
            print(f"   action_dequantized shape: {sample['action_dequantized'].shape}")
            print(f"   Quantized value range: [{sample['action_quantized'].min().item():.0f}, {sample['action_quantized'].max().item():.0f}]")
        
        # Get normalizer
        print("\n5. Testing normalizer...")
        normalizer = dataset.get_normalizer()
        action_normalizer = normalizer['action']
        print(f"   Action normalizer input_stats shape: {action_normalizer.input_stats['min'].shape}")
        print(f"   Expected: (25,)")
        
        if action_normalizer.input_stats['min'].shape[0] == expected_action_dim:
            print(f"\n✓ SUCCESS: Normalizer handles {expected_action_dim}-dim action correctly")
        else:
            print(f"\n✗ FAILURE: Normalizer dimension mismatch")
            return False
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        return True

if __name__ == '__main__':
    try:
        success = test_action_shape()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
