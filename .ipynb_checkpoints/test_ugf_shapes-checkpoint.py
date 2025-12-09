#!/usr/bin/env python3
"""Quick test to verify UGF model output shapes"""

import torch
import sys
sys.path.insert(0, '/home/jupyter-228w1a12b8/dec_7/IDD_AW/IDD_AW/IDDAW')

from models_LH import UncertaintyGatedFusionModel

print("="*80)
print("TESTING UGF-NET OUTPUT SHAPES")
print("="*80)

# Create model
model = UncertaintyGatedFusionModel(num_classes=30)
model.eval()

# Test inputs
batch_size = 2
img_size = (512, 1024)  # H, W
rgb = torch.randn(batch_size, 3, img_size[0], img_size[1])
nir = torch.randn(batch_size, 1, img_size[0], img_size[1])

print(f"\nInput shapes:")
print(f"  RGB: {rgb.shape}")
print(f"  NIR: {nir.shape}")

# Test training mode
print("\n" + "-"*80)
print("TRAINING MODE")
print("-"*80)
model.train()
output, rgb_aux, nir_aux, rgb_gate, nir_gate = model(rgb, nir)

print(f"Outputs:")
print(f"  Main output: {output.shape}")
print(f"  RGB aux: {rgb_aux.shape}")
print(f"  NIR aux: {nir_aux.shape}")
print(f"  RGB gate: {rgb_gate.shape}")
print(f"  NIR gate: {nir_gate.shape}")

# Verify shapes
expected_output = (batch_size, 30, img_size[0], img_size[1])
assert output.shape == expected_output, f"❌ Main output shape wrong! Expected {expected_output}, got {output.shape}"
assert rgb_aux.shape == expected_output, f"❌ RGB aux shape wrong! Expected {expected_output}, got {rgb_aux.shape}"
assert nir_aux.shape == expected_output, f"❌ NIR aux shape wrong! Expected {expected_output}, got {nir_aux.shape}"

print("\n✓ All training mode shapes correct!")

# Test inference mode
print("\n" + "-"*80)
print("INFERENCE MODE")
print("-"*80)
model.eval()
with torch.no_grad():
    output, rgb_gate, nir_gate = model(rgb, nir)

print(f"Outputs:")
print(f"  Main output: {output.shape}")
print(f"  RGB gate: {rgb_gate.shape}")
print(f"  NIR gate: {nir_gate.shape}")

assert output.shape == expected_output, f"❌ Output shape wrong! Expected {expected_output}, got {output.shape}"

print("\n✓ All inference mode shapes correct!")

# Model info
params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"\nModel parameters: {params:.2f}M")

print("\n" + "="*80)
print("✓ ALL TESTS PASSED!")
print("="*80)