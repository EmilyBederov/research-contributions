#!/usr/bin/env python3
"""
INT8 Quantization Deep Debugging Script
======================================
Systematically debug why INT8 model gives 0.0178 Dice score
"""

import torch
import torch.backends.quantized
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set quantization backend
try:
    torch.backends.quantized.engine = 'qnnpack'
    print("✅ Using QNNPACK quantization backend")
except:
    print("❌ Could not set QNNPACK backend")

def load_test_models():
    """Load all three models for comparison"""
    
    print("Loading models for debugging...")
    
    # You'll need to import your UNETR model here
    # from your_model_file import UNETR
    
    models = {}
    model_paths = {
        "original": "./pretrained_models/UNETR_model_best_acc.pth",
        "fp16": "./quantized_models/unetr_fp16_quantized.pth", 
        "int8": "./quantized_models/unetr_int8_quantized.pth"
    }
    
    for model_type, path in model_paths.items():
        if os.path.exists(path):
            try:
                # Create model (replace with your UNETR import)
                model = create_unetr_model()  # You'll need to define this
                
                # Load weights
                model_dict = torch.load(path, map_location='cpu')
                if isinstance(model_dict, dict) and 'state_dict' in model_dict:
                    state_dict = model_dict['state_dict']
                else:
                    state_dict = model_dict
                
                # Clean state dict
                clean_state_dict = {}
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    clean_state_dict[name] = v
                
                model.load_state_dict(clean_state_dict, strict=False)
                model.eval()
                
                # Apply quantization if needed
                if model_type == "int8":
                    model = torch.quantization.quantize_dynamic(
                        model, {torch.nn.Linear, torch.nn.Conv3d}, dtype=torch.qint8
                    )
                elif model_type == "fp16":
                    model = model.half()
                
                models[model_type] = model
                print(f"✅ Loaded {model_type} model")
                
            except Exception as e:
                print(f"❌ Failed to load {model_type}: {e}")
        else:
            print(f"❌ {model_type} model file not found")
    
    return models

def create_unetr_model():
    """Create UNETR model - replace with your actual model creation"""
    # You'll need to replace this with your actual UNETR model creation
    # For now, using a placeholder
    class PlaceholderModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(100, 14)
        
        def forward(self, x):
            return self.linear(x.view(x.size(0), -1))
    
    return PlaceholderModel()

def test_model_outputs_detailed(models):
    """Test model outputs with multiple test cases"""
    
    print("\n" + "="*60)
    print("DETAILED MODEL OUTPUT TESTING")
    print("="*60)
    
    # Create test inputs of different sizes
    test_cases = [
        ("small", torch.randn(1, 1, 32, 32, 32)),
        ("medium", torch.randn(1, 1, 64, 64, 64)),
        ("realistic", torch.randn(1, 1, 96, 96, 96))
    ]
    
    results = {}
    
    for test_name, test_input in test_cases:
        print(f"\nTesting with {test_name} input: {test_input.shape}")
        results[test_name] = {}
        
        for model_name, model in models.items():
            try:
                # Prepare input
                if model_name == "fp16":
                    test_input_model = test_input.half()
                else:
                    test_input_model = test_input.float()
                
                # Forward pass
                with torch.no_grad():
                    output = model(test_input_model)
                
                # Convert to float for analysis
                output_float = output.float()
                
                # Analyze output
                analysis = {
                    'shape': output.shape,
                    'dtype': output.dtype,
                    'min': float(output_float.min()),
                    'max': float(output_float.max()),
                    'mean': float(output_float.mean()),
                    'std': float(output_float.std()),
                    'has_nan': bool(torch.isnan(output_float).any()),
                    'has_inf': bool(torch.isinf(output_float).any()),
                    'all_zeros': bool(torch.all(output_float == 0)),
                    'num_zeros': int(torch.sum(output_float == 0))
                }
                
                results[test_name][model_name] = analysis
                
                print(f"  {model_name:>8}: shape={output.shape}, "
                      f"range=[{analysis['min']:.4f}, {analysis['max']:.4f}], "
                      f"mean={analysis['mean']:.4f}")
                
                # Flag problematic outputs
                if analysis['has_nan']:
                    print(f"    ❌ {model_name} has NaN values!")
                if analysis['has_inf']:
                    print(f"    ❌ {model_name} has Inf values!")
                if analysis['all_zeros']:
                    print(f"    ❌ {model_name} output is all zeros!")
                if abs(analysis['mean']) > 100:
                    print(f"    ⚠️  {model_name} has very large mean: {analysis['mean']:.2f}")
                
            except Exception as e:
                print(f"    ❌ {model_name} failed: {e}")
                results[test_name][model_name] = {'error': str(e)}
    
    return results

def compare_layer_outputs(models):
    """Compare intermediate layer outputs between models"""
    
    print("\n" + "="*60)
    print("INTERMEDIATE LAYER OUTPUT COMPARISON")
    print("="*60)
    
    if len(models) < 2:
        print("Need at least 2 models for comparison")
        return
    
    # Create test input
    test_input = torch.randn(1, 1, 32, 32, 32)
    
    # Hook to capture intermediate outputs
    intermediate_outputs = {}
    
    def hook_fn(name, model_name):
        def hook(module, input, output):
            key = f"{model_name}_{name}"
            intermediate_outputs[key] = output.detach().clone()
        return hook
    
    # Register hooks on first few layers
    hooks = []
    for model_name, model in models.items():
        layer_count = 0
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv3d)) and layer_count < 3:
                hook = module.register_forward_hook(hook_fn(name, model_name))
                hooks.append(hook)
                layer_count += 1
    
    # Run inference
    try:
        for model_name, model in models.items():
            if model_name == "fp16":
                input_model = test_input.half()
            else:
                input_model = test_input.float()
            
            with torch.no_grad():
                _ = model(input_model)
        
        # Analyze intermediate outputs
        print("Intermediate layer statistics:")
        for key, output in intermediate_outputs.items():
            output_float = output.float()
            print(f"  {key}: shape={output.shape}, "
                  f"mean={float(output_float.mean()):.6f}, "
                  f"std={float(output_float.std()):.6f}")
    
    except Exception as e:
        print(f"❌ Error in intermediate analysis: {e}")
    
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()

def test_quantization_ranges(models):
    """Test if quantization ranges are reasonable"""
    
    print("\n" + "="*60)
    print("QUANTIZATION RANGE ANALYSIS")
    print("="*60)
    
    if "int8" not in models:
        print("No INT8 model available")
        return
    
    int8_model = models["int8"]
    
    print("Analyzing INT8 quantization parameters:")
    
    quantized_layers = 0
    problematic_layers = 0
    
    for name, module in int8_model.named_modules():
        if hasattr(module, 'scale') and hasattr(module, 'zero_point'):
            quantized_layers += 1
            
            scale = float(module.scale)
            zero_point = int(module.zero_point)
            
            print(f"  {name}:")
            print(f"    Scale: {scale:.8f}")
            print(f"    Zero point: {zero_point}")
            
            # Check for problematic ranges
            if scale == 0:
                print(f"    ❌ Zero scale - will cause division by zero!")
                problematic_layers += 1
            elif scale < 1e-7:
                print(f"    ⚠️  Very small scale - may cause precision issues")
                problematic_layers += 1
            elif scale > 1e3:
                print(f"    ⚠️  Very large scale - may cause overflow")
                problematic_layers += 1
            
            if abs(zero_point) > 127:
                print(f"    ⚠️  Zero point outside INT8 range")
                problematic_layers += 1
    
    print(f"\nQuantization summary:")
    print(f"  Total quantized layers: {quantized_layers}")
    print(f"  Problematic layers: {problematic_layers}")
    
    if problematic_layers > 0:
        print(f"  ❌ Found {problematic_layers} layers with quantization issues!")
    else:
        print(f"  ✅ All quantization parameters look reasonable")

def test_simple_segmentation_task(models):
    """Test with a simple synthetic segmentation task"""
    
    print("\n" + "="*60)
    print("SYNTHETIC SEGMENTATION TEST")
    print("="*60)
    
    # Create simple synthetic data
    # Input: small 3D volume with clear structure
    test_input = torch.zeros(1, 1, 32, 32, 32)
    test_input[:, :, 8:24, 8:24, 8:24] = 1.0  # Central cube
    
    # Expected: should produce some meaningful segmentation
    print("Testing with synthetic cube (central region = 1.0)...")
    
    for model_name, model in models.items():
        try:
            if model_name == "fp16":
                input_model = test_input.half()
            else:
                input_model = test_input.float()
            
            with torch.no_grad():
                output = model(input_model)
            
            output_float = output.float()
            
            # Apply softmax to get probabilities
            probs = torch.softmax(output_float, dim=1)
            pred = torch.argmax(probs, dim=1)
            
            # Analyze prediction
            unique_labels = torch.unique(pred)
            label_counts = [(int(label), int(torch.sum(pred == label))) for label in unique_labels]
            
            print(f"  {model_name}:")
            print(f"    Output range: [{float(output_float.min()):.4f}, {float(output_float.max()):.4f}]")
            print(f"    Predicted labels: {label_counts}")
            
            # Check if prediction makes sense
            if len(unique_labels) == 1:
                print(f"    ⚠️  Only predicting single label: {int(unique_labels[0])}")
            elif len(unique_labels) > 10:
                print(f"    ⚠️  Too many predicted labels: {len(unique_labels)}")
            else:
                print(f"    ✅ Reasonable number of labels: {len(unique_labels)}")
        
        except Exception as e:
            print(f"  ❌ {model_name} failed: {e}")

def visualize_output_distributions(results):
    """Create visualizations of output distributions"""
    
    print("\n" + "="*60)
    print("CREATING OUTPUT DISTRIBUTION PLOTS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Model Output Analysis', fontsize=16, fontweight='bold')
    
    # Extract data for plotting
    models = []
    means = []
    stds = []
    ranges = []
    
    for test_name, test_results in results.items():
        if test_name == "small":  # Use small test case
            for model_name, analysis in test_results.items():
                if 'error' not in analysis:
                    models.append(model_name)
                    means.append(analysis['mean'])
                    stds.append(analysis['std'])
                    ranges.append(analysis['max'] - analysis['min'])
    
    if len(models) >= 2:
        # Plot 1: Mean values
        axes[0, 0].bar(models, means, alpha=0.7)
        axes[0, 0].set_title('Output Mean Values')
        axes[0, 0].set_ylabel('Mean')
        
        # Plot 2: Standard deviation
        axes[0, 1].bar(models, stds, alpha=0.7)
        axes[0, 1].set_title('Output Standard Deviation')
        axes[0, 1].set_ylabel('Std Dev')
        
        # Plot 3: Value ranges
        axes[1, 0].bar(models, ranges, alpha=0.7)
        axes[1, 0].set_title('Output Value Range')
        axes[1, 0].set_ylabel('Max - Min')
        
        # Plot 4: Summary comparison
        axes[1, 1].text(0.1, 0.7, f"Models compared: {len(models)}", fontsize=12)
        axes[1, 1].text(0.1, 0.5, f"Mean range: {min(means):.3f} to {max(means):.3f}", fontsize=10)
        axes[1, 1].text(0.1, 0.3, f"Std range: {min(stds):.3f} to {max(stds):.3f}", fontsize=10)
        axes[1, 1].set_title('Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('int8_debugging_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Saved analysis plots to 'int8_debugging_analysis.png'")
        plt.show()
    else:
        print("❌ Not enough model results for plotting")

def main_debug_workflow():
    """Run complete debugging workflow"""
    
    print("INT8 QUANTIZATION DEBUGGING WORKFLOW")
    print("="*60)
    print("This will help identify why INT8 gives 0.0178 Dice score")
    print("="*60)
    
    # Step 1: Load models
    print("\n🔄 STEP 1: Loading models...")
    models = load_test_models()
    
    if len(models) == 0:
        print("❌ No models loaded! Cannot proceed with debugging.")
        return
    
    print(f"✅ Loaded {len(models)} models: {list(models.keys())}")
    
    # Step 2: Test basic outputs
    print("\n🔄 STEP 2: Testing model outputs...")
    results = test_model_outputs_detailed(models)
    
    # Step 3: Compare intermediate layers
    print("\n🔄 STEP 3: Comparing intermediate layers...")
    compare_layer_outputs(models)
    
    # Step 4: Analyze quantization ranges
    print("\n🔄 STEP 4: Analyzing quantization parameters...")
    test_quantization_ranges(models)
    
    # Step 5: Test synthetic segmentation
    print("\n🔄 STEP 5: Testing synthetic segmentation...")
    test_simple_segmentation_task(models)
    
    # Step 6: Create visualizations
    print("\n🔄 STEP 6: Creating visualizations...")
    visualize_output_distributions(results)
    
    # Step 7: Final recommendations
    print("\n" + "="*60)
    print("DEBUGGING SUMMARY & RECOMMENDATIONS")
    print("="*60)
    
    if "int8" in models:
        print("Based on the analysis above:")
        print("1. Check for NaN/Inf values in INT8 outputs")
        print("2. Look for problematic quantization scales/zero_points")
        print("3. Compare output distributions between models")
        print("4. Consider using static quantization instead of dynamic")
        print("5. Try different quantization backends if available")
        print("\nIf INT8 outputs look completely wrong, the issue is likely:")
        print("- Poor quantization calibration")
        print("- MacOS QNNPACK compatibility problems")
        print("- Numerical instability in the quantized model")
    else:
        print("❌ INT8 model not available for debugging")

# INSTRUCTIONS FOR RUNNING:
print("""
INSTRUCTIONS TO RUN THIS DEBUGGING SCRIPT:
=========================================

1. First, you need to modify the script to import your actual UNETR model:
   - Replace 'create_unetr_model()' with your actual UNETR model creation
   - Import the necessary modules (UNETR, etc.)

2. Make sure your model files exist:
   - ./pretrained_models/UNETR_model_best_acc.pth
   - ./quantized_models/unetr_fp16_quantized.pth
   - ./quantized_models/unetr_int8_quantized.pth

3. Run the debugging workflow:
   main_debug_workflow()

This will systematically test your models and identify the INT8 issue.
""")

# Uncomment this line after you've set up the UNETR model import:
# main_debug_workflow()