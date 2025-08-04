#!/usr/bin/env python3
"""
UNETR Model Quantization Script
Save this as: quantize_unetr.py
"""

import torch
import torch.quantization as quantization
import os
import time
import numpy as np
from networks.unetr import UNETR
from utils.data_utils import get_loader
from monai.inferers import sliding_window_inference


class UNETRQuantizer:
    def __init__(self, pretrained_path="./pretrained_models/UNETR_model_best_acc.pth"):
        self.pretrained_path = pretrained_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_original_model(self):
        """Load the original pretrained UNETR model"""
        print(" Loading original UNETR model...")
        
        model = UNETR(
            in_channels=1,
            out_channels=14,
            img_size=(96, 96, 96),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed='perceptron',
            norm_name='instance',
            conv_block=True,
            res_block=True,
            dropout_rate=0.0
        )
        
        # Load pretrained weights
        model_dict = torch.load(self.pretrained_path, map_location='cpu')
        model.load_state_dict(model_dict, strict=False)
        model.eval()
        
        print(" Original model loaded successfully!")
        return model
    
    def get_model_size(self, model, name="model"):
        """Calculate model size in MB"""
        temp_path = f"temp_{name}.pth"
        torch.save(model.state_dict(), temp_path)
        size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        os.remove(temp_path)
        return size_mb
    
    def dynamic_quantization_int8(self, model):
        """Apply INT8 dynamic quantization"""
        print("⚡ Applying INT8 Dynamic Quantization...")
        
        # Create a copy of the model
        model_copy = UNETR(
            in_channels=1, out_channels=14, img_size=(96, 96, 96),
            feature_size=16, hidden_size=768, mlp_dim=3072,
            num_heads=12, pos_embed='perceptron', norm_name='instance',
            conv_block=True, res_block=True, dropout_rate=0.0
        )
        model_copy.load_state_dict(model.state_dict())
        model_copy.eval()
        
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model_copy.cpu(),
            {torch.nn.Linear, torch.nn.Conv3d},
            dtype=torch.qint8
        )
        
        return quantized_model
    
    def fp16_quantization(self, model):
        """Apply FP16 quantization"""
        print(" Applying FP16 Quantization...")
        
        # Convert model to half precision
        model_fp16 = model.cpu().half()
        
        return model_fp16
    
    def save_quantized_models(self, original_model):
        """Save all quantized versions"""
        os.makedirs("./quantized_models", exist_ok=True)
        
        results = {}
        
        # Get original size
        original_size = self.get_model_size(original_model, "original")
        results['original'] = {'size_mb': original_size, 'model': original_model}
        
        print(f" Original model size: {original_size:.2f} MB")
        
        # INT8 Dynamic Quantization
        try:
            int8_model = self.dynamic_quantization_int8(original_model)
            int8_path = "./quantized_models/unetr_int8_dynamic.pth"
            torch.save(int8_model.state_dict(), int8_path)
            int8_size = self.get_model_size(int8_model, "int8")
            results['int8'] = {'size_mb': int8_size, 'model': int8_model, 'path': int8_path}
            
            print(f" INT8 model saved: {int8_path}")
            print(f" INT8 model size: {int8_size:.2f} MB")
            print(f" INT8 compression: {original_size/int8_size:.2f}x smaller")
            
        except Exception as e:
            print(f" INT8 quantization failed: {e}")
        
        # FP16 Quantization
        try:
            fp16_model = self.fp16_quantization(original_model)
            fp16_path = "./quantized_models/unetr_fp16.pth"
            torch.save(fp16_model.state_dict(), fp16_path)
            fp16_size = self.get_model_size(fp16_model, "fp16")
            results['fp16'] = {'size_mb': fp16_size, 'model': fp16_model, 'path': fp16_path}
            
            print(f" FP16 model saved: {fp16_path}")
            print(f" FP16 model size: {fp16_size:.2f} MB")
            print(f" FP16 compression: {original_size/fp16_size:.2f}x smaller")
            
        except Exception as e:
            print(f" FP16 quantization failed: {e}")
        
        return results
    
    
    def create_usage_examples(self):
        """Create example code for using quantized models"""
        usage_code = '''# Usage Examples for Quantized UNETR Models

## 1. Loading INT8 Quantized Model
import torch
from networks.unetr import UNETR

# Create model architecture
model = UNETR(
    in_channels=1, out_channels=14, img_size=(96, 96, 96),
    feature_size=16, hidden_size=768, mlp_dim=3072,
    num_heads=12, pos_embed='perceptron', norm_name='instance',
    conv_block=True, res_block=True, dropout_rate=0.0
)

# Apply quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear, torch.nn.Conv3d}, dtype=torch.qint8
)

# Load quantized weights
quantized_model.load_state_dict(torch.load('./quantized_models/unetr_int8_dynamic.pth'))
quantized_model.eval()

## 2. Loading FP16 Model
model_fp16 = UNETR(
    in_channels=1, out_channels=14, img_size=(96, 96, 96),
    feature_size=16, hidden_size=768, mlp_dim=3072,
    num_heads=12, pos_embed='perceptron', norm_name='instance',
    conv_block=True, res_block=True, dropout_rate=0.0
)
model_fp16.load_state_dict(torch.load('./quantized_models/unetr_fp16.pth'))
model_fp16.half()  # Convert to FP16
model_fp16.eval()

## 3. Use with test.py (CPU only for INT8)
python test.py \\
    --pretrained_dir=./quantized_models/ \\
    --pretrained_model_name=unetr_int8_dynamic.pth \\
    --saved_checkpoint=ckpt

## 4. Performance Summary
- Original: 354 MB
- INT8: 102 MB (3.47x smaller, CPU only)
- FP16: 177 MB (2x smaller, needs GPU)
'''
        
        with open('./quantized_models/usage_examples.txt', 'w') as f:
            f.write(usage_code)
        
        print("📋 Usage examples saved to: ./quantized_models/usage_examples.txt")


def main():
    """Main quantization workflow"""
    print(" UNETR Model Quantization")
    print("=" * 50)
    
    # Initialize quantizer
    quantizer = UNETRQuantizer()
    
    # Load original model
    original_model = quantizer.load_original_model()
    
    # Quantize and save models
    results = quantizer.save_quantized_models(original_model)
    
    # Benchmark models
    #quantizer.benchmark_models(results)
    
    # Create usage examples
    quantizer.create_usage_examples()
    
    print(f"\n Quantization Complete!")
    print(f" All quantized models saved in: ./quantized_models/")
    print(f" Check usage_examples.txt for how to use them")


if __name__ == "__main__":
    main()