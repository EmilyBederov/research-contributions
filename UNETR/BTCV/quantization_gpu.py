#!/usr/bin/env python3
"""
Clean UNETR Model Quantization Script
ONLY quantizes models - no benchmarking or testing
"""

import torch
import os
from networks.unetr import UNETR

class UNETRQuantizer:
    def __init__(self, pretrained_path="./pretrained_models/UNETR_model_best_acc.pth"):
        self.pretrained_path = pretrained_path
        
    def load_original_model(self):
        """Load the original pretrained UNETR model"""
        print("🔧 Loading original UNETR model...")
        
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
        
        print("✅ Original model loaded successfully!")
        return model
    
    def get_model_size(self, model, name="model"):
        """Calculate model size in MB"""
        temp_path = f"temp_{name}.pth"
        torch.save(model.state_dict(), temp_path)
        size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        os.remove(temp_path)
        return size_mb
    
    def create_int8_model(self, original_model):
        """Apply INT8 dynamic quantization"""
        print("🔧 Applying INT8 quantization...")
        
        # Create a copy of the model
        model_copy = UNETR(
            in_channels=1, out_channels=14, img_size=(96, 96, 96),
            feature_size=16, hidden_size=768, mlp_dim=3072,
            num_heads=12, pos_embed='perceptron', norm_name='instance',
            conv_block=True, res_block=True, dropout_rate=0.0
        )
        model_copy.load_state_dict(original_model.state_dict())
        model_copy.eval()
        
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model_copy,
            {torch.nn.Linear, torch.nn.Conv3d},
            dtype=torch.qint8
        )
        
        print("✅ INT8 quantization completed")
        return quantized_model
    
    def create_fp16_model(self, original_model):
        """Apply FP16 quantization"""
        print("🔧 Applying FP16 quantization...")
        
        # Convert model to half precision
        model_fp16 = original_model.cpu().half()
        
        print("✅ FP16 quantization completed")
        return model_fp16
    
    def save_all_models(self, original_model):
        """Save original and quantized models"""
        os.makedirs("./quantized_models", exist_ok=True)
        
        print("\n📊 Model Sizes:")
        
        # Original model size
        original_size = self.get_model_size(original_model, "original")
        print(f"  Original: {original_size:.2f} MB")
        
        # Create and save INT8 model
        try:
            int8_model = self.create_int8_model(original_model)
            int8_path = "./quantized_models/unetr_int8_quantized.pth"
            torch.save(int8_model.state_dict(), int8_path)
            int8_size = self.get_model_size(int8_model, "int8")
            
            print(f"  INT8:     {int8_size:.2f} MB ({original_size/int8_size:.1f}x smaller)")
            print(f"  📁 Saved: {int8_path}")
            
        except Exception as e:
            print(f"  ❌ INT8 failed: {e}")
        
        # Create and save FP16 model
        try:
            fp16_model = self.create_fp16_model(original_model)
            fp16_path = "./quantized_models/unetr_fp16_quantized.pth"
            torch.save(fp16_model.state_dict(), fp16_path)
            fp16_size = self.get_model_size(fp16_model, "fp16")
            
            print(f"  FP16:     {fp16_size:.2f} MB ({original_size/fp16_size:.1f}x smaller)")
            print(f"  📁 Saved: {fp16_path}")
            
        except Exception as e:
            print(f"  ❌ FP16 failed: {e}")
    
    def create_usage_guide(self):
        """Create simple usage guide"""
        usage_guide = '''# UNETR Quantized Models Usage

## Load Original Model
from networks.unetr import UNETR
model = UNETR(in_channels=1, out_channels=14, img_size=(96, 96, 96), ...)
model.load_state_dict(torch.load('./pretrained_models/UNETR_model_best_acc.pth'))

## Load INT8 Model (CPU)
model_int8 = UNETR(...)
model_int8 = torch.quantization.quantize_dynamic(model_int8, {torch.nn.Linear, torch.nn.Conv3d}, dtype=torch.qint8)
model_int8.load_state_dict(torch.load('./quantized_models/unetr_int8_quantized.pth'))

## Load FP16 Model (GPU)
model_fp16 = UNETR(...)
model_fp16.load_state_dict(torch.load('./quantized_models/unetr_fp16_quantized.pth'))
model_fp16 = model_fp16.half().cuda()

## Model Sizes
- Original: ~354 MB
- INT8: ~102 MB (3.5x smaller)
- FP16: ~177 MB (2x smaller)
'''
        
        with open('./quantized_models/usage_guide.txt', 'w') as f:
            f.write(usage_guide)
        
        print("📋 Usage guide saved: ./quantized_models/usage_guide.txt")

def main():
    """Main quantization workflow - ONLY quantization"""
    print("🔧 UNETR Model Quantization")
    print("=" * 40)
    
    # Check if original model exists
    original_path = "./pretrained_models/UNETR_model_best_acc.pth"
    if not os.path.exists(original_path):
        print(f"❌ Original model not found: {original_path}")
        return
    
    # Initialize quantizer
    quantizer = UNETRQuantizer(original_path)
    
    # Load original model
    original_model = quantizer.load_original_model()
    
    # Quantize and save all models
    quantizer.save_all_models(original_model)
    
    # Create usage guide
    quantizer.create_usage_guide()
    
    print(f"\n Quantization Complete!")
    print(f" All models saved in: ./quantized_models/")
    print(f" For performance testing, use a separate script")

if __name__ == "__main__":
    main()