#!/usr/bin/env python3
"""
CPU-Compatible version of test.py for UNETR
This version fixes all GPU/CPU compatibility issues
"""

import argparse
import os
import numpy as np
import torch
from networks.unetr import UNETR
from trainer import dice
from utils.data_utils import get_loader
from monai.inferers import sliding_window_inference

parser = argparse.ArgumentParser(description="UNETR segmentation pipeline")
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="/dataset/dataset0/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name", default="UNETR_model_best_acc.pth", type=str, help="pretrained model name"
)
parser.add_argument("--mlp_dim", default=3072, type=int, help="mlp dimention in ViT encoder")
parser.add_argument("--hidden_size", default=768, type=int, help="hidden size dimention in ViT encoder")
parser.add_argument("--feature_size", default=16, type=int, help="feature size dimention")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
parser.add_argument("--num_heads", default=12, type=int, help="number of attention heads in ViT encoder")
parser.add_argument("--res_block", action="store_true", help="use residual blocks")
parser.add_argument("--conv_block", action="store_true", help="use conv blocks")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--pos_embed", default="perceptron", type=str, help="type of position embedding")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")

def load_pretrained_compatible(model, checkpoint_path):
    """Load pretrained model with CPU compatibility and key filtering"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Load checkpoint to CPU
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DataParallel)
    clean_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        clean_state_dict[name] = v
    
    # Get model's expected keys
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(clean_state_dict.keys())
    
    # Find compatible keys
    compatible_keys = model_keys.intersection(checkpoint_keys)
    incompatible_keys = checkpoint_keys - model_keys
    missing_keys = model_keys - checkpoint_keys
    
    print(f"✅ Compatible keys: {len(compatible_keys)}/{len(model_keys)}")
    
    if incompatible_keys:
        print(f"⚠️ Skipping {len(incompatible_keys)} incompatible keys")
    
    if missing_keys:
        print(f"⚠️ {len(missing_keys)} keys will use default initialization")
    
    # Create filtered state dict
    filtered_state_dict = {k: v for k, v in clean_state_dict.items() if k in compatible_keys}
    
    # Load with strict=False to handle missing/extra keys
    model.load_state_dict(filtered_state_dict, strict=False)
    print("✅ Model loaded successfully!")
    
    return model

def main():
    args = parser.parse_args()
    args.test_mode = True
    
    # Force CPU usage
    device = torch.device('cpu')
    print(f"🖥️ Using device: {device}")
    
    # Load data
    print("📁 Loading dataset...")
    val_loader = get_loader(args)
    
    # Setup model
    print("🧠 Initializing UNETR model...")
    model = UNETR(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        feature_size=args.feature_size,
        hidden_size=args.hidden_size,
        mlp_dim=args.mlp_dim,
        num_heads=args.num_heads,
        pos_embed=args.pos_embed,
        norm_name=args.norm_name,
        conv_block=True,
        res_block=True,
        dropout_rate=args.dropout_rate,
    )
    
    # Load pretrained weights
    pretrained_path = os.path.join(args.pretrained_dir, args.pretrained_model_name)
    model = load_pretrained_compatible(model, pretrained_path)
    
    # Set model to evaluation mode and move to CPU
    model.eval()
    model.to(device)
    
    print("🧪 Starting inference...")
    
    with torch.no_grad():
        dice_list_case = []
        for i, batch in enumerate(val_loader):
            # Move data to CPU (explicit)
            val_inputs = batch["image"].to(device)
            val_labels = batch["label"].to(device)
            
            img_name = f"case_{i+1:03d}"  # Simple fallback name
            print(f"📊 Processing case {i+1}: {img_name}")
            
            # Run inference
            val_outputs = sliding_window_inference(
                val_inputs, 
                (96, 96, 96), 
                4, 
                model, 
                overlap=args.infer_overlap
            )
            
            # Post-process outputs
            val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
            val_labels = val_labels.cpu().numpy()[:, 0, :, :, :]
            
            # Calculate Dice scores for each organ
            dice_list_sub = []
            for organ_id in range(1, 14):  # Organs 1-13
                organ_dice = dice(val_outputs[0] == organ_id, val_labels[0] == organ_id)
                dice_list_sub.append(organ_dice)
            
            mean_dice = np.mean(dice_list_sub)
            print(f"   Mean Organ Dice: {mean_dice:.4f}")
            dice_list_case.append(mean_dice)
        
        overall_dice = np.mean(dice_list_case)
        print(f"\n🎉 Overall Mean Dice: {overall_dice:.4f}")

if __name__ == "__main__":
    main()