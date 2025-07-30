#!/usr/bin/env python3
"""
UNETR Test with Visual Output - Save segmentation results as images
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from networks.unetr import UNETR
from trainer import dice
from utils.data_utils import get_loader
from monai.inferers import sliding_window_inference

# Setup argument parser (same as before)
parser = argparse.ArgumentParser(description="UNETR segmentation pipeline with visual output")
parser.add_argument("--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory")
parser.add_argument("--data_dir", default="/dataset/dataset0/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument("--pretrained_model_name", default="UNETR_model_best_acc.pth", type=str, help="pretrained model name")
parser.add_argument("--output_dir", default="./results/", type=str, help="directory to save results")
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

# Organ names and colors for visualization
ORGAN_NAMES = {
    0: "Background",
    1: "Spleen", 2: "Right Kidney", 3: "Left Kidney", 4: "Gallbladder",
    5: "Esophagus", 6: "Liver", 7: "Stomach", 8: "Aorta",
    9: "IVC", 10: "Portal Vein", 11: "Pancreas",
    12: "Right Adrenal", 13: "Left Adrenal"
}

# Create colormap for organs
COLORS = [
    [0, 0, 0],       # 0: Background (black)
    [1, 0, 0],       # 1: Spleen (red)
    [0, 1, 0],       # 2: Right Kidney (green)
    [0, 0, 1],       # 3: Left Kidney (blue)
    [1, 1, 0],       # 4: Gallbladder (yellow)
    [1, 0, 1],       # 5: Esophagus (magenta)
    [0, 1, 1],       # 6: Liver (cyan)
    [1, 0.5, 0],     # 7: Stomach (orange)
    [0.5, 0, 1],     # 8: Aorta (purple)
    [1, 0.5, 0.5],   # 9: IVC (pink)
    [0.5, 1, 0.5],   # 10: Portal Vein (light green)
    [0.5, 0.5, 1],   # 11: Pancreas (light blue)
    [1, 1, 0.5],     # 12: Right Adrenal (light yellow)
    [0.5, 1, 1]      # 13: Left Adrenal (light cyan)
]

def load_pretrained_compatible(model, checkpoint_path):
    """Load pretrained model with CPU compatibility"""
    print(f"Loading model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    clean_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        clean_state_dict[name] = v
    
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(clean_state_dict.keys())
    compatible_keys = model_keys.intersection(checkpoint_keys)
    
    print(f"✅ Compatible keys: {len(compatible_keys)}/{len(model_keys)}")
    
    filtered_state_dict = {k: v for k, v in clean_state_dict.items() if k in compatible_keys}
    model.load_state_dict(filtered_state_dict, strict=False)
    print("✅ Model loaded successfully!")
    
    return model

def save_segmentation_results(image, prediction, ground_truth, case_name, output_dir, dice_scores):
    """Save visualization of segmentation results"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Select middle slices for visualization
    mid_axial = image.shape[2] // 2
    mid_coronal = image.shape[1] // 2
    mid_sagittal = image.shape[0] // 2
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f'Case: {case_name} (Mean Dice: {np.mean(dice_scores):.3f})', fontsize=16)
    
    # Axial view
    axes[0, 0].imshow(image[:, :, mid_axial], cmap='gray')
    axes[0, 0].set_title('Axial - Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(image[:, :, mid_axial], cmap='gray', alpha=0.7)
    axes[0, 1].imshow(ground_truth[:, :, mid_axial], cmap=ListedColormap(COLORS), alpha=0.5, vmin=0, vmax=13)
    axes[0, 1].set_title('Axial - Ground Truth')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(image[:, :, mid_axial], cmap='gray', alpha=0.7)
    axes[0, 2].imshow(prediction[:, :, mid_axial], cmap=ListedColormap(COLORS), alpha=0.5, vmin=0, vmax=13)
    axes[0, 2].set_title('Axial - Prediction')
    axes[0, 2].axis('off')
    
    # Coronal view
    axes[1, 0].imshow(image[:, mid_coronal, :], cmap='gray')
    axes[1, 0].set_title('Coronal - Original Image')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(image[:, mid_coronal, :], cmap='gray', alpha=0.7)
    axes[1, 1].imshow(ground_truth[:, mid_coronal, :], cmap=ListedColormap(COLORS), alpha=0.5, vmin=0, vmax=13)
    axes[1, 1].set_title('Coronal - Ground Truth')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(image[:, mid_coronal, :], cmap='gray', alpha=0.7)
    axes[1, 2].imshow(prediction[:, mid_coronal, :], cmap=ListedColormap(COLORS), alpha=0.5, vmin=0, vmax=13)
    axes[1, 2].set_title('Coronal - Prediction')
    axes[1, 2].axis('off')
    
    # Sagittal view
    axes[2, 0].imshow(image[mid_sagittal, :, :], cmap='gray')
    axes[2, 0].set_title('Sagittal - Original Image')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(image[mid_sagittal, :, :], cmap='gray', alpha=0.7)
    axes[2, 1].imshow(ground_truth[mid_sagittal, :, :], cmap=ListedColormap(COLORS), alpha=0.5, vmin=0, vmax=13)
    axes[2, 1].set_title('Sagittal - Ground Truth')
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(image[mid_sagittal, :, :], cmap='gray', alpha=0.7)
    axes[2, 2].imshow(prediction[mid_sagittal, :, :], cmap=ListedColormap(COLORS), alpha=0.5, vmin=0, vmax=13)
    axes[2, 2].set_title('Sagittal - Prediction')
    axes[2, 2].axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{case_name}_segmentation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save dice scores summary
    with open(os.path.join(output_dir, f'{case_name}_scores.txt'), 'w') as f:
        f.write(f"Case: {case_name}\n")
        f.write(f"Mean Dice Score: {np.mean(dice_scores):.4f}\n\n")
        f.write("Individual Organ Scores:\n")
        for i, score in enumerate(dice_scores):
            organ_name = ORGAN_NAMES[i+1]
            f.write(f"{organ_name}: {score:.4f}\n")

def main():
    args = parser.parse_args()
    args.test_mode = True
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cpu')
    print(f"🖥️ Using device: {device}")
    print(f"📁 Results will be saved to: {args.output_dir}")
    
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
    
    model.eval()
    model.to(device)
    
    print("🧪 Starting inference with visual output...")
    
    with torch.no_grad():
        dice_list_case = []
        for i, batch in enumerate(val_loader):
            val_inputs = batch["image"].to(device)
            val_labels = batch["label"].to(device)
            
            case_name = f"case_{i+1:03d}"
            print(f"📊 Processing {case_name}...")
            
            # Run inference
            val_outputs = sliding_window_inference(
                val_inputs, 
                (96, 96, 96), 
                4, 
                model, 
                overlap=args.infer_overlap
            )
            
            # Post-process outputs
            val_outputs_softmax = torch.softmax(val_outputs, 1).cpu().numpy()
            val_outputs_argmax = np.argmax(val_outputs_softmax, axis=1).astype(np.uint8)
            val_labels_np = val_labels.cpu().numpy()[:, 0, :, :, :]
            val_inputs_np = val_inputs.cpu().numpy()[0, 0, :, :, :]
            
            # Calculate Dice scores
            dice_list_sub = []
            for organ_id in range(1, 14):
                organ_dice = dice(val_outputs_argmax[0] == organ_id, val_labels_np[0] == organ_id)
                dice_list_sub.append(organ_dice)
            
            mean_dice = np.mean(dice_list_sub)
            print(f"   Mean Dice: {mean_dice:.4f}")
            dice_list_case.append(mean_dice)
            
            # Save visual results
            save_segmentation_results(
                val_inputs_np,
                val_outputs_argmax[0],
                val_labels_np[0],
                case_name,
                args.output_dir,
                dice_list_sub
            )
            print(f"   💾 Saved visualization to {args.output_dir}/{case_name}_segmentation.png")
        
        overall_dice = np.mean(dice_list_case)
        print(f"\n🎉 Overall Mean Dice: {overall_dice:.4f}")
        print(f"📊 All results saved in: {args.output_dir}")

if __name__ == "__main__":
    main()