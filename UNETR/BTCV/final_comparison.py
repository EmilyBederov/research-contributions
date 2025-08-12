#!/usr/bin/env python3
"""
Simple UNETR Model Comparison with Metrics - Clean and Complete
"""

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
from networks.unetr import UNETR
from utils.data_utils import get_loader
from trainer import dice
from monai.inferers import sliding_window_inference
from scipy.spatial.distance import directed_hausdorff
from scipy import ndimage

def calculate_hd95(pred_mask, gt_mask, spacing=(1.5, 1.5, 2.0)):
    """Calculate HD95 distance"""
    if np.sum(pred_mask) == 0 and np.sum(gt_mask) == 0:
        return 0.0
    if np.sum(pred_mask) == 0 or np.sum(gt_mask) == 0:
        return 373.13  # Large penalty
    
    # Get surface points
    pred_surface = get_surface_points(pred_mask)
    gt_surface = get_surface_points(gt_mask)
    
    if len(pred_surface) == 0 or len(gt_surface) == 0:
        return 373.13
    
    # Apply spacing
    pred_surface_mm = pred_surface * np.array(spacing)
    gt_surface_mm = gt_surface * np.array(spacing)
    
    # Calculate distances
    dist_pred_to_gt = [np.min(np.linalg.norm(gt_surface_mm - p, axis=1)) for p in pred_surface_mm]
    dist_gt_to_pred = [np.min(np.linalg.norm(pred_surface_mm - g, axis=1)) for g in gt_surface_mm]
    
    all_distances = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
    return float(np.percentile(all_distances, 95))

def get_surface_points(mask):
    """Extract surface points from binary mask"""
    eroded = ndimage.binary_erosion(mask)
    surface = mask & ~eroded
    surface_coords = np.where(surface)
    return np.column_stack(surface_coords)

def load_model(model_path, model_type):
    """Load UNETR model"""
    print(f"Loading {model_type.upper()} model...")
    
    # Create model
    model = UNETR(in_channels=1, out_channels=14, img_size=(96, 96, 96),
                  feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12,
                  pos_embed="perceptron", norm_name="instance", 
                  conv_block=True, res_block=True, dropout_rate=0.0)
    
    # Load weights
    model_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(model_dict, strict=False)
    
    # Set device and quantization
    if model_type == "int8":
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear, torch.nn.Conv3d}, dtype=torch.qint8)
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        if model_type == "fp16":
            model = model.half()
    
    model.eval().to(device)
    print(f"✅ {model_type.upper()} loaded on {device}")
    return model, device

def run_inference_with_metrics(model, inputs, labels, device, model_type):
    """Run inference and calculate all metrics"""
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    if model_type == "fp16":
        inputs = inputs.half()
    
    # Time inference
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    
    with torch.no_grad():
        outputs = sliding_window_inference(inputs, (96, 96, 96), 4, model, overlap=0.5)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    inference_time = time.perf_counter() - start_time
    
    # Convert to segmentation
    outputs = torch.softmax(outputs, 1).cpu().numpy()
    pred_seg = np.argmax(outputs, axis=1)[0].astype(np.uint8)
    gt_seg = labels.cpu().numpy()[0, 0, :, :, :]
    
    # Calculate Dice scores
    dice_scores = []
    hd95_scores = []
    
    for organ_id in range(1, 14):
        pred_mask = (pred_seg == organ_id).astype(np.uint8)
        gt_mask = (gt_seg == organ_id).astype(np.uint8)
        
        # Dice
        if np.sum(gt_mask) == 0 and np.sum(pred_mask) == 0:
            dice_score = 1.0
        elif np.sum(gt_mask) == 0:
            dice_score = 0.0
        else:
            dice_score = dice(pred_mask, gt_mask)
        
        dice_scores.append(dice_score)
        
        # HD95
        hd95_score = calculate_hd95(pred_mask, gt_mask)
        hd95_scores.append(hd95_score)
    
    mean_dice = np.mean(dice_scores)
    mean_hd95 = np.mean(hd95_scores)
    
    return pred_seg, inference_time, mean_dice, mean_hd95

def create_overlay_comparison(ct_data, segmentations, case_name, output_dir="./results"):
    """Create overlay comparison on original CT"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Normalize CT for display
    ct_display = np.clip(ct_data, -175, 250)
    ct_display = (ct_display - ct_display.min()) / (ct_display.max() - ct_display.min() + 1e-8)
    
    # Get middle slice
    middle_slice_idx = ct_data.shape[2] // 2
    ct_slice = ct_display[:, :, middle_slice_idx]
    
    # Create comparison figure
    n_models = len(segmentations)
    fig, axes = plt.subplots(1, n_models + 1, figsize=(5*(n_models + 1), 5))
    
    # Colormap for organs
    from matplotlib.colors import ListedColormap
    colors = ['black', 'red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 
              'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'navy']
    cmap = ListedColormap(colors[:14])
    
    # Plot original CT
    axes[0].imshow(ct_slice, cmap='gray', origin='lower')
    axes[0].set_title('Original CT', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Plot overlays
    for i, (model_name, seg_data) in enumerate(segmentations.items()):
        seg_slice = seg_data[:, :, middle_slice_idx]
        
        # Show CT background
        axes[i+1].imshow(ct_slice, cmap='gray', origin='lower')
        
        # Overlay segmentation
        seg_masked = np.ma.masked_where(seg_slice == 0, seg_slice)
        axes[i+1].imshow(seg_masked, cmap=cmap, vmin=0, vmax=13, alpha=0.6, origin='lower')
        
        axes[i+1].set_title(f'{model_name.upper()} Overlay', fontsize=12, fontweight='bold')
        axes[i+1].axis('off')
    
    plt.suptitle(f'{case_name} - Axial Slice {middle_slice_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'{case_name}_overlay_comparison.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"📷 Saved: {save_path}")

def main():
    """Main function with complete metrics"""
    print("🔍 UNETR Model Comparison with Metrics")
    print("=" * 50)
    
    # Model paths
    models = {
        "original": "./pretrained_models/UNETR_model_best_acc.pth",
        "int8": "./quantized_models/unetr_int8_quantized.pth",
        "fp16": "./quantized_models/unetr_fp16_quantized.pth"
    }
    
    # Load data
    class Args:
        test_mode = True
        data_dir = "./dataset/"
        json_list = "dataset_0.json"
        workers = 4
        distributed = False
        space_x = 1.5; space_y = 1.5; space_z = 2.0
        a_min = -175.0; a_max = 250.0; b_min = 0.0; b_max = 1.0
        roi_x = 96; roi_y = 96; roi_z = 96
        RandFlipd_prob = 0.2; RandRotate90d_prob = 0.2
        RandScaleIntensityd_prob = 0.1; RandShiftIntensityd_prob = 0.1
    
    args = Args()
    val_loader = get_loader(args)
    
    # Get first batch
    batch = next(iter(val_loader))
    case_name = "test_case"
    
    print(f"\n📊 Processing {case_name}...")
    print(f"Input shape: {batch['image'].shape}")
    
    # Store results
    results = []
    segmentations = {}
    
    # Test each model
    for model_type, model_path in models.items():
        if not os.path.exists(model_path):
            print(f"⚠️ {model_type} model not found: {model_path}")
            continue
        
        try:
            print(f"\n🔄 Testing {model_type.upper()} model...")
            
            # Load model
            model, device = load_model(model_path, model_type)
            
            # Run inference with metrics
            pred_seg, inf_time, mean_dice, mean_hd95 = run_inference_with_metrics(
                model, batch["image"], batch["label"], device, model_type
            )
            
            # Store results
            result = {
                'Model': model_type.upper(),
                'Inference_Time_s': inf_time,
                'FPS': 1.0 / inf_time,
                'Mean_Dice': mean_dice,
                'Mean_HD95_mm': mean_hd95,
                'Device': str(device)
            }
            results.append(result)
            segmentations[model_type] = pred_seg
            
            print(f"  ⏱️ Time: {inf_time:.3f}s ({1.0/inf_time:.1f} FPS)")
            print(f"  🎯 Dice: {mean_dice:.4f}")
            print(f"  📏 HD95: {mean_hd95:.1f}mm")
            print(f"  💻 Device: {device}")
            
        except Exception as e:
            print(f"  ❌ {model_type} failed: {e}")
    
    # Create performance table
    if results:
        print(f"\n📊 PERFORMANCE SUMMARY")
        print("=" * 70)
        
        df = pd.DataFrame(results)
        
        # Add model sizes
        model_sizes = {"ORIGINAL": 354, "INT8": 102, "FP16": 177}
        df['Model_Size_MB'] = df['Model'].map(model_sizes)
        df['Size_Reduction'] = (354 / df['Model_Size_MB']).round(1)
        
        # Display table
        display_cols = ['Model', 'Inference_Time_s', 'FPS', 'Mean_Dice', 'Mean_HD95_mm', 'Model_Size_MB', 'Device']
        print(df[display_cols].to_string(index=False, float_format='%.3f'))
        
        # Save results
        df.to_csv('./results/performance_results.csv', index=False)
        print(f"\n💾 Results saved to: ./results/performance_results.csv")
    
    # Create overlay comparison
    if len(segmentations) >= 2:
        print(f"\n🖼️ Creating overlay comparison...")
        
        # Get original CT data (first case for simplicity)
        ct_data = batch["image"][0, 0].cpu().numpy()  # Remove batch and channel dims
        create_overlay_comparison(ct_data, segmentations, case_name)
    
    print(f"\n✅ Comparison Complete!")

if __name__ == "__main__":
    main()