#!/usr/bin/env python3
"""
Run UNETR models directly on CT test images and compare performance with overlays
"""

import os
import time
import json
import numpy as np
import pandas as pd
import torch
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from networks.unetr import UNETR
from trainer import dice
from monai.inferers import sliding_window_inference
from monai import transforms
from monai.metrics import HausdorffDistanceMetric
from scipy.spatial.distance import directed_hausdorff
from scipy import ndimage
import glob

def create_organ_colormap():
    """Create colormap for 14 organs with transparency for overlay"""
    colors = [
        [0, 0, 0, 0],        # 0: Background (transparent)
        [1, 0, 0, 0.6],      # 1: Spleen (red)
        [0, 1, 0, 0.6],      # 2: Right Kidney (green)
        [0, 0, 1, 0.6],      # 3: Left Kidney (blue)
        [1, 1, 0, 0.6],      # 4: Gallbladder (yellow)
        [1, 0, 1, 0.6],      # 5: Esophagus (magenta)
        [0, 1, 1, 0.6],      # 6: Liver (cyan)
        [1, 0.65, 0, 0.6],   # 7: Stomach (orange)
        [0.5, 0, 0.5, 0.6],  # 8: Aorta (purple)
        [0.65, 0.16, 0.16, 0.6], # 9: IVC (brown)
        [1, 0.75, 0.8, 0.6], # 10: Portal/Splenic Veins (pink)
        [0.5, 0.5, 0.5, 0.6], # 11: Pancreas (gray)
        [0.5, 0.5, 0, 0.6],  # 12: Right Adrenal (olive)
        [0, 0, 0.5, 0.6]     # 13: Left Adrenal (navy)
    ]
    return ListedColormap(colors)

def load_model(model_path, model_type="original"):
    """Load and return UNETR model"""
    
    print(f"🔧 Loading {model_type.upper()} model...")
    
    # Create model architecture
    model = UNETR(
        in_channels=1,
        out_channels=14,
        img_size=(96, 96, 96),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        conv_block=True,
        res_block=True,
        dropout_rate=0.0,
    )
    
    # Load weights
    model_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(model_dict, strict=False)
    
    # Apply quantization if needed
    if model_type == "int8":
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear, torch.nn.Conv3d}, dtype=torch.qint8
        )
        device = torch.device("cpu")  # INT8 runs on CPU
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_type == "fp16" and device.type == "cuda":
            model = model.half()
    
    model.eval()
    model.to(device)
    
    print(f"✅ {model_type.upper()} model loaded on {device}")
    return model, device

def preprocess_ct_image(ct_path):
    """Load and preprocess CT image for model input"""
    
    # Load CT image
    ct_img = nib.load(ct_path)
    ct_data = ct_img.get_fdata()
    
    # Apply MONAI transforms (same as training)
    transforms_list = transforms.Compose([
        transforms.EnsureChannelFirst(),
        transforms.Orientation(axcodes="RAS"),
        transforms.Spacing(pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
        transforms.ScaleIntensityRange(a_min=-175.0, a_max=250.0, b_min=0.0, b_max=1.0, clip=True),
        transforms.CropForeground(allow_smaller=True),
        transforms.ToTensor()
    ])
    
    # Apply transforms
    ct_tensor = transforms_list(ct_data)
    ct_tensor = ct_tensor.unsqueeze(0)  # Add batch dimension
    
    return ct_tensor, ct_data, ct_img.affine

def load_ground_truth(ct_path):
    """Load corresponding ground truth segmentation"""
    
    # Convert image path to label path
    label_path = ct_path.replace('/img/', '/label/').replace('img', 'label')
    
    if os.path.exists(label_path):
        label_img = nib.load(label_path)
        return label_img.get_fdata()
    else:
        print(f"⚠️ Ground truth not found: {label_path}")
        return None

def run_segmentation_with_timing(model, ct_tensor, device, model_type, overlap=0.5):
    """Run segmentation inference and measure timing"""
    
    # Move input to device
    ct_input = ct_tensor.to(device)
    if model_type == "fp16" and device.type == "cuda":
        ct_input = ct_input.half()
    
    # Warmup run
    with torch.no_grad():
        _ = model(ct_input)
    
    # Timed inference
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    
    with torch.no_grad():
        # Use sliding window inference
        outputs = sliding_window_inference(
            ct_input, (96, 96, 96), 4, model, overlap=overlap
        )
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    inference_time = end_time - start_time
    
    # Process outputs
    outputs = torch.softmax(outputs, 1).cpu().numpy()
    segmentation = np.argmax(outputs, axis=1)[0]  # Remove batch dimension
    
    return segmentation.astype(np.uint8), inference_time

def calculate_hd95(pred_mask, gt_mask, spacing=(1.5, 1.5, 2.0)):
    """
    Calculate Hausdorff Distance 95th percentile between prediction and ground truth
    
    Args:
        pred_mask: Binary prediction mask
        gt_mask: Binary ground truth mask  
        spacing: Voxel spacing in mm (x, y, z)
    
    Returns:
        HD95 distance in mm
    """
    
    if np.sum(pred_mask) == 0 and np.sum(gt_mask) == 0:
        return 0.0  # Both empty - perfect match
    
    if np.sum(pred_mask) == 0 or np.sum(gt_mask) == 0:
        return 373.13  # Large penalty for complete miss (diagonal of 96x96x96 @ 1.5mm spacing)
    
    # Get surface points using edge detection
    pred_surface = get_surface_points(pred_mask)
    gt_surface = get_surface_points(gt_mask)
    
    if len(pred_surface) == 0 or len(gt_surface) == 0:
        return 373.13
    
    # Apply spacing to convert to mm
    pred_surface_mm = pred_surface * np.array(spacing)
    gt_surface_mm = gt_surface * np.array(spacing)
    
    # Calculate directed Hausdorff distances
    try:
        # Distance from prediction to ground truth
        dist_pred_to_gt = np.array([
            np.min(np.linalg.norm(gt_surface_mm - p, axis=1)) 
            for p in pred_surface_mm
        ])
        
        # Distance from ground truth to prediction  
        dist_gt_to_pred = np.array([
            np.min(np.linalg.norm(pred_surface_mm - g, axis=1))
            for g in gt_surface_mm
        ])
        
        # Combine all distances
        all_distances = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
        
        # Return 95th percentile
        hd95 = np.percentile(all_distances, 95)
        
        return float(hd95)
        
    except Exception as e:
        print(f"    ⚠️ HD95 calculation error: {e}")
        return 373.13

def get_surface_points(mask):
    """Extract surface points from binary mask using morphological operations"""
    
    # Get edges using morphological operations
    eroded = ndimage.binary_erosion(mask)
    surface = mask & ~eroded
    
    # Get coordinates of surface points
    surface_coords = np.where(surface)
    surface_points = np.column_stack(surface_coords)
    
    return surface_points

def calculate_dice_and_hd95(pred_seg, gt_seg, spacing=(1.5, 1.5, 2.0)):
    """Calculate both Dice scores and HD95 for each organ"""
    
    if gt_seg is None:
        return None, None, None, None
    
    dice_scores = []
    hd95_scores = []
    organ_names = [
        "Spleen", "Right Kidney", "Left Kidney", "Gallbladder", "Esophagus", 
        "Liver", "Stomach", "Aorta", "IVC", "Portal/Splenic Veins", 
        "Pancreas", "Right Adrenal", "Left Adrenal"
    ]
    
    print(f"    📊 Calculating Dice & HD95 for {len(organ_names)} organs...")
    
    for organ_id in range(1, 14):  # Skip background (0)
        pred_mask = (pred_seg == organ_id).astype(np.uint8)
        gt_mask = (gt_seg == organ_id).astype(np.uint8)
        
        # Calculate Dice
        if np.sum(gt_mask) == 0 and np.sum(pred_mask) == 0:
            dice_score = 1.0  # Perfect match for absent organ
        elif np.sum(gt_mask) == 0:
            dice_score = 0.0  # False positive
        else:
            dice_score = dice(pred_mask, gt_mask)
        
        # Calculate HD95
        hd95_score = calculate_hd95(pred_mask, gt_mask, spacing)
        
        dice_scores.append(dice_score)
        hd95_scores.append(hd95_score)
        
        # Print per-organ results
        organ_name = organ_names[organ_id-1]
        print(f"      {organ_name:20s}: Dice={dice_score:.3f}, HD95={hd95_score:.1f}mm")
    
    mean_dice = np.mean(dice_scores)
    mean_hd95 = np.mean(hd95_scores)
    
    print(f"    🎯 Overall: Dice={mean_dice:.4f}, HD95={mean_hd95:.1f}mm")
    
    return dice_scores, hd95_scores, mean_dice, mean_hd95

def create_overlay_comparison(ct_data, segmentations, case_name, output_dir):
    """Create overlay comparison showing all models on original CT"""
    
    print(f"  🖼️ Creating overlay comparison for {case_name}")
    
    # Normalize CT for display
    ct_display = np.clip(ct_data, -175, 250)
    ct_display = (ct_display - ct_display.min()) / (ct_display.max() - ct_display.min() + 1e-8)
    
    # Get middle slices
    shape = ct_data.shape
    slices_info = [
        ('axial', shape[2] // 2, 2),
        ('coronal', shape[1] // 2, 1)
    ]
    
    cmap = create_organ_colormap()
    
    for view_name, slice_idx, axis in slices_info:
        # Create figure
        n_models = len(segmentations) + 1  # +1 for original CT
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 6))
        if n_models == 1:
            axes = [axes]
        
        # Extract slices
        if axis == 0:
            ct_slice = ct_display[slice_idx, :, :]
        elif axis == 1:
            ct_slice = ct_display[:, slice_idx, :]
        else:
            ct_slice = ct_display[:, :, slice_idx]
        
        # Plot original CT
        axes[0].imshow(ct_slice, cmap='gray', origin='lower')
        axes[0].set_title('Original CT', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Plot each model's segmentation overlay
        for i, (model_name, seg_data) in enumerate(segmentations.items()):
            # Extract segmentation slice
            if axis == 0:
                seg_slice = seg_data[slice_idx, :, :]
            elif axis == 1:
                seg_slice = seg_data[:, slice_idx, :]
            else:
                seg_slice = seg_data[:, :, slice_idx]
            
            # Display CT background
            axes[i+1].imshow(ct_slice, cmap='gray', origin='lower')
            
            # Overlay segmentation
            seg_masked = np.ma.masked_where(seg_slice == 0, seg_slice)
            axes[i+1].imshow(seg_masked, cmap=cmap, vmin=0, vmax=13, alpha=0.7, origin='lower')
            
            axes[i+1].set_title(f'{model_name.upper()}', fontsize=14, fontweight='bold')
            axes[i+1].axis('off')
        
        plt.suptitle(f'{case_name} - {view_name.title()} View (Slice {slice_idx})', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save comparison
        comparison_path = os.path.join(output_dir, f'{case_name}_{view_name}_overlay_comparison.png')
        plt.savefig(comparison_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"    ✅ Saved: {comparison_path}")

def main():
    """Main function to run direct CT segmentation comparison"""
    
    print("🏥 Direct CT Segmentation Comparison")
    print("=" * 50)
    
    # Configuration
    models_config = {
        "original": "./pretrained_models/UNETR_model_best_acc.pth",
        "int8": "./quantized_models/unetr_int8_dynamic.pth", 
        "fp16": "./quantized_models/unetr_fp16.pth"
    }
    
    # Create output directory
    output_dir = "./direct_ct_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find test CT images
    ct_images = []
    
    # Look for test images in dataset
    test_patterns = [
        "./dataset/Testing/img/*.nii.gz",
        "./dataset/Training/img/*.nii.gz"  # Use some training as test
    ]
    
    for pattern in test_patterns:
        ct_images.extend(glob.glob(pattern))
    
    if not ct_images:
        print("❌ No CT images found!")
        return
    
    # Limit to first 3 cases for demo
    ct_images = sorted(ct_images)[:3]
    print(f"📊 Found {len(ct_images)} CT images to process")
    
    # Results storage
    results = []
    all_segmentations = {}  # Store segmentations for comparison
    
    # Process each model
    for model_name, model_path in models_config.items():
        
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found: {model_path}")
            continue
        
        print(f"\n{'='*30}")
        print(f"🔍 Testing {model_name.upper()} Model")
        print(f"{'='*30}")
        
        # Load model
        model, device = load_model(model_path, model_name)
        
        # Process each CT image
        model_results = []
        model_segmentations = {}
        
        for ct_path in ct_images:
            case_name = os.path.basename(ct_path).replace('.nii.gz', '')
            print(f"  📄 Processing: {case_name}")
            
            try:
                # Preprocess CT
                ct_tensor, ct_data, ct_affine = preprocess_ct_image(ct_path)
                
                # Load ground truth
                gt_seg = load_ground_truth(ct_path)
                
                # Run segmentation with timing
                pred_seg, inference_time = run_segmentation_with_timing(
                    model, ct_tensor, device, model_name
                )
                
                # Calculate accuracy metrics
                dice_scores, hd95_scores, mean_dice, mean_hd95 = calculate_dice_and_hd95(
                    pred_seg, gt_seg, spacing=(1.5, 1.5, 2.0)
                )
                
                # Store results
                result = {
                    'Model': model_name.upper(),
                    'Case': case_name,
                    'Inference_Time_s': inference_time,
                    'Mean_Dice': mean_dice if mean_dice else 0.0,
                    'Mean_HD95_mm': mean_hd95 if mean_hd95 else 999.0,
                    'Device': str(device)
                }
                
                model_results.append(result)
                model_segmentations[case_name] = pred_seg
                
                print(f"    ⏱️ Time: {inference_time:.3f}s")
                if mean_dice is not None:
                    print(f"    🎯 Dice: {mean_dice:.4f}, HD95: {mean_hd95:.1f}mm")
                else:
                    print(f"    🎯 Dice: N/A, HD95: N/A")
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                continue
        
        results.extend(model_results)
        all_segmentations[model_name] = model_segmentations
        
        # Clean up model
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Create performance comparison table
    print(f"\n📊 Creating Performance Table...")
    
    if results:
        df = pd.DataFrame(results)
        
        # Calculate summary statistics
        summary_stats = df.groupby('Model').agg({
            'Inference_Time_s': ['mean', 'std', 'min', 'max'],
            'Mean_Dice': ['mean', 'std', 'count'],
            'Mean_HD95_mm': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        # Flatten column names
        summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
        summary_stats = summary_stats.reset_index()
        
        # Add model sizes
        model_sizes = {"ORIGINAL": 354, "INT8": 102, "FP16": 177}
        summary_stats['Model_Size_MB'] = summary_stats['Model'].map(model_sizes)
        summary_stats['Size_Reduction'] = (354 / summary_stats['Model_Size_MB']).round(1)
        summary_stats['FPS'] = (1 / summary_stats['Inference_Time_s_mean']).round(2)
        
        # Save detailed results
        detailed_path = os.path.join(output_dir, "detailed_results.csv")
        df.to_csv(detailed_path, index=False)
        
        # Save summary table
        summary_path = os.path.join(output_dir, "performance_summary.csv")
        summary_stats.to_csv(summary_path, index=False)
        
        print(f"✅ Detailed results: {detailed_path}")
        print(f"✅ Summary table: {summary_path}")
        
        # Display summary table
        print(f"\n📋 PERFORMANCE SUMMARY:")
        print("=" * 80)
        print(summary_stats.to_string(index=False))
        
        # Create performance plots
        create_performance_plots(summary_stats, output_dir)
    
    # Create overlay comparisons for each case
    print(f"\n🖼️ Creating Overlay Comparisons...")
    
    for ct_path in ct_images:
        case_name = os.path.basename(ct_path).replace('.nii.gz', '')
        
        # Get segmentations for this case from all models
        case_segmentations = {}
        for model_name, model_segs in all_segmentations.items():
            if case_name in model_segs:
                case_segmentations[model_name] = model_segs[case_name]
        
        if len(case_segmentations) >= 2:
            try:
                _, ct_data, _ = preprocess_ct_image(ct_path)
                ct_data = ct_data.squeeze() if ct_data.ndim > 3 else ct_data
                create_overlay_comparison(ct_data, case_segmentations, case_name, output_dir)
            except Exception as e:
                print(f"  ❌ Error creating overlay for {case_name}: {e}")
    
    print(f"\n🎉 Analysis Complete!")
    print(f"📁 All results saved to: {output_dir}")
    print(f"📊 Check the CSV files for detailed performance metrics")
    print(f"🖼️ Check the PNG files for visual comparisons on actual CT scans")

def create_performance_plots(summary_stats, output_dir):
    """Create performance visualization plots including HD95"""
    
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        models = summary_stats['Model']
        colors = ['#2E86AB', '#A23B72', '#F18F01'][:len(models)]
        
        # 1. Accuracy comparison (Dice)
        ax1.bar(models, summary_stats['Mean_Dice_mean'], 
                yerr=summary_stats['Mean_Dice_std'], 
                color=colors, alpha=0.8, capsize=5)
        ax1.set_ylabel('Mean Dice Score')
        ax1.set_title('Model Accuracy Comparison (Dice)')
        ax1.set_ylim(0, 1)
        
        # Add value labels
        for i, (model, dice) in enumerate(zip(models, summary_stats['Mean_Dice_mean'])):
            ax1.text(i, dice + 0.02, f'{dice:.3f}', ha='center', fontweight='bold')
        
        # 2. Inference time comparison
        ax2.bar(models, summary_stats['Inference_Time_s_mean'], 
                yerr=summary_stats['Inference_Time_s_std'],
                color=colors, alpha=0.8, capsize=5)
        ax2.set_ylabel('Inference Time (seconds)')
        ax2.set_title('Inference Speed Comparison')
        
        # Add value labels
        for i, (model, time_val) in enumerate(zip(models, summary_stats['Inference_Time_s_mean'])):
            ax2.text(i, time_val + 0.01, f'{time_val:.3f}s', ha='center', fontweight='bold')
        
        # 3. HD95 comparison (lower is better)
        ax3.bar(models, summary_stats['Mean_HD95_mm_mean'], 
                yerr=summary_stats['Mean_HD95_mm_std'],
                color=colors, alpha=0.8, capsize=5)
        ax3.set_ylabel('Mean HD95 Distance (mm)')
        ax3.set_title('Surface Distance Comparison (Lower is Better)')
        
        # Add value labels
        for i, (model, hd95) in enumerate(zip(models, summary_stats['Mean_HD95_mm_mean'])):
            ax3.text(i, hd95 + 1, f'{hd95:.1f}mm', ha='center', fontweight='bold')
        
        # 4. Model size comparison
        ax4.bar(models, summary_stats['Model_Size_MB'], color=colors, alpha=0.8)
        ax4.set_ylabel('Model Size (MB)')
        ax4.set_title('Model Size Comparison')
        
        # Add value labels
        for i, (model, size) in enumerate(zip(models, summary_stats['Model_Size_MB'])):
            ax4.text(i, size + 5, f'{size}MB', ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, "performance_plots_with_hd95.png")
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Performance plots with HD95: {plot_path}")
        
        # Create detailed comparison table plot
        create_metrics_table_plot(summary_stats, output_dir)
        
    except Exception as e:
        print(f"⚠️ Error creating plots: {e}")

def create_metrics_table_plot(summary_stats, output_dir):
    """Create a visual table showing all metrics"""
    
    try:
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare data for table
        table_data = []
        for _, row in summary_stats.iterrows():
            table_row = [
                row['Model'],
                f"{row['Mean_Dice_mean']:.4f} ± {row['Mean_Dice_std']:.4f}",
                f"{row['Mean_HD95_mm_mean']:.1f} ± {row['Mean_HD95_mm_std']:.1f}",
                f"{row['Inference_Time_s_mean']:.3f} ± {row['Inference_Time_s_std']:.3f}",
                f"{row['FPS']:.2f}",
                f"{row['Model_Size_MB']:.0f}",
                f"{row['Size_Reduction']:.1f}x"
            ]
            table_data.append(table_row)
        
        # Create table
        headers = ['Model', 'Dice Score', 'HD95 (mm)', 'Time (s)', 'FPS', 'Size (MB)', 'Reduction']
        table = ax.table(cellText=table_data, colLabels=headers, 
                        cellLoc='center', loc='center')
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)
        
        # Color code headers
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color code cells based on performance
        for i in range(len(table_data)):
            # Best Dice score (highest) - green
            if i == 0:  # Assuming sorted by performance
                table[(i+1, 1)].set_facecolor('#E8F5E8')
            
            # Best HD95 (lowest) - green  
            table[(i+1, 2)].set_facecolor('#E8F5E8' if i == 0 else '#FFF3E0')
            
            # Best speed (lowest time) - blue
            table[(i+1, 3)].set_facecolor('#E3F2FD' if i == 0 else '#FFF3E0')
        
        plt.title('UNETR Model Performance Comparison\nDice ↑ Better, HD95 ↓ Better, Time ↓ Better', 
                 fontsize=16, fontweight='bold', pad=20)
        
        table_path = os.path.join(output_dir, "metrics_comparison_table.png")
        plt.savefig(table_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Metrics table: {table_path}")
        
    except Exception as e:
        print(f"⚠️ Error creating metrics table: {e}")

if __name__ == "__main__":
    main()