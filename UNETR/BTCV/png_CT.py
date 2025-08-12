#!/usr/bin/env python3
"""
Create properly aligned segmentation overlays by resampling to match dimensions
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import glob
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

def create_organ_colormap():
    """Create colormap for the 14 organs with transparency"""
    colors = [
        [0, 0, 0, 0],        # 0: Background (transparent)
        [1, 0, 0, 0.7],      # 1: Spleen (red)
        [0, 1, 0, 0.7],      # 2: Right Kidney (green)
        [0, 0, 1, 0.7],      # 3: Left Kidney (blue)
        [1, 1, 0, 0.7],      # 4: Gallbladder (yellow)
        [1, 0, 1, 0.7],      # 5: Esophagus (magenta)
        [0, 1, 1, 0.7],      # 6: Liver (cyan)
        [1, 0.65, 0, 0.7],   # 7: Stomach (orange)
        [0.5, 0, 0.5, 0.7],  # 8: Aorta (purple)
        [0.65, 0.16, 0.16, 0.7], # 9: IVC (brown)
        [1, 0.75, 0.8, 0.7], # 10: Portal/Splenic Veins (pink)
        [0.5, 0.5, 0.5, 0.7], # 11: Pancreas (gray)
        [0.5, 0.5, 0, 0.7],  # 12: Right Adrenal (olive)
        [0, 0, 0.5, 0.7]     # 13: Left Adrenal (navy)
    ]
    return ListedColormap(colors)

def find_original_image(case_name):
    """Find the original CT image for a case"""
    
    # Read dataset JSON to get the mapping
    json_path = "./dataset/dataset_0.json"
    if os.path.exists(json_path):
        try:
            import json
            with open(json_path, 'r') as f:
                dataset_info = json.load(f)
            
            # Look in validation set
            if 'validation' in dataset_info:
                for i, case_info in enumerate(dataset_info['validation']):
                    if f"case_{i+1:03d}" == case_name:
                        if 'image' in case_info:
                            img_path = case_info['image']
                            if img_path.startswith('./'):
                                img_path = img_path[2:]
                            full_path = os.path.join("./dataset", img_path)
                            if os.path.exists(full_path):
                                return full_path
        except Exception as e:
            print(f"  ⚠️ Error reading dataset JSON: {e}")
    
    # Fallback to simple mapping
    case_num = int(case_name.replace('case_', ''))
    possible_paths = [
        f"./dataset/Training/img/img{case_num:04d}.nii.gz",
        f"./dataset/Testing/img/img{case_num:04d}.nii.gz",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

def resample_segmentation_to_original(seg_data, seg_affine, orig_shape, orig_affine):
    """Resample segmentation to match original image dimensions and spacing"""
    
    print(f"    🔄 Resampling segmentation...")
    print(f"       Seg shape: {seg_data.shape} → Orig shape: {orig_shape}")
    
    # If shapes already match, no resampling needed
    if seg_data.shape == orig_shape:
        print(f"       ✅ Shapes already match, no resampling needed")
        return seg_data
    
    # Create coordinate grids for both spaces
    seg_x = np.arange(seg_data.shape[0])
    seg_y = np.arange(seg_data.shape[1]) 
    seg_z = np.arange(seg_data.shape[2])
    
    # Create interpolator for segmentation (using nearest neighbor to preserve labels)
    interpolator = RegularGridInterpolator(
        (seg_x, seg_y, seg_z), 
        seg_data, 
        method='nearest',
        bounds_error=False, 
        fill_value=0
    )
    
    # Create target coordinate grid
    orig_x = np.linspace(0, seg_data.shape[0]-1, orig_shape[0])
    orig_y = np.linspace(0, seg_data.shape[1]-1, orig_shape[1])
    orig_z = np.linspace(0, seg_data.shape[2]-1, orig_shape[2])
    
    # Create meshgrid for interpolation
    X, Y, Z = np.meshgrid(orig_x, orig_y, orig_z, indexing='ij')
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # Interpolate
    resampled_flat = interpolator(points)
    resampled = resampled_flat.reshape(orig_shape)
    
    print(f"       ✅ Resampling complete: {resampled.shape}")
    return resampled.astype(np.uint8)

def create_aligned_overlay_image(original_data, seg_data, title, slice_idx, axis):
    """Create overlay with properly aligned segmentation"""
    
    # Extract slice from both images
    if axis == 0:
        original_slice = original_data[slice_idx, :, :]
        seg_slice = seg_data[slice_idx, :, :]
    elif axis == 1:
        original_slice = original_data[:, slice_idx, :]
        seg_slice = seg_data[:, slice_idx, :]
    else:
        original_slice = original_data[:, :, slice_idx]
        seg_slice = seg_data[:, :, slice_idx]
    
    # Normalize original image for display (CT windowing)
    original_slice = np.clip(original_slice, -175, 250)
    original_slice = (original_slice - original_slice.min()) / (original_slice.max() - original_slice.min() + 1e-8)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Display original image in grayscale
    ax.imshow(original_slice, cmap='gray', alpha=1.0, origin='lower')
    
    # Create masked segmentation (only show non-background)
    seg_masked = np.ma.masked_where(seg_slice == 0, seg_slice)
    
    # Overlay segmentation with transparency
    cmap = create_organ_colormap()
    ax.imshow(seg_masked, cmap=cmap, vmin=0, vmax=13, alpha=0.6, origin='lower')
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Add some info about the slice
    ax.text(0.02, 0.98, f'Slice {slice_idx}', transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            verticalalignment='top', fontsize=12)
    
    return fig

def process_case_with_proper_alignment(case_name, results_dirs, output_dir):
    """Process a case with proper alignment"""
    
    print(f"🔍 Processing case: {case_name}")
    
    # Find original image
    original_path = find_original_image(case_name)
    if not original_path:
        print(f"  ❌ Cannot find original image for {case_name}")
        return False
    
    # Load original image
    try:
        original_img = nib.load(original_path)
        original_data = original_img.get_fdata()
        original_affine = original_img.affine
        print(f"  ✅ Loaded original image: {original_path}")
        print(f"     Shape: {original_data.shape}")
    except Exception as e:
        print(f"  ❌ Error loading original image: {e}")
        return False
    
    # Process each model's results
    for result_dir in results_dirs:
        model_name = os.path.basename(result_dir).replace('results_', '')
        
        # Find segmentation file
        seg_files = glob.glob(os.path.join(result_dir, f"{case_name}_segmentation.nii.gz"))
        
        if not seg_files:
            print(f"  ⚠️ No segmentation found for {case_name} in {model_name}")
            continue
        
        seg_file = seg_files[0]
        
        try:
            # Load segmentation
            seg_img = nib.load(seg_file)
            seg_data = seg_img.get_fdata()
            seg_affine = seg_img.affine
            
            print(f"  📄 Processing {model_name} segmentation...")
            print(f"     Seg shape: {seg_data.shape}")
            
            # Resample segmentation to match original if needed
            if seg_data.shape != original_data.shape:
                seg_data_aligned = resample_segmentation_to_original(
                    seg_data, seg_affine, original_data.shape, original_affine
                )
            else:
                seg_data_aligned = seg_data
                print(f"    ✅ Shapes already match")
            
            # Create output directory
            case_output_dir = os.path.join(output_dir, f"{case_name}_{model_name}")
            os.makedirs(case_output_dir, exist_ok=True)
            
            # Create overlay images for different views
            shape = original_data.shape
            slices_info = [
                ('axial', shape[2] // 2, 2, 'Axial (Top-down)'),
                ('sagittal', shape[0] // 2, 0, 'Sagittal (Side)'),
                ('coronal', shape[1] // 2, 1, 'Coronal (Front)')
            ]
            
            for view_name, slice_idx, axis, description in slices_info:
                title = f'{case_name} - {model_name.upper()}\n{description} (Slice {slice_idx})'
                
                try:
                    fig = create_aligned_overlay_image(
                        original_data, seg_data_aligned, title, slice_idx, axis
                    )
                    
                    # Save overlay image
                    overlay_filename = os.path.join(case_output_dir, 
                                                  f'{case_name}_{model_name}_aligned_overlay_{view_name}_slice_{slice_idx}.png')
                    fig.savefig(overlay_filename, dpi=200, bbox_inches='tight', facecolor='white')
                    plt.close()
                    
                    print(f"    ✅ Saved aligned overlay: {overlay_filename}")
                    
                except Exception as e:
                    print(f"    ❌ Error creating {view_name} overlay: {e}")
                    continue
        
        except Exception as e:
            print(f"  ❌ Error processing {model_name}: {e}")
            continue
    
    return True

def create_side_by_side_aligned_comparisons(results_dirs, output_dir):
    """Create side-by-side comparison with proper alignment"""
    
    print(f"\n🖼️ Creating aligned side-by-side comparisons...")
    
    # Find common cases
    common_cases = []
    case_files = {}
    
    for result_dir in results_dirs:
        model_name = os.path.basename(result_dir).replace('results_', '')
        seg_files = glob.glob(os.path.join(result_dir, "*_segmentation.nii.gz"))
        
        for seg_file in seg_files:
            case_name = os.path.basename(seg_file).replace('_segmentation.nii.gz', '')
            if case_name not in case_files:
                case_files[case_name] = {}
            case_files[case_name][model_name] = seg_file
    
    # Find cases with all models
    for case_name, models in case_files.items():
        if len(models) == len(results_dirs):
            common_cases.append(case_name)
    
    print(f"📊 Found {len(common_cases)} common cases for comparison")
    
    if not common_cases:
        return
    
    # Create comparison directory
    comparison_dir = os.path.join(output_dir, "aligned_comparisons")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Process each case
    for case_name in sorted(common_cases)[:3]:  # Limit to first 3 cases for speed
        print(f"🔍 Creating aligned comparison for: {case_name}")
        
        # Find original image
        original_path = find_original_image(case_name)
        if not original_path:
            continue
        
        try:
            original_img = nib.load(original_path)
            original_data = original_img.get_fdata()
            original_affine = original_img.affine
        except:
            continue
        
        # Load and align all segmentations
        aligned_segmentations = {}
        for model_name, seg_file in case_files[case_name].items():
            try:
                seg_img = nib.load(seg_file)
                seg_data = seg_img.get_fdata()
                seg_affine = seg_img.affine
                
                # Align to original
                if seg_data.shape != original_data.shape:
                    seg_aligned = resample_segmentation_to_original(
                        seg_data, seg_affine, original_data.shape, original_affine
                    )
                else:
                    seg_aligned = seg_data
                
                aligned_segmentations[model_name] = seg_aligned
                
            except Exception as e:
                print(f"  ⚠️ Error aligning {model_name}: {e}")
                continue
        
        if len(aligned_segmentations) < 2:
            continue
        
        # Create comparison images
        shape = original_data.shape
        slices_info = [
            ('axial', shape[2] // 2, 2),
            ('coronal', shape[1] // 2, 1)  # Skip sagittal for speed
        ]
        
        for view_name, slice_idx, axis in slices_info:
            try:
                # Create figure
                n_models = len(aligned_segmentations)
                fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 8))
                if n_models == 1:
                    axes = [axes]
                
                # Extract slice from original
                if axis == 0:
                    original_slice = original_data[slice_idx, :, :]
                elif axis == 1:
                    original_slice = original_data[:, slice_idx, :]
                else:
                    original_slice = original_data[:, :, slice_idx]
                
                # Normalize original
                original_slice = np.clip(original_slice, -175, 250)
                original_slice = (original_slice - original_slice.min()) / (original_slice.max() - original_slice.min() + 1e-8)
                
                # Plot each model
                cmap = create_organ_colormap()
                for i, (model_name, seg_data) in enumerate(sorted(aligned_segmentations.items())):
                    # Extract segmentation slice
                    if axis == 0:
                        seg_slice = seg_data[slice_idx, :, :]
                    elif axis == 1:
                        seg_slice = seg_data[:, slice_idx, :]
                    else:
                        seg_slice = seg_data[:, :, slice_idx]
                    
                    # Display original
                    axes[i].imshow(original_slice, cmap='gray', alpha=1.0, origin='lower')
                    
                    # Overlay segmentation
                    seg_masked = np.ma.masked_where(seg_slice == 0, seg_slice)
                    axes[i].imshow(seg_masked, cmap=cmap, vmin=0, vmax=13, alpha=0.6, origin='lower')
                    
                    axes[i].set_title(f'{model_name.upper()}', fontsize=16, fontweight='bold')
                    axes[i].axis('off')
                
                plt.suptitle(f'{case_name} - {view_name.title()} Aligned Comparison (Slice {slice_idx})', 
                            fontsize=18, fontweight='bold')
                plt.tight_layout()
                
                # Save comparison
                comparison_filename = os.path.join(comparison_dir, 
                                                 f'{case_name}_aligned_comparison_{view_name}_slice_{slice_idx}.png')
                plt.savefig(comparison_filename, dpi=200, bbox_inches='tight', facecolor='white')
                plt.close()
                
                print(f"  ✅ Saved aligned comparison: {comparison_filename}")
                
            except Exception as e:
                print(f"  ❌ Error creating {view_name} comparison: {e}")
                continue

def main():
    """Main function with proper alignment"""
    
    print("🎯 Creating PROPERLY ALIGNED Segmentation Overlays")
    print("=" * 60)
    
    # Find results directories
    results_dirs = [d for d in glob.glob("results_*") if os.path.isdir(d)]
    
    if not results_dirs:
        print("❌ No results directories found")
        return
    
    print(f"📁 Found result directories: {results_dirs}")
    
    # Create output directory
    output_dir = "./aligned_overlay_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all cases
    all_cases = set()
    for result_dir in results_dirs:
        seg_files = glob.glob(os.path.join(result_dir, "*_segmentation.nii.gz"))
        for seg_file in seg_files:
            case_name = os.path.basename(seg_file).replace('_segmentation.nii.gz', '')
            all_cases.add(case_name)
    
    print(f"📊 Found {len(all_cases)} unique cases")
    
    # Process each case with proper alignment
    for case_name in sorted(all_cases):
        success = process_case_with_proper_alignment(case_name, results_dirs, output_dir)
        if not success:
            continue
    
    # Create aligned comparisons
    create_side_by_side_aligned_comparisons(results_dirs, output_dir)
    
    print(f"\n🎉 Aligned overlay creation complete!")
    print(f"📁 All properly aligned images saved to: {output_dir}")
    print(f"🔧 Fixed alignment issues by resampling segmentations to match original dimensions")
    print(f"🎯 Images should now be perfectly aligned!")

if __name__ == "__main__":
    main()