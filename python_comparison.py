#!/usr/bin/env python3
"""
Convert NIfTI files to viewable PNG images with proper organ colors
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import glob

def create_organ_colormap():
    """Create colormap for the 14 organs"""
    colors = [
        '#000000',  # 0: Background (black)
        '#FF0000',  # 1: Spleen (red)
        '#00FF00',  # 2: Right Kidney (green)
        '#0000FF',  # 3: Left Kidney (blue)
        '#FFFF00',  # 4: Gallbladder (yellow)
        '#FF00FF',  # 5: Esophagus (magenta)
        '#00FFFF',  # 6: Liver (cyan)
        '#FFA500',  # 7: Stomach (orange)
        '#800080',  # 8: Aorta (purple)
        '#A52A2A',  # 9: IVC (brown)
        '#FFC0CB',  # 10: Portal/Splenic Veins (pink)
        '#808080',  # 11: Pancreas (gray)
        '#808000',  # 12: Right Adrenal (olive)
        '#000080'   # 13: Left Adrenal (navy)
    ]
    return ListedColormap(colors)

def get_organ_names():
    """Get organ names for legend"""
    return [
        "Background", "Spleen", "Right Kidney", "Left Kidney", "Gallbladder",
        "Esophagus", "Liver", "Stomach", "Aorta", "IVC", 
        "Portal/Splenic Veins", "Pancreas", "Right Adrenal", "Left Adrenal"
    ]

def convert_nifti_to_png(nifti_path, output_dir, case_name, file_type):
    """Convert a single NIfTI file to PNG slices"""
    
    try:
        # Load NIfTI file
        img = nib.load(nifti_path)
        data = img.get_fdata()
        
        print(f"  📄 Processing: {os.path.basename(nifti_path)}")
        print(f"      Shape: {data.shape}")
        
        # Create colormap
        cmap = create_organ_colormap()
        
        # Get interesting slices (middle slices from each dimension)
        shape = data.shape
        slices_info = [
            ('axial', shape[2] // 2, 2, 'Top-down view'),
            ('sagittal', shape[0] // 2, 0, 'Side view'),
            ('coronal', shape[1] // 2, 1, 'Front view')
        ]
        
        # Create output directory for this case
        case_output_dir = os.path.join(output_dir, case_name)
        os.makedirs(case_output_dir, exist_ok=True)
        
        for view_name, slice_idx, axis, description in slices_info:
            # Extract slice
            if axis == 0:
                slice_data = data[slice_idx, :, :]
            elif axis == 1:
                slice_data = data[:, slice_idx, :]
            else:
                slice_data = data[:, :, slice_idx]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Display image
            im = ax.imshow(slice_data, cmap=cmap, vmin=0, vmax=13)
            ax.set_title(f'{case_name} - {file_type.title()}\n{description} (Slice {slice_idx})', 
                        fontsize=14, fontweight='bold')
            ax.axis('off')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Organ Labels', rotation=270, labelpad=20)
            
            # Save PNG
            png_filename = os.path.join(case_output_dir, 
                                      f'{case_name}_{file_type}_{view_name}_slice_{slice_idx}.png')
            plt.savefig(png_filename, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"      ✅ Saved: {png_filename}")
        
        return True
        
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return False

def create_comparison_images(results_dirs, output_dir):
    """Create side-by-side comparison images"""
    
    print("\n🖼️ Creating comparison images...")
    
    # Find common cases across all result directories
    all_cases = {}
    
    for result_dir in results_dirs:
        model_name = os.path.basename(result_dir).replace('results_', '')
        cases = {}
        
        # Find all segmentation files
        seg_files = glob.glob(os.path.join(result_dir, "*_segmentation.nii.gz"))
        
        for seg_file in seg_files:
            case_name = os.path.basename(seg_file).replace('_segmentation.nii.gz', '')
            cases[case_name] = seg_file
        
        all_cases[model_name] = cases
    
    # Find common cases
    if not all_cases:
        print("❌ No cases found")
        return
    
    # Get intersection of all case names
    common_cases = set(list(all_cases.values())[0].keys())
    for cases in all_cases.values():
        common_cases &= set(cases.keys())
    
    print(f"📊 Found {len(common_cases)} common cases: {sorted(common_cases)}")
    
    if not common_cases:
        print("❌ No common cases found across all models")
        return
    
    # Create comparison directory
    comparison_dir = os.path.join(output_dir, "comparisons")
    os.makedirs(comparison_dir, exist_ok=True)
    
    cmap = create_organ_colormap()
    
    # Process each common case
    for case_name in sorted(common_cases):
        print(f"🔍 Creating comparison for: {case_name}")
        
        # Load all model results for this case
        model_data = {}
        for model_name, cases in all_cases.items():
            if case_name in cases:
                try:
                    img = nib.load(cases[case_name])
                    model_data[model_name] = img.get_fdata()
                except Exception as e:
                    print(f"  ⚠️ Error loading {model_name}: {e}")
        
        if len(model_data) < 2:
            print(f"  ⚠️ Skipping {case_name}: insufficient data")
            continue
        
        # Load ground truth if available
        gt_data = None
        for model_name in model_data.keys():
            gt_file = os.path.join(f"results_{model_name}", f"{case_name}_ground_truth.nii.gz")
            if os.path.exists(gt_file):
                try:
                    gt_img = nib.load(gt_file)
                    gt_data = gt_img.get_fdata()
                    break
                except:
                    continue
        
        # Create comparison images for different views
        shape = list(model_data.values())[0].shape
        slices_info = [
            ('axial', shape[2] // 2, 2),
            ('sagittal', shape[0] // 2, 0),
            ('coronal', shape[1] // 2, 1)
        ]
        
        for view_name, slice_idx, axis in slices_info:
            # Create figure with subplots
            n_models = len(model_data)
            n_cols = n_models + (1 if gt_data is not None else 0)
            
            fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
            if n_cols == 1:
                axes = [axes]
            
            col_idx = 0
            
            # Plot ground truth first if available
            if gt_data is not None:
                if axis == 0:
                    gt_slice = gt_data[slice_idx, :, :]
                elif axis == 1:
                    gt_slice = gt_data[:, slice_idx, :]
                else:
                    gt_slice = gt_data[:, :, slice_idx]
                
                im = axes[col_idx].imshow(gt_slice, cmap=cmap, vmin=0, vmax=13)
                axes[col_idx].set_title('Ground Truth', fontsize=12, fontweight='bold')
                axes[col_idx].axis('off')
                col_idx += 1
            
            # Plot model results
            for model_name, data in sorted(model_data.items()):
                if axis == 0:
                    model_slice = data[slice_idx, :, :]
                elif axis == 1:
                    model_slice = data[:, slice_idx, :]
                else:
                    model_slice = data[:, :, slice_idx]
                
                axes[col_idx].imshow(model_slice, cmap=cmap, vmin=0, vmax=13) 
                axes[col_idx].set_title(f'{model_name.upper()}', fontsize=12, fontweight='bold')
                axes[col_idx].axis('off')
                col_idx += 1
            
            # Add colorbar
            plt.colorbar(im, ax=axes, orientation='horizontal', pad=0.1, shrink=0.8)
            
            plt.suptitle(f'{case_name} - {view_name.title()} View (Slice {slice_idx})', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save comparison
            comparison_filename = os.path.join(comparison_dir, 
                                             f'{case_name}_comparison_{view_name}_slice_{slice_idx}.png')
            plt.savefig(comparison_filename, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"  ✅ Saved: {comparison_filename}")

def create_legend():
    """Create a legend showing organ colors and names"""
    
    colors = create_organ_colormap().colors
    organ_names = get_organ_names()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create color patches
    for i, (color, name) in enumerate(zip(colors, organ_names)):
        ax.add_patch(plt.Rectangle((0, i), 1, 0.8, facecolor=color, edgecolor='black'))
        ax.text(1.2, i + 0.4, f"{i}: {name}", va='center', fontsize=12)
    
    ax.set_xlim(0, 6)
    ax.set_ylim(-0.5, len(organ_names))
    ax.set_title('Organ Segmentation Legend', fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    return fig

def main():
    """Main conversion workflow"""
    
    print("🖼️ Converting NIfTI files to PNG images")
    print("=" * 50)
    
    # Find all results directories
    results_dirs = [d for d in glob.glob("results_*") if os.path.isdir(d)]
    
    if not results_dirs:
        print("❌ No results directories found (looking for results_*)")
        return
    
    print(f"📁 Found result directories: {results_dirs}")
    
    # Create main output directory
    output_dir = "./png_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create and save legend
    print("\n🏷️ Creating organ legend...")
    legend_fig = create_legend()
    legend_path = os.path.join(output_dir, "organ_legend.png")
    legend_fig.savefig(legend_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Legend saved: {legend_path}")
    
    # Process each results directory
    for result_dir in results_dirs:
        model_name = os.path.basename(result_dir).replace('results_', '')
        print(f"\n📂 Processing {model_name.upper()} results...")
        
        # Find all NIfTI files
        nifti_files = glob.glob(os.path.join(result_dir, "*.nii.gz"))
        
        if not nifti_files:
            print(f"  ⚠️ No NIfTI files found in {result_dir}")
            continue
        
        print(f"  📄 Found {len(nifti_files)} NIfTI files")
        
        # Process each file
        for nifti_file in sorted(nifti_files):
            filename = os.path.basename(nifti_file)
            
            # Extract case name and file type
            if "_segmentation.nii.gz" in filename:
                case_name = filename.replace("_segmentation.nii.gz", "")
                file_type = "segmentation"
            elif "_ground_truth.nii.gz" in filename:
                case_name = filename.replace("_ground_truth.nii.gz", "")  
                file_type = "ground_truth"
            else:
                case_name = filename.replace(".nii.gz", "")
                file_type = "unknown"
            
            # Convert to PNG
            convert_nifti_to_png(nifti_file, output_dir, f"{case_name}_{model_name}", file_type)
    
    # Create comparison images
    create_comparison_images(results_dirs, output_dir)
    
    print(f"\n🎉 Conversion complete!")
    print(f"📁 All PNG images saved to: {output_dir}")
    print(f"🖼️ Check the 'comparisons' folder for side-by-side model comparisons")
    print(f"🏷️ Use 'organ_legend.png' to understand the colors")

if __name__ == "__main__":
    main()