#!/usr/bin/env python3

import os
import subprocess
import sys
from datetime import datetime
import shutil

def run_model_test(model_type, data_dir="./dataset", output_base_dir="./comparison_outputs"):
    """Run test for a specific model type"""
    
    print(f"\n{'='*50}")
    print(f"🔍 Testing {model_type.upper()} Model")
    print(f"{'='*50}")
    
    # Set up paths based on model type
    if model_type == "original":
        pretrained_dir = "./pretrained_models/"
        model_name = "UNETR_model_best_acc.pth"
    else:
        pretrained_dir = "./quantized_models/"
        if model_type == "int8":
            model_name = "unetr_int8_dynamic.pth"
        else:  # fp16
            model_name = "unetr_fp16.pth"
    
    # Create command
    cmd = [
        "python", "test_cpu.py",
        f"--data_dir={data_dir}",
        "--json_list=dataset_0.json",
        f"--pretrained_dir={pretrained_dir}",
        f"--pretrained_model_name={model_name}",
        "--saved_checkpoint=ckpt",
        f"--model_type={model_type}",
        "--save_outputs",
        f"--output_dir={output_base_dir}"
    ]
    
    # Run the test
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            print(f" {model_type.upper()} model test completed successfully")
            return True, result.stdout
        else:
            print(f" {model_type.upper()} model test failed:")
            print(result.stderr)
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f" {model_type.upper()} model test timed out")
        return False, "Timeout"
    except Exception as e:
        print(f" Error running {model_type.upper()} model: {e}")
        return False, str(e)


def create_consolidated_comparisons(output_base_dir):
    """Create consolidated comparison images with all model types"""
    
    print(f"\n{'='*50}")
    print(" Creating Consolidated Comparisons")
    print(f"{'='*50}")
    
    try:
        import matplotlib.pyplot as plt
        import nibabel as nib
        from matplotlib.colors import ListedColormap
        import numpy as np
        
        # Find all model output directories
        model_dirs = {}
        for item in os.listdir(output_base_dir):
            item_path = os.path.join(output_base_dir, item)
            if os.path.isdir(item_path):
                if item.startswith("original_"):
                    model_dirs["original"] = item_path
                elif item.startswith("int8_"):
                    model_dirs["int8"] = item_path
                elif item.startswith("fp16_"):
                    model_dirs["fp16"] = item_path
        
        if len(model_dirs) < 2:
            print(" Need at least 2 model outputs for comparison")
            return
        
        print(f" Found {len(model_dirs)} model outputs: {list(model_dirs.keys())}")
        
        # Create consolidated comparison directory
        comparison_dir = os.path.join(output_base_dir, "consolidated_comparisons")
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Find common cases across all models
        case_sets = []
        for model_name, model_dir in model_dirs.items():
            cases = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
            case_sets.append(set(cases))
        
        common_cases = set.intersection(*case_sets) if case_sets else set()
        print(f" Found {len(common_cases)} common cases: {sorted(list(common_cases))}")
        
        if not common_cases:
            print(" No common cases found across all models")
            return
        
        # Create colormap
        colors = ['black', 'red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 
                 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'navy']
        cmap = ListedColormap(colors[:14])
        
        # Process each common case
        for case_name in sorted(common_cases):
            print(f" Processing case: {case_name}")
            
            case_comparison_dir = os.path.join(comparison_dir, case_name)
            os.makedirs(case_comparison_dir, exist_ok=True)
            
            # Load segmentations from all models
            segmentations = {}
            gt_data = None
            
            for model_name, model_dir in model_dirs.items():
                case_dir = os.path.join(model_dir, case_name)
                
                # Load model segmentation
                seg_file = os.path.join(case_dir, f"{model_name}_segmentation.nii.gz")
                if os.path.exists(seg_file):
                    seg_img = nib.load(seg_file)
                    segmentations[model_name] = seg_img.get_fdata()
                
                # Load ground truth (only once)
                if gt_data is None:
                    gt_file = os.path.join(case_dir, "ground_truth.nii.gz")
                    if os.path.exists(gt_file):
                        gt_img = nib.load(gt_file)
                        gt_data = gt_img.get_fdata()
            
            if gt_data is None or len(segmentations) < 2:
                print(f" Skipping {case_name}: missing data")
                continue
            
            # Create comprehensive comparison
            shape = gt_data.shape
            middle_slices = [
                ('axial', shape[2] // 2, 2),
                ('sagittal', shape[0] // 2, 0), 
                ('coronal', shape[1] // 2, 1)
            ]
            
            for slice_name, slice_idx, axis in middle_slices:
                # Create figure with all models + ground truth
                n_cols = len(segmentations) + 1
                fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
                if n_cols == 1:
                    axes = [axes]
                
                # Plot ground truth
                if axis == 0:
                    gt_slice = gt_data[slice_idx, :, :]
                elif axis == 1:
                    gt_slice = gt_data[:, slice_idx, :]
                else:
                    gt_slice = gt_data[:, :, slice_idx]
                
                im0 = axes[0].imshow(gt_slice, cmap=cmap, vmin=0, vmax=13)
                axes[0].set_title('Ground Truth', fontsize=14, fontweight='bold')
                axes[0].axis('off')
                
                # Plot all model outputs
                model_names = sorted(segmentations.keys())
                for i, model_name in enumerate(model_names):
                    seg_data = segmentations[model_name]
                    
                    if axis == 0:
                        model_slice = seg_data[slice_idx, :, :]
                    elif axis == 1:
                        model_slice = seg_data[:, slice_idx, :]
                    else:
                        model_slice = seg_data[:, :, slice_idx]
                    
                    axes[i+1].imshow(model_slice, cmap=cmap, vmin=0, vmax=13)
                    axes[i+1].set_title(f'{model_name.upper()} Model', fontsize=14, fontweight='bold')
                    axes[i+1].axis('off')
                
                # Add colorbar
                plt.colorbar(im0, ax=axes, orientation='horizontal', pad=0.1, shrink=0.8)
                
                plt.suptitle(f'{case_name} - {slice_name.title()} View (Slice {slice_idx})', 
                           fontsize=16, fontweight='bold')
                plt.tight_layout()
                
                # Save comparison
                comparison_file = os.path.join(case_comparison_dir, 
                                             f'all_models_{slice_name}_slice_{slice_idx}.png')
                plt.savefig(comparison_file, dpi=200, bbox_inches='tight')
                plt.close()
                
                print(f"  💾 Saved: {comparison_file}")
        
        # Create summary HTML report
        create_html_report(comparison_dir, common_cases, model_dirs.keys())
        
        print(f"\n Consolidated comparisons saved to: {comparison_dir}")
        
    except ImportError:
        print(" matplotlib or nibabel not available for creating comparisons")
    except Exception as e:
        print(f" Error creating consolidated comparisons: {e}")


def create_html_report(comparison_dir, cases, model_types):
    """Create an HTML report for easy viewing of all comparisons"""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>UNETR Model Comparison Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; text-align: center; }}
            h2 {{ color: #666; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
            .case-section {{ margin-bottom: 40px; }}
            .image-grid {{ display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; }}
            .image-item {{ text-align: center; }}
            .image-item img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
            .summary {{ background-color: #f9f9f9; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
            .model-info {{ display: flex; justify-content: space-around; margin: 20px 0; }}
            .model-box {{ background: #e8f4f8; padding: 15px; border-radius: 8px; text-align: center; }}
        </style>
    </head>
    <body>
        <h1>🏥 UNETR Model Comparison Report</h1>
        
        <div class="summary">
            <h2>📊 Summary</h2>
            <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p><strong>Models Compared:</strong> {', '.join(model_types)}</p>
            <p><strong>Cases Analyzed:</strong> {len(cases)}</p>
            
            <div class="model-info">
                <div class="model-box">
                    <h3>🔵 Original Model</h3>
                    <p>Full precision<br>~354 MB</p>
                </div>
                <div class="model-box">
                    <h3>🟡 INT8 Model</h3>
                    <p>8-bit quantization<br>~102 MB (3.5x smaller)</p>
                </div>
                <div class="model-box">
                    <h3>🟢 FP16 Model</h3>
                    <p>Half precision<br>~177 MB (2x smaller)</p>
                </div>
            </div>
        </div>
    """
    
    # Add each case
    for case_name in sorted(cases):
        html_content += f"""
        <div class="case-section">
            <h2>📋 Case: {case_name}</h2>
            <div class="image-grid">
        """
        
        case_dir = os.path.join(comparison_dir, case_name)
        if os.path.exists(case_dir):
            # Find all comparison images for this case
            images = [f for f in os.listdir(case_dir) if f.endswith('.png')]
            images.sort()
            
            for img in images:
                img_path = os.path.join(case_name, img)
                view_name = img.replace('all_models_', '').replace('.png', '').replace('_', ' ').title()
                html_content += f"""
                <div class="image-item">
                    <h4>{view_name}</h4>
                    <img src="{img_path}" alt="{view_name}">
                </div>
                """
        
        html_content += """
            </div>
        </div>
        """
    
    html_content += """
        <div class="summary">
            <h2>🔍 How to Interpret the Images</h2>
            <ul>
                <li><strong>Ground Truth:</strong> The manually annotated reference segmentation</li>
                <li><strong>Colors:</strong> Each organ has a unique color (see legend)</li>
                <li><strong>Views:</strong> Axial (top-down), Sagittal (side), Coronal (front)</li>
                <li><strong>Quality Assessment:</strong> Compare how well each model matches ground truth</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    report_path = os.path.join(comparison_dir, "comparison_report.html")
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"📄 HTML report saved: {report_path}")


def main():
    """Main comparison workflow"""
    
    print("🔬 UNETR Model Comparison Suite")
    print("=" * 60)
    
    # Configuration
    data_dir = "./dataset"
    output_base_dir = "./comparison_outputs"
    models_to_test = ["original", "int8", "fp16"]
    
    # Check prerequisites
    print("🔍 Checking prerequisites...")
    
    # Check if dataset exists
    if not os.path.exists(os.path.join(data_dir, "dataset_0.json")):
        print(f"❌ Dataset not found at {data_dir}/dataset_0.json")
        return
    
    # Check if models exist
    missing_models = []
    if not os.path.exists("./pretrained_models/UNETR_model_best_acc.pth"):
        missing_models.append("original")
    if not os.path.exists("./quantized_models/unetr_int8_dynamic.pth"):
        missing_models.append("int8")
    if not os.path.exists("./quantized_models/unetr_fp16.pth"):
        missing_models.append("fp16")
    
    if missing_models:
        print(f"❌ Missing models: {missing_models}")
        print("💡 Run quantization.py first to create quantized models")
        return
    
    print("✅ All prerequisites found")
    
    # Create base output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Test each model
    results = {}
    successful_models = []
    
    for model_type in models_to_test:
        success, output = run_model_test(model_type, data_dir, output_base_dir)
        results[model_type] = {"success": success, "output": output}
        
        if success:
            successful_models.append(model_type)
            # Extract dice score and timing from output
            try:
                lines = output.split('\n')
                for line in lines:
                    if "Overall Mean Dice:" in line:
                        dice_score = float(line.split(":")[-1].strip())
                        results[model_type]["dice"] = dice_score
                    elif "Average Inference Time:" in line:
                        # Extract time value (format: "1.234s ± 0.056s")
                        time_part = line.split(":")[-1].strip()
                        avg_time = float(time_part.split("s")[0].strip())
                        results[model_type]["avg_time"] = avg_time
                    elif "Average FPS:" in line:
                        fps = float(line.split(":")[-1].strip())
                        results[model_type]["fps"] = fps
            except Exception as e:
                print(f"⚠️ Error parsing results for {model_type}: {e}")
                results[model_type]["dice"] = None
                results[model_type]["avg_time"] = None
                results[model_type]["fps"] = None
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 TESTING SUMMARY")
    print(f"{'='*60}")
    
    for model_type in models_to_test:
        result = results[model_type]
        status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
        dice = f"Dice: {result.get('dice', 'N/A'):.4f}" if result.get('dice') else "Dice: N/A"
        time_info = f"Time: {result.get('avg_time', 'N/A'):.3f}s" if result.get('avg_time') else "Time: N/A"
        fps_info = f"FPS: {result.get('fps', 'N/A'):.2f}" if result.get('fps') else "FPS: N/A"
        print(f"{model_type.upper():>10}: {status:>10} | {dice} | {time_info} | {fps_info}")
    
    # Create consolidated comparisons if we have multiple successful models
    if len(successful_models) >= 2:
        create_consolidated_comparisons(output_base_dir)
        
        # Create performance comparison
        create_performance_summary(results, output_base_dir)
        
        print(f"\n🎉 Comparison complete!")
        print(f"📁 All results saved to: {output_base_dir}")
        print(f"🌐 Open comparison_report.html in a browser to view results")
        
    else:
        print(f"\n⚠️ Need at least 2 successful model runs for comparison")
        print(f"   Successful models: {successful_models}")


def create_performance_summary(results, output_dir):
    """Create a performance summary chart"""
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Extract data for plotting
        models = []
        dice_scores = []
        inference_times = []
        fps_values = []
        model_sizes = {"original": 354, "int8": 102, "fp16": 177}  # MB
        
        for model_type, result in results.items():
            if result["success"] and result.get("dice"):
                models.append(model_type.upper())
                dice_scores.append(result["dice"])
                inference_times.append(result.get("avg_time", 0))
                fps_values.append(result.get("fps", 0))
        
        if len(models) < 2:
            return
        
        # Create comprehensive performance comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        colors = ['#2E86AB', '#A23B72', '#F18F01'][:len(models)]
        
        # 1. Dice scores comparison
        bars1 = ax1.bar(models, dice_scores, color=colors, alpha=0.8)
        ax1.set_ylabel('Dice Score', fontsize=12)
        ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 1)
        
        for bar, score in zip(bars1, dice_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Model size comparison
        sizes = [model_sizes.get(model.lower(), 0) for model in models]
        bars2 = ax2.bar(models, sizes, color=colors, alpha=0.8)
        ax2.set_ylabel('Model Size (MB)', fontsize=12)
        ax2.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
        
        for bar, size in zip(bars2, sizes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{size} MB', ha='center', va='bottom', fontweight='bold')
        
        # 3. Inference time comparison
        if any(t > 0 for t in inference_times):
            bars3 = ax3.bar(models, inference_times, color=colors, alpha=0.8)
            ax3.set_ylabel('Avg Inference Time (s)', fontsize=12)
            ax3.set_title('Inference Speed Comparison', fontsize=14, fontweight='bold')
            
            for bar, time_val in zip(bars3, inference_times):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{time_val:.3f}s', ha='center', va='bottom', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'Timing data not available', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Inference Speed Comparison', fontsize=14, fontweight='bold')
        
        # 4. FPS comparison
        if any(f > 0 for f in fps_values):
            bars4 = ax4.bar(models, fps_values, color=colors, alpha=0.8)
            ax4.set_ylabel('Frames Per Second (FPS)', fontsize=12)
            ax4.set_title('Throughput Comparison', fontsize=14, fontweight='bold')
            
            for bar, fps in zip(bars4, fps_values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{fps:.2f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'FPS data not available', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Throughput Comparison', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save performance chart
        perf_path = os.path.join(output_dir, "performance_comparison.png")
        plt.savefig(perf_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Performance comparison saved: {perf_path}")
        
        # Create efficiency plot (Dice vs Size vs Speed)
        if len(models) >= 2 and any(t > 0 for t in inference_times):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Accuracy vs Size
            scatter1 = ax1.scatter(sizes, dice_scores, c=colors, s=200, alpha=0.7)
            for i, model in enumerate(models):
                ax1.annotate(model, (sizes[i], dice_scores[i]), 
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=12, fontweight='bold')
            
            ax1.set_xlabel('Model Size (MB)', fontsize=12)
            ax1.set_ylabel('Dice Score', fontsize=12)
            ax1.set_title('Accuracy vs Size Trade-off', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Accuracy vs Speed
            scatter2 = ax2.scatter(inference_times, dice_scores, c=colors, s=200, alpha=0.7)
            for i, model in enumerate(models):
                ax2.annotate(model, (inference_times[i], dice_scores[i]), 
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=12, fontweight='bold')
            
            ax2.set_xlabel('Inference Time (s)', fontsize=12)
            ax2.set_ylabel('Dice Score', fontsize=12)
            ax2.set_title('Accuracy vs Speed Trade-off', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            eff_path = os.path.join(output_dir, "efficiency_comparison.png")
            plt.savefig(eff_path, dpi=200, bbox_inches='tight')
            plt.close()
            
            print(f"⚡ Efficiency comparison saved: {eff_path}")
    except ImportError:
        print("❌ matplotlib or numpy not available for creating performance summary")
    except Exception as e:
        print(f"❌ Error creating performance summary: {e}")
        
def create_performance_table(results, output_dir):
    """Create a comprehensive performance table"""
    
    try:
        import pandas as pd
        
        # Prepare data for table
        table_data = []
        model_sizes = {"original": 354, "int8": 102, "fp16": 177}
        
        for model_type, result in results.items():
            if result["success"]:
                row = {
                    'Model': model_type.upper(),
                    'Dice Score': f"{result.get('dice', 0):.4f}" if result.get('dice') else "N/A",
                    'Avg Inference Time (s)': f"{result.get('avg_time', 0):.3f}" if result.get('avg_time') else "N/A",
                    'FPS': f"{result.get('fps', 0):.2f}" if result.get('fps') else "N/A",
                    'Model Size (MB)': model_sizes.get(model_type, "N/A"),
                    'Size Reduction': f"{354/model_sizes.get(model_type, 354):.1f}x" if model_sizes.get(model_type) else "1.0x",
                    'Status': "✅ SUCCESS"
                }
            else:
                row = {
                    'Model': model_type.upper(),
                    'Dice Score': "FAILED",
                    'Avg Inference Time (s)': "FAILED", 
                    'FPS': "FAILED",
                    'Model Size (MB)': model_sizes.get(model_type, "N/A"),
                    'Size Reduction': f"{354/model_sizes.get(model_type, 354):.1f}x" if model_sizes.get(model_type) else "1.0x",
                    'Status': "❌ FAILED"
                }
            
            table_data.append(row)
        
        # Create DataFrame and save as CSV
        df = pd.DataFrame(table_data)
        csv_path = os.path.join(output_dir, "performance_summary.csv")
        df.to_csv(csv_path, index=False)
        
        print(f"📊 Performance table saved: {csv_path}")
        
        # Also save as formatted text
        txt_path = os.path.join(output_dir, "performance_summary.txt")
        with open(txt_path, 'w') as f:
            f.write("UNETR Model Performance Comparison\n")
            f.write("=" * 60 + "\n\n")
            f.write(df.to_string(index=False))
            f.write("\n\n")
            f.write("Key Metrics:\n")
            f.write("- Dice Score: Segmentation accuracy (higher is better)\n")
            f.write("- Inference Time: Time per case (lower is better)\n") 
            f.write("- FPS: Frames per second (higher is better)\n")
            f.write("- Size Reduction: Compression vs original model\n")
        
        print(f"📝 Performance summary saved: {txt_path}")
        
    except ImportError:
        print("⚠️ pandas not available for creating performance table")
    except Exception as e:
        print(f"⚠️ Error creating performance table: {e}")


if __name__ == "__main__":
    main()