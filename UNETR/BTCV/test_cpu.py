#!/usr/bin/env python3
"""
Fixed test script for UNETR quantized models
This version handles path issues and quantized model loading
"""

import argparse
import os
import numpy as np
import torch
from networks.unetr import UNETR
from trainer import dice
from utils.data_utils import get_loader
from monai.inferers import sliding_window_inference
import nibabel as nib
from datetime import datetime
import time
import psutil

parser = argparse.ArgumentParser(description="UNETR segmentation pipeline")
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="./dataset/", type=str, help="dataset directory")  # Fixed default path
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name", default="UNETR_model_best_acc.pth", type=str, help="pretrained model name"
)
parser.add_argument(
    "--saved_checkpoint", default="ckpt", type=str, help="Supports torchscript or ckpt pretrained checkpoint type"
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
parser.add_argument("--quantized", action="store_true", help="load quantized model")
parser.add_argument("--model_type", default="original", choices=["original", "int8", "fp16"], help="quantized model type")
parser.add_argument("--save_outputs", action="store_true", help="save segmentation outputs for comparison")
parser.add_argument("--output_dir", default="./comparison_outputs", type=str, help="directory to save outputs")


def benchmark_model_performance(model, sample_input, device, model_type, warmup_runs=3, benchmark_runs=10):
    """Benchmark model inference speed and memory usage"""
    
    print(f"🔥 Warming up {model_type.upper()} model ({warmup_runs} runs)...")
    model.eval()
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            if model_type == "fp16" and device.type == "cuda":
                sample_input = sample_input.half()
            _ = model(sample_input)
    
    # Benchmark runs
    print(f"⏱️ Benchmarking {model_type.upper()} model ({benchmark_runs} runs)...")
    times = []
    memory_usage = []
    
    with torch.no_grad():
        for i in range(benchmark_runs):
            # Memory before inference
            if device.type == "cuda":
                torch.cuda.synchronize()
                memory_before = torch.cuda.memory_allocated() / 1024**2  # MB
            else:
                memory_before = psutil.Process().memory_info().rss / 1024**2  # MB
            
            # Time inference
            start_time = time.perf_counter()
            
            if model_type == "fp16" and device.type == "cuda":
                sample_input_fp16 = sample_input.half()
                output = model(sample_input_fp16)
            else:
                output = model(sample_input)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            inference_time = end_time - start_time
            times.append(inference_time)
            
            # Memory after inference
            if device.type == "cuda":
                memory_after = torch.cuda.memory_allocated() / 1024**2  # MB
            else:
                memory_after = psutil.Process().memory_info().rss / 1024**2  # MB
            
            memory_usage.append(memory_after - memory_before)
    
    # Calculate statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    avg_memory = np.mean(memory_usage)
    
    benchmark_results = {
        'avg_inference_time': avg_time,
        'std_inference_time': std_time,
        'min_inference_time': min_time,
        'max_inference_time': max_time,
        'avg_memory_usage': avg_memory,
        'fps': 1.0 / avg_time,
        'times': times
    }
    
    print(f"📊 {model_type.upper()} Benchmark Results:")
    print(f"   Average inference time: {avg_time:.3f}s ± {std_time:.3f}s")
    print(f"   FPS: {1.0/avg_time:.2f}")
    print(f"   Memory usage: {avg_memory:.1f} MB")
    print(f"   Range: {min_time:.3f}s - {max_time:.3f}s")
    
    return benchmark_results
    """Save segmentation outputs and create comparison visualizations"""
    
    # Create output directory structure
    case_dir = os.path.join(output_dir, case_name.replace('.nii.gz', ''))
    os.makedirs(case_dir, exist_ok=True)
    
    # Save raw segmentation
    seg_filename = os.path.join(case_dir, f"{model_type}_segmentation.nii.gz")
    
    # Create NIfTI image with proper header (copy from original if available)
    if hasattr(original_image, 'affine'):
        affine = original_image.affine
        header = original_image.header
    else:
        # Default affine if not available
        affine = np.eye(4)
        header = None
    
    # Save segmentation
    seg_img = nib.Nifti1Image(outputs.astype(np.uint8), affine, header)
    nib.save(seg_img, seg_filename)
    
    # Save ground truth (only once per case)
    gt_filename = os.path.join(case_dir, "ground_truth.nii.gz")
    if not os.path.exists(gt_filename):
        gt_img = nib.Nifti1Image(labels.astype(np.uint8), affine, header)
        nib.save(gt_img, gt_filename)
    
    print(f"💾 Saved {model_type} segmentation: {seg_filename}")
    
    return case_dir


def create_comparison_slices(case_dir, case_name):
    """Create comparison images showing different model outputs side by side"""
    
    # Look for available model outputs
    model_files = {}
    gt_file = None
    
    for file in os.listdir(case_dir):
        if file.endswith('_segmentation.nii.gz'):
            model_name = file.replace('_segmentation.nii.gz', '')
            model_files[model_name] = os.path.join(case_dir, file)
        elif file == 'ground_truth.nii.gz':
            gt_file = os.path.join(case_dir, file)
    
    if len(model_files) < 2 or gt_file is None:
        return  # Need at least 2 models and ground truth for comparison
    
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        
        # Load all segmentations
        segmentations = {}
        for model_name, file_path in model_files.items():
            seg_img = nib.load(file_path)
            segmentations[model_name] = seg_img.get_fdata()
        
        # Load ground truth
        gt_img = nib.load(gt_file)
        gt_data = gt_img.get_fdata()
        
        # Create colormap for organs (14 classes)
        colors = ['black', 'red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 
                 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'navy']
        cmap = ListedColormap(colors[:14])
        
        # Get middle slices from each dimension
        shape = gt_data.shape
        slices_to_show = [
            ('axial', shape[2] // 2, 2),
            ('sagittal', shape[0] // 2, 0),
            ('coronal', shape[1] // 2, 1)
        ]
        
        for slice_name, slice_idx, axis in slices_to_show:
            # Create figure
            n_models = len(segmentations) + 1  # +1 for ground truth
            fig, axes = plt.subplots(1, n_models, figsize=(4*n_models, 4))
            if n_models == 1:
                axes = [axes]
            
            # Plot ground truth
            if axis == 0:
                gt_slice = gt_data[slice_idx, :, :]
            elif axis == 1:
                gt_slice = gt_data[:, slice_idx, :]
            else:
                gt_slice = gt_data[:, :, slice_idx]
            
            axes[0].imshow(gt_slice, cmap=cmap, vmin=0, vmax=13)
            axes[0].set_title('Ground Truth')
            axes[0].axis('off')
            
            # Plot model outputs
            for i, (model_name, seg_data) in enumerate(segmentations.items()):
                if axis == 0:
                    model_slice = seg_data[slice_idx, :, :]
                elif axis == 1:
                    model_slice = seg_data[:, slice_idx, :]
                else:
                    model_slice = seg_data[:, :, slice_idx]
                
                axes[i+1].imshow(model_slice, cmap=cmap, vmin=0, vmax=13)
                axes[i+1].set_title(f'{model_name.upper()} Model')
                axes[i+1].axis('off')
            
            plt.tight_layout()
            
            # Save comparison image
            comparison_filename = os.path.join(case_dir, f'comparison_{slice_name}_slice_{slice_idx}.png')
            plt.savefig(comparison_filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"🖼️ Saved comparison: {comparison_filename}")
    
    except ImportError:
        print("⚠️ matplotlib not available, skipping visual comparisons")
    except Exception as e:
        print(f"⚠️ Error creating comparisons: {e}")


def create_organ_legend():
    """Create a legend showing organ labels and colors"""
    organ_names = [
        "Background", "Spleen", "Right Kidney", "Left Kidney", "Gallbladder",
        "Esophagus", "Liver", "Stomach", "Aorta", "IVC", 
        "Portal/Splenic Veins", "Pancreas", "Right Adrenal", "Left Adrenal"
    ]
    
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        import matplotlib.patches as patches
        
        colors = ['black', 'red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 
                 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'navy']
        
        fig, ax = plt.subplots(figsize=(8, 10))
        
        for i, (organ, color) in enumerate(zip(organ_names, colors)):
            rect = patches.Rectangle((0, i), 1, 0.8, facecolor=color, edgecolor='black')
            ax.add_patch(rect)
            ax.text(1.1, i+0.4, f"{i}: {organ}", va='center', fontsize=10)
        
        ax.set_xlim(0, 4)
        ax.set_ylim(-0.5, len(organ_names))
        ax.set_title('Organ Segmentation Legend', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        return fig
    
    except ImportError:
        return None
    """
    Load pretrained model with compatibility handling
    """
    print(f"🔄 Loading model from: {pretrained_path}")
    
    # Load the state dict
    checkpoint = torch.load(pretrained_path, map_location=torch.device('cpu'))
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("📦 Found 'state_dict' in checkpoint")
    else:
        state_dict = checkpoint
        print("📦 Using checkpoint directly as state_dict")
    
    # Remove 'module.' prefix if present (from DataParallel training)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # remove `module.`
        new_state_dict[name] = v
    
    # Filter out incompatible keys
    model_keys = set(model.state_dict().keys())
    pretrained_keys = set(new_state_dict.keys())
    
    # Find compatible keys
    compatible_keys = model_keys.intersection(pretrained_keys)
    incompatible_keys = pretrained_keys - model_keys
    missing_keys = model_keys - pretrained_keys
    
    print(f"✅ Compatible keys: {len(compatible_keys)}")
    if incompatible_keys:
        print(f"⚠️ Incompatible keys (ignored): {len(incompatible_keys)}")
        for key in list(incompatible_keys)[:5]:  # Show first 5
            print(f"   - {key}")
        if len(incompatible_keys) > 5:
            print(f"   ... and {len(incompatible_keys) - 5} more")
    
    if missing_keys:
        print(f"⚠️ Missing keys (keeping default): {len(missing_keys)}")
        for key in list(missing_keys)[:5]:  # Show first 5
            print(f"   - {key}")
        if len(missing_keys) > 5:
            print(f"   ... and {len(missing_keys) - 5} more")
    
    # Create filtered state dict with only compatible keys
    filtered_state_dict = {k: v for k, v in new_state_dict.items() if k in compatible_keys}
    
    # Load the filtered state dict
    model.load_state_dict(filtered_state_dict, strict=False)
    
    return model


def load_quantized_model(args, pretrained_pth):
    """Load quantized model based on type"""
    
    if args.model_type == "int8":
        print("🔧 Loading INT8 quantized model...")
        
        # Create base model
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
        
        # Apply quantization
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear, torch.nn.Conv3d}, dtype=torch.qint8
        )
        
        # Load quantized weights
        model = load_pretrained_model_compatible(model, pretrained_pth)
        print("✅ INT8 quantized model loaded")
        
    elif args.model_type == "fp16":
        print("🔧 Loading FP16 quantized model...")
        
        # Create base model
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
        
        # Load weights and convert to FP16
        model = load_pretrained_model_compatible(model, pretrained_pth)
        model = model.half()
        print("✅ FP16 quantized model loaded")
        
    else:  # original
        print("🔧 Loading original model...")
        
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
        
        model = load_pretrained_model_compatible(model, pretrained_pth)
        print("✅ Original model loaded")
    
    return model


def main():
    args = parser.parse_args()
    args.test_mode = True
    
    # Fix paths
    args.data_dir = os.path.abspath(args.data_dir)
    args.pretrained_dir = os.path.abspath(args.pretrained_dir)
    
    print(f"📁 Data directory: {args.data_dir}")
    print(f"📁 Pretrained directory: {args.pretrained_dir}")
    
    # Check if dataset exists
    json_path = os.path.join(args.data_dir, args.json_list)
    if not os.path.exists(json_path):
        print(f"❌ Dataset JSON not found: {json_path}")
        print("💡 Available files in data directory:")
        if os.path.exists(args.data_dir):
            files = os.listdir(args.data_dir)
            for f in files[:10]:  # Show first 10 files
                print(f"   - {f}")
        else:
            print(f"   Data directory does not exist: {args.data_dir}")
        return
    
    # Load validation data
    print("📊 Loading validation data...")
    val_loader = get_loader(args)
    
    # Set up model paths
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    
    print(f"🎯 Device: {device}")
    print(f"📂 Model path: {pretrained_pth}")
    
    # Check if model file exists
    if not os.path.exists(pretrained_pth):
        print(f"❌ Model file not found: {pretrained_pth}")
        print("💡 Available files in pretrained directory:")
        if os.path.exists(pretrained_dir):
            files = os.listdir(pretrained_dir)
            for f in files:
                print(f"   - {f}")
        return
    
    # Load model
    if args.saved_checkpoint == "torchscript":
        print("🔧 Loading TorchScript model...")
        model = torch.jit.load(pretrained_pth)
    elif args.saved_checkpoint == "ckpt":
        model = load_quantized_model(args, pretrained_pth)
    else:
        raise ValueError(f"Unsupported checkpoint type: {args.saved_checkpoint}")
    
    model.eval()
    
    # Move model to device (be careful with quantized models)
    if args.model_type == "int8":
        # INT8 quantized models typically run on CPU
        device = torch.device("cpu")
        print("🔧 INT8 model running on CPU")
    elif args.model_type == "fp16":
        if device.type == "cuda":
            model = model.to(device)
        else:
            print("⚠️ FP16 model needs GPU, falling back to CPU with float32")
            model = model.float().to(device)
    else:
        model = model.to(device)
    
    print(f"🎯 Model loaded on: {device}")
    
    # Set up output directory if saving outputs
    if args.save_outputs:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args.output_dir, f"{args.model_type}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Saving outputs to: {output_dir}")
        
        # Create organ legend
        try:
            legend_fig = create_organ_legend()
            if legend_fig:
                legend_path = os.path.join(args.output_dir, "organ_legend.png")
                legend_fig.savefig(legend_path, dpi=150, bbox_inches='tight')
                print(f"🏷️ Saved organ legend: {legend_path}")
        except Exception as e:
            print(f"⚠️ Could not create legend: {e}")
    else:
        output_dir = None
    
    # Run inference and benchmark
    print("🔍 Starting inference with performance monitoring...")
    
    # Prepare for benchmarking
    inference_times = []
    case_results = []
    total_start_time = time.perf_counter()
    
    with torch.no_grad():
        dice_list_case = []
        
        for i, batch in enumerate(val_loader):
            case_start_time = time.perf_counter()
            
            # Handle device placement for inputs
            if device.type == "cuda":
                val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            else:
                val_inputs, val_labels = (batch["image"].cpu(), batch["label"].cpu())
            
            # Handle FP16 inputs
            if args.model_type == "fp16" and device.type == "cuda":
                val_inputs = val_inputs.half()
            
            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            print(f"Inference on case {img_name}")
            
            # Time the actual inference
            inference_start = time.perf_counter()
            
            # Run inference
            try:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                
                val_outputs = sliding_window_inference(
                    val_inputs, (96, 96, 96), 4, model, overlap=args.infer_overlap
                )
                
                if device.type == "cuda":
                    torch.cuda.synchronize()
                
                inference_end = time.perf_counter()
                inference_time = inference_end - inference_start
                inference_times.append(inference_time)
                
                # Convert outputs back to float32 for processing
                if args.model_type == "fp16":
                    val_outputs = val_outputs.float()
                
                val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
                val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
                val_labels = val_labels.cpu().numpy()[:, 0, :, :, :]
                
                # Save segmentation outputs if requested
                if args.save_outputs and output_dir:
                    try:
                        # Get original image for proper NIfTI header
                        original_img_path = batch["image_meta_dict"]["filename_or_obj"][0]
                        original_img = nib.load(original_img_path)
                        
                        case_dir = save_segmentation_outputs(
                            val_outputs[0], val_labels[0], original_img, 
                            img_name, args.model_type, output_dir
                        )
                        
                        # Create comparison if multiple models have been run
                        create_comparison_slices(case_dir, img_name)
                        
                    except Exception as e:
                        print(f"⚠️ Error saving outputs for {img_name}: {e}")
                
                # Calculate dice scores
                dice_list_sub = []
                for j in range(1, 14):
                    organ_Dice = dice(val_outputs[0] == j, val_labels[0] == j)
                    dice_list_sub.append(organ_Dice)
                
                mean_dice = np.mean(dice_list_sub)
                case_end_time = time.perf_counter()
                total_case_time = case_end_time - case_start_time
                
                print(f"Mean Organ Dice: {mean_dice:.4f} | Inference time: {inference_time:.3f}s | Total time: {total_case_time:.3f}s")
                
                dice_list_case.append(mean_dice)
                case_results.append({
                    'case_name': img_name,
                    'dice_score': mean_dice,
                    'inference_time': inference_time,
                    'total_time': total_case_time,
                    'organ_dice_scores': dice_list_sub
                })
                
            except Exception as e:
                print(f"❌ Error during inference on {img_name}: {e}")
                continue
        
        total_end_time = time.perf_counter()
        total_inference_time = total_end_time - total_start_time
        
        if dice_list_case:
            overall_mean_dice = np.mean(dice_list_case)
            avg_inference_time = np.mean(inference_times)
            std_inference_time = np.std(inference_times)
            total_cases = len(dice_list_case)
            
            print(f"\n{'='*60}")
            print(f"🎉 PERFORMANCE SUMMARY - {args.model_type.upper()} MODEL")
            print(f"{'='*60}")
            print(f"📊 Overall Mean Dice Score: {overall_mean_dice:.4f}")
            print(f"📈 Cases Processed: {total_cases}")
            print(f"⏱️ Average Inference Time: {avg_inference_time:.3f}s ± {std_inference_time:.3f}s")
            print(f"🚀 Average FPS: {1.0/avg_inference_time:.2f}")
            print(f"⏰ Total Processing Time: {total_inference_time:.1f}s")
            print(f"📏 Time per Case: {total_inference_time/total_cases:.2f}s")
            
            # Performance breakdown by case
            print(f"\n📋 Per-Case Performance:")
            print(f"{'Case':<15} {'Dice':<8} {'Inf. Time':<10} {'Total Time':<12}")
            print("-" * 50)
            for result in case_results:
                print(f"{result['case_name']:<15} {result['dice_score']:<8.4f} {result['inference_time']:<10.3f} {result['total_time']:<12.3f}")
            
            # Save detailed results
            if args.save_outputs and output_dir:
                results_summary = {
                    'model_type': args.model_type,
                    'overall_dice': overall_mean_dice,
                    'avg_inference_time': avg_inference_time,
                    'std_inference_time': std_inference_time,
                    'total_processing_time': total_inference_time,
                    'fps': 1.0/avg_inference_time,
                    'total_cases': total_cases,
                    'case_results': case_results,
                    'device': str(device),
                    'timestamp': datetime.now().isoformat()
                }
                
                import json
                results_file = os.path.join(output_dir, f"{args.model_type}_performance_results.json")
                with open(results_file, 'w') as f:
                    json.dump(results_summary, f, indent=2)
                
                print(f"\n💾 Detailed results saved to: {results_file}")
            
            if args.save_outputs and output_dir:
                print(f"\n📁 All outputs saved to: {output_dir}")
                print("🖼️ Use the comparison images to visually assess model differences")
                print("💡 Tip: Run this script with different --model_type values to compare all models")
        else:
            print("❌ No cases processed successfully")


if __name__ == "__main__":
    main()