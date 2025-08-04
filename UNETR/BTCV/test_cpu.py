#!/usr/bin/env python3
"""
Simple performance test for UNETR models - just adds timing to existing test.py
"""

import argparse
import os
import numpy as np
import torch
import time
from networks.unetr import UNETR
from trainer import dice
from utils.data_utils import get_loader
from monai.inferers import sliding_window_inference
import nibabel as nib

parser = argparse.ArgumentParser(description="UNETR segmentation pipeline with timing")
parser.add_argument("--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory")
parser.add_argument("--data_dir", default="./dataset/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument("--pretrained_model_name", default="UNETR_model_best_acc.pth", type=str, help="pretrained model name")
parser.add_argument("--saved_checkpoint", default="ckpt", type=str, help="Supports torchscript or ckpt pretrained checkpoint type")
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
parser.add_argument("--save_outputs", action="store_true", help="save segmentation outputs")
parser.add_argument("--output_dir", default="./outputs", type=str, help="directory to save outputs")


def main():
    args = parser.parse_args()
    args.test_mode = True
    
    # Load data
    val_loader = get_loader(args)
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    
    print(f"🎯 Device: {device}")
    print(f"📂 Model: {pretrained_pth}")
    
    # Create output directory if saving
    if args.save_outputs:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"💾 Saving outputs to: {args.output_dir}")
    
    # Load model
    if args.saved_checkpoint == "torchscript":
        model = torch.jit.load(pretrained_pth)
    elif args.saved_checkpoint == "ckpt":
        model = UNETR(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            feature_size=args.feature_size,
            hidden_size=args.hidden_size,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            pos_embed="perceptron",  # Fixed value
            norm_name="instance",    # Fixed value
            conv_block=True,
            res_block=True,
            dropout_rate=args.dropout_rate,
        )
        
        # Load model weights with error handling
        try:
            model_dict = torch.load(pretrained_pth, map_location=device)
            model.load_state_dict(model_dict, strict=False)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"⚠️ Model loading warning: {e}")
            try:
                # Try loading with CPU mapping
                model_dict = torch.load(pretrained_pth, map_location='cpu')
                model.load_state_dict(model_dict, strict=False)
                print("✅ Model loaded with CPU mapping")
            except Exception as e2:
                print(f"❌ Failed to load model: {e2}")
                return
    
    model.eval()
    model.to(device)
    
    print("🔍 Starting inference with timing...")
    
    # Track performance
    dice_scores = []
    inference_times = []
    total_start = time.time()
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            # Move data to device
            val_inputs = batch["image"].to(device)
            val_labels = batch["label"].to(device)
            
            # Get image name safely
            try:
                img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            except (KeyError, IndexError):
                img_name = f"case_{i+1:03d}"
            
            print(f"Processing: {img_name}")
            
            # Time the inference
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            start_time = time.time()
            
            # Run inference
            val_outputs = sliding_window_inference(
                val_inputs, (96, 96, 96), 4, model, overlap=args.infer_overlap
            )
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.time()
            inference_time = end_time - start_time
            inference_times.append(inference_time)
            
            # Process outputs
            val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
            val_labels = val_labels.cpu().numpy()[:, 0, :, :, :]
            
            # Calculate dice scores
            dice_list_sub = []
            for j in range(1, 14):
                organ_dice = dice(val_outputs[0] == j, val_labels[0] == j)
                dice_list_sub.append(organ_dice)
            
            mean_dice = np.mean(dice_list_sub)
            dice_scores.append(mean_dice)
            
            # Save outputs if requested
            if args.save_outputs:
                try:
                    # Save segmentation as NIfTI
                    seg_filename = os.path.join(args.output_dir, f"{img_name.replace('.nii.gz', '')}_segmentation.nii.gz")
                    seg_img = nib.Nifti1Image(val_outputs[0].astype(np.uint8), np.eye(4))
                    nib.save(seg_img, seg_filename)
                    
                    # Save ground truth
                    gt_filename = os.path.join(args.output_dir, f"{img_name.replace('.nii.gz', '')}_ground_truth.nii.gz")
                    gt_img = nib.Nifti1Image(val_labels[0].astype(np.uint8), np.eye(4))
                    nib.save(gt_img, gt_filename)
                    
                    print(f"  💾 Saved: {seg_filename}")
                    
                except Exception as e:
                    print(f"  ⚠️ Save error: {e}")
            
            print(f"  Dice: {mean_dice:.4f} | Time: {inference_time:.3f}s")
    
    total_time = time.time() - total_start
    
    # Print results
    print(f"\n{'='*50}")
    print(f"📊 PERFORMANCE RESULTS")
    print(f"{'='*50}")
    print(f"🎯 Overall Mean Dice: {np.mean(dice_scores):.4f}")
    print(f"⏱️ Average inference time: {np.mean(inference_times):.3f}s ± {np.std(inference_times):.3f}s")
    print(f"🚀 Average FPS: {1.0/np.mean(inference_times):.2f}")
    print(f"⏰ Total time: {total_time:.1f}s")
    print(f"📈 Cases processed: {len(dice_scores)}")
    
    # Show per-case breakdown
    print(f"\n📋 Per-case results:")
    for i, (dice_score, inf_time) in enumerate(zip(dice_scores, inference_times)):
        print(f"  Case {i+1}: Dice={dice_score:.4f}, Time={inf_time:.3f}s")


if __name__ == "__main__":
    main()