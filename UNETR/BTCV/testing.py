#!/usr/bin/env python3
"""
BTCV Testing Data Setup Script for UNETR
This script organizes only what you need for testing the pretrained UNETR model
"""

import os
import shutil
import zipfile
import json
from pathlib import Path

def setup_testing_data(base_path="./dataset", num_test_cases=6):
    """
    Setup BTCV data for UNETR testing only
    
    Args:
        base_path: Path to your dataset directory
        num_test_cases: Number of cases to use for testing (default: 6)
    """
    base_path = Path(base_path)
    base_path.mkdir(exist_ok=True)
    
    print("🧪 Setting up BTCV data for TESTING only...")
    
    # Step 1: Extract zip files
    zip_files = list(base_path.glob("*.zip"))
    for zip_file in zip_files:
        print(f"📦 Extracting {zip_file}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(base_path)
        print(f"✅ Extracted {zip_file}")
    
    # Step 2: Organize structure
    rawdata_path = base_path / "RawData"
    if rawdata_path.exists():
        print("📁 Organizing structure for testing...")
        
        training_src = rawdata_path / "Training"
        training_dst = base_path / "Training"
        
        if training_src.exists():
            if training_dst.exists():
                shutil.rmtree(training_dst)
            shutil.move(str(training_src), str(training_dst))
            print("✅ Moved Training folder")
            
        # Remove RawData folder
        if rawdata_path.exists():
            shutil.rmtree(rawdata_path)
            print("🗑️ Cleaned up RawData folder")
    
    # Step 3: Create minimal JSON for testing (validation split only)
    print(f"📝 Creating test dataset with {num_test_cases} cases...")
    
    training_img = base_path / "Training" / "img"
    training_label = base_path / "Training" / "label"
    
    if not training_img.exists() or not training_label.exists():
        print("❌ Training folders not found!")
        return False
    
    img_files = sorted(list(training_img.glob("*.nii.gz")))
    
    # Use last few cases for validation/testing
    validation_data = []
    for img_file in img_files[-num_test_cases:]:
        case_num = img_file.stem.replace("img", "").replace(".nii", "")
        label_file = training_label / f"label{case_num}.nii.gz"
        
        if label_file.exists():
            validation_data.append({
                "image": f"./Training/img/{img_file.name}",
                "label": f"./Training/label/{label_file.name}"
            })
    
    # Create minimal JSON (only validation needed for testing)
    dataset_json = {
        "validation": validation_data
    }
    
    json_file = base_path / "dataset_0.json"
    with open(json_file, 'w') as f:
        json.dump(dataset_json, f, indent=2)
    
    print(f"✅ Created dataset_0.json with {len(validation_data)} test cases")
    
    # Step 4: Verify setup
    print("\n🔍 Verification:")
    print(f"✅ Test cases ready: {len(validation_data)}")
    print(f"✅ Images available: {len(img_files)}")
    print(f"✅ JSON file created: {json_file.exists()}")
    
    print(f"\n📋 Testing structure:")
    print(f"{base_path}/")
    print("├── dataset_0.json (validation cases only)")
    print("└── Training/")
    print("    ├── img/ (CT scans)")
    print("    └── label/ (ground truth)")
    
    return True

def verify_pretrained_model():
    """Verify the pretrained model is in place"""
    model_path = Path("./pretrained_models/UNETR_model_best_acc.pth")
    
    if model_path.exists():
        print("✅ Pretrained model found")
        return True
    else:
        print("❌ Pretrained model NOT found")
        print("💡 Download it from:")
        print("   https://developer.download.nvidia.com/assets/Clara/monai/research/UNETR_model_best_acc.pth")
        print("   Place it in: ./pretrained_models/UNETR_model_best_acc.pth")
        return False

def show_test_command():
    """Show the command to run testing"""
    print("\n🚀 Ready to test! Run this command:")
    print("=" * 60)
    print("python test.py \\")
    print("    --data_dir=./dataset/ \\")
    print("    --json_list=dataset_0.json \\")
    print("    --pretrained_dir=./pretrained_models/ \\")
    print("    --pretrained_model_name=UNETR_model_best_acc.pth \\")
    print("    --saved_checkpoint=ckpt \\")
    print("    --infer_overlap=0.5")
    print("=" * 60)

if __name__ == "__main__":
    print("🧪 UNETR Testing Setup")
    print("=" * 40)
    
    # Setup testing data
    success = setup_testing_data("./dataset", num_test_cases=6)
    
    if success:
        # Check pretrained model
        verify_pretrained_model()
        
        # Show test command
        show_test_command()
        
        print("\n💖 Setup complete! You're ready to test UNETR!")
    else:
        print("❌ Setup failed. Check your data files.")