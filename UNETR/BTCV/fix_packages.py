#!/usr/bin/env python3
"""
Fix the dataset JSON file to match your actual file structure
"""

import json
import os
from pathlib import Path

def fix_dataset_json(dataset_dir="./dataset"):
    """Fix the JSON file paths to match actual file structure"""
    dataset_dir = Path(dataset_dir)
    json_file = dataset_dir / "dataset_0.json"
    
    if not json_file.exists():
        print("❌ dataset_0.json not found!")
        return False
    
    # Load the current JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print("🔧 Fixing JSON file paths...")
    
    # Check what files actually exist
    training_img_dir = dataset_dir / "Training" / "img"
    training_label_dir = dataset_dir / "Training" / "label"
    
    if not training_img_dir.exists():
        print(f"❌ {training_img_dir} not found!")
        return False
    
    if not training_label_dir.exists():
        print(f"❌ {training_label_dir} not found!")
        return False
    
    # Get actual available files
    img_files = sorted(list(training_img_dir.glob("*.nii.gz")))
    label_files = sorted(list(training_label_dir.glob("*.nii.gz")))
    
    print(f"📁 Found {len(img_files)} images and {len(label_files)} labels")
    
    # Create new dataset with correct paths
    # Use the last 6 cases for validation (testing)
    validation_cases = []
    for img_file in img_files[-6:]:  # Last 6 cases
        case_num = img_file.stem.replace("img", "").replace(".nii", "")
        label_file = training_label_dir / f"label{case_num}.nii.gz"
        
        if label_file.exists():
            validation_cases.append({
                "image": f"./Training/img/{img_file.name}",
                "label": f"./Training/label/{label_file.name}"
            })
    
    # Create training cases (optional, but good to have)
    training_cases = []
    for img_file in img_files[:-6]:  # All but last 6
        case_num = img_file.stem.replace("img", "").replace(".nii", "")
        label_file = training_label_dir / f"label{case_num}.nii.gz"
        
        if label_file.exists():
            training_cases.append({
                "image": f"./Training/img/{img_file.name}",
                "label": f"./Training/label/{label_file.name}"
            })
    
    # Create new JSON structure
    new_data = {
        "description": "BTCV dataset for UNETR testing",
        "labels": {
            "0": "background",
            "1": "spleen",
            "2": "right kidney", 
            "3": "left kidney",
            "4": "gallbladder",
            "5": "esophagus",
            "6": "liver",
            "7": "stomach",
            "8": "aorta",
            "9": "inferior vena cava",
            "10": "portal vein and splenic vein",
            "11": "pancreas",
            "12": "right adrenal gland",
            "13": "left adrenal gland"
        },
        "training": training_cases,
        "validation": validation_cases
    }
    
    # Backup original and save new JSON
    backup_file = json_file.with_suffix('.json.backup')
    if json_file.exists():
        import shutil
        shutil.copy(json_file, backup_file)
        print(f"📋 Backed up original to {backup_file}")
    
    with open(json_file, 'w') as f:
        json.dump(new_data, f, indent=2)
    
    print(f"✅ Fixed dataset_0.json:")
    print(f"   📊 Training cases: {len(training_cases)}")
    print(f"   🧪 Validation cases: {len(validation_cases)}")
    
    # Verify the paths exist
    print("\n🔍 Verifying file paths...")
    missing_files = []
    for case in validation_cases:
        img_path = dataset_dir / case["image"].lstrip("./")
        label_path = dataset_dir / case["label"].lstrip("./")
        
        if not img_path.exists():
            missing_files.append(str(img_path))
        if not label_path.exists():
            missing_files.append(str(label_path))
    
    if missing_files:
        print(f"❌ Missing files:")
        for f in missing_files[:5]:  # Show first 5
            print(f"   - {f}")
        return False
    else:
        print("✅ All validation files exist!")
        return True

if __name__ == "__main__":
    print("🔧 Fixing Dataset JSON Paths")
    print("=" * 40)
    
    success = fix_dataset_json("./dataset")
    
    if success:
        print("\n🎉 JSON file fixed! You can now run:")
        print("python test_cpu.py \\")
        print("    --data_dir=./dataset/ \\")
        print("    --json_list=dataset_0.json \\")
        print("    --pretrained_dir=./pretrained_models/ \\")
        print("    --pretrained_model_name=UNETR_model_best_acc.pth \\")
        print("    --infer_overlap=0.5")
    else:
        print("\n❌ Failed to fix JSON file. Check your data structure.")