#!/usr/bin/env python3
"""
Find the dataset structure and locate original images
"""

import os
import glob

def find_dataset_structure():
    """Find and display the dataset structure"""
    
    print("🔍 Searching for dataset structure...")
    print("=" * 50)
    
    # Common dataset locations
    possible_locations = [
        "./dataset",
        "../dataset", 
        "./data",
        "../data",
        "./BTCV",
        "../BTCV",
        ".",
        ".."
    ]
    
    found_datasets = []
    
    for location in possible_locations:
        if os.path.exists(location):
            print(f"\n📁 Checking: {os.path.abspath(location)}")
            
            # Look for common patterns
            nii_files = glob.glob(os.path.join(location, "**/*.nii.gz"), recursive=True)
            json_files = glob.glob(os.path.join(location, "**/*.json"), recursive=True)
            
            if nii_files or json_files:
                found_datasets.append(location)
                print(f"  ✅ Found {len(nii_files)} .nii.gz files")
                print(f"  ✅ Found {len(json_files)} .json files")
                
                # Show directory structure
                for root, dirs, files in os.walk(location):
                    level = root.replace(location, '').count(os.sep)
                    indent = ' ' * 2 * level
                    print(f"{indent}📂 {os.path.basename(root)}/")
                    subindent = ' ' * 2 * (level + 1)
                    for file in files[:5]:  # Show first 5 files
                        if file.endswith(('.nii.gz', '.json')):
                            print(f"{subindent}📄 {file}")
                    if len(files) > 5:
                        print(f"{subindent}... and {len(files) - 5} more files")
                    if level > 2:  # Don't go too deep
                        break
    
    if not found_datasets:
        print("❌ No dataset found in common locations")
        print("\n💡 Please check:")
        print("   - Is your dataset in a different location?")
        print("   - Do you have the BTCV dataset downloaded?")
        print("   - Are the .nii.gz files accessible?")
    
    return found_datasets

def find_original_images_for_cases():
    """Find original images that match our result cases"""
    
    print(f"\n🎯 Looking for original images matching result cases...")
    print("=" * 50)
    
    # Get case names from results
    result_cases = []
    for result_dir in glob.glob("results_*"):
        seg_files = glob.glob(os.path.join(result_dir, "*_segmentation.nii.gz"))
        for seg_file in seg_files:
            case_name = os.path.basename(seg_file).replace('_segmentation.nii.gz', '')
            if case_name not in result_cases:
                result_cases.append(case_name)
    
    print(f"📋 Result cases: {result_cases}")
    
    # Search for matching original images
    all_nii_files = glob.glob("**/*.nii.gz", recursive=True)
    
    matches = {}
    for case_name in result_cases:
        print(f"\n🔍 Searching for original image of {case_name}:")
        
        possible_matches = []
        for nii_file in all_nii_files:
            filename = os.path.basename(nii_file)
            
            # Various naming patterns
            if (case_name in filename or 
                case_name.replace('case_', 'img') in filename or
                filename.replace('img', 'case_') == f"{case_name}.nii.gz" or
                filename.replace('.nii.gz', '') == case_name.replace('case_', 'img')):
                
                possible_matches.append(nii_file)
        
        if possible_matches:
            print(f"  ✅ Found potential matches:")
            for match in possible_matches:
                print(f"    📄 {match}")
            matches[case_name] = possible_matches
        else:
            print(f"  ❌ No matches found")
    
    return matches

def main():
    """Main function to analyze dataset structure"""
    
    # Find dataset structure
    datasets = find_dataset_structure()
    
    # Find original images for our cases
    matches = find_original_images_for_cases()
    
    print(f"\n📊 SUMMARY")
    print("=" * 30)
    print(f"Datasets found: {len(datasets)}")
    print(f"Case matches found: {len(matches)}")
    
    if matches:
        print(f"\n💡 SOLUTION: Update the overlay script paths:")
        print("Replace the find_original_image function with these paths:")
        for case_name, paths in matches.items():
            if paths:
                print(f"  {case_name}: {paths[0]}")

if __name__ == "__main__":
    main()