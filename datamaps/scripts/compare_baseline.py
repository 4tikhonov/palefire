import json
import argparse
import sys
from pathlib import Path
from unidecode import unidecode

def normalize_name(name):
    """Normalize names to handle accents and casing differences."""
    return unidecode(name).strip().lower()

def compare_baselines(baseline_path: str, test_path: str):
    if not Path(baseline_path).exists():
        print(f"Error: Baseline file {baseline_path} not found.")
        sys.exit(1)
        
    if not Path(test_path).exists():
        print(f"Error: Test data file {test_path} not found.")
        sys.exit(1)
        
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)
        
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
        
    baseline_points = baseline_data.get("data", [])
    test_points = test_data.get("data", [])
    
    baseline_dict = {normalize_name(item["location"]): item for item in baseline_points}
    test_dict = {normalize_name(item["location"]): item for item in test_points}
    
    total_baseline = len(baseline_dict)
    if total_baseline == 0:
        print("Warning: No baseline data points found to compare.")
        sys.exit(0)
        
    matched = 0
    mismatched = []
    missing = []
    
    print("=== Baseline Comparison Report ===")
    print(f"Comparing {test_path} against {baseline_path}\n")
    
    for loc, b_item in baseline_dict.items():
        if loc not in test_dict:
            missing.append(b_item["location"])
            continue
            
        t_item = test_dict[loc]
        b_val = b_item.get("value_range", "").strip()
        t_val = t_item.get("value_range", "").strip()
        
        if b_val == t_val:
            matched += 1
        else:
            mismatched.append({
                "location": b_item["location"],
                "expected": b_val,
                "actual": t_val
            })
            
    # Calculate extra items found in test but not in baseline
    extra = [t_item["location"] for loc, t_item in test_dict.items() if loc not in baseline_dict]
    
    accuracy = (matched / total_baseline) * 100
    
    print(f"Accuracy: {accuracy:.1f}% ({matched}/{total_baseline} locations matched exactly)\n")
    
    if mismatched:
        print("--- Mismatched Values ---")
        for m in mismatched:
            print(f"Location: {m['location']} | Expected: '{m['expected']}' | Actual: '{m['actual']}'")
        print()
        
    if missing:
        print("--- Missing Locations (in baseline but not found in test) ---")
        for m in missing:
            print(f"- {m}")
        print()
        
    if extra:
        print("--- Extra Locations (found in test but not in baseline) ---")
        for e in extra:
            print(f"- {e}")
        print()
        
    if matched == total_baseline and not extra:
        print("✅ Perfect match! The test data exactly matches the baseline.")
    else:
        print("❌ Test data differs from baseline.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare extracted JSON data against a ground truth baseline.")
    parser.add_argument("baseline", help="Path to the baseline ground truth JSON")
    parser.add_argument("test", help="Path to the extracted test JSON")

    args = parser.parse_args()
    compare_baselines(args.baseline, args.test)
