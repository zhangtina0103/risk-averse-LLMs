#!/usr/bin/env python3
"""
DPO Dataset Validation Script

Run this script to validate the quality and consistency of your DPO training data.
"""

import json
import re
from typing import Dict, List, Tuple

def load_dataset(filepath: str) -> List[Dict]:
    """Load JSON dataset."""
    with open(filepath, 'r') as f:
        return json.load(f)

def check_stray_numbers(data: List[Dict]) -> Tuple[int, List[int]]:
    """Check for stray numbers/letters at end of responses."""
    issues = []
    for i, item in enumerate(data):
        if re.search(r'[\.\s]\d+\s*$', item['chosen']) or \
           re.search(r'\s+[a-z]\s*$', item['chosen'], re.IGNORECASE):
            issues.append(i)
        if re.search(r'[\.\s]\d+\s*$', item['rejected']) or \
           re.search(r'\s+[a-z]\s*$', item['rejected'], re.IGNORECASE):
            issues.append(i)
    return len(issues), issues

def check_conclusions(data: List[Dict]) -> Dict[str, int]:
    """Check that responses have proper conclusions."""
    stats = {
        'chosen_with_conclusion': 0,
        'rejected_with_conclusion': 0,
        'chosen_missing': [],
        'rejected_missing': []
    }
    
    for i, item in enumerate(data):
        if 'Therefore, I select' in item['chosen'] or 'I select' in item['chosen']:
            stats['chosen_with_conclusion'] += 1
        else:
            stats['chosen_missing'].append(i)
            
        if 'Therefore, I select' in item['rejected'] or 'I select' in item['rejected']:
            stats['rejected_with_conclusion'] += 1
        else:
            stats['rejected_missing'].append(i)
    
    return stats

def check_utility_functions(data: List[Dict]) -> Dict[str, int]:
    """Check for proper utility function usage."""
    stats = {
        'chosen_exponential': 0,
        'rejected_linear': 0,
        'chosen_no_utility': [],
        'rejected_no_utility': []
    }
    
    for i, item in enumerate(data):
        if 'e^(-' in item['chosen']:
            stats['chosen_exponential'] += 1
        else:
            stats['chosen_no_utility'].append(i)
            
        if 'u(w) = w' in item['rejected'] or 'u(w) = 1' in item['rejected']:
            stats['rejected_linear'] += 1
        else:
            stats['rejected_no_utility'].append(i)
    
    return stats

def validate_dataset(filepath: str, show_details: bool = False):
    """Run all validation checks on dataset."""
    print("=" * 70)
    print(f"VALIDATING: {filepath}")
    print("=" * 70)
    
    # Load data
    try:
        data = load_dataset(filepath)
        print(f"✅ Successfully loaded {len(data)} samples\n")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return
    
    # Check for stray numbers
    print("📝 Checking for stray trailing numbers/letters...")
    stray_count, stray_indices = check_stray_numbers(data)
    if stray_count == 0:
        print("   ✅ No stray numbers found")
    else:
        print(f"   ⚠️  Found {stray_count} responses with stray numbers")
        if show_details:
            print(f"      Indices: {stray_indices[:10]}...")
    
    # Check conclusions
    print("\n📝 Checking for proper conclusions...")
    conclusion_stats = check_conclusions(data)
    chosen_pct = 100 * conclusion_stats['chosen_with_conclusion'] / len(data)
    rejected_pct = 100 * conclusion_stats['rejected_with_conclusion'] / len(data)
    
    print(f"   Chosen:   {conclusion_stats['chosen_with_conclusion']}/{len(data)} ({chosen_pct:.1f}%)")
    print(f"   Rejected: {conclusion_stats['rejected_with_conclusion']}/{len(data)} ({rejected_pct:.1f}%)")
    
    if chosen_pct >= 99 and rejected_pct >= 99:
        print("   ✅ Conclusion coverage is excellent")
    elif chosen_pct >= 95 and rejected_pct >= 95:
        print("   ⚠️  Conclusion coverage is acceptable but could be improved")
    else:
        print("   ❌ Low conclusion coverage - review dataset")
    
    # Check utility functions
    print("\n📝 Checking utility function usage...")
    utility_stats = check_utility_functions(data)
    chosen_util_pct = 100 * utility_stats['chosen_exponential'] / len(data)
    rejected_util_pct = 100 * utility_stats['rejected_linear'] / len(data)
    
    print(f"   Chosen (exponential):  {utility_stats['chosen_exponential']}/{len(data)} ({chosen_util_pct:.1f}%)")
    print(f"   Rejected (linear ref): {utility_stats['rejected_linear']}/{len(data)} ({rejected_util_pct:.1f}%)")
    
    if chosen_util_pct >= 95:
        print("   ✅ Utility function consistency is excellent")
    elif chosen_util_pct >= 80:
        print("   ⚠️  Utility function consistency is acceptable")
    else:
        print("   ❌ Low utility function consistency - review dataset")
    
    # Length statistics
    print("\n📝 Response length statistics...")
    chosen_lengths = [len(item['chosen']) for item in data]
    rejected_lengths = [len(item['rejected']) for item in data]
    
    print(f"   Chosen responses:  {min(chosen_lengths):,} - {max(chosen_lengths):,} chars (avg: {sum(chosen_lengths)/len(chosen_lengths):,.0f})")
    print(f"   Rejected responses: {min(rejected_lengths):,} - {max(rejected_lengths):,} chars (avg: {sum(rejected_lengths)/len(rejected_lengths):,.0f})")
    
    # Sample responses
    if show_details:
        print("\n📝 Sample responses (last 100 chars)...")
        for i in [0, len(data)//2, len(data)-1]:
            print(f"\n   Sample {i+1}:")
            print(f"   Chosen:   ...{data[i]['chosen'][-100:]}")
            print(f"   Rejected: ...{data[i]['rejected'][-100:]}")
    
    # Overall verdict
    print("\n" + "=" * 70)
    if stray_count == 0 and chosen_pct >= 99 and chosen_util_pct >= 95:
        print("✅ DATASET IS READY FOR TRAINING")
    elif stray_count == 0 and chosen_pct >= 95:
        print("⚠️  DATASET IS USABLE BUT HAS MINOR ISSUES")
    else:
        print("❌ DATASET HAS ISSUES - REVIEW RECOMMENDED")
    print("=" * 70)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python validate_dataset.py <path_to_json> [--details]")
        print("\nExample:")
        print("  python validate_dataset.py dpo_data_cleaned.json")
        print("  python validate_dataset.py dpo_data_test_50.json --details")
        sys.exit(1)
    
    filepath = sys.argv[1]
    show_details = '--details' in sys.argv
    
    validate_dataset(filepath, show_details)
