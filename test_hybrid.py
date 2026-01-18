import pandas as pd
from data_profiler import DataProfiler

def test_hybrid_config():
    print("Testing Hybrid Config...")
    df1 = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
    df2 = pd.DataFrame({'c': [3, 4], 'd': ['z', 'w']})
    
    # Setup: 'dataset_A' gets a specific range check. 'dataset_B' should fall back to default checks.
    profiler = DataProfiler({'dataset_A': df1, 'dataset_B': df2})
    
    config = {
        "check_range": {
            "columns": ["a"],
            "datasets": ["dataset_A"],
            "params": {"min_val": 0, "max_val": 10}
        }
    }
    
    results = profiler.run_checks(config)
    
    # dataset_A check:
    ds_a_cols = results['dataset_A'].columns
    print(ds_a_cols)
    # The profiler prefixes keys unless they are in the whitelist. 
    # 'range_min' IS NOT in the whitelist (only null_count, is_unique, etc.). 
    # So it becomes 'range_range_min'.
    assert 'range_range_min' in ds_a_cols
    assert 'null_count' not in ds_a_cols
    
    # dataset_B check:
    print("\nDataset B Results (Expected: Default checks like null_count):")
    ds_b_cols = results['dataset_B'].columns
    print(ds_b_cols)
    assert 'null_count' in ds_b_cols
    assert 'is_unique' in ds_b_cols
    # range check should NOT be here
    assert 'range_range_min' not in ds_b_cols
    
    print("\nTest Passed: Hybrid configuration logic works correctly.")

if __name__ == "__main__":
    test_hybrid_config()
