import pandas as pd
import os
from data_profiler import DataProfiler

def test_save_to_csv():
    print("Testing save_to_csv...")
    
    # Create sample dataframes
    df1 = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
    df2 = pd.DataFrame({'a': [3, 4], 'b': ['z', 'w']})
    
    # Initialize profiler with dictionary
    profiler = DataProfiler({'dataset_A': df1, 'dataset_B': df2})
    
    # Run checks (implicitly records results)
    profiler.run_checks()
    
    # Save to CSV
    filename = "test_merged_results.csv"
    if os.path.exists(filename):
        os.remove(filename)
        
    profiler.save_to_csv(filename)
    
    # Verify file exists
    assert os.path.exists(filename)
    
    # Verify content
    saved_df = pd.read_csv(filename)
    print("\nSaved CSV Content:")
    print(saved_df)
    
    assert 'dataset_name' in saved_df.columns
    assert 'dataset_A' in saved_df['dataset_name'].values
    assert 'dataset_B' in saved_df['dataset_name'].values
    assert len(saved_df) == 4 # 2 cols * 2 datasets
    
    print("\nTest Passed: save_to_csv logic works correctly.")
    
    # Cleanup
    if os.path.exists(filename):
        os.remove(filename)

if __name__ == "__main__":
    test_save_to_csv()
