import pandas as pd
from data_profiler import DataProfiler

def verify_single_dataframe():
    print("Testing Single DataFrame input...")
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    profiler = DataProfiler(df)
    results = profiler.run_checks()
    print(f"Result keys: {list(results.keys())}")
    assert len(results) == 1
    assert 'dataset_0' in results

def verify_dict_dataframes():
    print("\nTesting Dictionary of DataFrames input...")
    df1 = pd.DataFrame({'a': [1, 2]})
    df2 = pd.DataFrame({'b': [3, 4]})
    data = {'train': df1, 'test': df2}
    profiler = DataProfiler(data)
    results = profiler.run_checks()
    print(f"Result keys: {list(results.keys())}")
    assert len(results) == 2
    assert 'train' in results
    assert 'test' in results

if __name__ == "__main__":
    try:
        verify_single_dataframe()
        verify_dict_dataframes()
        print("\nVerification Successful: Requirements met.")
    except AssertionError as e:
        print(f"\nVerification Failed: {e}")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
