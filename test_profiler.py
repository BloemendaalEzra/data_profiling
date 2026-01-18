import pandas as pd
from data_profiler import DataProfiler

def test_profiler():
    df = pd.DataFrame({'od': [1, 2, 3], 'col': ['a', 'b', 'c']})
    profiler = DataProfiler(df)
    results = profiler.run_checks()
    print("Profiler ran successfully on Pandas DataFrame.")
    print(results['dataset_0'])

if __name__ == "__main__":
    test_profiler()
