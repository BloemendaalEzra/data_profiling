import pandas as pd
import numpy as np
from typing import Union, List, Dict, Optional, Any
try:
    from pyspark.sql import DataFrame as SparkDataFrame
    import pyspark.sql.functions as F
    from pyspark.sql.types import NumericType, StringType
except ImportError:
    class SparkDataFrame: pass # Dummy mostly for type hinting if not installed
    pass

class DataProfiler:
    """
    A class to perform data profiling checks on pandas and PySpark DataFrames.

    This class handles one or multiple datasets and allows for flexible configuration of checks.

    Usage:
    ------
    1. Initialize the profiler with data:
       - Single DataFrame: `DataProfiler(df)` (named 'dataset_0' by default)
       - List of DataFrames: `DataProfiler([df1, df2])` (named 'dataset_0', 'dataset_1', ...)
       - Dictionary: `DataProfiler({'train': df_train, 'test': df_test})`

    2. Run checks using `run_checks(check_config=...)`.

    Configuration (`check_config`):
    -------------------------------
    Pass a dictionary to `run_checks` to specify which checks to run.
    If `check_config` is None, a default suite (data types, uniqueness, nulls, column length) is run on ALL columns/datasets.

    Structure:
    {
        "check_method_name": {
            "columns": ["col_A", "col_B"],  # Optional: List of columns to check. If omitted, checks ALL columns.
            "datasets": ["dataset_0"],      # Optional: List of dataset names to check. If omitted, checks ALL datasets.
            "params": { ... }               # Optional: Key-value pairs of parameters matching the check method's signature.
        },
        ...
    }

    Available Checks & Params:
    --------------------------
    - `check_nulls`: Checks for missing values.
    - `check_uniqueness`: Checks for unique values and duplicates.
    - `check_column_length`: Statistics on string lengths (min, max, avg).
    - `check_data_types`: Reports data types.
    - `check_range`: Checks if numeric values fall within a range.
        - params: `min_val` (numeric), `max_val` (numeric)
    - `check_date_range`: Checks if dates fall within a range.
        - params: `min_date` (str/datetime), `max_date` (str/datetime)
    - `check_distinct_values`: Lists unique values in the column.

    Example:
    --------
    profiler = DataProfiler(my_df)
    config = {
        "check_nulls": {"columns": ["id", "name"]},
        "check_range": {
            "columns": ["age"],
            "params": {"min_val": 0, "max_val": 100}
        }
    }
    results = profiler.run_checks(config)
    # results['dataset_0'] is a DataFrame with columns as rows and metrics as columns.
    """
    def __init__(self, data: Union[pd.DataFrame, SparkDataFrame, List[Union[pd.DataFrame, SparkDataFrame]], Dict[str, Union[pd.DataFrame, SparkDataFrame]]]):
        self.dfs = {}
        
        # Helper to check if object is a Spark DF (robust to import errors/missing lib)
        def is_spark(obj):
            return 'pyspark.sql.dataframe.DataFrame' in str(type(obj))

        if isinstance(data, pd.DataFrame) or is_spark(data):
            self.dfs["dataset_0"] = data
        elif isinstance(data, list):
            for i, df in enumerate(data):
                self.dfs[f"dataset_{i}"] = df
        elif isinstance(data, dict):
            self.dfs = data
        else:
            raise ValueError("Data must be a Pandas/Spark DataFrame, list, or dictionary of DataFrames.")
        
        # Structure: self.results[dataset_name][check_name][column_name] = result
        self.results = {name: {} for name in self.dfs.keys()}

    def _is_spark(self, df) -> bool:
        """Check if dataframe is a Spark DataFrame."""
        return 'pyspark.sql.dataframe.DataFrame' in str(type(df))

    def _get_target_columns(self, df: pd.DataFrame, columns: Union[str, List[str], None]) -> List[str]:
        if columns is None:
            return list(df.columns)
        if isinstance(columns, str):
            return [columns]
        return columns

    def _should_process_dataset(self, dataset_name: str, target_datasets: List[str] = None) -> bool:
        if target_datasets is None:
            return True
        return dataset_name in target_datasets

    def _record_result(self, dataset_name, check_name, column, result):
        if check_name not in self.results[dataset_name]:
            self.results[dataset_name][check_name] = {}
        self.results[dataset_name][check_name][column] = result

    def check_uniqueness(self, columns: Union[str, List[str], None] = None, datasets: List[str] = None):
        """Check for uniqueness in specific column(s) or all columns."""
        report = {}
        for name, df in self.dfs.items():
            if not self._should_process_dataset(name, datasets):
                continue

            targets = self._get_target_columns(df, columns)
            ds_report = {}
            
            for col in targets:
                if col not in df.columns:
                    ds_report[col] = "Column not found"
                    continue
                
                if self._is_spark(df):
                    total_count = df.count()
                    distinct_count = df.select(col).distinct().count()
                    is_unique = (total_count == distinct_count)
                    duplicate_count = total_count - distinct_count
                else:
                    is_unique = df[col].is_unique
                    duplicate_count = df[col].duplicated().sum()
                
                res = {"is_unique": is_unique, "duplicate_count": duplicate_count}
                
                ds_report[col] = res
                self._record_result(name, "uniqueness", col, res)
            report[name] = ds_report
        return report

    def check_column_length(self, columns: Union[str, List[str], None] = None, datasets: List[str] = None):
        """Check length stats for string columns."""
        report = {}
        for name, df in self.dfs.items():
            if not self._should_process_dataset(name, datasets):
                continue
                
            targets = self._get_target_columns(df, columns)
            ds_report = {}
            for col in targets:
                if col not in df.columns:
                    ds_report[col] = "Column not found"
                    continue

                if self._is_spark(df):
                    # Spark Check
                    # Check if string type
                    dtype = dict(df.dtypes)[col]
                    # Spark dtypes are like 'string', 'int' etc. 
                    if dtype.startswith('string') or dtype.startswith('varchar'):
                        stats_row = df.select(
                            F.min(F.length(F.col(col))).alias('min_length'),
                            F.max(F.length(F.col(col))).alias('max_length'),
                            F.mean(F.length(F.col(col))).alias('avg_length')
                        ).collect()[0]
                        stats = stats_row.asDict()
                    else:
                        stats = {"count": df.select(col).count()}
                else:
                    # Pandas Check
                    if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                        lengths = df[col].astype(str).str.len()
                        stats = {
                            "min_length": lengths.min(),
                            "max_length": lengths.max(),
                            "avg_length": lengths.mean()
                        }
                    else:
                        stats = {"count": len(df[col])}
                
                ds_report[col] = stats
                self._record_result(name, "column_length", col, stats)
            report[name] = ds_report
        return report

    def check_nulls(self, columns: Union[str, List[str], None] = None, datasets: List[str] = None):
        """Check for nulls in specific column(s) or all columns."""
        report = {}
        for name, df in self.dfs.items():
            if not self._should_process_dataset(name, datasets):
                continue
                
            targets = self._get_target_columns(df, columns)
            ds_report = {}
            for col in targets:
                if col not in df.columns:
                    ds_report[col] = "Column not found"
                    continue
                
                if self._is_spark(df):
                    # Spark Check - simple isNull count
                    # Note: isnan is separate in Spark, usually we check isNull
                    null_count = df.filter(F.col(col).isNull() | F.isnan(col) if dict(df.dtypes)[col] in ['double', 'float'] else F.col(col).isNull()).count()
                else:
                    null_count = df[col].isnull().sum()

                ds_report[col] = {"null_count": null_count}
                self._record_result(name, "nulls", col, {"null_count": null_count})
            report[name] = ds_report
        return report

    def check_range(self, min_val, max_val, columns: Union[str, List[str], None] = None, datasets: List[str] = None):
        """Check if numeric columns are within a range."""
        report = {}
        for name, df in self.dfs.items():
            if not self._should_process_dataset(name, datasets):
                continue
                
            targets = self._get_target_columns(df, columns)
            ds_report = {}
            for col in targets:
                if col not in df.columns:
                    ds_report[col] = "Column not found"
                    continue
                
                if self._is_spark(df):
                    # Spark Check
                    dtype = dict(df.dtypes)[col]
                    # Simple check for numeric types by name
                    if dtype not in ['int', 'bigint', 'float', 'double', 'long', 'smallint', 'tinyint', 'decimal']:
                        continue
                    
                    # Logic: count values NOT between min and max
                    # between in Spark is inclusive
                    out_of_range_count = df.filter(~F.col(col).between(min_val, max_val)).count()
                else:
                    # Pandas Check
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        continue 

                    out_of_range = df[~df[col].between(min_val, max_val, inclusive='both')]
                    out_of_range_count = len(out_of_range)

                res = {
                    "range_min": min_val,
                    "range_max": max_val,
                    "out_of_range_count": out_of_range_count,
                }
                ds_report[col] = res
                self._record_result(name, "range", col, res)
            report[name] = ds_report
        return report

    def check_date_range(self, min_date: str, max_date: str, columns: Union[str, List[str], None] = None, datasets: List[str] = None):
        """Check if date columns are within a specific range."""
        min_ts = pd.to_datetime(min_date)
        max_ts = pd.to_datetime(max_date)
        
        report = {}
        for name, df in self.dfs.items():
            if not self._should_process_dataset(name, datasets):
                continue
                
            targets = self._get_target_columns(df, columns)
            ds_report = {}
            for col in targets:
                if col not in df.columns:
                    ds_report[col] = "Column not found"
                    continue
                
                try:
                    if self._is_spark(df):
                        # Spark Check
                        # Ensure col is date/timestamp or can be cast? 
                        # Assuming usage on date columns. Spark filter logic:
                        out_of_range_count = df.filter(
                            (F.col(col) < min_ts) | (F.col(col) > max_ts)
                        ).count()
                    else:
                        # Pandas Check
                        series_dates = pd.to_datetime(df[col], errors='coerce')
                        if series_dates.isna().all() and not df[col].isna().all():
                            continue 

                        out_of_range = series_dates[(series_dates < min_ts) | (series_dates > max_ts)]
                        out_of_range_count = len(out_of_range)

                    res = {
                        "date_range_min": str(min_ts.date()),
                        "date_range_max": str(max_ts.date()),
                        "date_out_of_range_count": out_of_range_count,
                    }
                    ds_report[col] = res
                    self._record_result(name, "date_range", col, res)

                except Exception as e:
                    ds_report[col] = {"error": str(e)}
            
            report[name] = ds_report
        return report

    def check_distinct_values(self, columns: Union[str, List[str], None] = None, datasets: List[str] = None):
        """Check distinct values in column(s)."""
        report = {}
        for name, df in self.dfs.items():
            if not self._should_process_dataset(name, datasets):
                continue
                
            targets = self._get_target_columns(df, columns)
            ds_report = {}
            for col in targets:
                if col not in df.columns:
                    ds_report[col] = "Column not found"
                    continue
                
                if self._is_spark(df):
                    vals = [x[0] for x in df.select(col).distinct().collect()]
                else:
                    vals = df[col].unique().tolist()
                    
                res = {"distinct_values": vals} # Standardize to dict
                ds_report[col] = res
                self._record_result(name, "distinct_values", col, res)
            report[name] = ds_report
        return report

    def check_data_types(self, columns: Union[str, List[str], None] = None, datasets: List[str] = None):
        """Check data types of all columns."""
        report = {}
        for name, df in self.dfs.items():
            if not self._should_process_dataset(name, datasets):
                continue
                
            # Respect 'columns' argument if provided
            targets = self._get_target_columns(df, columns)
            
            if self._is_spark(df):
                full_dtypes = dict(df.dtypes)
                dtypes = {col: full_dtypes[col] for col in targets if col in full_dtypes}
            else:
                dtypes = df[targets].dtypes.apply(lambda x: x.name).to_dict()
            
            # Record per column for easier flattening
            for col, dtype_name in dtypes.items():
                self._record_result(name, "data_types", col, {"dtype": dtype_name})
            
            report[name] = dtypes
        return report

    def run_checks(self, check_config: Dict[str, Dict] = None, clear_previous: bool = True):
        """
        Orchestrate checks based on configuration.
        Config structure:
        {
            "check_name": { "columns": ["ColA"], "datasets": ["dataset_0"], "params": {"min_val": 0, ...} }
        }
        If check_config is None, runs default checks (uniqueness, nulls, lengths, types) on all columns.
        If clear_previous is True (default), previous results are discarded before running new checks.
        """
        if clear_previous:
            self.results = {name: {} for name in self.dfs.keys()}

        if check_config is None:
            # Default Checks
            self.check_data_types()
            self.check_uniqueness()
            self.check_nulls()
            self.check_column_length()
        else:
            # Custom Checks
            for method_name, settings in check_config.items():
                method = getattr(self, method_name, None)
                if not method:
                    print(f"Warning: Method {method_name} not found.")
                    continue
                
                if settings is None: settings = {} # Should be dict or None
                
                cols = settings.get("columns", None)
                datasets = settings.get("datasets", None)
                params = settings.get("params", {})
                
                # Execute
                method(columns=cols, datasets=datasets, **params)
        
        return self._get_results_as_dataframes()

    def _get_results_as_dataframes(self) -> Dict[str, pd.DataFrame]:
        flattened_results = {}
        
        for ds_name, checks in self.results.items():
            # We want a DataFrame where Index = Column Name, Columns = Attribute (e.g. null_count, is_unique)
            # checks structure: { 'uniqueness': { 'ID': {'is_unique': True...} } }
            
            # First, gather all columns mentioned across all checks
            all_cols = set()
            for check_data in checks.values():
                all_cols.update(check_data.keys())
            
            # Prepare rows
            rows = {}
            for col in all_cols:
                rows[col] = {}

            # Populate rows
            for check_name, col_data in checks.items():
                for col, res in col_data.items():
                    if isinstance(res, dict):
                        for k, v in res.items():
                            # Create a unique column name for the metric, e.g. "uniqueness_is_unique"
                            # If the key is generic like 'count', prefix it.
                            # If existing key is fairly unique like 'min_length', keep it?
                            # Safer to prefix to avoid collisions.
                            
                            # Exception: check_data_types just returns 'dtype' key?
                            if check_name == 'data_types' and k == 'dtype':
                                metric_name = 'dtype'
                            else:
                                metric_name = f"{check_name}_{k}"
                            
                            rows[col][metric_name] = v
                    else:
                        rows[col][check_name] = res
            
            if not rows:
                flattened_results[ds_name] = pd.DataFrame() # Return empty DF if no checks run
            else:
                flattened_results[ds_name] = pd.DataFrame.from_dict(rows, orient='index')
            
        return flattened_results
