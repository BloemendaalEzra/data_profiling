from pyspark.sql import SparkSession
from data_profiler import DataProfiler
import pandas as pd

def main():
    # 1. Initialize Spark Session
    # Note: You need Java installed for PySpark to work locally.
    print("Initializing Spark Session...")
    try:
        spark = SparkSession.builder \
            .appName("DataProfilerExample") \
            .master("local[*]") \
            .getOrCreate()
        spark.sparkContext.setLogLevel("ERROR") # Reduce logging noise
    except Exception as e:
        print(f"Error starting Spark: {e}")
        print("Ensure you have 'pyspark' installed and a compatible Java Runtime (JRE/JDK) in your path.")
        return

    # 2. Create a sample PySpark DataFrame
    # Including some nulls and duplicates to test the profiler
    data = [
        (1, "Alice", 50000.0, "2023-01-15"),
        (2, "Bob", 60000.0, "2023-02-10"),
        (3, "Charlie", 75000.0, "2023-03-01"),
        (4, None, 55000.0, "2023-01-20"),         # Null Name
        (5, "Dave", None, "2023-04-05"),          # Null Salary
        (1, "Alice", 50000.0, "2023-01-15"),      # Duplicate Row
        (7, "Eve", 40000.0, "2025-01-01"),        # Future Date
        (8, "Frank", 120000.0, "2020-01-01")      # High Salary
    ]
    
    columns = ["ID", "Name", "Salary", "JoinDate"]
    
    print("\nCreating Mock PySpark DataFrame...")
    df_spark = spark.createDataFrame(data, columns)
    
    # Cast JoinDate to real date type for date checks
    from pyspark.sql.functions import to_date
    df_spark = df_spark.withColumn("JoinDate", to_date("JoinDate"))
    
    print("Data Preview:")
    df_spark.show()

    # 3. Initialize Data Profiler
    # The profiler automatically detects it's a Spark DataFrame
    print("Initializing Profiler...")
    profiler = DataProfiler(df_spark)

    # 4. Define specific checks (optional)
    check_config = {
        "check_nulls": None, # Run on all columns
        "check_uniqueness": {"columns": ["ID"]},
        "check_range": {
            "columns": ["Salary"], 
            "params": {"min_val": 45000, "max_val": 100000}
        },
        "check_date_range": {
            "columns": ["JoinDate"],
            "params": {"min_date": "2022-01-01", "max_date": "2024-12-31"}
        }
    }

    # 5. Run Checks
    print("Running checks...")
    results = profiler.run_checks(check_config)

    # 6. Display Results
    # Results are returned as a dictionary of Pandas DataFrames
    print("\n--- Profiling Results ---")
    if "dataset_0" in results:
        print(results["dataset_0"].to_string())

    spark.stop()

if __name__ == "__main__":
    main()
