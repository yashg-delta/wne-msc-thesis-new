import pandas as pd
import numpy as np

# Check all files for missing data
files = [
    '/home/ubuntu/wne-msc-thesis-new/artifacts/btc-usdt-5m:v5/btc-usdt-5m-in-sample-0.csv',
    '/home/ubuntu/wne-msc-thesis-new/artifacts/btc-usdt-5m:v5/btc-usdt-5m-out-of-sample-0.csv'
]

for file_path in files:
    print(f"\n=== Checking {file_path.split('/')[-1]} ===")
    
    df = pd.read_csv(file_path)
    df['open_time'] = pd.to_datetime(df['open_time'])
    
    # Check for missing values in any column
    missing_values = df.isnull().sum()
    print(f"Missing values per column:")
    print(missing_values[missing_values > 0])
    
    if missing_values.sum() == 0:
        print("No missing values found in any column")
    
    # Check time intervals
    time_diffs = df['open_time'].diff().dropna()
    expected_interval = pd.Timedelta(minutes=5)
    
    # Count interval consistency
    consistent_intervals = (time_diffs == expected_interval).sum()
    total_intervals = len(time_diffs)
    
    print(f"Total data points: {len(df)}")
    print(f"Consistent 5-min intervals: {consistent_intervals}/{total_intervals}")
    
    if consistent_intervals != total_intervals:
        print("Time interval distribution:")
        print(time_diffs.value_counts().head())
        
        # Show examples of inconsistent intervals
        inconsistent = time_diffs[time_diffs != expected_interval]
        if len(inconsistent) > 0:
            print(f"Found {len(inconsistent)} inconsistent intervals")
            print("Examples:")
            for i, (idx, interval) in enumerate(inconsistent.head().items()):
                print(f"  Row {idx}: {interval} (expected 5 min)")
                if i >= 4:
                    break
    else:
        print("All time intervals are exactly 5 minutes - no missing data!")
    
    # Check for duplicated timestamps
    duplicates = df['open_time'].duplicated().sum()
    print(f"Duplicate timestamps: {duplicates}")
    
    # Show date range
    print(f"Date range: {df['open_time'].iloc[0]} to {df['open_time'].iloc[-1]}")