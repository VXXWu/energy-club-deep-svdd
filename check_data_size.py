import os
import glob
import pandas as pd
import sys

def check_size(root):
    search_paths = [
        os.path.join(root, 'hai-21.03', '*.csv.gz'),
        os.path.join(root, 'hai-21.03', '*.csv'),
        os.path.join(root, 'hai-23.05', '*.csv.gz'),
        os.path.join(root, 'hai-23.05', '*.csv'),
    ]
    
    files = []
    for path in search_paths:
        found = sorted(glob.glob(path))
        if found:
             # Check size
             if os.path.getsize(found[0]) > 1000:
                 files = found
                 break
    
    print(f"Found files: {files}")
    total_rows = 0
    for f in files:
        try:
            df = pd.read_csv(f)
            print(f"File {f}: {len(df)} rows, {len(df.columns)} cols, {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            total_rows += len(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    print(f"Total rows: {total_rows}")

if __name__ == '__main__':
    check_size('data/hai')
