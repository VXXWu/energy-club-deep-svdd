from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import glob
import os
from base.base_dataset import BaseADDataset

class HAITorchDataset(Dataset):
    def __init__(self, root, train=True, window_size=64, stride=1, scaler=None):
        super().__init__()
        self.root = root
        self.train = train
        self.window_size = window_size
        self.stride = stride
        
        self.data, self.labels, self.scaler = self._load_data(scaler)
        # self.windows, self.window_labels = self._create_windows() # Removed for memory efficiency

    def _load_data(self, scaler=None):
        # Search for HAI CSV files
        # Prioritize hai-21.03 since 23.05 seems to be LFS pointers without quota
        search_paths = [
            os.path.join(self.root, 'hai-21.03', '*.csv.gz'),
            os.path.join(self.root, 'hai-21.03', '*.csv'),
            os.path.join(self.root, 'hai-23.05', '*.csv.gz'), # Try gz for 23.05 too
            os.path.join(self.root, 'hai-23.05', '*.csv'),
            os.path.join(self.root, '*.csv.gz'),
            os.path.join(self.root, '*.csv')
        ]
        
        files = []
        for path in search_paths:
            found = sorted(glob.glob(path))
            if found:
                # Check if these are LFS pointers
                # Just check the first file
                try:
                    if found[0].endswith('.gz'):
                         # GZ files are likely real if they exist and are not tiny
                         if os.path.getsize(found[0]) > 1000:
                             files = found
                             break
                    else:
                        # CSV file, check size or content
                        if os.path.getsize(found[0]) > 1000:
                             files = found
                             break
                except:
                    continue

        if not files:
            raise FileNotFoundError(f"No HAI CSV files found in {self.root}")

        # Filter train/test files
        if self.train:
            files = [f for f in files if 'train' in os.path.basename(f).lower()]
        else:
            files = [f for f in files if 'test' in os.path.basename(f).lower()]

        if not files:
            raise FileNotFoundError(f"No {'training' if self.train else 'test'} files found in {self.root}")

        print(f"Loading files: {files}")
        
        dfs = []
        for f in files:
            try:
                df = pd.read_csv(f)
                # Check if it's a git lfs pointer
                if len(df) < 10 and 'version https://git-lfs.github.com/spec/v1' in df.to_string():
                    raise ValueError(f"File {f} is a Git LFS pointer. Please download the actual file.")
                
                # Drop timestamp (first column)
                # And drop attack labels if present (usually last columns)
                # HAI structure: time, ...tags..., attack, ...
                # We need to identify tag columns.
                # Assuming first column is time.
                # Attack column is usually named 'attack' or 'Attack'
                
                # Simple heuristic: drop first column (time) and any column with 'attack' in name
                drop_cols = [df.columns[0]]
                for col in df.columns:
                    if 'attack' in col.lower():
                        drop_cols.append(col)
                
                data_df = df.drop(columns=drop_cols)
                
                # For labels, if test, we need the attack column
                if not self.train:
                    # Find attack column
                    attack_col = None
                    for col in df.columns:
                        if 'attack' in col.lower():
                            attack_col = col
                            break
                    if attack_col:
                        labels = df[attack_col].values
                    else:
                        labels = np.zeros(len(df))
                else:
                    labels = np.zeros(len(df))
                
                dfs.append((data_df.values, labels))
            except Exception as e:
                print(f"Error loading {f}: {e}")
                raise

        # Concatenate
        all_data = np.concatenate([d[0] for d in dfs], axis=0)
        all_labels = np.concatenate([d[1] for d in dfs], axis=0)
        
        # Normalize
        if scaler is None:
            scaler = MinMaxScaler()
            all_data = scaler.fit_transform(all_data)
        else:
            all_data = scaler.transform(all_data)
        
        # Convert to torch tensor immediately to save memory if possible, or keep as numpy
        # Keeping as numpy is fine, but let's ensure float32
        all_data = all_data.astype(np.float32)
        
        return all_data, all_labels, scaler

    def __getitem__(self, index):
        # Lazy window slicing
        # Adjust index for stride if needed, but assuming stride=1 for simplicity in len calculation
        # If stride > 1, index mapping is needed: start = index * stride
        start = index * self.stride
        end = start + self.window_size
        
        window = self.data[start:end]
        # Label is max of window labels (if any point is anomaly, window is anomaly)
        target = self.labels[start:end].max()
        
        # Convert to torch tensor
        window = torch.from_numpy(window)
        
        # Return index as well for Deep SVDD
        return window, target, index

    def __len__(self):
        # Number of windows
        if len(self.data) < self.window_size:
            return 0
        return (len(self.data) - self.window_size) // self.stride + 1

class HAIDataset(BaseADDataset):
    def __init__(self, root: str):
        super().__init__(root)
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([0])
        self.outlier_classes = tuple([1])

        self.train_set = HAITorchDataset(root=root, train=True)
        # Pass the fitted scaler from train_set to test_set
        self.test_set = HAITorchDataset(root=root, train=False, scaler=self.train_set.scaler)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test, num_workers=num_workers)
        return train_loader, test_loader
