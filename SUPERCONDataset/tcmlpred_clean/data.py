import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SuperconductorDataset(Dataset):

    def __init__(self, X, y_reg, y_cls):

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_reg = torch.tensor(y_reg, dtype=torch.float32)
        self.y_cls = torch.tensor(y_cls, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        return (
            self.X[idx],
            self.y_reg[idx],
            self.y_cls[idx]
        )


def load_data(csv_path, test_size, random_state):

    df = pd.read_csv(csv_path)

    y_reg_raw = df["Tc"].to_numpy(dtype=float)
    y_cls = (y_reg_raw > 0).astype(int)
    
    y_std = float(np.std(y_reg_raw))
    scaler_y = StandardScaler()
    y_reg_scaled = scaler_y.fit_transform(y_reg_raw.reshape(-1, 1)).flatten()
    

    X = df.drop(columns=["Tc","formula","composition","category"])
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
        X,
        y_reg_scaled,
        y_cls,
        test_size=test_size,
        random_state=random_state
    )

    train_ds = SuperconductorDataset(
        X_train,
        y_reg_train,
        y_cls_train
    )

    test_ds = SuperconductorDataset(
        X_test,
        y_reg_test,
        y_cls_test
    )

    return train_ds, test_ds, X.shape[1], y_std