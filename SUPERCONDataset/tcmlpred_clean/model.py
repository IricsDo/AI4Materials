import torch
import torch.nn as nn
import torch.nn.functional as F

# class TcMLPred(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()

#         # Mạng dùng chung (Shared layers)
#         self.shared = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.BatchNorm1d(128),      # Bổ sung BatchNorm
#             nn.ReLU(),
#             nn.Dropout(0.2),          # Bổ sung Dropout (tắt ngẫu nhiên 20% neuron)

#             nn.Linear(128, 64),
#             nn.BatchNorm1d(64),       # Bổ sung BatchNorm
#             nn.ReLU(),
#             nn.Dropout(0.2)
#         )

#         # Nhánh Hồi quy (Regression Head) - Dự đoán giá trị Tc
#         self.reg_head = nn.Sequential(
#             nn.Linear(64, 64),
#             nn.BatchNorm1d(64),       # Bổ sung BatchNorm
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(64, 1)
#         )

#         # Nhánh Phân loại (Classification Head) - Dự đoán có phải siêu dẫn không
#         self.cls_head = nn.Sequential(
#             nn.Linear(64, 32),
#             nn.BatchNorm1d(32),       # Bổ sung BatchNorm
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(32, 2)
#         )
        
#         self.log_var_reg = nn.Parameter(torch.zeros(1))
#         self.log_var_cls = nn.Parameter(torch.zeros(1))
        
#     def forward(self, x):
#         h = self.shared(x)

#         tc_pred = self.reg_head(h)
#         cls_pred = self.cls_head(h)

#         return tc_pred, cls_pred
    
class TcMLPred(nn.Module):
    def __init__(self, input_dim):
        
        super().__init__()
        

        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        # regression head
        self.reg_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

        # classification head
        self.cls_head = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,2)
        )
        
        # learnable uncertainty weights
        self.log_var_reg = nn.Parameter(torch.zeros(1))
        self.log_var_cls = nn.Parameter(torch.zeros(1))
        
    def forward(self,x):

        h = self.relu(self.bn1(self.fc1(x)))

        res = h
        h = self.relu(self.bn2(self.fc2(h)))
        h = h + res              # residual connection

        h = self.dropout(self.relu(self.bn3(self.fc3(h))))

        tc_pred = self.reg_head(h)
        cls_pred = self.cls_head(h)

        return tc_pred, cls_pred
    
    def multitask_loss(self, tc_pred, cls_pred, tc_true, cls_true):

        tc_true = tc_true.unsqueeze(1)   # match shape [B,1]

        reg_loss = F.mse_loss(tc_pred, tc_true)
        cls_loss = F.cross_entropy(cls_pred, cls_true)

        loss = torch.exp(-self.log_var_reg)*reg_loss + self.log_var_reg \
            + torch.exp(-self.log_var_cls)*cls_loss + self.log_var_cls

        return loss

import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.0, model_path='best_model.pt'):
        """
        Args:
            patience (int): Số lượng epoch chịu đựng nếu validation loss không giảm.
            min_delta (float): Mức giảm tối thiểu để được coi là có cải thiện.
            model_path (str): Nơi lưu trọng số mô hình tốt nhất.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.model_path = model_path
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss, model):
        # Nếu loss tốt hơn best_loss (vượt qua cả min_delta)
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0 # Reset lại bộ đếm
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        """Lưu lại mô hình khi validation loss giảm."""
        torch.save(model.state_dict(), self.model_path)