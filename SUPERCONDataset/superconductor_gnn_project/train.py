import torch
import torch.nn as nn
import numpy as np
import config
from data import load_graph_data
from model import MultiTaskGNN

# ==========================================
# 1. CLASS EARLY STOPPING (Theo dõi RMSE Kelvin của Tc)
# ==========================================
class EarlyStopping:
    def __init__(self, patience=30, min_delta=0.001, model_path='best_model.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.model_path = model_path
        self.counter = 0
        self.best_rmse = np.inf
        self.early_stop = False

    def __call__(self, current_rmse, model):
        if current_rmse < self.best_rmse - self.min_delta:
            self.best_rmse = current_rmse
            torch.save(model.state_dict(), self.model_path)
            self.counter = 0
            return True 
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

# ==========================================
# 2. KHỞI TẠO MÔI TRƯỜNG & DỮ LIỆU
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Đang sử dụng thiết bị: {device}")

# Load data với scalar_dict chứa các thông số chuẩn hóa của cả 3 Task
train_loader, test_loader, node_dim, scalar_dict = load_graph_data(
    csv_path="data/raw/featurized_multi_task.csv", # File mới từ Materials Project
    batch_size=config.BATCH_SIZE, 
    test_size=config.TEST_SIZE,
    random_state=config.RANDOM_STATE
)

y_std_tc = scalar_dict['y_std']

model = MultiTaskGNN(input_dim=node_dim, hidden_dim=config.HIDDEN_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=1e-4)

# Định nghĩa các hàm Loss thô (Raw Loss)
reg_loss_fn = nn.HuberLoss(delta=1.0) # Dùng cho cả 3 đầu ra vì đều là Regression

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6
)

early_stopping = EarlyStopping(
    patience=40, min_delta=0.001, model_path=config.MODEL_PATH
)

# ==========================================
# 3. VÒNG LẶP HUẤN LUYỆN CHÍNH
# ==========================================
print("\n🔥 BẮT ĐẦU HUẤN LUYỆN MULTI-TASK (Tc + Stability + Structure)...")

for epoch in range(config.EPOCHS):
    current_lr = optimizer.param_groups[0]['lr']

    # --- PHA TRAIN ---
    model.train()
    total_loss = 0
    total_train_mse_tc = 0
    total_samples_train = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass nhận 3 đầu ra
        tc_pred, energy_pred, vol_pred = model(batch)
        
        # 1. Tính toán Loss thô cho từng Task
        loss_tc_raw = reg_loss_fn(tc_pred.squeeze(), batch.y)
        loss_energy_raw = reg_loss_fn(energy_pred.squeeze(), batch.y_energy)
        loss_vol_raw = reg_loss_fn(vol_pred.squeeze(), batch.y_volume)

        # 2. Uncertainty Weighting (Cân bằng tự động)
        # Task Tc
        prec_tc = torch.exp(-model.log_var_tc)
        loss_tc_w = prec_tc * loss_tc_raw + model.log_var_tc
        
        # Task Energy (Stability)
        prec_energy = torch.exp(-model.log_var_energy)
        loss_energy_w = prec_energy * loss_energy_raw + model.log_var_energy
        
        # Task Volume (Structure)
        prec_vol = torch.exp(-model.log_var_vol)
        loss_vol_w = prec_vol * loss_vol_raw + model.log_var_vol

        # Tổng Loss Multi-task
        loss = loss_tc_w + loss_energy_w + loss_vol_w

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # Lưu MSE của riêng Tc để tính RMSE Kelvin
        mse_tc = nn.functional.mse_loss(tc_pred.squeeze(), batch.y)
        total_train_mse_tc += mse_tc.item() * batch.num_graphs
        total_samples_train += batch.num_graphs

    train_rmse_kelvin = np.sqrt(total_train_mse_tc / total_samples_train) * y_std_tc
    avg_loss = total_loss / len(train_loader)

    # --- PHA VALIDATION ---
    model.eval()
    total_val_mse_tc = 0
    total_samples_val = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            tc_pred, _, _ = model(batch)
            
            val_mse_batch = nn.functional.mse_loss(tc_pred.squeeze(), batch.y, reduction='sum')
            total_val_mse_tc += val_mse_batch.item()
            total_samples_val += batch.num_graphs

    val_rmse_kelvin = np.sqrt(total_val_mse_tc / total_samples_val) * y_std_tc

    # In Log (Theo dõi đồng thời cả Loss tổng và RMSE của mục tiêu chính Tc)
    status = "⭐" if val_rmse_kelvin < early_stopping.best_rmse else "  "
    print(f"{status} Epoch {epoch:03d} | LR: {current_lr:.1e} | Loss: {avg_loss:>7.4f} | Tc-RMSE Tr: {train_rmse_kelvin:>6.2f}K | Tc-RMSE Val: {val_rmse_kelvin:>6.2f}K")

    # --- CẬP NHẬT CHIẾN THUẬT ---
    scheduler.step(val_rmse_kelvin)
    early_stopping(val_rmse_kelvin, model)
    
    if early_stopping.early_stop:
        print(f"\n🛑 Early Stopping! Không có cải thiện sau {early_stopping.patience} Epoch.")
        print(f"🏆 RMSE Kelvin tốt nhất đạt được: {early_stopping.best_rmse:.4f} K")
        break

print("\n✅ Quá trình huấn luyện hoàn tất.")