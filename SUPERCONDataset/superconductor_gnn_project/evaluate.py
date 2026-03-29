import torch
import torch.nn as nn
import numpy as np
import config
from data import load_graph_data
from model import MultiTaskGNN

# ==========================================
# 1. KHỞI TẠO MÔI TRƯỜNG & DỮ LIỆU
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Đang chạy đánh giá Multi-Task trên: {device}")

# Load dữ liệu và lấy scalar_dict để giải nén đơn vị
_, test_loader, node_dim, scalar_dict = load_graph_data(
    csv_path="data/raw/featurized_multi_task.csv", 
    batch_size=config.BATCH_SIZE, 
    test_size=config.TEST_SIZE,
    random_state=config.RANDOM_STATE
)

# Trích xuất các tham số chuẩn hóa
y_std_tc = scalar_dict['y_std']
e_std = scalar_dict['energy_std']
v_std = scalar_dict['vol_std']

# ==========================================
# 2. KHỞI TẠO VÀ NẠP MÔ HÌNH
# ==========================================
model = MultiTaskGNN(input_dim=node_dim, hidden_dim=config.HIDDEN_DIM).to(device)
model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
model.eval()

# Danh sách lưu kết quả để tính chỉ số cuối cùng
all_tc_true, all_tc_pred = [], []
all_e_true, all_e_pred = [], []
all_v_true, all_v_pred = [], []

print("\n📊 Đang tiến hành đánh giá chi tiết trên tập Test...")

# ==========================================
# 3. VÒNG LẶP ĐÁNH GIÁ (INFERENCE)
# ==========================================
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        
        # Model trả về 3 đầu ra
        tc_p, e_p, v_p = model(batch)
        
        # Lưu kết quả (đưa về CPU và chuyển thành numpy)
        all_tc_true.extend(batch.y.cpu().numpy())
        all_tc_pred.extend(tc_p.squeeze().cpu().numpy())
        
        all_e_true.extend(batch.y_energy.cpu().numpy())
        all_e_pred.extend(e_p.squeeze().cpu().numpy())
        
        all_v_true.extend(batch.y_volume.cpu().numpy())
        all_v_pred.extend(v_p.squeeze().cpu().numpy())

# Chuyển thành mảng numpy để tính toán
all_tc_true, all_tc_pred = np.array(all_tc_true), np.array(all_tc_pred)
all_e_true, all_e_pred = np.array(all_e_true), np.array(all_e_pred)
all_v_true, all_v_pred = np.array(all_v_true), np.array(all_v_pred)

# ==========================================
# 4. TÍNH TOÁN CHỈ SỐ VẬT LÝ THỰC TẾ
# ==========================================

# 1. Tc RMSE (Kelvin)
tc_rmse = np.sqrt(np.mean((all_tc_true - all_tc_pred)**2)) * y_std_tc

# 2. Energy MAE (eV/atom) - Dùng MAE để dễ hình dung sai số năng lượng
e_mae = np.mean(np.abs(all_e_true - all_e_pred)) * e_std

# 3. Volume MAE (A^3/atom)
v_mae = np.mean(np.abs(all_v_true - all_v_pred)) * v_std

# ==========================================
# 5. IN KẾT QUẢ ĐA CHIỀU
# ==========================================
print("\n" + "═"*50)
print("       BÁO CÁO ĐÁNH GIÁ VẬT LIỆU (MULTI-TASK)")
print("═"*50)
print(f"🌡️  Nhiệt độ (Tc RMSE)     : {tc_rmse:>8.4f} K")
print(f"💎  Độ bền (Energy MAE)    : {e_mae:>8.4f} eV/atom")
print(f"📦  Cấu trúc (Volume MAE)  : {v_mae:>8.4f} Å³/atom")
print("═"*50)

# Tính thêm hệ số tương quan R2 cho Tc để xem độ khớp
from sklearn.metrics import r2_score
r2_tc = r2_score(all_tc_true, all_tc_pred)
print(f"📈  R² Score (Tc)          : {r2_tc:>8.4f}")
print("═"*50 + "\n")

import matplotlib.pyplot as plt
import seaborn as sns

def plot_multitask_results(all_tc_true, all_tc_pred, all_e_true, all_e_pred, all_v_true, all_v_pred, scalar_dict):
    # Giải nén đơn vị thực tế cho biểu đồ
    tc_true_k = all_tc_true * scalar_dict['y_std']
    tc_pred_k = all_tc_pred * scalar_dict['y_std']
    
    e_true_ev = all_e_true * scalar_dict['energy_std']
    e_pred_ev = all_e_pred * scalar_dict['energy_std']
    
    v_true_a3 = all_v_true * scalar_dict['vol_std']
    v_pred_a3 = all_v_pred * scalar_dict['vol_std']

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    sns.set_style("whitegrid")

    # 1. Biểu đồ Tc (Kelvin)
    sns.regplot(x=tc_true_k, y=tc_pred_k, ax=axes[0], 
                scatter_kws={'alpha':0.4, 's':10, 'color': 'royalblue'}, 
                line_kws={'color': 'red', 'ls': '--'})
    axes[0].set_title(f'Nhiệt độ Siêu dẫn (Tc)\nRMSE: {tc_rmse:.2f} K', fontsize=14)
    axes[0].set_xlabel('Thực tế (K)')
    axes[0].set_ylabel('Dự đoán (K)')

    # 2. Biểu đồ Formation Energy (eV/atom)
    sns.regplot(x=e_true_ev, y=e_pred_ev, ax=axes[1], 
                scatter_kws={'alpha':0.4, 's':10, 'color': '#2ecc71'}, 
                line_kws={'color': 'red', 'ls': '--'})
    axes[1].set_title(f'Năng lượng Hình thành (Stability)\nMAE: {e_mae:.4f} eV/atom', fontsize=14)
    axes[1].set_xlabel('Thực tế (eV/atom)')
    axes[1].set_ylabel('Dự đoán (eV/atom)')

    # 3. Biểu đồ Volume (Å³/atom)
    sns.regplot(x=v_true_a3, y=v_pred_a3, ax=axes[2], 
                scatter_kws={'alpha':0.4, 's':10, 'color': 'orange'}, 
                line_kws={'color': 'red', 'ls': '--'})
    axes[2].set_title(f'Thể tích nguyên tử (Structure)\nMAE: {v_mae:.4f} Å³/atom', fontsize=14)
    axes[2].set_xlabel('Thực tế (Å³/atom)')
    axes[2].set_ylabel('Dự đoán (Å³/atom)')

    plt.tight_layout()
    plt.savefig('evaluation_scatter_plots.png', dpi=300)
    print("\n✅ Đã lưu biểu đồ tại: evaluation_scatter_plots.png")
    plt.show()

# Gọi hàm sau khi đã tính toán xong các mảng all_tc_true, ...
plot_multitask_results(all_tc_true, all_tc_pred, all_e_true, all_e_pred, all_v_true, all_v_pred, scalar_dict)