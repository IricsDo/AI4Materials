import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch_geometric.data import Data

import config
from data import get_element_features, load_graph_data
from model import TcMLPred_v2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VAL_BATCH_SIZE = 32  # Tăng lên tùy thuộc vào bộ nhớ GPU (VRAM)
STEPS = 500          # Có thể giảm bước nếu mô hình GATv2 hội tụ nhanh
print(f"🚀 Bắt đầu Validate Inverse Design trên {device}...")

# 1. Load Data (Lấy test_loader đã có thuộc tính .formula)
_, test_loader, node_dim, y_std = load_graph_data(
    config.CSV_PATH, 
    batch_size=VAL_BATCH_SIZE, 
    test_size=config.TEST_SIZE,
    random_state=config.RANDOM_STATE
)

# 2. Chuẩn bị không gian 88 nguyên tố
candidate_elements = []
node_features = []
invalid_symbols = ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn']
for z in range(1, 95):
    from pymatgen.core import Element
    elem = Element.from_Z(z)
    if elem.symbol not in invalid_symbols:
        candidate_elements.append(elem.symbol)
        node_features.append(get_element_features(elem.symbol))

x_single = torch.tensor(node_features, dtype=torch.float32).to(device)
n_nodes = len(candidate_elements)
# Tạo edge_index cho một đồ thị đơn
edge_index_single = torch.tensor([[i, j] for i in range(n_nodes) for j in range(n_nodes)], 
                              dtype=torch.long).t().to(device)
batch_idx_all = torch.zeros(n_nodes, dtype=torch.long).to(device)

# 3. Load Model
model = TcMLPred_v2(input_dim=node_dim).to(device)
model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
model.eval()
for param in model.parameters(): param.requires_grad = False

results = []
# Duyệt qua toàn bộ tập Test
for b_idx, batch in enumerate(test_loader):
    batch = batch.to(device)
    curr_batch_size = batch.num_graphs
    target_tc_scaled = batch.y.view(-1, 1)
    
    # Lấy công thức gốc trực tiếp từ batch (đã fix trong data.py)
    true_formula = batch.formula[0] 
    
    # Tối ưu hóa ngược
    fracs_logits = torch.ones(curr_batch_size * n_nodes, 1, device=device) * -2.0
    fracs_logits.requires_grad = True
    
    optimizer = optim.Adam([fracs_logits], lr=config.INVERSE_LR * 2)
    
    # Chuẩn bị dữ liệu đồ thị lặp lại cho Batch
    # Nhân bản node features và edge index cho mọi mẫu trong batch
    x_batch = x_single.repeat(curr_batch_size, 1)
    
    # Tạo edge_index cho Batch (phải cộng offset cho từng đồ thị)
    edge_indices = []
    for i in range(curr_batch_size):
        edge_indices.append(edge_index_single + (i * n_nodes))
    edge_index_batch = torch.cat(edge_indices, dim=1)
    
    # Tạo batch_idx: [0,0... (n_nodes lần), 1,1... (n_nodes lần)]
    batch_idx = torch.arange(curr_batch_size, device=device).repeat_interleave(n_nodes)
    
    for step in range(STEPS): # Tăng lên nếu muốn chính xác hơn
        optimizer.zero_grad()
        
        # Softmax trên từng đồ thị riêng biệt (reshape lại để tính)
        fracs_reshaped = fracs_logits.view(curr_batch_size, n_nodes, 1)
        fracs = torch.softmax(fracs_reshaped, dim=1).view(-1, 1)
        
        temp_data = Data(x=x_batch, edge_index=edge_index_batch, batch=batch_idx, fracs=fracs)
        pred_scaled, _ = model(temp_data)
        l1_penalty = 0.01 * torch.norm(fracs, p=1)
        loss = torch.mean((pred_scaled - target_tc_scaled)**2) + l1_penalty
        
        loss.backward()
        optimizer.step()
    
    # Trích xuất kết quả
    final_fracs_tensor = torch.softmax(fracs_logits.view(curr_batch_size, n_nodes), dim=1)
    final_fracs_np = final_fracs_tensor.detach().cpu().numpy()
    for i in range(curr_batch_size):
        sample_fracs = final_fracs_np[i]
        
        ai_dict = {
            candidate_elements[j]: float(sample_fracs[j]) 
            for j in range(n_nodes) if sample_fracs[j] >= config.THRESHOLD_FRAC
        }

        results.append({
            "True_Formula": batch.formula[i],
            "Target_Tc_K": round(target_tc_scaled[i].item() * y_std, 2),
            "AI_Formula_Raw": ai_dict
        })

    print(f"✅ Đã xong Batch {b_idx + 1} ({len(results)} mẫu)")

# Lưu kết quả
df_res = pd.DataFrame(results)
df_res.to_csv("results/inverse_validation_full.csv", index=False)
print("💾 Đã lưu kết quả đối chiếu vào results/inverse_validation_full.csv")