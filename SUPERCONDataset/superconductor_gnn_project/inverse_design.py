import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from pymatgen.core import Element
from torch_geometric.data import Data
from mendeleev import element as m_elem

import config
from data import get_element_features, load_graph_data
from model import MultiTaskGNN
from fractions import Fraction

EPSILON_TEMP = 1e-4
EPSILON_LOSS = 1e-6
PATIENCE = 3
stable_count = 0
prev_temp = 0.0
prev_loss = 0.0

print(f"⚙️ Đang tối ưu hóa cấu trúc cho mục tiêu {config.TARGET_TC} K...")
# Danh sách các phi kim quan trọng trong siêu dẫn (có thể mở rộng)
NON_METALS = ['O', 'N', 'F', 'S', 'Se', 'P', 'Cl', 'Br']

def simplify_formula(elements_dict):
    # Tìm giá trị nhỏ nhất để chia lấy tỷ lệ
    min_val = min(elements_dict.values())
    simplified = {k: round(v / min_val, 2) for k, v in elements_dict.items()}
    return simplified

# ==========================================
# 1. KHỞI TẠO MÔI TRƯỜNG & DỮ LIỆU VẬT LÝ
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Khởi động AI Designer (Physical Constraints Enabled) trên {device}...\n")

# Lấy y_std để quy đổi Tc
try:
    df = pd.read_csv(config.CSV_PATH, usecols=['Tc'])
    y_std = float(np.std(df['Tc'].to_numpy()))
except:
    y_std = 1.0

# Thu thập đặc trưng và hằng số vật lý cho 88 nguyên tố
candidate_elements = []
node_features = []
valence_list = []
volume_pa_list = []

invalid_symbols = ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn']
for z in range(1, 95):
    symbol = Element.from_Z(z).symbol
    if symbol not in invalid_symbols:
        candidate_elements.append(symbol)
        node_features.append(get_element_features(symbol))
        
        # Lấy dữ liệu từ mendeleev cho hàm Loss vật lý
        m = m_elem(symbol)
        v = m.oxistates[0] if m.oxistates else 0 # Lấy trạng thái oxy hóa phổ biến nhất
        valence_list.append(float(v))
        # Ép kiểu vdw_radius về float trước khi tính lũy thừa
        r_raw = m.vdw_radius
        if r_raw is not None:
            r_vdw = float(str(r_raw)) 
        else:
            r_vdw = 100.0 # Giá trị mặc định (pm) nếu không tìm thấy
            
        vol = r_vdw ** 3
        # Dùng bán kính Van der Waals làm proxy cho Volume per atom
        volume_pa_list.append(vol)

# Chuyển sang Tensor
x_tensor = torch.tensor(node_features, dtype=torch.float32).to(device)
valences = torch.tensor(valence_list, dtype=torch.float32).to(device)
volumes = torch.tensor(volume_pa_list, dtype=torch.float32).to(device)

n_nodes = len(candidate_elements)
edge_index = torch.tensor([[i, j] for i in range(n_nodes) for j in range(n_nodes)], 
                          dtype=torch.long).t().contiguous().to(device)
batch_idx = torch.zeros(n_nodes, dtype=torch.long).to(device)

# ==========================================
# 2. ĐỊNH NGHĨA HÀM LOSS VẬT LÝ (PHYSICS CORE)
# ==========================================
def calculate_physical_loss(fracs, candidate_elements, curr_e_form, curr_vol, target_tc):
    """Ràng buộc AI tuân thủ các quy luật hóa lý"""
    
    loss = 0.0
    
    # 1. RÀNG BUỘC ĐỘ BỀN (Formation Energy)
    # E_form càng âm càng tốt. Nếu > 0.1 eV/atom là cực kỳ kém bền.
    if curr_e_form > 0.05:
        loss += torch.exp(curr_e_form * 3.0) # Phạt nặng theo hàm mũ
    else:
        loss += curr_e_form * 0.5 # Khuyến khích E_form âm sâu hơn

    # 2. RÀNG BUỘC THỂ TÍCH (Volume per atom)
    # Khoảng vật lý an toàn: 10 - 35 Å³/atom
    # Dưới 10: Quá nén (chỉ tồn tại ở áp suất cực cao)
    # Trên 35: Quá rỗng (cấu trúc không ổn định)
    v_min, v_max = 10.0, 35.0
    if curr_vol < v_min:
        loss += 2.0 * (v_min - curr_vol)**2
    elif curr_vol > v_max:
        loss += 2.0 * (curr_vol - v_max)**2
        
    # 1. Valence Balance (Tổng hóa trị phải gần bằng 0)
    valence_sum = torch.sum(fracs.squeeze() * valences)
    loss += 1.0 * (valence_sum)**2
    
    non_metal_sum = torch.sum(fracs[non_metal_indices])
    if non_metal_sum < 0.15:
        loss += 1.0 * (0.15 - non_metal_sum)**2
        
    # 2. Lattice Strain (Dựa trên GSvolume_pa)
    # Khuyến khích sự chênh lệch kích thước nguyên tử cho vật liệu Tc cao (>70K)
    mean_vol = torch.sum(fracs.squeeze() * volumes)
    avg_dev_vol = torch.sum(fracs.squeeze() * torch.abs(volumes - mean_vol))
    
    if target_tc > 70:
        loss += torch.clamp(1.0 / (avg_dev_vol + 1e-4), max=100.0)
    else:
        loss += avg_dev_vol # Siêu dẫn truyền thống thường đồng nhất

    # 3. Ambient Pressure Constraint (Hạn chế Hydrogen quá cao)
    h_idx = candidate_elements.index('H')
    loss += torch.relu(fracs[h_idx].squeeze() - 0.2)**2 # Phạt nếu H > 20%  
      
    # --- RÀNG BUỘC 1: PHẢI CÓ PHI KIM (Non-metal constraint) ---
    # Tính tổng tỷ lệ của các nguyên tố phi kim
    non_metal_sum = torch.sum(fracs[non_metal_indices])
    # Nếu tổng phi kim < 20% (0.2), bắt đầu phạt nặng
    if non_metal_sum.item() < 0.2:
        loss += 1.0 * (0.2 - non_metal_sum)**2

    # --- RÀNG BUỘC 2: TÍNH THƯA THỚT (Sparsity - tối đa 5 nguyên tố) ---
    # Dùng L1-norm để ép các nguyên tố không quan trọng về 0
    # Càng nhiều nguyên tố có tỷ lệ đáng kể, loss càng cao
    # Ép các nguyên tố chiếm tỷ lệ quá nhỏ (<1%) về hẳn mức 0
    loss += 1.0 * torch.norm(fracs, p=1)
    
    # Phạt thêm nếu số lượng nguyên tố có tỷ lệ > 1% vượt quá 5
    n_active = torch.sum(torch.where(fracs > 0.01, 1.0, 0.0))
    if n_active > 5:
        loss += 1.0 * (n_active - 5)**2

    # --- RÀNG BUỘC 3: ĐỘ ÂM ĐIỆN (Tùy chọn) ---
    # Siêu dẫn nhiệt độ cao thường cần sự kết hợp giữa kim loại và phi kim có độ âm điện lệch nhau
    # (Bạn có thể thêm logic tính Delta Electronegativity ở đây)
    
    return loss

# ==========================================
# 3. NẠP MÔ HÌNH VÀ TỐI ƯU HÓA
# ==========================================
model = MultiTaskGNN(input_dim=x_tensor.shape[1]).to(device)
model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
model.eval()
for param in model.parameters(): param.requires_grad = False

fracs_logits = torch.ones(n_nodes, 1, device=device) * 0.1
fracs_logits.requires_grad_(True)
optimizer = optim.Adam([fracs_logits], lr=config.INVERSE_LR * 2) # Tăng LR vì hàm Loss phức tạp hơn

print(f"🎯 Mục tiêu: Tc = {config.TARGET_TC} K | Áp suất: 1 atm")
print(f"⚙️ Đang tối ưu hóa cấu trúc hóa học...\n")

# Lấy index của các phi kim này trong danh sách candidate_elements
non_metal_indices = [i for i, elem in enumerate(candidate_elements) if elem in NON_METALS]
_, _, node_dim, scalar_dict = load_graph_data(
    csv_path="data/raw/featurized_multi_task.csv", 
    batch_size=config.BATCH_SIZE
)

y_std_tc = scalar_dict['y_std']
e_std = scalar_dict['energy_std']
e_mean = scalar_dict['energy_mean']
v_mean = scalar_dict['vol_mean']
v_std = scalar_dict['vol_std']

for step in range(config.INVERSE_STEPS):
    optimizer.zero_grad()
    fracs = torch.softmax(fracs_logits, dim=0)
    
    # Dự đoán Tc từ GNN
    graph_data = Data(x=x_tensor, edge_index=edge_index, batch=batch_idx, fracs=fracs)
    tc_pred, e_pred, v_pred = model(graph_data)
      
    # Giải nén đơn vị
    tc_pred_k = tc_pred.squeeze() * y_std_tc
    c_e = (e_pred.squeeze() * e_std) + e_mean
    
    c_v = (v_pred.squeeze() * v_std) + v_mean
    
    
    # Hàm Loss tổng hợp
    loss_tc = (tc_pred_k - config.TARGET_TC)**2
    loss_physics = calculate_physical_loss(fracs, candidate_elements,c_e, c_v, config.TARGET_TC)
    
    
    # Trọng số để luật vật lý có tiếng nói mạnh mẽ
    total_loss = loss_tc + 1.0 * loss_physics 
    
    total_loss.backward()
    optimizer.step()
    
    curr_temp = tc_pred_k.item()
    curr_loss = total_loss.item()
    
    delta_temp = abs(curr_temp - prev_temp)
    delta_loss = abs(curr_loss - prev_loss)
    
    if delta_temp <= EPSILON_TEMP and delta_loss <= EPSILON_LOSS:
        stable_count += 1
    else:
        stable_count = 0 # Reset nếu bị vọt ra ngoài ngưỡng
    
    prev_temp = curr_temp
    prev_loss = curr_loss
    
    if stable_count >= PATIENCE:
        print(f"✨ Đã hội tụ tại bước {step+1}! | Stable: {stable_count}/{PATIENCE}")
        print(f"🎯 Tc cuối: {tc_pred_k.item():.2f} K (Lệch: {delta_temp:.2f} K)")
        print(f"⚖️ Total Loss: {curr_loss:.4f}")
        break
    
    if (step + 1) % 500 == 0:
        print(f"  -> Bước {step+1:04d} | Tc: {tc_pred_k.item():>6.2f} K | Total Loss: {curr_loss:.4f} | Stable: {stable_count}/{PATIENCE}")

# ==========================================
# 4. TRÍCH XUẤT VÀ LỌC KẾT QUẢ
# ==========================================
final_fracs = torch.softmax(fracs_logits, dim=0).detach().cpu().numpy().flatten()
generated_elements = [(candidate_elements[i], f) for i, f in enumerate(final_fracs) if f >= config.THRESHOLD_FRAC]
generated_elements.sort(key=lambda x: x[1], reverse=True)

simple_dict = simplify_formula({s: f for s, f in generated_elements})
print(f"🧪 Công thức rút gọn: {simple_dict}")

total_selected_frac = sum([x[1] for x in generated_elements])

print("\n" + "="*50)
print("🎉 ĐÃ TÌM THẤY VẬT LIỆU ỨNG VIÊN (PHYSICS-VALIDATED)")
print("="*50)

formula_str = ""
for symbol, frac in generated_elements:
    norm_f = frac / total_selected_frac
    coeff = round(norm_f * 10, 2) # Hiển thị theo hệ số 10 nguyên tử
    formula_str += f"{symbol}{coeff}"
    print(f"  • {symbol:<2}: {norm_f*100:>5.1f}%")

print("-" * 50)
print(f"🧪 Công thức đề xuất: {formula_str}")
print(f"💡 Lưu ý: Công thức đã được tối ưu hóa cho ứng suất mạng (Lattice Strain)")
print("="*50)



with open(config.GENERATED_MATERIALS_PATH, "a") as f:
    f.write(f"{formula_str},{config.TARGET_TC},Physics_Constrained\n")