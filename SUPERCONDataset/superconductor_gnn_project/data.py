import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pymatgen.core import Composition, Element

# ==========================================
# 1. HÀM TRÍCH XUẤT ĐẶC TRƯNG NGUYÊN TỐ
# ==========================================
def get_element_features(element_symbol):
    try:
        elem = Element(element_symbol)
        features = [
            float(elem.Z),                         # Số hiệu nguyên tử
            float(elem.X) if elem.X else 0.0,      # Độ âm điện
            float(elem.group) if elem.group else 0.0,
            float(elem.row) if elem.row else 0.0,
            float(elem.atomic_mass)
        ]
    except:
        features = [0.0] * 5
    return features

# ==========================================
# 2. HÀM CHUYỂN CÔNG THỨC THÀNH ĐỒ THỊ (MULTI-TASK)
# ==========================================
def formula_to_graph_multitask(formula, tc_s, energy_s, vol_s):
    """Đóng gói Tc, Formation Energy và Volume vào 1 Graph."""
    try:
        comp = Composition(formula)
        elements = list(comp.keys())
        n_nodes = len(elements)
        
        node_features = [get_element_features(e.symbol) for e in elements]
        fractions = [comp.get_atomic_fraction(e) for e in elements]
            
        x = torch.tensor(node_features, dtype=torch.float32)
        fracs = torch.tensor(fractions, dtype=torch.float32).view(-1, 1)
        
        # Fully Connected Edges
        edge_index = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # Đóng gói 3 nhãn (Targets)
        y_tc = torch.tensor([tc_s], dtype=torch.float32)
        y_energy = torch.tensor([energy_s], dtype=torch.float32)
        y_volume = torch.tensor([vol_s], dtype=torch.float32)

        graph = Data(x=x, edge_index=edge_index, y=y_tc, fracs=fracs)
        graph.y_energy = y_energy
        graph.y_volume = y_volume
        graph.formula = formula 
        
        return graph
    except:
        return None

# ==========================================
# 3. GRAPH DATASET CLASS
# ==========================================
class SuperconductorGraphDataset(Dataset):
    def __init__(self, graphs):
        super().__init__(None, None, None)
        self.graphs = graphs
    def len(self): return len(self.graphs)
    def get(self, idx): return self.graphs[idx]

# ==========================================
# 4. HÀM LOAD DATA CẬP NHẬT (MULTI-TASK)
# ==========================================
def load_graph_data(csv_path, batch_size, test_size=0.2, random_state=42):
    print(f"🚀 Đang nạp dữ liệu đa mục tiêu từ: {csv_path}")
    
    # Đọc 3 cột mục tiêu mới
    cols = ['formula', 'Tc', 'formation_energy', 'volume_per_atom']
    df = pd.read_csv(csv_path, usecols=cols).dropna()
    
    # --- CHUẨN HÓA DỮ LIỆU ---
    scaler_tc = StandardScaler()
    scaler_energy = StandardScaler()
    scaler_vol = StandardScaler()

    # Fit và Transform từng mục tiêu
    y_tc_scaled = scaler_tc.fit_transform(df[['Tc']]).flatten()
    y_energy_scaled = scaler_energy.fit_transform(df[['formation_energy']]).flatten()
    y_vol_scaled = scaler_vol.fit_transform(df[['volume_per_atom']]).flatten()

    # Lưu std của Tc để phục vụ tính RMSE thực tế
    y_std_tc = float(scaler_tc.scale_[0]) # type: ignore
    
    # Dictionary chứa các tham số scaler để dùng cho Inverse Design sau này
    scalar_dict = {
        'y_std': y_std_tc,
        'tc_mean': float(scaler_tc.mean_[0]), # type: ignore
        'energy_std': float(scaler_energy.scale_[0]), # type: ignore
        'energy_mean': float(scaler_energy.mean_[0]), # type: ignore
        'vol_std': float(scaler_vol.scale_[0]), # type: ignore
        'vol_mean': float(scaler_vol.mean_[0]) # type: ignore
    }

    formulas = df['formula'].tolist()
    graphs = []
    
    print("🛠️ Đang chuyển đổi thành đồ thị Multi-task...")
    for i in range(len(df)):
        g = formula_to_graph_multitask(
            formulas[i], 
            y_tc_scaled[i], 
            y_energy_scaled[i], 
            y_vol_scaled[i]
        )
        if g is not None:
            graphs.append(g)
            
    print(f"✅ Đã tạo thành công {len(graphs)} đồ thị!")

    train_graphs, test_graphs = train_test_split(
        graphs, test_size=test_size, random_state=random_state
    )

    train_loader = DataLoader(SuperconductorGraphDataset(train_graphs), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(SuperconductorGraphDataset(test_graphs), batch_size=batch_size, shuffle=False)
    
    node_dim = graphs[0].x.shape[1]

    return train_loader, test_loader, node_dim, scalar_dict