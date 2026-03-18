import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool

class TcMLPred_v1(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(TcMLPred_v1, self).__init__()
        
        # 1. Nhúng Node (Node Embedding)
        # Chuyển 5 đặc trưng vật lý ban đầu thành vector 64 chiều
        self.node_emb = nn.Linear(input_dim, hidden_dim)
        
        # 2. Message Passing (Các lớp Tích chập Đồ thị)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # 3. MLP Head cho Regression (Dự đoán Tc)
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 4. MLP Head cho Classification (Phân loại siêu dẫn)
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 5. Tự động cân bằng Loss (Uncertainty Weighting)
        self.log_var_reg = nn.Parameter(torch.zeros(1))
        self.log_var_cls = nn.Parameter(torch.zeros(1))

    def forward(self, data):
        """
        data: Đối tượng Batch của PyTorch Geometric chứa các đồ thị đã được gộp.
        """
        # Bóc tách dữ liệu từ Graph object
        x, edge_index, batch_idx, fracs = data.x, data.edge_index, data.batch, data.fracs
        
        # --- BƯỚC 1: CẬP NHẬT TRẠNG THÁI NODE (MESSAGE PASSING) ---
        x = F.relu(self.node_emb(x))
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        
        # --- BƯỚC 2: GLOBAL POOLING (FRACTIONAL READOUT) ---
        # ĐÂY LÀ CHÌA KHÓA CHO INVERSE DESIGN!
        # Nhân mỗi vector nguyên tố với tỷ lệ phần trăm của nó trong hợp chất
        x_weighted = x * fracs 
        
        # Cộng dồn các node lại thành 1 vector duy nhất đại diện cho cả phân tử
        # Hàm global_add_pool sẽ tự động biết node nào thuộc đồ thị nào nhờ biến 'batch_idx'
        graph_embed = global_add_pool(x_weighted, batch_idx)
        
        # --- BƯỚC 3: DỰ ĐOÁN ĐẦU RA ---
        tc_pred = self.reg_head(graph_embed)
        cls_pred = self.cls_head(graph_embed)
        
        return tc_pred, cls_pred
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool

class TcMLPred_v2(nn.Module):
    def __init__(self, input_dim, hidden_dim=128): # Tăng hidden_dim lên 128
        super(TcMLPred_v2, self).__init__()
        
        self.node_emb = nn.Linear(input_dim, hidden_dim)
        
        # Dùng GATv2: Cơ chế Attention mạnh mẽ hơn GCN
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, heads=2, concat=False)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=2, concat=False)
        self.conv3 = GATv2Conv(hidden_dim, hidden_dim, heads=2, concat=False)
        
        # Batch Normalization giúp ổn định gradient
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Regression Head sâu hơn với LeakyReLU
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.log_var_reg = nn.Parameter(torch.zeros(1))
        self.log_var_cls = nn.Parameter(torch.zeros(1))

    def forward(self, data):
        x, edge_index, batch_idx, fracs = data.x, data.edge_index, data.batch, data.fracs
        
        # Node Embedding ban đầu
        h = self.node_emb(x)
        
        # Lớp Conv 1 + Skip Connection
        h1 = self.conv1(h, edge_index)
        h1 = F.leaky_relu(self.bn1(h1), 0.1)
        h = h + h1 # Residual link
        
        # Lớp Conv 2 + Skip Connection
        h2 = self.conv2(h, edge_index)
        h2 = F.leaky_relu(self.bn2(h2), 0.1)
        h = h + h2 # Residual link
        
        # Readout: Nhân với tỷ lệ fracs
        # Chỗ này cực kỳ quan trọng cho Inverse Design
        x_weighted = h * fracs 
        graph_embed = global_add_pool(x_weighted, batch_idx)
        
        return self.reg_head(graph_embed), self.cls_head(graph_embed)
    

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool

class MultiTaskGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(MultiTaskGNN, self).__init__()
        
        # 1. Khởi tạo Node Embedding
        self.node_emb = nn.Linear(input_dim, hidden_dim)
        
        # 2. Backbone: 3 lớp GATv2 với Batch Normalization
        # Giữ nguyên cấu trúc Attention và BatchNorm từ TcMLPred_v2
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, heads=2, concat=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=2, concat=False)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.conv3 = GATv2Conv(hidden_dim, hidden_dim, heads=2, concat=False)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # 3. Heads dự đoán riêng biệt (Dùng cấu trúc Sequential sâu như reg_head cũ)
        
        # Nhánh 1: Dự đoán Tc (Regression)
        self.tc_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Nhánh 2: Dự đoán Formation Energy (Regression)
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Nhánh 3: Dự đoán Volume per Atom (Regression)
        self.vol_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 4. Tham số học trọng số Loss (Uncertainty Weighting)
        self.log_var_tc = nn.Parameter(torch.zeros(1))
        self.log_var_energy = nn.Parameter(torch.zeros(1))
        self.log_var_vol = nn.Parameter(torch.zeros(1))
        
    def forward(self, data):
        x, edge_index, batch_idx, fracs = data.x, data.edge_index, data.batch, data.fracs
        
        # --- PHASE 1: Node Feature Learning ---
        h = self.node_emb(x)
        
        # Lớp Conv 1 + Residual
        h1 = self.conv1(h, edge_index)
        h1 = F.leaky_relu(self.bn1(h1), 0.1)
        h = h + h1 
        
        # Lớp Conv 2 + Residual
        h2 = self.conv2(h, edge_index)
        h2 = F.leaky_relu(self.bn2(h2), 0.1)
        h = h + h2
        
        # Lớp Conv 3 + Residual
        h3 = self.conv3(h, edge_index)
        h3 = F.leaky_relu(self.bn3(h3), 0.1)
        h = h + h3
        
        # --- PHASE 2: Readout (Nhân với tỷ lệ nguyên tố) ---
        # Đây là bước quan trọng nhất cho Inverse Design
        x_weighted = h * fracs 
        graph_embed = global_add_pool(x_weighted, batch_idx)
        
        # --- PHASE 3: Task-specific Heads ---
        tc_out = self.tc_head(graph_embed)
        energy_out = self.energy_head(graph_embed)
        vol_out = self.vol_head(graph_embed)
        
        return tc_out, energy_out, vol_out