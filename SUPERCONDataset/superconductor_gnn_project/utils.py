import torch
import numpy as np
import random
import os

def classification_accuracy(cls_pred, y_cls):
    """
    Tính độ chính xác (Accuracy) cho bài toán phân loại nhị phân.
    
    Args:
        cls_pred (Tensor): Đầu ra chưa qua hàm kích hoạt (logits) từ mô hình.
        y_cls (Tensor): Nhãn thực tế (0 hoặc 1).
        
    Returns:
        float: Tỷ lệ đoán đúng (từ 0.0 đến 1.0)
    """
    # 1. Đưa logits qua hàm Sigmoid để biến thành xác suất (0 -> 1)
    probs = torch.sigmoid(cls_pred)
    
    # 2. Ngưỡng 0.5: >= 0.5 là siêu dẫn (1), ngược lại là (0)
    predictions = (probs >= 0.5).float()
    
    # 3. Đếm số lượng dự đoán khớp với nhãn thực tế
    correct = (predictions == y_cls).sum().item()
    
    # 4. Trả về tỷ lệ trung bình
    accuracy = correct / len(y_cls)
    
    return accuracy


def set_seed(seed=42):
    """
    Cố định tất cả các mầm ngẫu nhiên (Random Seeds).
    Điều này đảm bảo mỗi lần bạn chạy lại code, mô hình sẽ khởi tạo 
    trọng số giống hệt nhau và chia tập train/test giống hệt nhau, 
    giúp dễ dàng debug và so sánh các phiên bản mô hình.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # Ép PyTorch sử dụng các thuật toán deterministic (tính toán cố định)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Đã cố định Random Seed ở mức: {seed}")