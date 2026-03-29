import torch
import numpy as np
from torch.utils.data import DataLoader

import config
from data import load_data
from model import TcMLPred
from utils import classification_accuracy

# 1. Cập nhật: Nhận thêm y_std từ load_data
# Lưu ý: Sửa lại đường dẫn file CSV cho đúng với thực tế của bạn
train_ds, test_ds, input_dim, y_std = load_data(
    "../archive/featurized.csv", # Trước đó bạn dùng file này, hãy kiểm tra lại
    config.TEST_SIZE,
    config.RANDOM_STATE
)

test_loader = DataLoader(
    test_ds,
    batch_size=config.BATCH_SIZE,
    shuffle=False # Tập test không nên/không cần shuffle
)

# Khởi tạo và nạp trọng số mô hình tốt nhất
model = TcMLPred(input_dim)
model.load_state_dict(torch.load(config.MODEL_PATH))
model.eval()

reg_loss = torch.nn.MSELoss()

total_mse_scaled = 0
total_acc = 0

with torch.no_grad():
    for x, y_reg, y_cls in test_loader:
        tc_pred, cls_pred = model(x)

        # Cộng dồn MSE (ở không gian đã bị scale)
        total_mse_scaled += reg_loss(
            tc_pred.squeeze(),
            y_reg
        ).item()

        # Cộng dồn Accuracy
        total_acc += classification_accuracy(
            cls_pred,
            y_cls
        )

# 2. Cập nhật: Tính trung bình và chuyển đổi đơn vị
avg_mse_scaled = total_mse_scaled / len(test_loader)

# Khôi phục RMSE về nhiệt độ Kelvin
rmse_kelvin = np.sqrt(avg_mse_scaled) * y_std

avg_acc = total_acc / len(test_loader)

print("=== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST ===")
print(f"Regression RMSE      : {rmse_kelvin:.4f} Kelvin")
print(f"Classification Acc   : {avg_acc:.4f} (Tương đương {avg_acc * 100:.2f}%)")