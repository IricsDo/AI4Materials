import torch
import numpy as np
from torch.utils.data import DataLoader

import config
from data import load_data
from model import TcMLPred
from utils import classification_accuracy
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error, 
    r2_score
)

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
all_preds = []
all_probs = []
all_labels = []

all_tc_pred = []
all_tc_true = []

with torch.no_grad():
    for x, y_reg, y_cls in test_loader:

        tc_pred, cls_pred = model(x)

        total_mse_scaled += reg_loss(
            tc_pred.squeeze(),
            y_reg
        ).item()


        # regression
        all_tc_pred.extend(tc_pred.squeeze().numpy())
        all_tc_true.extend(y_reg.numpy())
        
        # classification
        probs = torch.softmax(cls_pred, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.numpy())
        all_probs.extend(probs[:,1].numpy())   # probability class 1
        all_labels.extend(y_cls.numpy())

        total_acc += classification_accuracy(cls_pred, y_cls)

precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
roc_auc = roc_auc_score(all_labels, all_probs)

mae = mean_absolute_error(all_tc_true, all_tc_pred)
r2 = r2_score(all_tc_true, all_tc_pred)

cm = confusion_matrix(all_labels, all_preds)

avg_mse_scaled = total_mse_scaled / len(test_loader)
rmse_kelvin = np.sqrt(avg_mse_scaled) * y_std
avg_acc = total_acc / len(test_loader)

print("=== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST ===")

print("----- Regression -----")
print(f"RMSE      : {rmse_kelvin:.4f} Kelvin")
print(f"MAE       : {mae:.4f}")
print(f"R² Score  : {r2:.4f}")

print("\n----- Classification -----")
print(f"Accuracy  : {avg_acc:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC AUC   : {roc_auc:.4f}")

print("\nConfusion Matrix")
print(cm)