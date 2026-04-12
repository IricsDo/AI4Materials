import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Đọc file test_results.csv (CGCNN mặc định không tạo header cho file này)
df = pd.read_csv('test_results.csv', header=None, names=['ID', 'Actual', 'Predicted'])

# Lấy dữ liệu thực tế (Target) và dự đoán (Prediction)
y_true = df['Actual']
y_pred = df['Predicted']

# 1. Tính toán các chỉ số đánh giá mô hình
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print(f"--- KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ---")
print(f"MAE  (Sai số tuyệt đối trung bình): {mae:.3f} K")
print(f"RMSE (Căn bậc hai sai số toàn phương): {rmse:.3f} K")
print(f"R^2  (Hệ số xác định): {r2:.3f}")

# 2. Thiết lập vẽ biểu đồ phân tán (Scatter Plot)
plt.figure(figsize=(8, 8))
plt.scatter(y_true, y_pred, alpha=0.7, edgecolors='w', s=80, c='royalblue', label='Vật liệu (Tập Test)')

# Vẽ đường y = x (Đường thẳng lý tưởng, nếu AI đoán đúng 100% thì các điểm sẽ nằm trên đây)
min_val = min(min(y_true), min(y_pred))
max_val = max(max(y_true), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Dự đoán hoàn hảo')

# 3. Làm đẹp biểu đồ
plt.xlabel(r'Nhiệt độ $T_c$ thực tế (K)', fontsize=14)
plt.ylabel(r'Nhiệt độ $T_c$ dự đoán (K)', fontsize=14)
plt.title('Đánh giá Mô hình CGCNN: Thực tế vs. Dự đoán', fontsize=16)

# Chèn hộp văn bản chứa các chỉ số vào góc trái biểu đồ
textstr = '\n'.join((
    r'$MAE=%.3f$' % (mae, ),
    r'$RMSE=%.3f$' % (rmse, ),
    r'$R^2=%.3f$' % (r2, )))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

plt.legend(loc='lower right', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()

# 4. Lưu biểu đồ thành file ảnh chất lượng cao
plt.savefig('scatter_plot_Tc.png', dpi=300)
print("Đã lưu biểu đồ thành công vào file 'scatter_plot_Tc.png'")