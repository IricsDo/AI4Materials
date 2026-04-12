import pandas as pd

# 1. Đọc file dữ liệu gốc (thường không có dòng tiêu đề)
# Đổi 'id_prop.csv' thành tên file thực tế của bạn nếu nó khác nhé
file_path = r"/home/hoanguyen/ws_duy/AI4Materials/cgcnn_data/id_prop.csv" 
df = pd.read_csv(file_path, header=None)

# 2. Lọc bỏ các giá trị Tc <= 0 (giả sử cột 0 là ID vật liệu, cột 1 là Tc)
df_cleaned = df[df[1] > 0]

# 3. Lưu ra một file mới
df_cleaned.to_csv(r'/home/hoanguyen/ws_duy/AI4Materials/cgcnn_data/id_prop_cleaned.csv', index=False, header=None)

print(f"--- ĐÃ LỌC DỮ LIỆU THÀNH CÔNG ---")
print(f"Số lượng vật liệu ban đầu: {len(df)}")
print(f"Số lượng vật liệu giữ lại (Tc > 0): {len(df_cleaned)}")
print(f"Đã loại bỏ {len(df) - len(df_cleaned)} vật liệu không siêu dẫn.")