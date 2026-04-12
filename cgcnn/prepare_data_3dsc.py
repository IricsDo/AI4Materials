import os
import pandas as pd
import shutil

# 1. Khai báo đường dẫn (bạn hãy kiểm tra lại đường dẫn chính xác trong máy)
# Theo tài liệu của 3DSC, đường dẫn cif là tương đối so với thư mục superconductors_3D
path_3dsc_repo = r"/home/hoanguyen/ws_duy/AI4Materials/3DSC/superconductors_3D" 
csv_path = r"/home/hoanguyen/ws_duy/AI4Materials/3DSC/MP/3DSC_MP.csv" # Hoặc tìm đúng file 3DSC_MP.csv
output_dir = r"/home/hoanguyen/ws_duy/AI4Materials/cgcnn_data"

# Tạo thư mục đầu ra cho CGCNN
os.makedirs(output_dir, exist_ok=True)

# 2. Đọc dữ liệu từ 3DSC
df = pd.read_csv(csv_path, skiprows=1)


# Tạo list để lưu dữ liệu cho file id_prop.csv
id_prop_data = []

print("Đang xử lý và copy file CIF...")
for index, row in df.iterrows():
    # Lấy đường dẫn cif và giá trị Tc
    relative_cif_path = row['cif']
    tc_value = row['tc']
    
    # Tạo một ID duy nhất (vì 3DSC có thể có nhiều chất trùng công thức nhưng khác cấu trúc)
    # Ví dụ: file cif tên là 'mp-123.cif' thì ID là 'mp-123'
    cif_filename = os.path.basename(relative_cif_path)
    material_id = cif_filename.replace('.cif', f'_{index}') # Thêm index để chống trùng lặp
    new_cif_name = f"{material_id}.cif"
    
    # Đường dẫn file gốc và file đích
    src_cif = os.path.join(path_3dsc_repo, relative_cif_path)
    dst_cif = os.path.join(output_dir, new_cif_name)
    
    # Copy file nếu tồn tại
    if os.path.exists(src_cif):
        shutil.copy(src_cif, dst_cif)
        id_prop_data.append([material_id, tc_value])
    else:
        print(f"Cảnh báo: Không tìm thấy {src_cif}")

# 3. Tạo file id_prop.csv cho CGCNN (Không có header)
output_csv = os.path.join(output_dir, "id_prop.csv")
df_out = pd.DataFrame(id_prop_data)
df_out.to_csv(output_csv, index=False, header=False)

print(f"Hoàn tất! Đã gom {len(id_prop_data)} file CIF vào thư mục '{output_dir}'.")