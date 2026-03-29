import pandas as pd
import numpy as np
from pymatgen.core import Composition
import json
import re  # Thêm re để làm sạch chuỗi nếu cần

def get_comp_vector(formula_dict, all_elements):
    """Chuyển dict công thức thành vector tỷ lệ chuẩn hóa trên 88 nguyên tố"""
    vec = np.zeros(len(all_elements))
    total = sum(formula_dict.values())
    if total == 0: return vec
    
    for i, elem in enumerate(all_elements):
        if elem in formula_dict:
            vec[i] = formula_dict[elem] / total
    return vec

# 1. Load kết quả từ file Validation đã chạy ở bước trước
try:
    results_df = pd.read_csv("results/inverse_validation_full.csv")
    # Load lại file gốc để lấy Ground Truth (Công thức thật)
    print(f"✅ Đã nạp {len(results_df)} mẫu từ kết quả Validation.")
except FileNotFoundError:
    print("Vui lòng chạy file inverse_validation.py trước để tạo dữ liệu!")
    exit()

# 2. Chuẩn bị danh sách nguyên tố chuẩn (giống trong inverse_validation.py)
from pymatgen.core import Element
all_elements = [Element.from_Z(z).symbol for z in range(1, 95) 
                if Element.from_Z(z).symbol not in ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn']]

evaluation_report = []

print("🚀 Đang so sánh AI Formula với Ground Truth...")

for _, row in results_df.iterrows():
    
    # Lấy trực tiếp từ hàng hiện tại
    true_formula_str = str(row['True_Formula'])
    try:
        true_comp_dict = Composition(true_formula_str).get_el_amt_dict()
    except:
        continue # Bỏ qua nếu công thức lỗi
        
    # Lấy công thức AI (Chuyển string dict thành dict thật)
    raw_str = row['AI_Formula_Raw']
    
    # Sửa lỗi format: Thay dấu nháy đơn thành nháy kép để đúng chuẩn JSON
    # và loại bỏ các đối tượng lạ như np.float nếu có
    clean_str = raw_str.replace("'", '"')
    
    try:
        ai_comp_dict = json.loads(clean_str)
    except json.JSONDecodeError:
        # Nếu vẫn lỗi, ta dùng re để trích xuất thủ công (phương án dự phòng)
        # Tìm tất cả các cặp "Key": Value
        pairs = re.findall(r'\"(\w+)\":\s*([\d\.]+)', clean_str)
        ai_comp_dict = {k: float(v) for k, v in pairs}
        
    if not ai_comp_dict: continue
    
    # Chuyển cả 2 thành vector 88 chiều
    v_true = get_comp_vector(true_comp_dict, all_elements)
    v_ai = get_comp_vector(ai_comp_dict, all_elements)
    
    # --- TÍNH TOÁN CHỈ SỐ ---
    # 1. Cosine Similarity (Độ tương đồng cấu trúc)
    norm = (np.linalg.norm(v_true) * np.linalg.norm(v_ai))
    cos_sim = np.dot(v_true, v_ai) / norm if norm != 0 else 0
    
    # 2. Element Recall (Khả năng nhận diện đúng nguyên tố)
    true_elems = set(true_comp_dict.keys())
    ai_elems = set(ai_comp_dict.keys())
    recall = len(true_elems.intersection(ai_elems)) / len(true_elems)
    
    # 3. MAE (Sai số tỷ lệ trung bình)
    mae = np.mean(np.abs(v_true - v_ai))

    evaluation_report.append({
        "True_Formula": true_formula_str,
        "Cosine_Similarity": cos_sim,
        "Element_Recall": recall,
        "MAE": mae
    })

# 3. TỔNG KẾT VÀ IN BÁO CÁO
final_df = pd.DataFrame(evaluation_report)

print("\n" + "="*50)
print("📊 BÁO CÁO ĐÁNH GIÁ THIẾT KẾ NGƯỢC (INVERSE DESIGN)")
print("="*50)
print(f"✅ Độ tương đồng trung bình (Cosine Sim): {final_df['Cosine_Similarity'].mean()*100:.2f}%")
print(f"✅ Khả năng tìm đúng nguyên tố (Recall)  : {final_df['Element_Recall'].mean()*100:.2f}%")
print(f"✅ Sai số tỷ lệ trung bình (MAE)        : {final_df['MAE'].mean():.4f}")
print("="*50)

# Lưu báo cáo chi tiết
final_df.to_csv("results/inverse_final_evaluation.csv", index=False)
print("💾 Đã lưu báo cáo chi tiết tại: results/inverse_final_evaluation.csv")