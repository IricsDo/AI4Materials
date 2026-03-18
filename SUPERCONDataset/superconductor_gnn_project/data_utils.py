from mp_api.client import MPRester
import pandas as pd

def augment_with_mp_data(csv_path, api_key):
    mpr = MPRester(api_key)
    df = pd.read_csv(csv_path)
    
    # Tạo dictionary để map formula -> formation_energy
    # Lưu ý: Supercon có nhiều chất, ta sẽ query theo công thức hóa học
    print("📡 Đang truy vấn Materials Project để lấy Formation Energy...")
    
    # Ví dụ query mẫu (Trong thực tế bạn nên loop hoặc query hàng loạt)
    # Lưu ý: $E_{form}$ đơn vị eV/atom. Càng âm càng bền.
    results = mpr.summary.search(fields=["formula_pretty", "formation_energy_per_atom", "volume"])
    
    # ... Logic để merge dữ liệu này vào file featurized.csv hiện tại của bạn ...
    # Tạo cột: 'formation_energy' và 'volume_per_atom'