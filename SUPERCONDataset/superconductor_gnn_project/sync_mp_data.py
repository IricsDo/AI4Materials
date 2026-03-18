import pandas as pd
from mp_api.client import MPRester
from tqdm import tqdm
import time

MP_API = "28Un4sx89agTjSG1vQF7U5PXEKGU0PDj"


def get_physical_properties(df_path):
    df = pd.read_csv(df_path)
    # Lấy danh sách các công thức duy nhất để tiết kiệm số lần gọi API
    unique_formulas = df['formula'].unique().tolist()
    mp_data = {}

    with MPRester(MP_API) as mpr:
        print(f"📡 Đang truy vấn {len(unique_formulas)} vật liệu từ Materials Project...")
        
        # Truy vấn theo lô (batch) để tránh bị block API
        for i in tqdm(range(0, len(unique_formulas), 100)):
            batch = unique_formulas[i:i+100]
            try:
                # 1. Cập nhật đường dẫn: materials.summary.search
                # 2. Thêm "elements" vào danh sách fields
                docs = mpr.materials.summary.search(
                    formula=batch, 
                    fields=["formula_pretty", "formation_energy_per_atom", "volume", "elements"]
                )

                for doc in docs:
                    f = doc.formula_pretty
                    # Lấy pha có formation energy thấp nhất (bền nhất)
                    if f not in mp_data or doc.formation_energy_per_atom < mp_data[f]['e_form']:
                        # Tính volume per atom: Tổng volume / số lượng nguyên tử trong công thức
                        n_atoms = len(doc.elements) if doc.elements else 1
                        mp_data[f] = {
                            'e_form': doc.formation_energy_per_atom,
                            'volume': doc.volume / n_atoms
                        }
            except Exception as e:
                print(f"Lỗi tại batch {i}: {e}")

    # Map dữ liệu ngược lại dataframe gốc
    
    mp_df = pd.DataFrame.from_dict(mp_data, orient='index')

    mp_df = mp_df.rename(columns={
        'e_form': 'formation_energy',
        'volume': 'volume_per_atom'
    })

    df = df.merge(
        mp_df[['formation_energy', 'volume_per_atom']],
        left_on='formula',
        right_index=True,
        how='left'
    )

    df['formation_energy'].fillna(0.5, inplace=True)     # Mặc định 0.5 (không bền) nếu không tìm thấy
    df['volume_per_atom'].fillna(15.0, inplace=True)     # Mặc định 15.0 A^3

    df.to_csv("data/raw/featurized_multi_task.csv", index=False)
    print("✅ Đã tạo xong file: data/raw/featurized_multi_task.csv")

# Chạy script
get_physical_properties("data/raw/featurized.csv")