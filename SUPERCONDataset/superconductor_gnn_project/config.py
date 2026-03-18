import os

# ==========================================
# 1. ĐƯỜNG DẪN THƯ MỤC VÀ FILE (PATHS)
# ==========================================
# Khởi tạo thư mục tự động nếu chưa có
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# File chứa dữ liệu gốc (đảm bảo file này có cột 'formula' và 'Tc')
# CSV_PATH = "data/raw/featurized.csv" 
CSV_PATH = "data/raw/featurized_multi_task.csv" 


# Nơi lưu trọng số mô hình GNN tốt nhất sau khi train
# MODEL_PATH = "models/best_gnn_model.pt"
# MODEL_PATH = "models/gatv2_supercon_model.pth"
MODEL_PATH = "models/gatv2_gnn_multi_task_model.pth"

# Nơi lưu danh sách vật liệu mới được sinh ra từ Inverse Design
GENERATED_MATERIALS_PATH = "results/generated_materials.csv"


# ==========================================
# 2. THÔNG SỐ DỮ LIỆU (DATA)
# ==========================================
TEST_SIZE = 0.2         # Dành 20% dữ liệu cho tập Test/Validation
RANDOM_STATE = 42       # Cố định random seed để kết quả có thể lặp lại (Reproducible)
BATCH_SIZE = 128         # Số lượng đồ thị (graphs) đưa vào mô hình trong 1 lần cập nhật trọng số


# ==========================================
# 3. THÔNG SỐ HUẤN LUYỆN GNN (TRAINING)
# ==========================================
EPOCHS = 10000            # Tổng số vòng lặp tối đa
LR = 1e-2               # Tốc độ học (Learning Rate) khởi tạo của mô hình
HIDDEN_DIM = 128         # Số chiều của các lớp ẩn trong Graph Neural Network

# Thông số Scheduler (Giảm Learning Rate)
SCHEDULER_PATIENCE = 7  # Đợi 7 epoch, nếu loss không giảm thì hạ LR
SCHEDULER_FACTOR = 0.5  # Mức độ giảm LR (0.5 nghĩa là giảm một nửa)
MIN_LR = 1e-6           # Mức LR nhỏ nhất, không giảm thêm nữa

# Thông số Early Stopping (Dừng sớm)
ES_PATIENCE = 20        # Đợi 20 epoch, nếu loss không giảm thì ngắt vòng lặp
ES_MIN_DELTA = 0.05     # Mức cải thiện tối thiểu (đơn vị: Kelvin) để được tính là có giảm


# ==========================================
# 4. THÔNG SỐ INVERSE DESIGN (THIẾT KẾ NGƯỢC)
# ==========================================
TARGET_TC = 93.0       # Mức nhiệt độ siêu dẫn mục tiêu bạn muốn tìm (VD: 150 Kelvin)
INVERSE_STEPS = 100000    # Số bước Gradient Ascent để tối ưu hóa công thức hóa học
INVERSE_LR = 0.1       # Tốc độ học cho pha Inverse (thường lớn hơn LR lúc train một chút)
THRESHOLD_FRAC = 0.01   # Loại bỏ các nguyên tố có tỷ lệ phần trăm nhỏ hơn 1% (chống nhiễu)

