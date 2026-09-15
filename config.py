from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional, List, Dict, Any
import os
import json
from pathlib import Path


@dataclass
class TrainConfig:
    """
    Cấu hình toàn bộ hệ thống huấn luyện cho mô hình Deep LSTM (Drowsiness Detection).
    Hỗ trợ 2 chế độ:
    1. Chế độ Tensor .pt siêu tốc (Fast Preloaded Tensor Mode - Khuyến nghị & mặc định theo datn4ni2.ipynb).
    2. Chế độ Video thô trích xuất qua CNN (Legacy Raw Video Mode).
    """

    # ---- 1. DATASET CONFIGURATION (TENSOR .PT & RAW VIDEO) ----
    use_preloaded_pt: bool = True                              # True: Dùng Tensor .pt đã trích xuất | False: Đọc video thô qua OpenCV
    train_pt: str = "extracted_features_pt/features_sust_train.pt"  # Đường dẫn tệp tensor train
    val_pt: str = "extracted_features_pt/features_sust_val.pt"      # Đường dẫn tệp tensor validation
    dataset_dir: str = r"D:\Project\AI\dataset\SUST"           # Đường dẫn tới thư mục video thô (nếu dùng chế độ raw video)
    seq_len: int = 120                                         # Số lượng khung hình cố định cho mỗi video (120 cho SUST dataset)
    image_size: Tuple[int, int] = (480, 480)                   # Kích thước khung hình (Height, Width)
    video_exts: Tuple[str, ...] = (".avi", ".mp4", ".mkv")      # Các định dạng video hợp lệ
    train_ratio: float = 0.8                                   # Tỷ lệ chia tập huấn luyện (80% train, 20% validation)
    split_by_subject: bool = True                              # Chia dataset theo người tham gia (Subject-independent split)

    # ---- 2. DATALOADER CONFIGURATION ----
    batch_size: int = 64                                       # Kích thước batch size (64 cho tensor .pt trên GPU T4/RTX)
    num_workers: int = 0                                       # Số lượng worker (0 là tối ưu nhất khi tensor đã nạp sẵn vào RAM)
    pin_memory: bool = True                                    # Tối ưu hóa chuyển dữ liệu lên GPU memory
    shuffle: bool = True                                       # Trộn ngẫu nhiên tập huấn luyện
    drop_last: bool = False                                    # Không bỏ rơi batch cuối cùng để đánh giá trọn vẹn tập dữ liệu
    persistent_workers: bool = False                           # Giữ nguyên worker giữa các epoch
    prefetch_factor: Optional[int] = None                      # Số lượng batch prefetch cho mỗi worker
    seed: int = 42                                             # Seed khởi tạo ngẫu nhiên để đảm bảo tính lặp lại (reproducibility)
    use_dummy_cnn: bool = False                                # Tự động trích xuất đặc trưng qua Dummy CNN trong collate_fn (chỉ cho raw video)

    # ---- 3. MODEL ARCHITECTURE CONFIGURATION ----
    cnn_manifest_path: str = r"/outsrc/myCNN\checkpoints_ftCOCO\model_mainfest.json"
    cnn_weights_path: str = r"/outsrc/myCNN\checkpoints_ftCOCO\ft_step00091000.pt"
    cnn_neck_channels: Tuple[int, int, int] = (224, 448, 640)     # Kênh thực tế của (p3, p4, p5) từ PAFPN
    cnn_strides: Tuple[int, int, int] = (8, 16, 32)                # Strides tương ứng của (p3, p4, p5)
    cnn_num_features: int = 3                                      # Số lượng tầng đặc trưng đầu vào (p3, p4, p5)
    cnn_out_channels: int = 1312                                   # Tổng số kênh khi ghép nối (224 + 448 + 640 = 1312)
    cnn_spatial_size: Tuple[int, int] = (15, 15)                   # Độ phân giải đặc trưng không gian tầng sâu nhất p5
    spatial_fusion: str = "concat"                                 # Phương thức kết hợp: 'concat' | 'sum' | 'mean'
    adapter_dropout: float = 0.1                                   # Tỷ lệ Dropout sau Spatial Feature Adapter
    use_norm: bool = True                                          # Sử dụng LayerNorm trong Spatial Adapter và Dropout trong FC head
    input_dim: int = 256                                           # Kích thước vector đặc trưng x^t sau Spatial Adapter đưa vào LSTM
    hidden_dim: int = 256                                          # Số lượng đơn vị ẩn (hidden units) trong từng khối LSTM
    num_layers: int = 3                                            # Số lớp LSTM xếp chồng (Deep LSTM - 3 lớp)
    num_classes: int = 2                                           # Số lượng lớp đầu ra phân loại (2: Tỉnh táo vs Buồn ngủ)
    dropout: float = 0.2                                           # Tỷ lệ Dropout giữa các lớp LSTM

    # ---- 4. LOSS CONFIGURATION ----
    loss_type: str = "ce"                                      # Loại loss: "ce" (CrossEntropyLoss chuẩn) hoặc "bce" (BCELoss)
    bce_eps: float = 1e-7                                      # Hằng số epsilon kẹp giá trị tránh log(0) (nếu dùng bce)
    bce_reduction: str = "mean"                                # Cách gom nhóm loss ('mean' | 'sum' | 'none')
    pos_weight: Optional[float] = None                         # Trọng số cho lớp dương (1: Buồn ngủ) khi mất cân bằng dữ liệu

    # ---- 5. OPTIMIZER & SCHEDULER CONFIGURATION ----
    epochs: int = 40                                           # Tổng số epoch huấn luyện (40 epochs chỉ mất ~50s với tensor .pt)
    lr0: float = 1e-3                                          # Learning rate khởi tạo
    lr_min_factor: float = 0.01                                # Hệ số lr tối thiểu: lr_min = lr0 * lr_min_factor (1e-5)
    weight_decay: float = 1e-4                                 # Trọng số phạt L2 regularization
    warmup_epochs: float = 1.0                                 # Số epoch khởi động mềm (Warmup)
    optimizer: str = "adamw"                                   # Bộ tối ưu hóa: "adamw" | "adam" | "sgd"
    betas: Tuple[float, float] = (0.9, 0.999)                  # Tham số betas cho Adam/AdamW
    momentum: float = 0.9                                      # Động lượng Momentum khi optimizer="sgd"
    grad_clip_norm: float = 1.0                                # Ngưỡng cắt gradient (Gradient Clipping) tránh bùng nổ gradient
    use_scheduler: bool = True                                 # Bật/Tắt bộ điều chỉnh Learning Rate Scheduler
    scheduler_type: str = "cosine"                             # Loại Scheduler: "cosine" | "step" | "plateau"

    # ---- 6. TENSORBOARD & LOGGING CONFIGURATION ----
    tb_log_dir: str = "runs"                                   # Thư mục lưu log cho TensorBoard
    log_dir: str = "./logs"                                    # Thư mục lưu log text (.log)
    experiment_name: str = "lstm_sust_t4"                      # Tên bài thử nghiệm (experiment)
    checkpoint_dir: str = "lstm_experiment_results/checkpoints" # Thư mục lưu checkpoint mô hình (.pth)
    save_ckpt_interval_epochs: int = 5                         # Số epoch giữa 2 lần lưu checkpoint định kỳ
    save_best_only: bool = False                               # True: Chỉ lưu best checkpoint | False: Lưu định kỳ + last/best
    ckpt_keep_last: int = 3                                    # Số lượng checkpoint định kỳ giữ lại
    resume: str = ""                                           # Đường dẫn file checkpoint để huấn luyện tiếp (rỗng = train từ đầu)

    # ---- 7. RUNTIME & HARDWARE CONFIGURATION ----
    device: str = "cuda"                                       # Thiết bị tính toán ("cuda" | "cpu")
    amp: bool = True                                           # Bật Tự động ép kiểu chính xác hỗn hợp (Automatic Mixed Precision FP16)
    log_interval: int = 10                                     # Số step giữa các lần in log tiến trình chi tiết
    val_interval_epochs: int = 1                               # Số epoch giữa 2 lần đánh giá tập Validation

    def __post_init__(self):
        """Kiểm tra tính hợp lệ của tham số cấu hình và tự động điều chỉnh theo môi trường."""
        # 1. Tự động nhận diện môi trường Kaggle
        if os.path.exists("/kaggle"):
            if not os.path.exists(self.train_pt):
                kaggle_train = "/kaggle/input/datasets/nyvantran6634/sust4ni1/features_sust_train.pt"
                kaggle_val   = "/kaggle/input/datasets/nyvantran6634/sust4ni1/features_sust_val.pt"
                if os.path.exists(kaggle_train):
                    self.train_pt = kaggle_train
                    self.val_pt = kaggle_val
            self.checkpoint_dir = "/kaggle/working/checkpoints"
            self.tb_log_dir = "/kaggle/working/runs"

        # 2. Ràng buộc các tham số dữ liệu & dataloader
        assert self.seq_len > 0, "seq_len phải > 0"
        assert self.batch_size > 0, "batch_size phải > 0"
        assert 0.0 < self.train_ratio < 1.0, "train_ratio phải nằm trong khoảng (0.0, 1.0)"
        assert self.num_workers >= 0, "num_workers không được âm"
        assert self.epochs > 0, "epochs phải > 0"
        assert self.lr0 > 0.0, "lr0 phải > 0"

        # 3. Xử lý logic khi num_workers == 0
        if self.num_workers == 0:
            if self.persistent_workers:
                self.persistent_workers = False
            if self.prefetch_factor is not None:
                self.prefetch_factor = None

        # 4. Ràng buộc tham số mô hình
        assert self.input_dim > 0, "input_dim phải > 0"
        assert self.hidden_dim > 0, "hidden_dim phải > 0"
        assert self.num_layers > 0, "num_layers phải > 0"
        assert self.num_classes > 0, "num_classes phải > 0"

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi Config sang dạng Dictionary."""
        return asdict(self)

    def save_json(self, json_path: str) -> None:
        """Lưu cấu hình ra file JSON."""
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)
        print(f"[TrainConfig] Đã lưu cấu hình vào: {path.resolve()}")

    @classmethod
    def load_json(cls, json_path: str) -> "TrainConfig":
        """Nạp cấu hình từ file JSON."""
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file config JSON: {json_path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Chuyển đổi list thành tuple cho các trường kiểu Tuple
        tuple_fields = ["image_size", "video_exts", "cnn_neck_channels", "cnn_strides", "cnn_spatial_size", "betas"]
        for fld in tuple_fields:
            if fld in data and isinstance(data[fld], list):
                data[fld] = tuple(data[fld])

        return cls(**data)
