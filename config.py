from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional, List, Dict, Any
import json
from pathlib import Path


@dataclass
class TrainConfig:
    """
    Cấu hình toàn bộ hệ thống huấn luyện cho mô hình Deep LSTM (Drowsiness Detection).
    Bao gồm cấu hình Dữ liệu (Dataset), Dataloader, Kiến trúc mô hình (CNN + LSTM),
    Hàm mất mát (Loss), Bộ tối ưu hóa (Optimizer & Scheduler), Logging (TensorBoard) và Runtime.
    """

    # ---- 1. DATASET CONFIGURATION ----
    dataset_dir: str = r"D:\Project\AI\dataset\VBDDD-dataset"  # Đường dẫn tới thư mục VBDDD-dataset
    seq_len: int = 60                                           # Số lượng khung hình cố định cho mỗi video (60 cho FI-DDD / VBDDD)
    image_size: Tuple[int, int] = (480, 480)                   # Kích thước khung hình (Height, Width)
    video_exts: Tuple[str, ...] = (".avi", ".mp4", ".mkv")      # Các định dạng video hợp lệ
    train_ratio: float = 0.8                                   # Tỷ lệ chia tập huấn luyện (80% train, 20% validation)
    split_by_subject: bool = True                              # Chia dataset theo người tham gia (Subject-independent split)

    # ---- 2. DATALOADER CONFIGURATION ----
    batch_size: int = 4                                        # Kích thước batch size cho huấn luyện và đánh giá
    num_workers: int = 0                                       # Số lượng tiến trình worker nạp dữ liệu (0 cho Windows/Debug)
    pin_memory: bool = True                                    # Tối ưu hóa chuyển dữ liệu lên GPU memory
    shuffle: bool = True                                       # Trộn ngẫu nhiên tập huấn luyện
    drop_last: bool = True                                     # Bỏ qua batch cuối cùng nếu không đủ kích thước batch_size
    persistent_workers: bool = False                           # Giữ nguyên worker giữa các epoch (yêu cầu num_workers > 0)
    prefetch_factor: Optional[int] = None                      # Số lượng batch prefetch cho mỗi worker (yêu cầu num_workers > 0)
    seed: int = 42                                             # Seed khởi tạo ngẫu nhiên để đảm bảo tính lặp lại (reproducibility)
    use_dummy_cnn: bool = True                                 # Tự động trích xuất đặc trưng qua Dummy CNN trong collate_fn

    # ---- 3. MODEL ARCHITECTURE CONFIGURATION ----
    cnn_manifest_path: str = r"D:\Project\AI\myLSTM\myCNN\checkpoints_ftCOCO\model_mainfest.json"
    cnn_weights_path: str = r"D:\Project\AI\myLSTM\myCNN\checkpoints_ftCOCO\ft_step00091000.pt"
    cnn_out_channels: int = 640                                # Số kênh đầu ra của mô hình CNN feature extractor (640 theo kientruc.md)
    cnn_spatial_size: Tuple[int, int] = (15, 15)               # Độ phân giải đặc trưng không gian (15, 15 theo kientruc.md)
    input_dim: int = 256                                       # Kích thước vector đặc trưng x^t sau Spatial Adapter đưa vào LSTM
    hidden_dim: int = 256                                      # Số lượng đơn vị ẩn (hidden units) trong từng khối LSTM
    num_layers: int = 3                                        # Số lớp LSTM xếp chồng (Deep LSTM - 3 lớp)
    num_classes: int = 2                                       # Số lượng lớp đầu ra phân loại (2: Tỉnh táo vs Buồn ngủ)
    dropout: float = 0.0                                       # Tỷ lệ Dropout giữa các lớp LSTM

    # ---- 4. LOSS CONFIGURATION ----
    loss_type: str = "bce"                                     # Loại loss: "bce" (DrowsinessBCELoss)
    bce_eps: float = 1e-7                                      # Hằng số epsilon kẹp giá trị tránh log(0)
    bce_reduction: str = "mean"                                # Cách gom nhóm loss ('mean' | 'sum' | 'none')
    pos_weight: Optional[float] = None                         # Trọng số cho lớp dương (1: Buồn ngủ) khi mất cân bằng dữ liệu

    # ---- 5. OPTIMIZER & SCHEDULER CONFIGURATION ----
    epochs: int = 10                                           # Tổng số epoch huấn luyện
    lr0: float = 1e-3                                          # Learning rate khởi tạo / sau warmup
    lr_min_factor: float = 0.01                                # Hệ số lr tối thiểu: lr_min = lr0 * lr_min_factor
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
    experiment_name: str = "lstm_drowsiness"                   # Tên bài thử nghiệm (experiment)
    checkpoint_dir: str = "./checkpoints"                      # Thư mục lưu checkpoint mô hình (.pth)
    save_ckpt_interval_epochs: int = 1                         # Số epoch giữa 2 lần lưu checkpoint định kỳ
    save_best_only: bool = False                               # True: Chỉ lưu best checkpoint | False: Lưu định kỳ + last/best
    ckpt_keep_last: int = 3                                    # Số lượng checkpoint định kỳ giữ lại (<=0 để giữ tất cả)
    resume: str = ""                                           # Đường dẫn file checkpoint để huấn luyện tiếp (rỗng = train từ đầu)

    # ---- 7. RUNTIME & HARDWARE CONFIGURATION ----
    device: str = "cuda"                                       # Thiết bị tính toán ("cuda" | "cpu")
    amp: bool = False                                          # Bật/Tắt Tự động ép kiểu chính xác hỗn hợp (Automatic Mixed Precision)
    log_interval: int = 10                                     # Số step giữa các lần in log tiến trình chi tiết
    val_interval_epochs: int = 1                               # Số epoch giữa 2 lần đánh giá tập Validation

    def __post_init__(self):
        """Kiểm tra tính hợp lệ của tham số cấu hình và tự động điều chỉnh nếu cần."""
        # 1. Ràng buộc các tham số dữ liệu & dataloader
        assert self.seq_len > 0, "seq_len phải > 0"
        assert self.batch_size > 0, "batch_size phải > 0"
        assert 0.0 < self.train_ratio < 1.0, "train_ratio phải nằm trong khoảng (0.0, 1.0)"
        assert self.num_workers >= 0, "num_workers không được âm"
        assert self.epochs > 0, "epochs phải > 0"
        assert self.lr0 > 0.0, "lr0 phải > 0"

        # 2. Xử lý logic khi num_workers == 0
        if self.num_workers == 0:
            if self.persistent_workers:
                print("[TrainConfig][Warning] persistent_workers=True yêu cầu num_workers > 0. "
                      "Tự động đặt lại persistent_workers=False.")
                self.persistent_workers = False
            if self.prefetch_factor is not None:
                print("[TrainConfig][Warning] prefetch_factor chỉ có tác dụng khi num_workers > 0. "
                      "Tự động đặt lại prefetch_factor=None.")
                self.prefetch_factor = None

        # 3. Ràng buộc tham số mô hình
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
        if "image_size" in data and isinstance(data["image_size"], list):
            data["image_size"] = tuple(data["image_size"])
        if "video_exts" in data and isinstance(data["video_exts"], list):
            data["video_exts"] = tuple(data["video_exts"])
        if "cnn_spatial_size" in data and isinstance(data["cnn_spatial_size"], list):
            data["cnn_spatial_size"] = tuple(data["cnn_spatial_size"])
        if "betas" in data and isinstance(data["betas"], list):
            data["betas"] = tuple(data["betas"])

        return cls(**data)



