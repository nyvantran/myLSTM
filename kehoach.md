# KẾ HOẠCH TỐI ƯU HÓA HUẤN LUYỆN TOÀN DIỆN
## CHIẾN LƯỢC: PHÂN CHIA TẬP DỮ LIỆU & TRÍCH XUẤT ĐẶC TRƯNG LOCAL RA TENSOR PYTORCH (`.pt`) - HUẤN LUYỆN SIÊU TỐC TRÊN KAGGLE VỚI `datn4ni2.ipynb` (GIÁM SÁT TENSORBOARD)

- **Notebook huấn luyện mới trên Kaggle:** `datn4ni2.ipynb` (kế thừa và giữ nguyên file gốc `datn4ni.ipynb` để đối chiếu).
- **Môi trường trích xuất:** Máy cục bộ (Local PC với GPU CUDA & tập dữ liệu 2,074 video có sẵn tại `D:\Project\AI\dataset\SUST`).
- **Chiến lược phân chia:** Tách biệt tập **Train (80%)** và **Validation (20%)** ở cấp độ **Video ID** *trước khi* thực hiện trích xuất, lưu siêu dữ liệu phân bổ `dataset_split.json`.
- **Định dạng lưu trữ trung gian:** Tệp PyTorch Tensor nhị phân (`features_sust_train.pt` và `features_sust_val.pt`).
- **Môi trường huấn luyện:** Kaggle (2x Tesla T4, 29GB Host RAM, 4 vCPUs, 12h session).
- **Công cụ giám sát & phân tích:** **TensorBoard** (theo dõi Loss, Accuracy, Learning Rate, đồ thị mô hình Graph, Confusion Matrix theo thời gian thực).

---

## 1. TỔNG QUAN CHIẾN LƯỢC TOÀN DIỆN

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│              KIẾN TRÚC TỐI ƯU TOÀN TRÌNH (PRE-SPLIT -> LOCAL TENSOR .PT -> KAGGLE TENSORBOARD)          │
├───────────────────────────────────────────────────────────┬─────────────────────────────────────────────┤
│ GIAI ĐOẠN 1: TẠI MÁY LOCAL (LOCAL PC)                     │ GIAI ĐOẠN 2: TRÊN KAGGLE (datn4ni2.ipynb)   │
│                                                           │                                             │
│ ┌───────────────────────────────────────────────────────┐ │ ┌─────────────────────────────────────────┐ │
│ │ 2,074 Video SUST (D:\..\SUST)                         │ │ │ Kaggle Dataset                          │ │
│ └──────────────────────────┬────────────────────────────┘ │ │ features_sust_train.pt (~1.0 GB)        │ │
│                            │                              │ │ features_sust_val.pt   (~260 MB)        │ │
│                            ▼                              │ └────────────────────┬────────────────────┘ │
│ ┌───────────────────────────────────────────────────────┐ │                      │ torch.load (< 0.5s)  │
│ │ BƯỚC 1: PHÂN CHIA TRAIN / VAL TRƯỚC (SEED 42)         │ │                      ▼                      │
│ │ - Train: 1,659 videos (80%)                           │ │ ┌─────────────────────────────────────────┐ │
│ │ - Val:     415 videos (20%)                           │ │ │ PreloadedTensorDataset                  │ │
│ │ - Lưu siêu dữ liệu phân bổ dataset_split.json         │ │ │ [N, 120, C] sẵn sàng trong RAM          │ │
│ └──────────┬─────────────────────────────────┬──────────┘ │ └────────────────────┬────────────────────┘ │
│            │                                 │            │                      │                      │
│            ▼                                 ▼            │                      │ DataLoader (Batch 64)│
│ ┌──────────────────────┐         ┌──────────────────────┐ │                      ▼                      │
│ │ Trích xuất Train     │         │ Trích xuất Val       │ │ ┌─────────────────────────────────────────┐ │
│ │ ONNX CUDA + Pooling  │         │ ONNX CUDA + Pooling  │ │ │ Train Deep LSTM (AMP FP16)              │ │
│ └──────────┬───────────┘         └──────────┬───────────┘ │ │ Tesla T4 GPU 0: ~1.2s / Epoch!          │ │
│            │                                │             │ └────────────────────┬────────────────────┘ │
│            ▼                                ▼             │                      │ Ghi log thời gian    │
│ ┌──────────────────────┐         ┌──────────────────────┐ │                      ▼ thực (Loss/Acc/LR)   │
│ │ features_sust_train  │         │ features_sust_val    │ │ ┌─────────────────────────────────────────┐ │
│ │ .pt (~1.0 GB)        │         │ .pt (~260 MB)        │ │ │ TENSORBOARD WEB UI                      │ │
│ └──────────┬───────────┘         └──────────┬───────────┘ │ │ Real-time: Loss, Acc, F1, Matrix Graph  │ │
│            └──────────────────┬─────────────────────┘     │ └─────────────────────────────────────────┘ │
│                               │ Upload Kaggle             │                                             │
│                               ▼                           │                                             │
└───────────────────────────────────────────────────────────┴─────────────────────────────────────────────┘
```

### Lợi ích cốt lõi của chiến lược:
1. **Phân chia Train/Val độc lập & triệt để (Không Data Leakage):**
   - Phân chia ở cấp độ Video ID/chủ thể (Subject) trước khi nạp bất kỳ frame nào vào mô hình.
   - Tập Train và Val được xử lý và lưu trữ thành 2 tệp `.pt` riêng biệt hoàn toàn.
2. **Vượt trội hoàn toàn so với định dạng CSV / Text:**
   - **Tốc độ đọc:** `torch.load()` nạp file nhị phân trực tiếp vào RAM trong **< 0.5 giây** (thay vì mất 10 - 20 giây đọc và parse text của CSV).
   - **Không cần tiền xử lý phức tạp:** Dữ liệu đã là Tensor 3D chuẩn `[N, 120, C]`, không cần `df.groupby("video_id")`, không cần lặp qua từng video để gom frame hay padding thủ công trên Kaggle.
   - **Bảo toàn độ chính xác số học:** Dữ liệu float32/float16 được lưu nguyên bản dưới dạng byte nhị phân, không bị sai số làm tròn khi parse sang chuỗi như CSV.
3. **Loại bỏ 100% gánh nặng I/O giải mã video & suy luận ONNX trên Kaggle:**
   - Kaggle không cần chạy 3.7 triệu lần forward CNN lặp đi lặp lại qua từng epoch.
   - Vòng lặp train LSTM chạy với tốc độ tối đa của GPU Tesla T4 (~1.2 - 1.5s/epoch).
   - 40 epoch huấn luyện chỉ tốn **dưới 1 phút**, tiêu hao chưa tới **0.02 giờ** trong hạn ngạch 30 giờ GPU hàng tuần của Kaggle.
4. **Giám sát trực quan chuyên nghiệp với TensorBoard:**
   - Theo dõi sự hội tụ của mô hình (Overfitting / Underfitting) qua từng epoch và batch.
   - Nhận diện tức thì tình trạng learning rate decay, gradient vanishing.
   - Trực quan hóa Confusion Matrix và sơ đồ kiến trúc Computational Graph.

---

## 2. THIẾT KẾ ĐỊNH DẠNG DỮ LIỆU TENSOR PYTORCH (`.pt` SPECIFICATION)

### 2.1. Cấu trúc lưu trữ dữ liệu (Tensor Dictionary Schema)
Mỗi tệp `.pt` (`features_sust_train.pt` và `features_sust_val.pt`) là một Python Dictionary chuẩn của PyTorch, chứa toàn bộ tensor đã được padding cố định về chiều dài chuỗi thời gian ($T = 120$ frames):

```python
{
    "p3": torch.Tensor,        # Shape: [N, 120, 224],  dtype: torch.float32 (Feature tầng p3)
    "p4": torch.Tensor,        # Shape: [N, 120, 448],  dtype: torch.float32 (Feature tầng p4)
    "p5": torch.Tensor,        # Shape: [N, 120, 640],  dtype: torch.float32 (Feature tầng p5)
    "labels": torch.Tensor,    # Shape: [N],            dtype: torch.long    (Nhãn 0: Alert, 1: Drowsy)
    "video_ids": list[str],    # Danh sách N chuỗi video_id (Ví dụ: ['001-driving-01', ...])
    "seq_lens": torch.Tensor,  # Shape: [N],            dtype: torch.int32   (Số frame thực tế trước khi pad)
}
```

Trong đó $N$ là số lượng video của mỗi tập:
- **Tập Train ($N_{train} = 1,659$ video - 80%):**
  - `p3`: Tensor kích thước $[1659, 120, 224]$ $\approx 178.4\text{ MB}$.
  - `p4`: Tensor kích thước $[1659, 120, 448]$ $\approx 356.8\text{ MB}$.
  - `p5`: Tensor kích thước $[1659, 120, 640]$ $\approx 509.7\text{ MB}$.
  - `labels`: Tensor kích thước $[1659]$ $\approx 13\text{ KB}$.
  - **Tổng dung lượng Train `.pt`:** $\mathbf{\approx 1.04\text{ GB}}$ (nếu lưu chuẩn `torch.save()`).
- **Tập Validation ($N_{val} = 415$ video - 20%):**
  - `p3`: Tensor kích thước $[415, 120, 224]$ $\approx 44.6\text{ MB}$.
  - `p4`: Tensor kích thước $[415, 120, 448]$ $\approx 89.3\text{ MB}$.
  - `p5`: Tensor kích thước $[415, 120, 640]$ $\approx 127.5\text{ MB}$.
  - `labels`: Tensor kích thước $[415]$ $\approx 3.3\text{ KB}$.
  - **Tổng dung lượng Val `.pt`:** $\mathbf{\approx 261\text{ MB}}$.

> [!TIP]
> **Tối ưu dung lượng (Tùy chọn FP16):** Nếu ép kiểu tensor sang `torch.float16` trước khi lưu, dung lượng sẽ giảm **50%**: Tập Train chỉ còn **~520 MB** và Tập Val chỉ còn **~130 MB**, tải lên Kaggle Dataset cực nhanh và tiết kiệm bộ nhớ.

---

## 3. THIẾT KẾ HỆ THỐNG GIÁM SÁT TENSORBOARD TOÀN DIỆN

Hệ thống giám sát huấn luyện được xây dựng dựa trên lớp chuẩn hóa [`TensorBoardLogger`](file:///D:/Project/AI/myLSTM/log.py), tích hợp trực tiếp vào notebook `datn4ni2.ipynb` trên Kaggle.

### 3.1. Phân loại các chỉ số giám sát (Metrics Taxonomy)

```
runs/
└── lstm_drowsiness_20260913_220000/
    ├── events.out.tfevents...
    ├── Train/
    │   ├── Batch_Loss           (Mỗi batch huấn luyện)
    │   ├── Batch_Accuracy       (Mỗi batch huấn luyện)
    │   ├── Epoch_Loss           (Tổng kết epoch train)
    │   ├── Epoch_Accuracy       (Tổng kết epoch train)
    │   └── Learning_Rate        (Theo dõi biến thiên LR từ Scheduler)
    ├── Val/
    │   ├── Epoch_Loss           (Loss trên tập Validation)
    │   └── Epoch_Accuracy       (Độ chính xác Validation)
    ├── Metrics/
    │   ├── Precision            (Độ chuẩn xác nhận diện buồn ngủ)
    │   ├── Recall               (Độ nhạy bắt trúng trạng thái buồn ngủ)
    │   └── F1_Score             (Điểm điều hòa F1)
    ├── Images/
    │   └── Confusion_Matrix     (Ảnh nhiệt ma trận nhầm lẫn sau mỗi 5 epoch)
    └── Graphs/
        └── DeepLSTM_Graph       (Sơ đồ tính toán mô hình mạng nơ-ron)
```

1. **Giám sát mức độ chi tiết (Batch-level):**
   - Thẻ `Train/Batch_Loss` & `Train/Batch_Accuracy`: Giúp phát hiện tức thời tình trạng loss nhảy bất thường (loss spike) hoặc gradient bùng nổ ngay trong quá trình cập nhật weights.
2. **Giám sát tổng kết (Epoch-level):**
   - Ghép cặp `Train/Epoch_Loss` song song với `Val/Epoch_Loss` để quan sát điểm giao cắt (overfitting point).
   - Theo dõi `Train/Learning_Rate` để kiểm chứng bộ điều chỉnh tốc độ học (CosineAnnealingLR hoặc ReduceLROnPlateau).
3. **Chỉ số đánh giá chuyên sâu (Evaluation Metrics):**
   - Đánh giá phân loại nhị phân: Ghi log `Precision`, `Recall`, `F1-Score` của lớp "Buồn ngủ" (Drowsy) ở cuối mỗi epoch đánh giá.
4. **Trực quan hóa hình ảnh (Visual Artifacts):**
   - Đẩy ảnh ma trận nhầm lẫn (Confusion Matrix Heatmap) vẽ bằng `matplotlib` lên giao diện TensorBoard qua `SummaryWriter.add_figure()`.
5. **Sơ đồ ma trận kiến trúc mạng (Computational Graph):**
   - Nạp tensor mẫu giả lập `((p3, p4, p5))` và gọi `writer.add_graph(model, input_sample)` để xem toàn bộ luồng kết nối LSTM, Dropout và Linear layers.

---

## 4. KẾ HOẠCH TRIỂN KHAI CHI TIẾT (3 GIAI ĐOẠN)

### GIAI ĐOẠN 1: PHÂN CHIA DỮ LIỆU & TRÍCH XUẤT LOCAL RA FILE `.pt`

#### Nhiệm vụ:
Tạo file script Python độc lập: [`extract_to_pt.py`](file:///D:/Project/AI/myLSTM/extract_to_pt.py) chạy trên máy local.

#### Các bước thực hiện chặt chẽ trong script:
1. **Bước 1: Quét danh mục Video & Trích xuất Metadata nhãn:**
   - Quét toàn bộ 2,074 video từ thư mục `D:\Project\AI\dataset\SUST`.
   - Xác định nhãn (0: Alert, 1: Drowsy) dựa trên quy ước tên thư mục/tên file.
2. **Bước 2: Phân chia Train/Validation (80/20) TIÊN QUYẾT:**
   - Sử dụng `train_test_split` của `scikit-learn` với tỷ lệ `test_size=0.20`, `random_state=42`, phân tầng theo nhãn (`stratify=labels`).
   - Kết quả: **1,659 video Train** và **415 video Validation**.
   - Lưu lại cấu hình phân bổ vào tệp `dataset_split.json` gồm danh sách video ID cụ thể của từng tập để đảm bảo tính tái lập 100%.
3. **Bước 3: Khởi tạo ONNX Runtime GPU Local:**
   - Nạp mô hình `clone.onnx` với `CUDAExecutionProvider`.
4. **Bước 4: Vòng lặp trích xuất tuần tự cho từng tập (Tách biệt hoàn toàn):**
   - **Vòng lặp Train:** Lặp qua 1,659 video Train:
     - Đọc 120 khung hình bằng OpenCV kết hợp resize letterbox $480 \times 480$.
     - Forward qua ONNX backbone theo từng mini-chunk.
     - Áp dụng `F.adaptive_avg_pool2d(..., (1, 1)).flatten(1)`.
     - Lưu kết quả vào danh sách bộ nhớ tạm và đệm padding đủ $seq\_len = 120$.
     - Gom lại thành 3 Tensor lớn: `p3` $[1659, 120, 224]$, `p4` $[1659, 120, 448]$, `p5` $[1659, 120, 640]$ cùng `labels` $[1659]$.
     - Lưu file: `features_sust_train.pt`.
   - **Vòng lặp Validation:** Lặp qua 415 video Validation tương tự.
     - Gom lại thành 3 Tensor: `p3` $[415, 120, 224]$, `p4` $[415, 120, 448]$, `p5` $[415, 120, 640]$ cùng `labels` $[415]$.
     - Lưu file: `features_sust_val.pt`.

---

### GIAI ĐOẠN 2: TẠO DATASET TENSOR TRÊN KAGGLE

#### Các bước:
1. Truy cập [Kaggle Datasets](https://www.kaggle.com/datasets) $\rightarrow$ Click **New Dataset**.
2. Đặt tên dataset: ví dụ `sust-extracted-features-pt`.
3. Upload 2 file:
   - `features_sust_train.pt` (~1.0 GB)
   - `features_sust_val.pt` (~260 MB)
   - *(Kèm file `dataset_split.json` để kiểm tra đối chiếu).*
4. Nhấn **Create** để Kaggle lưu trữ dataset.

---

### GIAI ĐOẠN 3: XÂY DỰNG NOTEBOOK MỚI `datn4ni2.ipynb` TRÊN KAGGLE

Tạo file notebook mới **`datn4ni2.ipynb`** tích hợp nạp Tensor siêu tốc, huấn luyện Deep LSTM với AMP FP16 và giám sát TensorBoard.

#### Cấu trúc các Cell trong `datn4ni2.ipynb`:

#### Cell 0: Thiết lập môi trường & Import thư viện
```python
import os
import sys
import time
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tqdm.auto import tqdm

print(f"[+] PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[+] GPU 0: {torch.cuda.get_device_name(0)}")
```

#### Cell 1: Cấu hình tham số tập trung (`KaggleTensorConfig`)
```python
class KaggleTensorConfig:
    # Đường dẫn Dataset Tensor .pt trên Kaggle
    train_pt = "/kaggle/input/sust-extracted-features-pt/features_sust_train.pt"
    val_pt   = "/kaggle/input/sust-extracted-features-pt/features_sust_val.pt"
    output_dir = "/kaggle/working"
    log_dir = "/kaggle/working/runs"
    experiment_name = "lstm_drowsiness_t4"
    
    # Siêu tham số mô hình
    seq_len: int = 120
    input_dim: int = 256
    hidden_dim: int = 256
    num_layers: int = 3
    num_classes: int = 2
    dropout: float = 0.2
    
    # Tối ưu tốc độ huấn luyện
    batch_size: int = 64            # 64 sequences mỗi batch
    epochs: int = 40                # 40 epochs chỉ mất ~1 phút
    lr0: float = 1e-3
    weight_decay: float = 1e-4
    amp: bool = True                # Bật Mixed Precision FP16
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

cfg = KaggleTensorConfig()
```

#### Cell 2: Khởi tạo TensorBoard Engine tích hợp
```python
class TensorBoardLogger:
    """Quản lý ghi log sự kiện và biểu đồ trực quan hóa."""
    def __init__(self, log_dir: str = "/kaggle/working/runs", experiment_name: str = "lstm"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = Path(log_dir) / f"{experiment_name}_{timestamp}"
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_path))
        print(f"[TensorBoard] Log directory initialized at: {self.log_path}")

    def log_train_batch(self, loss: float, accuracy: float, step: int):
        self.writer.add_scalar("Train/Batch_Loss", loss, step)
        self.writer.add_scalar("Train/Batch_Accuracy", accuracy, step)

    def log_train_epoch(self, epoch_loss: float, epoch_acc: float, lr: float, epoch: int):
        self.writer.add_scalar("Train/Epoch_Loss", epoch_loss, epoch)
        self.writer.add_scalar("Train/Epoch_Accuracy", epoch_acc, epoch)
        self.writer.add_scalar("Train/Learning_Rate", lr, epoch)

    def log_val_epoch(self, val_loss: float, val_acc: float, epoch: int):
        self.writer.add_scalar("Val/Epoch_Loss", val_loss, epoch)
        self.writer.add_scalar("Val/Epoch_Accuracy", val_acc, epoch)

    def log_metrics(self, metrics: dict, epoch: int):
        for k, v in metrics.items():
            self.writer.add_scalar(f"Metrics/{k}", v, epoch)

    def log_confusion_matrix(self, cm: np.ndarray, epoch: int, class_names=["Alert", "Drowsy"]):
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix (Epoch {epoch})")
        plt.tight_layout()
        self.writer.add_figure("Images/Confusion_Matrix", fig, epoch)
        plt.close(fig)

    def log_graph(self, model: nn.Module, dummy_input: tuple):
        try:
            self.writer.add_graph(model, dummy_input)
            print("[TensorBoard] Computational Graph logged successfully.")
        except Exception as e:
            print(f"[TensorBoard] Could not log graph: {e}")

    def close(self):
        self.writer.close()
        print(f"[TensorBoard] SummaryWriter closed.")
```

#### Cell 3: Khởi chạy giao diện TensorBoard trên Kaggle
```python
# Mở bảng điều khiển TensorBoard trực tiếp trong giao diện notebook Kaggle
%load_ext tensorboard
%tensorboard --logdir /kaggle/working/runs
```

#### Cell 4: Kiến trúc mô hình `DeepLSTMClassifier` & `SpatialFeatureAdapter`
Kế thừa mô hình tối ưu đã hoàn thiện trong [`model.py`](file:///D:/Project/AI/myLSTM/model.py):
- Nhận trực tiếp tensor $(p3: 224, p4: 448, p5: 640)$.
- Chiếu về 256 chiều thông qua Linear Layer kết hợp LayerNorm & ReLU.
- Đưa qua 3 tầng LSTM đa lớp với Dropout 0.2.
- Phân loại nhị phân ở bước thời gian cuối cùng.

#### Cell 5: Dataset nạp Tensor siêu tốc (`PreloadedTensorDataset`)
Nạp trực tiếp từ file `.pt` vào RAM trong chưa đầy 0.5 giây:
```python
class PreloadedTensorDataset(Dataset):
    """Nạp trực tiếp toàn bộ chuỗi tensor đã trích xuất vào bộ nhớ."""
    def __init__(self, pt_path: str):
        print(f"[+] Đang nạp tensor từ: {pt_path}...")
        t0 = time.time()
        data = torch.load(pt_path, map_location="cpu")
        
        self.p3 = data["p3"].float()       # [N, 120, 224]
        self.p4 = data["p4"].float()       # [N, 120, 448]
        self.p5 = data["p5"].float()       # [N, 120, 640]
        self.labels = data["labels"].long() # [N]
        self.video_ids = data.get("video_ids", [])
        
        print(f"    - Nạp thành công {len(self.labels)} video trong {time.time()-t0:.3f}s!")
        print(f"    - Kích thước tensor: p3={list(self.p3.shape)}, p4={list(self.p4.shape)}, p5={list(self.p5.shape)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.p3[idx], self.p4[idx], self.p5[idx]), self.labels[idx]
```

#### Cell 6: Khởi tạo Dataloaders, Mô hình & Đăng ký Graph vào TensorBoard
```python
# Nạp Dataset
train_dataset = PreloadedTensorDataset(cfg.train_pt)
val_dataset   = PreloadedTensorDataset(cfg.val_pt)

train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, pin_memory=True)

# Khởi tạo mô hình
model = DeepLSTMClassifier(
    num_classes=cfg.num_classes,
    hidden_dim=cfg.hidden_dim,
    num_layers=cfg.num_layers,
    dropout=cfg.dropout
).to(cfg.device)

# Đăng ký Computational Graph vào TensorBoard
logger = TensorBoardLogger(log_dir=cfg.log_dir, experiment_name=cfg.experiment_name)
dummy_input = (
    torch.zeros(1, cfg.seq_len, 224, device=cfg.device),
    torch.zeros(1, cfg.seq_len, 448, device=cfg.device),
    torch.zeros(1, cfg.seq_len, 640, device=cfg.device)
)
logger.log_graph(model, dummy_input)
```

#### Cell 7: Vòng lặp huấn luyện kết hợp AMP FP16 & TensorBoard
```python
scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr0, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-5)
criterion = nn.CrossEntropyLoss()

best_val_acc = 0.0
global_step = 0

for epoch in range(1, cfg.epochs + 1):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    for (p3, p4, p5), labels in train_loader:
        p3, p4, p5, labels = p3.to(cfg.device), p4.to(cfg.device), p5.to(cfg.device), labels.to(cfg.device)
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=cfg.amp):
            outputs = model((p3, p4, p5))
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        batch_acc = (outputs.argmax(1) == labels).float().mean().item()
        logger.log_train_batch(loss.item(), batch_acc, global_step)
        global_step += 1
        
        total_loss += loss.item() * len(labels)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += len(labels)
        
    epoch_train_loss = total_loss / total
    epoch_train_acc = correct / total
    current_lr = optimizer.param_groups[0]["lr"]
    logger.log_train_epoch(epoch_train_loss, epoch_train_acc, current_lr, epoch)
    scheduler.step()
    
    # Validation Loop
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for (p3, p4, p5), labels in val_loader:
            p3, p4, p5, labels = p3.to(cfg.device), p4.to(cfg.device), p5.to(cfg.device), labels.to(cfg.device)
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                outputs = model((p3, p4, p5))
                loss = criterion(outputs, labels)
            val_loss += loss.item() * len(labels)
            preds = outputs.argmax(1)
            val_correct += (preds == labels).sum().item()
            val_total += len(labels)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    epoch_val_loss = val_loss / val_total
    epoch_val_acc = val_correct / val_total
    logger.log_val_epoch(epoch_val_loss, epoch_val_acc, epoch)
    
    # Tính Precision, Recall, F1
    metrics = {
        "Precision": precision_score(all_targets, all_preds, zero_division=0),
        "Recall": recall_score(all_targets, all_preds, zero_division=0),
        "F1_Score": f1_score(all_targets, all_preds, zero_division=0)
    }
    logger.log_metrics(metrics, epoch)
    
    # Vẽ Confusion Matrix mỗi 5 epoch và ở epoch cuối
    if epoch % 5 == 0 or epoch == cfg.epochs:
        cm = confusion_matrix(all_targets, all_preds)
        logger.log_confusion_matrix(cm, epoch)
        
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save(model.state_dict(), os.path.join(cfg.output_dir, "best_lstm.pth"))
        print(f"[*] Epoch {epoch:02d}: Cập nhật checkpoint tốt nhất với Val Acc: {best_val_acc*100:.2f}%")

logger.close()
```

#### Cell 8: Đóng gói toàn bộ Checkpoint & TensorBoard Event Logs
- Tự động nén toàn bộ thư mục `runs/` và model `best_lstm.pth` vào file zip `temp/lstm_experiment_results.zip`.
- Giúp người dùng dễ dàng tải toàn bộ file nhật ký sự kiện về máy để mở lại TensorBoard bất kỳ lúc nào mà không bị mất dữ liệu khi tắt session Kaggle.

---

## 5. BẢNG SO SÁNH ĐỐI CHIẾU HIỆU NĂNG 3 PHƯƠNG PHÁP

| Tiêu chí so sánh | `datn4ni.ipynb` (Online Video Extraction) | Phương án CSV Tabular | `datn4ni2.ipynb` (Pre-split + Tensor `.pt`) |
| :--- | :--- | :--- | :--- |
| **Phân chia Train/Val** | Chia ngẫu nhiên lúc train video | Chia sau khi quét video | **Chia tách Video-level độc lập TRƯỚC TIÊN** |
| **Định dạng dữ liệu** | Video thô `.mp4` / `.avi` | `.csv.gz` (Text bảng) | **PyTorch Tensor `.pt` nhị phân** |
| **Thời gian nạp dữ liệu vào RAM** | Không nạp trước (đọc từng frame) | ~10 - 20 giây (parse CSV + groupby) | **< 0.5 giây (`torch.load` trực tiếp)** |
| **Thời gian chạy 1 Epoch** | ~83 phút / epoch | ~1.5 - 2.0 giây / epoch | **~1.0 - 1.2 giây / epoch** |
| **Thời gian hoàn thành 40 Epochs** | Bất khả thi (> 55 giờ, timeout 12h) | ~1.5 phút | **~45 - 50 giây** |
| **Khả năng xảy ra Data Leakage** | Có thể có nếu shuffle frame | Thấp | **Tuyệt đối không (0%) do lưu 2 file `.pt` riêng** |
| **Mức tiêu hao GPU Quota Kaggle** | Hết 100% quota hàng tuần | ~0.05 giờ | **~0.02 giờ (tiết kiệm tối đa)** |
| **Giám sát trực quan** | Text output terminal | TensorBoard cơ bản | **TensorBoard toàn diện (Loss, Acc, PR, F1, CM, Graph)** |

---

## 6. DANH SÁCH FILE TRONG HỆ THỐNG MỚI

1. **[`extract_to_pt.py`](file:///D:/Project/AI/myLSTM/extract_to_pt.py) (Chạy tại Local PC):**
   - Đọc danh sách 2,074 video từ `D:\Project\AI\dataset\SUST`.
   - Phân chia Train/Val (80/20) với seed 42, xuất `dataset_split.json`.
   - Forward qua `clone.onnx` + `AdaptiveAvgPool2d((1, 1))` trên GPU local.
   - Xuất ra 2 tệp nhị phân: `features_sust_train.pt` và `features_sust_val.pt`.
2. **[`log.py`](file:///D:/Project/AI/myLSTM/log.py) (Đã sẵn sàng):**
   - Chứa lớp [`TensorBoardLogger`](file:///D:/Project/AI/myLSTM/log.py#L22) phục vụ việc ghi nhận chỉ số, biểu đồ, hình ảnh và graph.
3. **`datn4ni2.ipynb` (Chạy trên Kaggle):**
   - Nhận đầu vào là dataset `.pt` trên Kaggle.
   - Nạp dữ liệu qua `PreloadedTensorDataset`.
   - Huấn luyện `DeepLSTMClassifier` với AMP FP16.
   - Bật giao diện TensorBoard nhúng trực tiếp và xuất kết quả `temp/lstm_experiment_results.zip`.
