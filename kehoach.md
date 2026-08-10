# KẾ HOẠCH CÂN BẰNG THAM SỐ CẤU HÌNH KIẾN TRÚC MÔ HÌNH (MODEL ARCHITECTURE CONFIGURATION)

> **Tài liệu tham chiếu:** `myCNN/src/backbone_neck.py`, `myCNN/checkpoints_ftCOCO/model_mainfest.json`, `config.py`, `model.py`, `kientruc.md`.  
> **Mục tiêu:** Phân tích đặc tính thực tế các tầng đầu ra Neck của mạng `NMSFreeDetector` ($p_3, p_4, p_5$) nhằm cân bằng và chuẩn hóa toàn bộ tham số trong phân vùng `MODEL ARCHITECTURE CONFIGURATION` trước khi áp dụng vào source code.

---

## 1. Phân Tích Thực Tế Đầu Ra Của CNN Neck (`PAFPN`)

Theo cấu hình manifest (`model_mainfest.json`) và triển khai trong `myCNN/src/backbone_neck.py`:
* **Ảnh đầu vào tiêu chuẩn:** $H \times W = 480 \times 480$, 3 kênh màu (RGB).
* **Backbone (`Backbone`):** `backbone_w = (56, 112, 224, 448, 640)`, `strides = (8, 16, 32)`.
* **Neck (`PAFPN`):** Nhận 3 mức đặc trưng với kênh `neck_chs = (224, 448, 640)` và tiến hành liên kết đa thang đo (Top-down FPN + Bottom-up PAN).

### Bảng thông số chi tiết 3 tầng đặc trưng đầu ra của Neck:

| Tầng đặc trưng | Stride | Kích thước không gian ($H \times W$) | Số kênh ($C$) | Ngữ nghĩa đặc trưng trích xuất |
| :--- | :---: | :---: | :---: | :--- |
| **$p_3$ (Low-level)** | 8 | $60 \times 60$ | **224** | Chi tiết cạnh, vân bề mặt, mi mắt, chớp mắt, khóe miệng |
| **$p_4$ (Mid-level)** | 16 | $30 \times 30$ | **448** | Bộ phận khuôn mặt: vùng mắt, mũi, miệng mở (ngáp), cử động đầu |
| **$p_5$ (High-level)** | 32 | $15 \times 15$ | **640** | Ngữ nghĩa toàn cục: tư thế đầu gục, trạng thái buồn ngủ / tỉnh táo |

* **Tổng số kênh khi kết hợp 3 tầng ($p_3 + p_4 + p_5$):** $224 + 448 + 640 = \mathbf{1312}$ kênh.
* *(Trường hợp đồng nhất số kênh bằng phép chiếu/stack trực tiếp: $3 \times 640 = \mathbf{1920}$ kênh).*

---

## 2. Đánh Giá Điểm Bất Cập Của Cấu Hình Hiện Tại

Cấu hình cũ trong `config.py` được thiết kế khi chỉ nhận duy nhất tầng $p_5$:
```python
# Cấu hình cũ trong config.py:
cnn_out_channels: int = 640
cnn_spatial_size: Tuple[int, int] = (15, 15)
input_dim: int = 256
hidden_dim: int = 256
num_layers: int = 3
num_classes: int = 2
dropout: float = 0.0
```

### Các hạn chế:
1. **Thiếu thông tin đa thang đo:** Cấu hình cũ chỉ khai báo `cnn_out_channels = 640`, bỏ qua hoàn toàn 224 kênh của $p_3$ và 448 kênh của $p_4$.
2. **Cố định không gian đơn lẻ:** Chỉ lưu `cnn_spatial_size = (15, 15)` thay vì cấu hình động theo cả 3 độ phân giải $(60, 60)$, $(30, 30)$, $(15, 15)$.
3. **Thiếu tham số điều khiển Fusion:** Chưa có tham số cấu hình loại liên kết (`fusion = "concat" | "conv" | "attention" | "sum" | "mean"`), số lượng tầng `num_features = 3`, và dropout cho adapter.
4. **Tỷ lệ nén và cân bằng kích thước:** Tỷ lệ nén từ $1312 \to 256$ ($\approx 5.125 \times$) là tỷ lệ vàng giúp giảm chiều dữ liệu đưa vào LSTM mà không gây nghẽn cổ chai thông tin (information bottleneck).

---

## 3. Bảng Cân Bằng Tham Số `MODEL ARCHITECTURE CONFIGURATION` Mới

Đề xuất cập nhật phân vùng **3. MODEL ARCHITECTURE CONFIGURATION** trong `TrainConfig` như sau:

```python
    # ---- 3. MODEL ARCHITECTURE CONFIGURATION ----
    cnn_manifest_path: str = r"D:\Project\AI\myLSTM\myCNN\checkpoints_ftCOCO\model_mainfest.json"
    cnn_weights_path: str = r"D:\Project\AI\myLSTM\myCNN\checkpoints_ftCOCO\ft_step00091000.pt"
    
    # 3.1 Cấu hình kênh và không gian của CNN Neck đầu ra
    cnn_neck_channels: Tuple[int, int, int] = (224, 448, 640)     # Kênh thực tế của (p3, p4, p5) từ PAFPN
    cnn_strides: Tuple[int, int, int] = (8, 16, 32)                # Strides tương ứng của (p3, p4, p5)
    cnn_num_features: int = 3                                      # Số lượng tầng đặc trưng đầu vào (p3, p4, p5)
    cnn_out_channels: int = 1312                                   # Tổng số kênh khi ghép nối (224 + 448 + 640 = 1312)
    cnn_spatial_size: Tuple[int, int] = (15, 15)                   # Kích thước không gian tầng sâu nhất p5
    
    # 3.2 Cấu hình Spatial Feature Adapter (Bộ chuyển đổi / nén đặc trưng không gian)
    spatial_fusion: str = "concat"                                 # Phương thức kết hợp: 'concat' | 'conv' | 'attention' | 'sum' | 'mean'
    adapter_dropout: float = 0.1                                   # Tỷ lệ Dropout sau Spatial Adapter (chống overfitting)
    input_dim: int = 256                                           # Kích thước vector đặc trưng x^t sau Adapter đưa vào LSTM
    
    # 3.3 Cấu hình Deep LSTM Classifier
    hidden_dim: int = 256                                          # Số lượng đơn vị ẩn (hidden units) trong từng khối LSTM
    num_layers: int = 3                                            # Số lớp LSTM xếp chồng (Deep LSTM - 3 lớp)
    dropout: float = 0.2                                           # Tỷ lệ Dropout giữa các lớp LSTM (khi num_layers > 1)
    num_classes: int = 2                                           # Số lượng lớp đầu ra phân loại (2: Tỉnh táo vs Buồn ngủ)
```

### So sánh Cấu Hình Cũ vs Cấu Hình Mới Cân Bằng:

| Tham số | Giá trị Cũ | Giá trị Mới Cân Bằng | Lý do và Lợi ích |
| :--- | :---: | :---: | :--- |
| `cnn_neck_channels` | *(Chưa có)* | `(224, 448, 640)` | Khớp chính xác với cấu hình Neck của `NMSFreeDetector` |
| `cnn_strides` | *(Chưa có)* | `(8, 16, 32)` | Xác định tỉ lệ downsample cho từng tầng |
| `cnn_num_features` | *(Chưa có)* | `3` | Xác định rõ số lượng tầng đặc trưng nhận vào ($p_3, p_4, p_5$) |
| `cnn_out_channels` | `640` | `1312` | Phản ánh đúng tổng số kênh $(224 + 448 + 640)$ |
| `spatial_fusion` | *(Chưa có)* | `"concat"` | Hỗ trợ chuyển đổi linh hoạt các phương pháp kết hợp đặc trưng |
| `adapter_dropout` | *(Chưa có)* | `0.1` | Regularization ngay sau tầng nén không gian |
| `input_dim` | `256` | `256` | Giữ chuẩn $256$ chiều tối ưu cho Deep LSTM |
| `hidden_dim` | `256` | `256` | Đồng bộ $256$ đơn vị ẩn theo tài liệu `kientruc.md` |
| `num_layers` | `3` | `3` | Giữ cấu trúc 3 tầng LSTM sâu |
| `dropout` | `0.0` | `0.2` | Tăng cường tính khái quát hóa, chống học vẹt chuỗi thời gian |
| `num_classes` | `2` | `2` | Nhị phân: Tỉnh táo (0) vs Buồn ngủ (1) |

---

## 4. Phân Tích Dung Lượng Bộ Nhớ & Tham Số Tính Toán (Parameter Budget)

| Thành phần mô hình | Công thức tính tham số | Số lượng tham số | Dung lượng bộ nhớ |
| :--- | :--- | :---: | :---: |
| **Spatial Adapter (Projection)** | $1312 \times 256 + 256$ | **336,128** ($\approx 0.34\text{ M}$) | $\approx 1.35\text{ MB}$ |
| **LSTM Layer 1** | $4 \times [(256 + 256) \times 256 + 256]$ | **525,312** ($\approx 0.53\text{ M}$) | $\approx 2.10\text{ MB}$ |
| **LSTM Layer 2** | $4 \times [(256 + 256) \times 256 + 256]$ | **525,312** ($\approx 0.53\text{ M}$) | $\approx 2.10\text{ MB}$ |
| **LSTM Layer 3** | $4 \times [(256 + 256) \times 256 + 256]$ | **525,312** ($\approx 0.53\text{ M}$) | $\approx 2.10\text{ MB}$ |
| **FC Output ($W^R, b^R$)** | $256 \times 2 + 2$ | **514** | $\approx 2.05\text{ KB}$ |
| **TỔNG CỘNG** | | **1,912,578** ($\approx \mathbf{1.91\text{ M}}$) | $\approx \mathbf{7.65\text{ MB}}$ |

* **Đánh giá hiệu năng:**
  - Nhánh thời gian chỉ tiêu tốn $\approx 1.91\text{M}$ tham số và $\approx 7.65\text{ MB}$ VRAM.
  - Thời gian suy luận cực nhanh ($< 2\text{ ms}$ cho chuỗi 60 frames trên GPU).
  - Hoàn toàn đảm bảo khả năng chạy thời gian thực (Real-time Inference) $> 100\text{ FPS}$.

---

## 5. Kế Hoạch Triển Khai Vào Source Code

Sau khi hoàn thiện và rà soát file `kehoach.md`, các bước cập nhật source code sẽ được thực hiện theo thứ tự:

1. **Bước 1 — Cập nhật `config.py`:**
   - Thêm các thuộc tính mới vào `TrainConfig`: `cnn_neck_channels`, `cnn_strides`, `cnn_num_features`, `spatial_fusion`, `adapter_dropout`.
   - Cập nhật hàm `load_json()` để chuyển đổi tuple cho `cnn_neck_channels` và `cnn_strides`.
   - Thêm assertion kiểm tra tính hợp lệ của các tham số mới trong `__post_init__()`.

2. **Bước 2 — Cập nhật `model.py`:**
   - Điều chỉnh `SpatialFeatureAdapter` nhận `in_channels` linh hoạt: có thể là số nguyên (1312 hoặc 640 hoặc 1920) hoặc Tuple kênh `(224, 448, 640)`.
   - Khi nhận đầu vào $p_3, p_4, p_5$ với số kênh khác nhau $(224, 448, 640)$, adapter tự động pool không gian riêng biệt từng tầng rồi ghép thành vector 1312 chiều, sau đó chiếu về 256 chiều.
   - Hỗ trợ đầy đủ cả dạng tensor 6D $[bz, seqlen, 3, 640, 15, 15]$ và dạng tuple `(p3, p4, p5)` với các kích thước tự nhiên.

3. **Bước 3 — Cập nhật `dataloader.py`:**
   - Đảm bảo `CNNCollateFn` trích xuất đồng bộ $p_3, p_4, p_5$ tương thích với cấu hình mới.

4. **Bước 4 — Chạy kiểm thử toàn diện:**
   - Chạy test độc lập `python config.py`, `python model.py`, và kiểm tra toàn bộ pipeline dữ liệu.
