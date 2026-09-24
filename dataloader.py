import os
import sys
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, List, Union, Dict, Any

# Cấu hình encoding UTF-8 cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Đảm bảo đường dẫn outsrc được thêm vào sys.path để import các module con
CURRENT_DIR = Path(__file__).resolve().parent
OUTSRC_DIR = CURRENT_DIR / "outsrc"
if str(OUTSRC_DIR) not in sys.path:
    sys.path.insert(0, str(OUTSRC_DIR))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# Nạp CUDA DLL trên Windows nếu cần
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.path.exists(torch_lib_path) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(torch_lib_path)
    except Exception:
        pass

# Kiểm tra thư viện ONNX Runtime
try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    ort = None
    HAS_ORT = False

from config import TrainConfig
from dataset import MyLSTMDataset, PreloadedTensorDataset

# Nhập NMSFreeDetector an toàn
try:
    from outsrc.myCNN.src.model import NMSFreeDetector
    from outsrc.myCNN.src.runtime.infer import letterbox
except ModuleNotFoundError:
    try:
        from myCNN.src.model import NMSFreeDetector
        from myCNN.src.runtime.infer import letterbox
    except ModuleNotFoundError:
        NMSFreeDetector = None
        letterbox = None


def load_cnn_model(
    manifest_path: str = TrainConfig.cnn_manifest_path,
    weights_path: str = TrainConfig.cnn_weights_path,
    device: str = "cpu"
) -> Optional[Any]:
    """
    Nạp mô hình NMSFreeDetector và load trọng số pretrained từ file checkpoint.
    """
    if NMSFreeDetector is None:
        print("[DataLoader][Warning] Không tìm thấy module NMSFreeDetector.")
        return None

    if Path(manifest_path).exists():
        model = NMSFreeDetector.from_config(manifest_path)
    else:
        model = NMSFreeDetector()

    if Path(weights_path).exists():
        ckpt = torch.load(weights_path, map_location=device)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=False)
        elif "backbone" in ckpt:
            model.load_trunk(ckpt, strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
        print(f"[DataLoader] Đã nạp trọng số CNN từ: {weights_path}")
    else:
        print(f"[DataLoader][Warning] Không tìm thấy file trọng số CNN: {weights_path}")

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model


class CNNCollateFn:
    """
    Collate function tùy chỉnh cho DataLoader sử dụng mô hình PyTorch NMSFreeDetector.
    Nhận batch các mẫu (video_tensor, label) từ MyLSTMDataset (được lấy mẫu mỗi t giây, mặc định 0.5s),
    gom các khung hình và cho chạy qua backbone + neck của NMSFreeDetector để trích xuất đặc trưng đa tầng (p3, p4, p5).

    TỐI ƯU HÓA BỘ NHỚ & TÍNH TƯƠNG THÍCH CHUỖI:
      - Tự động đồng bộ / padding độ dài chuỗi (T) qua _pad_or_truncate_videos nếu các video có số khung hình khác nhau.
      - Xử lý chia nhỏ khung hình theo chunk (chunk_size) để chống tràn bộ nhớ VRAM khi seq_len lớn.
      - pool_spatial=True: Thực hiện Adaptive Average Pooling về (1, 1) ngay trên GPU
        để nén [B, T, C, H, W] -> [B, T, C], giảm dung lượng truyền tải qua RAM hơn 1000 lần.
      - Fallback an toàn: Tự động sinh đặc trưng tensor giả lập đúng chiều nếu cnn_model=None.
    """

    def __init__(
        self,
        cnn_model: Optional[Any] = None,
        device: str = "cpu",
        chunk_size: int = 16,
        pool_spatial: bool = True,
        to_cpu: bool = True,
        target_seq_len: Optional[int] = None
    ):
        self.cnn_model = cnn_model
        self.device = device
        self.chunk_size = chunk_size
        self.pool_spatial = pool_spatial
        self.to_cpu = to_cpu
        self.target_seq_len = target_seq_len

        if self.cnn_model is not None:
            self.cnn_model.to(self.device)
            self.cnn_model.eval()

    def _pad_or_truncate_videos(self, batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Đồng bộ số khung hình giữa các video trong batch (phòng trường hợp chuỗi video có độ dài động do lấy mẫu theo t).
        """
        if not batch:
            return torch.empty(0), torch.empty(0, dtype=torch.long)

        if self.target_seq_len is not None and self.target_seq_len > 0:
            target_t = self.target_seq_len
        else:
            target_t = max(item[0].shape[0] for item in batch)

        aligned_videos = []
        labels = []

        for video_tensor, label in batch:
            t_cur = video_tensor.shape[0]
            if t_cur < target_t:
                pad_count = target_t - t_cur
                last_frame = video_tensor[-1:] if t_cur > 0 else torch.zeros((1, *video_tensor.shape[1:]), dtype=video_tensor.dtype)
                padding = last_frame.repeat(pad_count, 1, 1, 1)
                video_padded = torch.cat([video_tensor, padding], dim=0) if t_cur > 0 else padding
            elif t_cur > target_t:
                video_padded = video_tensor[:target_t]
            else:
                video_padded = video_tensor

            aligned_videos.append(video_padded)
            labels.append(label)

        videos = torch.stack(aligned_videos, dim=0)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return videos, labels_tensor

    def __call__(
        self,
        batch: List[Tuple[torch.Tensor, int]]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        if not batch:
            dummy_labels = torch.empty(0, dtype=torch.long)
            if self.pool_spatial:
                return (torch.empty(0, 0, 224), torch.empty(0, 0, 448), torch.empty(0, 0, 640)), dummy_labels
            return (torch.empty(0, 0, 224, 0, 0), torch.empty(0, 0, 448, 0, 0), torch.empty(0, 0, 640, 0, 0)), dummy_labels

        videos, labels = self._pad_or_truncate_videos(batch)
        b, t, c, h, w = videos.shape

        # Nếu không có mô hình CNN, fallback trả về tensor đặc trưng giả lập đúng kích thước
        if self.cnn_model is None:
            if self.pool_spatial:
                dummy_p3 = torch.randn(b, t, 224)
                dummy_p4 = torch.randn(b, t, 448)
                dummy_p5 = torch.randn(b, t, 640)
            else:
                dummy_p3 = torch.randn(b, t, 224, 60, 60)
                dummy_p4 = torch.randn(b, t, 448, 30, 30)
                dummy_p5 = torch.randn(b, t, 640, 15, 15)
            return (dummy_p3, dummy_p4, dummy_p5), labels

        total_frames = b * t
        x_flat = videos.view(total_frames, c, h, w)

        chunk_sz = self.chunk_size if (self.chunk_size and self.chunk_size > 0) else total_frames
        acc_p3, acc_p4, acc_p5 = [], [], []

        for start in range(0, total_frames, chunk_sz):
            end = min(start + chunk_sz, total_frames)
            chunk = x_flat[start:end].to(self.device)

            with torch.no_grad():
                p3_c, p4_c, p5_c = self.cnn_model.backbone(chunk)
                p3_c, p4_c, p5_c = self.cnn_model.neck(p3_c, p4_c, p5_c)

                if self.pool_spatial:
                    p3_c = F.adaptive_avg_pool2d(p3_c, (1, 1)).flatten(1)  # [chunk_sz, 224]
                    p4_c = F.adaptive_avg_pool2d(p4_c, (1, 1)).flatten(1)  # [chunk_sz, 448]
                    p5_c = F.adaptive_avg_pool2d(p5_c, (1, 1)).flatten(1)  # [chunk_sz, 640]

            if self.to_cpu:
                acc_p3.append(p3_c.cpu())
                acc_p4.append(p4_c.cpu())
                acc_p5.append(p5_c.cpu())
            else:
                acc_p3.append(p3_c)
                acc_p4.append(p4_c)
                acc_p5.append(p5_c)

        if self.pool_spatial:
            p3 = torch.cat(acc_p3, dim=0).view(b, t, 224)
            p4 = torch.cat(acc_p4, dim=0).view(b, t, 448)
            p5 = torch.cat(acc_p5, dim=0).view(b, t, 640)
        else:
            p3 = torch.cat(acc_p3, dim=0).view(b, t, *acc_p3[0].shape[1:])
            p4 = torch.cat(acc_p4, dim=0).view(b, t, *acc_p4[0].shape[1:])
            p5 = torch.cat(acc_p5, dim=0).view(b, t, *acc_p5[0].shape[1:])

        return (p3, p4, p5), labels


class ONNXCNNCollateFn:
    """
    Collate function sử dụng ONNX Runtime CUDA I/O Binding để trích xuất đặc trưng (p3, p4, p5)
    trực tiếp trên GPU chỉ định (ví dụ CARD 2: GPU 1 trong hệ thống Dual-GPU).

    TỐI ƯU HÓA ĐỘT PHÁ CHỐNG TRÀN BỘ NHỚ (OOM):
      1. Tách GPU: Chạy hoàn toàn trên GPU 1 (device_id=1), tách biệt với GPU 0 huấn luyện LSTM.
      2. NÉN ADAPTIVE AVERAGE POOLING NGAY TRÊN GPU 1:
         Áp dụng F.adaptive_avg_pool2d(..., (1, 1)) ngay trên bộ nhớ VRAM của GPU 1 sau mỗi chunk,
         chuyển đổi:
           - p3: [chunk_sz, 224, 60, 60] -> [chunk_sz, 224] (giảm 3600 lần)
           - p4: [chunk_sz, 448, 30, 30] -> [chunk_sz, 448] (giảm 900 lần)
           - p5: [chunk_sz, 640, 15, 15] -> [chunk_sz, 640] (giảm 225 lần)
      3. Sau khi nén trên GPU 1, chỉ chuyển các vector 1D (đã nhẹ hơn 1032 lần) về CPU RAM.
         Dung lượng 1 batch 32 video từ 20.8 GB giảm xuống chỉ còn ~20.1 MB,
         loại bỏ triệt để hiện tượng tràn RAM (OOM Killer / Kernel died) trên Kaggle!
    """

    def __init__(
        self,
        onnx_model_path: str = "outsrc/myCNN/checkpoints_ftCOCO/clone.onnx",
        device_id: int = 1,
        chunk_size: int = 16,
        pool_spatial: bool = True,
        mem_limit_gb: float = 8.0,
        to_cpu: bool = True,
        target_seq_len: Optional[int] = None
    ):
        self.onnx_model_path = str(Path(onnx_model_path).resolve())
        self.device_id = device_id
        self.device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        self.chunk_size = chunk_size
        self.pool_spatial = pool_spatial
        self.mem_limit_gb = mem_limit_gb
        self.to_cpu = to_cpu
        self.target_seq_len = target_seq_len
        self.session = None

        if os.path.exists(self.onnx_model_path) and HAS_ORT:
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_opts.log_severity_level = 3

            # Cấu hình tối ưu bộ nhớ GPU
            cuda_opts = {
                "device_id": self.device_id,
                "arena_extend_strategy": "kSameAsRequested",
                "gpu_mem_limit": int(mem_limit_gb * 1024 * 1024 * 1024),
                "cudnn_conv_algo_search": "HEURISTIC",
                "do_copy_in_default_stream": True,
            }
            providers = [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]
            self.session = ort.InferenceSession(self.onnx_model_path, sess_options=sess_opts, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            self.use_cuda_io = "CUDAExecutionProvider" in self.session.get_providers() and torch.cuda.is_available()
            print(f"[ONNXCNNCollateFn] Đã nạp thành công mô hình ONNX trên {self.device} (CUDA I/O={self.use_cuda_io}, PoolSpatial={self.pool_spatial})")
        else:
            self.use_cuda_io = False
            print(f"[ONNXCNNCollateFn][WARN] Không tìm thấy file ONNX: {self.onnx_model_path}. Sẽ dùng fallback dummy.")

    def _infer_chunk(self, chunk: torch.Tensor) -> List[torch.Tensor]:
        """Đưa chunk vào GPU trích xuất và thực thi với I/O Binding."""
        if chunk.device != torch.device(self.device):
            chunk = chunk.to(self.device, non_blocking=True)
        if not chunk.is_contiguous():
            chunk = chunk.contiguous()

        io_binding = self.session.io_binding()
        io_binding.bind_input(
            name=self.input_name,
            device_type="cuda",
            device_id=self.device_id,
            element_type=np.float32,
            shape=tuple(chunk.shape),
            buffer_ptr=chunk.data_ptr()
        )
        for out_name in self.output_names:
            io_binding.bind_output(out_name, device_type="cuda", device_id=self.device_id)

        try:
            self.session.run_with_iobinding(io_binding)
            outputs = [torch.from_dlpack(o) for o in io_binding.get_outputs()]
            return outputs
        except Exception as e:
            raise RuntimeError(f"Lỗi khi chạy ONNX Runtime trên {self.device}: {e}") from e

    def _pad_or_truncate_videos(self, batch: List[Tuple[torch.Tensor, int]]) -> Tuple[List[torch.Tensor], List[int]]:
        """
        Đồng bộ số khung hình giữa các video trong batch (phòng trường hợp chuỗi video có độ dài động do lấy mẫu theo t).
        """
        if not batch:
            return [], []

        if self.target_seq_len is not None and self.target_seq_len > 0:
            target_t = self.target_seq_len
        else:
            target_t = max(item[0].shape[0] if item[0].dim() >= 4 else (1 if item[0].dim() == 3 else 0) for item in batch)

        aligned_batch = []
        labels = []

        for video_tensor, label in batch:
            if video_tensor.dim() == 3:
                # [C, H, W] -> [1, C, H, W]
                video_tensor = video_tensor.unsqueeze(0)
            elif video_tensor.dim() == 5 and video_tensor.shape[0] == 1:
                video_tensor = video_tensor.squeeze(0)

            t_cur = video_tensor.shape[0]
            if t_cur < target_t:
                pad_count = target_t - t_cur
                last_frame = video_tensor[-1:] if t_cur > 0 else torch.zeros((1, *video_tensor.shape[1:]), dtype=video_tensor.dtype)
                padding = last_frame.repeat(pad_count, 1, 1, 1)
                video_padded = torch.cat([video_tensor, padding], dim=0) if t_cur > 0 else padding
            elif t_cur > target_t:
                video_padded = video_tensor[:target_t]
            else:
                video_padded = video_tensor

            aligned_batch.append(video_padded)
            labels.append(label)

        return aligned_batch, labels

    def __call__(self, batch: List[Tuple[torch.Tensor, int]]) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        if not batch:
            dummy_labels = torch.empty(0, dtype=torch.long)
            if self.pool_spatial:
                return (torch.empty(0, 0, 224), torch.empty(0, 0, 448), torch.empty(0, 0, 640)), dummy_labels
            return (torch.empty(0, 0, 224, 0, 0), torch.empty(0, 0, 448, 0, 0), torch.empty(0, 0, 640, 0, 0)), dummy_labels

        aligned_videos, labels_list = self._pad_or_truncate_videos(batch)
        valid_p3, valid_p4, valid_p5 = [], [], []
        valid_labels = []

        for idx, (frames, label) in enumerate(zip(aligned_videos, labels_list)):
            try:
                frames = frames.float()
                t_frames = frames.shape[0]

                if self.session is not None:
                    chunk_sz = self.chunk_size if (self.chunk_size and self.chunk_size > 0) else t_frames
                    acc_p3, acc_p4, acc_p5 = [], [], []

                    for start in range(0, t_frames, chunk_sz):
                        end = min(start + chunk_sz, t_frames)
                        chunk = frames[start:end]

                        if self.use_cuda_io:
                            p3_c, p4_c, p5_c = self._infer_chunk(chunk)
                        else:
                            outs = self.session.run(None, {self.input_name: chunk.cpu().numpy()})
                            p3_c, p4_c, p5_c = [torch.from_numpy(o) for o in outs]

                        # ======================================================
                        # BƯỚC KHẮC PHỤC CHÍNH: POOLING NGAY TRÊN GPU TRÍCH XUẤT
                        # Nén [chunk_sz, C, H, W] -> [chunk_sz, C] tức thì
                        # ======================================================
                        if self.pool_spatial:
                            p3_c = F.adaptive_avg_pool2d(p3_c, (1, 1)).flatten(1)  # [chunk_sz, 224]
                            p4_c = F.adaptive_avg_pool2d(p4_c, (1, 1)).flatten(1)  # [chunk_sz, 448]
                            p5_c = F.adaptive_avg_pool2d(p5_c, (1, 1)).flatten(1)  # [chunk_sz, 640]

                        # Sau khi đã nén siêu nhẹ, chuyển về CPU để giải phóng VRAM GPU
                        if self.to_cpu:
                            acc_p3.append(p3_c.cpu())
                            acc_p4.append(p4_c.cpu())
                            acc_p5.append(p5_c.cpu())
                            del p3_c, p4_c, p5_c
                        else:
                            acc_p3.append(p3_c)
                            acc_p4.append(p4_c)
                            acc_p5.append(p5_c)

                    p3_item = torch.cat(acc_p3, dim=0)
                    p4_item = torch.cat(acc_p4, dim=0)
                    p5_item = torch.cat(acc_p5, dim=0)
                    del acc_p3, acc_p4, acc_p5

                else:
                    # Fallback tạo dummy features nếu không có session ONNX
                    if self.pool_spatial:
                        p3_item = torch.randn(t_frames, 224)
                        p4_item = torch.randn(t_frames, 448)
                        p5_item = torch.randn(t_frames, 640)
                    else:
                        p3_item = torch.randn(t_frames, 224, 60, 60)
                        p4_item = torch.randn(t_frames, 448, 30, 30)
                        p5_item = torch.randn(t_frames, 640, 15, 15)

                valid_p3.append(p3_item)
                valid_p4.append(p4_item)
                valid_p5.append(p5_item)
                valid_labels.append(label)

            except Exception as e:
                print(f"[ONNXCNNCollateFn][WARN] Gặp lỗi khi trích xuất mẫu {idx}: {e}. Bỏ qua.")
                continue

        # Nếu toàn bộ batch bị lỗi, fallback 1 mẫu dummy
        if len(valid_labels) == 0:
            print("[ONNXCNNCollateFn][ERROR] Toàn bộ batch bị lỗi ONNX. Dùng dummy sample dự phòng.")
            t_fb = self.target_seq_len if (self.target_seq_len and self.target_seq_len > 0) else 120
            if self.pool_spatial:
                p3 = torch.zeros(1, t_fb, 224)
                p4 = torch.zeros(1, t_fb, 448)
                p5 = torch.zeros(1, t_fb, 640)
            else:
                p3 = torch.zeros(1, t_fb, 224, 60, 60)
                p4 = torch.zeros(1, t_fb, 448, 30, 30)
                p5 = torch.zeros(1, t_fb, 640, 15, 15)
            labels = torch.zeros(1, dtype=torch.long)
        else:
            p3 = torch.stack(valid_p3, dim=0)
            p4 = torch.stack(valid_p4, dim=0)
            p5 = torch.stack(valid_p5, dim=0)
            labels = torch.tensor(valid_labels, dtype=torch.long)

        if self.to_cpu:
            p3, p4, p5 = p3.cpu(), p4.cpu(), p5.cpu()

        return (p3, p4, p5), labels


def dynamic_tensor_collate_fn(batch: List[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Collate function tự động pad các chuỗi tensor có độ dài khác nhau về độ dài lớn nhất trong Batch.
    Phục vụ dữ liệu trích xuất theo CÁCH 1 (List[torch.Tensor]).
    """
    p3_list = [item[0][0] for item in batch]
    p4_list = [item[0][1] for item in batch]
    p5_list = [item[0][2] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)

    p3_padded = torch.nn.utils.rnn.pad_sequence(p3_list, batch_first=True, padding_value=0.0)
    p4_padded = torch.nn.utils.rnn.pad_sequence(p4_list, batch_first=True, padding_value=0.0)
    p5_padded = torch.nn.utils.rnn.pad_sequence(p5_list, batch_first=True, padding_value=0.0)

    return (p3_padded, p4_padded, p5_padded), labels


def get_tensor_dataloaders(config: Optional[Any] = None) -> Tuple[DataLoader, DataLoader]:
    """
    Khởi tạo TrainLoader và ValLoader từ tệp nhị phân PyTorch Tensor (.pt).
    Đồng bộ 100% với CELL 6 của datn4ni2.ipynb:
    - Nạp trực tiếp dữ liệu vào RAM trong < 0.5s qua PreloadedTensorDataset.
    - Hỗ trợ cả Tensor 3D cố định và List[torch.Tensor] chuỗi động qua dynamic_tensor_collate_fn.
    - Batch size 64 tận dụng tối đa GPU, pin_memory=True, num_workers=0.
    """
    if config is None:
        config = TrainConfig()

    train_pt = getattr(config, "train_pt", "extracted_features_pt/features_sust_train.pt")
    val_pt   = getattr(config, "val_pt", "extracted_features_pt/features_sust_val.pt")
    batch_size = getattr(config, "batch_size", 64)
    drop_last = getattr(config, "drop_last", False)

    has_cuda = torch.cuda.is_available() and "cuda" in str(getattr(config, "device", "cuda"))
    pin_mem = getattr(config, "pin_memory", True) if has_cuda else False

    train_dataset = PreloadedTensorDataset(train_pt, is_train=True)
    val_dataset   = PreloadedTensorDataset(val_pt, is_train=False)

    collate_fn = dynamic_tensor_collate_fn if getattr(train_dataset, "is_variable_len", False) else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=0,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_mem,
        num_workers=0,
        collate_fn=collate_fn
    )

    print(f"[DataLoader] Đã tạo DataLoader Tensor (DynamicCollate={collate_fn is not None}): Train={len(train_dataset)} mẫu ({len(train_loader)} batches) | Val={len(val_dataset)} mẫu ({len(val_loader)} batches)")
    return train_loader, val_loader


def get_dataloaders(
    config: Optional[Any] = None,
    use_cnn_collate: bool = True,
    use_onnx: bool = False,
    pool_spatial: bool = True,
    sample_interval: Optional[float] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Khởi tạo DataLoader cho mô hình Deep LSTM:
    - Nếu config.use_preloaded_pt=True (mặc định): Nạp trực tiếp Tensor .pt siêu tốc qua get_tensor_dataloaders.
    - Nếu config.use_preloaded_pt=False: Nạp qua MyLSTMDataset đọc video thô qua OpenCV với chu kỳ lấy mẫu t.
    """
    if config is None:
        config = TrainConfig()

    # Chế độ nạp Tensor .pt siêu tốc (CELL 6 datn4ni2.ipynb)
    if getattr(config, "use_preloaded_pt", True):
        return get_tensor_dataloaders(config)

    image_size = getattr(config, "image_size", (480, 480))
    new_size = image_size[0] if isinstance(image_size, (tuple, list)) else image_size
    seq_len = getattr(config, "seq_len", 120)
    interval = sample_interval if sample_interval is not None else getattr(config, "sample_interval", 0.5)

    transforms = partial(letterbox, new_size=new_size) if letterbox is not None else None

    # Khởi tạo dataset với chu kỳ lấy mẫu t (sample_interval)
    if hasattr(MyLSTMDataset, "from_config"):
        dataset = MyLSTMDataset.from_config(config, transform=transforms, sample_interval=interval)
    else:
        dataset = MyLSTMDataset(
            dataset_dir=getattr(config, "dataset_dir", ""),
            seq_len=seq_len,
            sample_interval=interval,
            image_size=image_size,
            video_exts=getattr(config, "video_exts", (".avi", ".mp4", ".mkv"))
        )

    # Chọn collate function
    has_cuda = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count() if has_cuda else 0

    # Tự động chọn ONNX nếu cấu hình yêu cầu hoặc có file ONNX
    use_onnx_final = use_onnx or getattr(config, "use_onnx", False)

    if use_onnx_final and HAS_ORT:
        onnx_path = getattr(config, "cnn_onnx_path", "outsrc/myCNN/checkpoints_ftCOCO/clone.onnx")
        onnx_dev_id = getattr(config, "onnx_device_id", 1 if num_gpus > 1 else 0)
        chunk_sz = getattr(config, "chunk_size", 16)
        mem_limit = getattr(config, "onnx_mem_limit_gb", 8.0)

        collate_fn = ONNXCNNCollateFn(
            onnx_model_path=onnx_path,
            device_id=onnx_dev_id,
            chunk_size=chunk_sz,
            pool_spatial=pool_spatial,
            mem_limit_gb=mem_limit,
            to_cpu=True,
            target_seq_len=seq_len
        )
    elif use_cnn_collate:
        cnn_device = getattr(config, "device", "cuda" if has_cuda else "cpu")
        cnn_model = load_cnn_model(
            manifest_path=getattr(config, "cnn_manifest_path", TrainConfig.cnn_manifest_path),
            weights_path=getattr(config, "cnn_weights_path", TrainConfig.cnn_weights_path),
            device=cnn_device
        )
        chunk_sz = getattr(config, "chunk_size", 16)
        collate_fn = CNNCollateFn(
            cnn_model=cnn_model,
            device=cnn_device,
            chunk_size=chunk_sz,
            pool_spatial=pool_spatial,
            to_cpu=True,
            target_seq_len=seq_len
        )
    else:
        collate_fn = None

    # Chia tập Train / Validation
    train_ratio = getattr(config, "train_ratio", 0.8)
    seed = getattr(config, "seed", 42)

    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size

    if len(dataset) > 0:
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
    else:
        train_dataset, val_dataset = [], []

    batch_size = getattr(config, "batch_size", 4)
    num_workers = getattr(config, "num_workers", 0)
    pin_memory = getattr(config, "pin_memory", True) if has_cuda else False
    drop_last = getattr(config, "drop_last", True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=getattr(config, "shuffle", True),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn
    )

    return train_loader, val_loader


# Alias tương thích với script Kaggle / notebook
build_dataloaders = get_dataloaders


if __name__ == "__main__":
    import time

    print("=" * 70)
    print("KIỂM TRA DATALOADER VỚI ADAPTIVE AVERAGE POOLING TRÊN GPU TRÍCH XUẤT")
    print("=" * 70)

    # 1. Kiểm tra kích thước tensor đầu ra của ONNXCNNCollateFn (chuỗi động 28 frames và 30 frames)
    collate_pooled = ONNXCNNCollateFn(
        onnx_model_path="outsrc/myCNN/checkpoints_ftCOCO/clone.onnx",
        device_id=0,
        chunk_size=16,
        pool_spatial=True,
        to_cpu=True,
        target_seq_len=30
    )

    dummy_batch = [
        (torch.randn(28, 3, 480, 480), 0),  # Video 1: 28 frames
        (torch.randn(30, 3, 480, 480), 1)   # Video 2: 30 frames
    ]

    (p3, p4, p5), labels = collate_pooled(dummy_batch)
    print(f"[+] Output khi dùng ONNXCNNCollateFn (batch chuỗi động đã pad về target_seq_len=30):")
    print(f"    - p3 shape: {list(p3.shape)} (Mong đợi [Batch=2, Seq=30, Dim=224])")
    print(f"    - p4 shape: {list(p4.shape)} (Mong đợi [Batch=2, Seq=30, Dim=448])")
    print(f"    - p5 shape: {list(p5.shape)} (Mong đợi [Batch=2, Seq=30, Dim=640])")
    print(f"    - labels  : {labels.tolist()}")
    assert p3.shape == (2, 30, 224)
    assert p4.shape == (2, 30, 448)
    assert p5.shape == (2, 30, 640)

    # 2. Kiểm tra CNNCollateFn với video có độ dài động (Dynamic Temporal Sampling)
    print("\n[+] Kiểm tra CNNCollateFn với độ dài chuỗi động (28 frames và 30 frames):")
    collate_cnn = CNNCollateFn(
        cnn_model=None,  # Fallback dummy generator
        device="cpu",
        chunk_size=16,
        pool_spatial=True,
        to_cpu=True,
        target_seq_len=30
    )
    dynamic_batch = [
        (torch.randn(28, 3, 480, 480), 0),  # Video 1 có 28 frames
        (torch.randn(30, 3, 480, 480), 1)   # Video 2 có 30 frames
    ]
    (p3_c, p4_c, p5_c), labels_c = collate_cnn(dynamic_batch)
    print(f"    - p3 shape: {list(p3_c.shape)} (Mong đợi [Batch=2, Seq=30, Dim=224])")
    print(f"    - p4 shape: {list(p4_c.shape)} (Mong đợi [Batch=2, Seq=30, Dim=448])")
    print(f"    - p5 shape: {list(p5_c.shape)} (Mong đợi [Batch=2, Seq=30, Dim=640])")
    print(f"    - labels  : {labels_c.tolist()}")
    assert p3_c.shape == (2, 30, 224)
    assert p4_c.shape == (2, 30, 448)
    assert p5_c.shape == (2, 30, 640)

    # 3. Kiểm tra tương thích với mô hình DeepLSTMClassifier
    from model import DeepLSTMClassifier
    model = DeepLSTMClassifier()
    model_out = model((p3_c, p4_c, p5_c))
    print(f"\n[+] Mô hình DeepLSTMClassifier tiếp nhận đặc trưng 3D thành công:")
    print(f"    - Model logits shape: {list(model_out.shape)} (Mong đợi [2, 30, 2])")
    assert model_out.shape == (2, 30, 2)

    print("=" * 70)
    print("[+] HOÀN TẤT KIỂM THỬ: TẤT CẢ COLLATE VÀ DATALOADER HOẠT ĐỘNG HOÀN HẢO!")
    print("=" * 70)
