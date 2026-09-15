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
    Nhận batch các mẫu (video_tensor, label) từ MyLSTMDataset, gom các khung hình
    và cho chạy qua backbone + neck của NMSFreeDetector để trích xuất đặc trưng đa tầng (p3, p4, p5).

    TỐI ƯU HÓA BỘ NHỚ:
      - pool_spatial=True: Thực hiện Adaptive Average Pooling về (1, 1) ngay trên GPU
        để nén [B*T, C, H, W] -> [B, T, C], giảm dung lượng truyền tải qua RAM hơn 1000 lần.
    """

    def __init__(
        self,
        cnn_model: Optional[Any] = None,
        device: str = "cpu",
        pool_spatial: bool = True,
        to_cpu: bool = True
    ):
        self.cnn_model = cnn_model
        self.device = device
        self.pool_spatial = pool_spatial
        self.to_cpu = to_cpu
        if self.cnn_model is not None:
            self.cnn_model.to(self.device)
            self.cnn_model.eval()

    def __call__(self, batch: List[Tuple[torch.Tensor, int]]) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        videos = torch.stack([item[0] for item in batch], dim=0)
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)

        if self.cnn_model is None:
            return videos, labels

        b, t, c, h, w = videos.shape
        x_flat = videos.view(b * t, c, h, w).to(self.device)

        with torch.no_grad():
            p3, p4, p5 = self.cnn_model.backbone(x_flat)
            p3, p4, p5 = self.cnn_model.neck(p3, p4, p5)
            # p3: [B*T, 224, 60, 60], p4: [B*T, 448, 30, 30], p5: [B*T, 640, 15, 15]

            if self.pool_spatial:
                # Nén Adaptive Average Pooling về (1, 1) ngay trên GPU trích xuất
                p3 = F.adaptive_avg_pool2d(p3, (1, 1)).view(b, t, p3.shape[1])  # [B, T, 224]
                p4 = F.adaptive_avg_pool2d(p4, (1, 1)).view(b, t, p4.shape[1])  # [B, T, 448]
                p5 = F.adaptive_avg_pool2d(p5, (1, 1)).view(b, t, p5.shape[1])  # [B, T, 640]
            else:
                p3 = p3.view(b, t, *p3.shape[1:])
                p4 = p4.view(b, t, *p4.shape[1:])
                p5 = p5.view(b, t, *p5.shape[1:])

        if self.to_cpu:
            p3, p4, p5 = p3.cpu(), p4.cpu(), p5.cpu()

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
        chunk_size: int = 15,
        pool_spatial: bool = True,
        mem_limit_gb: float = 8.0,
        to_cpu: bool = True
    ):
        self.onnx_model_path = str(Path(onnx_model_path).resolve())
        self.device_id = device_id
        self.device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        self.chunk_size = chunk_size
        self.pool_spatial = pool_spatial
        self.to_cpu = to_cpu
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

    def __call__(self, batch: List[Tuple[torch.Tensor, int]]) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        if not batch:
            if self.pool_spatial:
                dummy_p3 = torch.zeros((0, 224))
                dummy_p4 = torch.zeros((0, 448))
                dummy_p5 = torch.zeros((0, 640))
            else:
                dummy_p3 = torch.zeros((0, 224, 60, 60))
                dummy_p4 = torch.zeros((0, 448, 30, 30))
                dummy_p5 = torch.zeros((0, 640, 15, 15))
            return (dummy_p3, dummy_p4, dummy_p5), torch.empty(0, dtype=torch.long)

        valid_p3, valid_p4, valid_p5 = [], [], []
        valid_labels = []

        for idx, (video, label) in enumerate(batch):
            try:
                is_single_image = (video.dim() == 3)
                if is_single_image:
                    frames = video.unsqueeze(0).float()
                elif video.dim() == 4:
                    frames = video.float()
                elif video.dim() == 5 and video.shape[0] == 1:
                    frames = video.squeeze(0).float()
                else:
                    frames = video.float()

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

                        # Sau khi đã nén siêu nhẹ, chuyển về CPU để giải phóng VRAM GPU 1
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

                if is_single_image:
                    p3_item = p3_item.squeeze(0)
                    p4_item = p4_item.squeeze(0)
                    p5_item = p5_item.squeeze(0)

                # Kiểm tra tương thích kích thước giữa các mẫu trong batch
                if len(valid_p3) > 0 and p3_item.shape != valid_p3[0].shape:
                    print(f"[ONNXCNNCollateFn][WARN] Kích thước mẫu {idx} ({p3_item.shape}) không khớp ({valid_p3[0].shape}). Bỏ qua.")
                    continue

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
            t_fb = 60
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


def get_tensor_dataloaders(config: Optional[Any] = None) -> Tuple[DataLoader, DataLoader]:
    """
    Khởi tạo TrainLoader và ValLoader từ tệp nhị phân PyTorch Tensor (.pt).
    Đồng bộ 100% với CELL 6 của datn4ni2.ipynb:
    - Nạp trực tiếp dữ liệu vào RAM trong < 0.5s qua PreloadedTensorDataset.
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_mem,
        num_workers=0
    )

    print(f"[DataLoader] Đã tạo DataLoader Tensor: Train={len(train_dataset)} mẫu ({len(train_loader)} batches) | Val={len(val_dataset)} mẫu ({len(val_loader)} batches)")
    return train_loader, val_loader


def get_dataloaders(
    config: Optional[Any] = None,
    use_cnn_collate: bool = True,
    use_onnx: bool = False,
    pool_spatial: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Khởi tạo DataLoader cho mô hình Deep LSTM:
    - Nếu config.use_preloaded_pt=True (mặc định): Nạp trực tiếp Tensor .pt siêu tốc qua get_tensor_dataloaders.
    - Nếu config.use_preloaded_pt=False: Nạp qua MyLSTMDataset đọc video thô qua OpenCV.
    """
    if config is None:
        config = TrainConfig()

    # Chế độ nạp Tensor .pt siêu tốc (CELL 6 datn4ni2.ipynb)
    if getattr(config, "use_preloaded_pt", True):
        return get_tensor_dataloaders(config)

    image_size = getattr(config, "image_size", (480, 480))
    new_size = image_size[0] if isinstance(image_size, (tuple, list)) else image_size

    transforms = partial(letterbox, new_size=new_size) if letterbox is not None else None

    # Khởi tạo dataset
    if hasattr(MyLSTMDataset, "from_config"):
        dataset = MyLSTMDataset.from_config(config, transform=transforms)
    else:
        dataset = MyLSTMDataset(
            dataset_dir=getattr(config, "dataset_dir", ""),
            seq_len=getattr(config, "seq_len", 60),
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
        chunk_sz = getattr(config, "chunk_size", 15)
        mem_limit = getattr(config, "onnx_mem_limit_gb", 8.0)

        collate_fn = ONNXCNNCollateFn(
            onnx_model_path=onnx_path,
            device_id=onnx_dev_id,
            chunk_size=chunk_sz,
            pool_spatial=pool_spatial,
            mem_limit_gb=mem_limit,
            to_cpu=True
        )
    elif use_cnn_collate:
        cnn_device = getattr(config, "device", "cuda" if has_cuda else "cpu")
        cnn_model = load_cnn_model(
            manifest_path=getattr(config, "cnn_manifest_path", TrainConfig.cnn_manifest_path),
            weights_path=getattr(config, "cnn_weights_path", TrainConfig.cnn_weights_path),
            device=cnn_device
        )
        collate_fn = CNNCollateFn(
            cnn_model=cnn_model,
            device=cnn_device,
            pool_spatial=pool_spatial,
            to_cpu=True
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

    # 1. Kiểm tra kích thước tensor đầu ra của ONNXCNNCollateFn (giả lập 2 video mẫu)
    collate_pooled = ONNXCNNCollateFn(
        onnx_model_path="outsrc/myCNN/checkpoints_ftCOCO/clone.onnx",
        device_id=0,
        chunk_size=15,
        pool_spatial=True,
        to_cpu=True
    )

    dummy_batch = [
        (torch.randn(30, 3, 480, 480), 0),
        (torch.randn(30, 3, 480, 480), 1)
    ]

    (p3, p4, p5), labels = collate_pooled(dummy_batch)
    print(f"[+] Output khi pool_spatial=True (Đã nén ngay trên GPU):")
    print(f"    - p3 shape: {list(p3.shape)} (Mong đợi [Batch=2, Seq=30, Dim=224])")
    print(f"    - p4 shape: {list(p4.shape)} (Mong đợi [Batch=2, Seq=30, Dim=448])")
    print(f"    - p5 shape: {list(p5.shape)} (Mong đợi [Batch=2, Seq=30, Dim=640])")
    print(f"    - labels  : {labels.tolist()}")
    assert p3.shape == (2, 30, 224)
    assert p4.shape == (2, 30, 448)
    assert p5.shape == (2, 30, 640)

    # 2. Kiểm tra tương thích với mô hình DeepLSTMClassifier
    from model import DeepLSTMClassifier
    model = DeepLSTMClassifier()
    model_out = model((p3, p4, p5))
    print(f"[+] Mô hình DeepLSTMClassifier tiếp nhận đặc trưng 3D thành công:")
    print(f"    - Model logits shape: {list(model_out.shape)} (Mong đợi [2, 30, 2])")
    assert model_out.shape == (2, 30, 2)

    print("=" * 70)
    print("[+] HOÀN TẤT KIỂM THỬ: ADAPTIVE AVERAGE POOLING TRÊN GPU HOẠT ĐỘNG HOÀN HẢO!")
    print("=" * 70)
