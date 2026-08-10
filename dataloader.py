from functools import partial
from pathlib import Path
from typing import Optional, Tuple, List, Any

import torch
from torch.utils.data import DataLoader, random_split

from config import TrainConfig
from dataset import MyLSTMDataset
from myCNN.src.model import NMSFreeDetector
from myCNN.src.runtime.infer import letterbox


def load_cnn_model(
        manifest_path: str = TrainConfig.cnn_manifest_path,
        weights_path: str = TrainConfig.cnn_weights_path,
        device: str = "cpu"
) -> NMSFreeDetector:
    """
    Nạp mô hình NMSFreeDetector và load trọng số pretrained từ file checkpoint.
    """
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
    Collate function tùy chỉnh cho DataLoader.
    Nhận batch các mẫu (video_tensor, label) từ MyLSTMDataset, gom các khung hình
    và cho chạy qua backbone + neck của NMSFreeDetector để trích xuất feature maps đa mức (p3, p4, p5)
    dạng 6D kích thước [Batch, Seq_Len, 3, 640, 15, 15] mà vẫn giữ nguyên thứ tự chuỗi dữ liệu.
    """

    def __init__(self, cnn_model: Optional[NMSFreeDetector] = None, device: str = "cpu"):
        self.cnn_model = cnn_model
        self.device = device
        if self.cnn_model is not None:
            self.cnn_model.to(self.device)
            self.cnn_model.eval()

    def __call__(self, batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Gom nhóm batch video_tensors [B, Seq_Len, C, H, W] và labels [B]
        videos = torch.stack([item[0] for item in batch], dim=0)
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)

        if self.cnn_model is None:
            return videos, labels

        b, t, c, h, w = videos.shape

        # 2. Flatten theo chiều (B * T) để trích xuất đặc trưng qua CNN
        # Thứ tự các khung hình (b0_t0, b0_t1, ..., b1_t0, b1_t1, ...) được giữ nguyên tuyệt đối
        x_flat = videos.view(b * t, c, h, w).to(self.device)

        with torch.no_grad():
            p3, p4, p5 = self.cnn_model.backbone(x_flat)
            p3, p4, p5 = self.cnn_model.neck(p3, p4, p5)
            # p3: [B*T, 224, 60, 60], p4: [B*T, 448, 30, 30], p5: [B*T, 640, 15, 15]

        # 3. Reshape từng tầng đặc trưng về chuỗi 5D và đưa về CPU
        p3_5d = p3.view(b, t, *p3.shape[1:]).cpu()
        p4_5d = p4.view(b, t, *p4.shape[1:]).cpu()
        p5_5d = p5.view(b, t, *p5.shape[1:]).cpu()

        return (p3_5d, p4_5d, p5_5d), labels


def get_dataloaders(
        config: Optional[TrainConfig] = None,
        use_cnn_collate: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Khởi tạo MyLSTMDataset (như ví dụ trong dataset.py) và trả về train_loader, val_loader.
    Sử dụng collate_fn để nạp khung hình qua backbone và neck của NMSFreeDetector.
    """
    if config is None:
        config = TrainConfig()

    # Khởi tạo transform letterbox như trong ví dụ dataset.py
    transforms = partial(letterbox, new_size=config.image_size[0])

    # Khởi tạo dataset trực tiếp từ TrainConfig
    dataset = MyLSTMDataset.from_config(config, transform=transforms)

    # Xác định device
    device = config.device if (torch.cuda.is_available() and config.device == "cuda") else "cpu"

    # Nạp CNN feature extractor
    cnn_model = None
    if use_cnn_collate:
        cnn_model = load_cnn_model(
            manifest_path=config.cnn_manifest_path,
            weights_path=config.cnn_weights_path,
            device=device
        )

    collate_fn = CNNCollateFn(cnn_model=cnn_model, device=device)

    # Chia tập Train / Validation
    train_size = int(len(dataset) * config.train_ratio)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
        collate_fn=collate_fn
    )

    return train_loader, val_loader


if __name__ == "__main__":
    import sys
    import time

    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    print("=== KIỂM TRA & BENCHMARK DATALOADER VỚI CNN FEATURE EXTRACTOR COLLATE_FN ===")

    # Khởi tạo config để test benchmark
    cfg = TrainConfig(batch_size=2, seq_len=60)

    print(f"[+] Khởi tạo DataLoader: batch_size={cfg.batch_size}, seq_len={cfg.seq_len}, device={cfg.device}...")
    train_loader, val_loader = get_dataloaders(config=cfg, use_cnn_collate=True)

    print(f"[+] Số lượng batch trong train_loader: {len(train_loader)}")
    print(f"[+] Số lượng batch trong val_loader: {len(val_loader)}")

    # -------------------------------------------------------------------------
    # BENCHMARK THỜI GIAN LOAD 1 BATCH DỮ LIỆU
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("--- BENCHMARK QUÁ TRÌNH LOAD 1 BATCH (READ VIDEO + CNN FEATURE EXTRACT) ---")
    print("=" * 60)

    loader_iter = iter(train_loader)

    if torch.cuda.is_available() and cfg.device == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    features, labels = next(loader_iter)

    if torch.cuda.is_available() and cfg.device == "cuda":
        torch.cuda.synchronize()

    elapsed_time = time.perf_counter() - start_time

    if isinstance(features, (tuple, list)):
        batch_size, seq_len = features[0].shape[0], features[0].shape[1]
        shape_info = [list(f.shape) for f in features]
        print(f"-> Kích thước batch đặc trưng đa tầng (p3, p4, p5): {shape_info}")
    else:
        batch_size, seq_len = features.shape[0], features.shape[1]
        print(f"-> Kích thước batch đặc trưng: {list(features.shape)}")

    total_frames = batch_size * seq_len
    fps = total_frames / elapsed_time if elapsed_time > 0 else 0.0
    ms_per_frame = (elapsed_time * 1000) / total_frames if total_frames > 0 else 0.0

    print(f"-> Kích thước batch nhãn (Labels) [B]: {list(labels.shape)}")
    print(f"-> Nhãn thực tế: {labels.tolist()}")
    print("-" * 60)
    print(
        f"-> Tổng thời gian load 1 batch ({batch_size} video, {total_frames} frames): {elapsed_time:.4f} giây ({elapsed_time * 1000:.2f} ms)")
    print(f"-> Thời gian xử lý trung bình mỗi frame: {ms_per_frame:.2f} ms/frame")
    print(f"-> Tốc độ xử lý (Throughput): {fps:.2f} FPS (frames/sec)")
    print("=" * 60)
