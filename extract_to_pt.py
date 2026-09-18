#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
File: extract_to_pt.py
Mục đích:
    Triển khai GIAI ĐOẠN 1 của Kế hoạch tối ưu hóa huấn luyện:
    1. Quét toàn bộ video từ tập dữ liệu (hỗ trợ .mp4, .avi, .mkv, .mov,...).
    2. Phân chia Train/Validation (80/20) TRƯỚC TIÊN theo nhãn (Stratified Split, seed=42)
       và lưu cấu hình phân bổ vào 'dataset_split.json' để đảm bảo không rò rỉ dữ liệu (No Data Leakage).
    3. Trích xuất đặc trưng không gian (p3, p4, p5) qua mô hình ONNX CUDA ('clone.onnx')
       kết hợp Adaptive Average Pooling (1, 1).
    4. Hỗ trợ cơ chế Resume / Cache thông minh (lưu từng video vào cache tạm thời để không mất tiến trình khi dừng).
    5. Đóng gói kết quả thành 2 tệp PyTorch Tensor nhị phân:
       - 'features_sust_train.pt'
       - 'features_sust_val.pt'
"""

import os
import sys
import time
import json
import random
import argparse
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional

from augment import DetectionAugmenter, config as DEFAULT_AUG_CONFIG

# Đảm bảo console Windows hỗ trợ in UTF-8 không bị lỗi charmap
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm

# Nạp thư viện CUDA DLL từ PyTorch nếu chạy trên môi trường Windows
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.path.exists(torch_lib_path) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(torch_lib_path)
    except Exception as e:
        print(f"[WARN] Không thể thêm torch DLL directory: {e}")

try:
    import onnxruntime as ort
except ImportError as e:
    raise ImportError("Vui lòng cài đặt onnxruntime-gpu: pip install onnxruntime-gpu") from e


# ==============================================================================
# 1. TIỀN XỬ LÝ ẢNH & RUNTIME ONNX VỚI CUDA ZERO-COPY
# ==============================================================================
def letterbox(image: np.ndarray, new_size: int = 480, color=(114, 114, 114)) -> np.ndarray:
    """Resize ảnh giữ nguyên tỉ lệ (aspect ratio) với padding đồng màu."""
    h, w = image.shape[:2]
    scale = min(new_size / h, new_size / w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((new_size, new_size, 3), color, dtype=image.dtype)
    pad_left = (new_size - new_w) // 2
    pad_top = (new_size - new_h) // 2
    canvas[pad_top: pad_top + new_h, pad_left: pad_left + new_w] = resized
    return canvas


class CloneONNXCUDARuntime:
    """Quản lý suy luận ONNX Runtime trên GPU CUDA với I/O Binding."""

    def __init__(self, onnx_model_path: str, device_id: int = 0):
        self.onnx_model_path = str(onnx_model_path)
        self.device_id = device_id

        if not os.path.exists(self.onnx_model_path):
            raise FileNotFoundError(f"Không tìm thấy file ONNX: {self.onnx_model_path}")

        self.session_options = ort.SessionOptions()
        self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session_options.log_severity_level = 3

        cuda_options = {
            "device_id": self.device_id,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "HEURISTIC",
            "do_copy_in_default_stream": True,
        }
        providers = [
            ("CUDAExecutionProvider", cuda_options),
            "CPUExecutionProvider"
        ]

        print(f"[+] Khởi tạo ONNX Runtime InferenceSession trên CUDA:{self.device_id}...")
        self.session = ort.InferenceSession(
            self.onnx_model_path,
            sess_options=self.session_options,
            providers=providers,
        )

        active = self.session.get_providers()
        print(f"[+] Active Providers: {active}")
        if "CUDAExecutionProvider" not in active:
            raise RuntimeError("CUDAExecutionProvider chưa được kích hoạt!")

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def forward(self, input_tensor: torch.Tensor) -> List[torch.Tensor]:
        """Suy luận trực tiếp từ PyTorch CUDA Tensor sang PyTorch CUDA Tensors (Zero-copy DLPack)."""
        io_binding = self.session.io_binding()
        io_binding.bind_input(
            name=self.input_name,
            device_type="cuda",
            device_id=self.device_id,
            element_type=np.float32,
            shape=tuple(input_tensor.shape),
            buffer_ptr=input_tensor.data_ptr(),
        )
        for out_name in self.output_names:
            io_binding.bind_output(out_name, device_type="cuda", device_id=self.device_id)

        self.session.run_with_iobinding(io_binding)
        raw_outputs = io_binding.get_outputs()
        return [torch.from_dlpack(out) for out in raw_outputs]


# ==============================================================================
# 2. HÀM ĐỌC VIDEO & TRÍCH XUẤT ĐẶC TRƯNG MỖI VIDEO
# ==============================================================================
def read_and_sample_video_frames(
        video_path: Path,
        seq_len: int = 120,
        img_size: int = 480
) -> Tuple[List[np.ndarray], int]:
    """
    Đọc nhanh video tuần tự, letterbox sang kích thước cố định và định dạng RGB (uint8).
    
    Returns:
        frames_rgb: Danh sách seq_len ảnh numpy RGB uint8 [img_size, img_size, 3]
        actual_frame_count: Số khung hình thực tế trích xuất được từ video trước khi padding
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    if total_frames <= 0:
        # Dự phòng trường hợp metadata của video không trả về frame count (thường gặp ở một số file .avi)
        raw_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            raw_frames.append(frame)
        cap.release()

        n_raw = len(raw_frames)
        if n_raw > 0:
            indices = set(torch.linspace(0, max(0, n_raw - 1), seq_len).long().tolist())
            for idx, frame in enumerate(raw_frames):
                if idx in indices:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    lb = letterbox(rgb, new_size=img_size)
                    frames.append(lb)
    else:
        indices = set(torch.linspace(0, max(0, total_frames - 1), seq_len).long().tolist())
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx in indices:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                lb = letterbox(rgb, new_size=img_size)
                frames.append(lb)
            idx += 1
        cap.release()

    actual_frames = len(frames)
    if actual_frames == 0:
        # Trường hợp rỗng dự phòng
        frames = [np.zeros((img_size, img_size, 3), dtype=np.uint8) for _ in range(seq_len)]
    else:
        while len(frames) < seq_len:
            frames.append(frames[-1].copy())
    frames = frames[:seq_len]
    return frames, actual_frames


def forward_video_chunks(
        frames_rgb: List[np.ndarray],
        runner: CloneONNXCUDARuntime,
        chunk_size: int = 24,
        device: str = "cuda:0"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward danh sách khung hình RGB qua mô hình ONNX CUDA theo từng mini-chunk
    kết hợp Adaptive Average Pooling (1, 1).
    
    Returns:
        p3_tensor: [seq_len, 224] on CPU
        p4_tensor: [seq_len, 448] on CPU
        p5_tensor: [seq_len, 640] on CPU
    """
    seq_len = len(frames_rgb)
    tensor_list = [
        torch.from_numpy(np.ascontiguousarray(f.transpose(2, 0, 1))).float() / 255.0
        for f in frames_rgb
    ]
    video_tensor = torch.stack(tensor_list, dim=0)  # [seq_len, 3, H, W]

    # Forward theo mini-chunks để tiết kiệm VRAM và tăng tốc
    p3_chunks, p4_chunks, p5_chunks = [], [], []
    for i in range(0, seq_len, chunk_size):
        chunk = video_tensor[i: i + chunk_size].to(device, non_blocking=True)
        outs = runner.forward(chunk)
        # outs[0]: [B, 224, 60, 60], outs[1]: [B, 448, 30, 30], outs[2]: [B, 640, 15, 15]
        p3_chunks.append(F.adaptive_avg_pool2d(outs[0], (1, 1)).flatten(1).cpu())
        p4_chunks.append(F.adaptive_avg_pool2d(outs[1], (1, 1)).flatten(1).cpu())
        p5_chunks.append(F.adaptive_avg_pool2d(outs[2], (1, 1)).flatten(1).cpu())

    p3_tensor = torch.cat(p3_chunks, dim=0)  # [seq_len, 224]
    p4_tensor = torch.cat(p4_chunks, dim=0)  # [seq_len, 448]
    p5_tensor = torch.cat(p5_chunks, dim=0)  # [seq_len, 640]
    return p3_tensor, p4_tensor, p5_tensor


def extract_single_video_features(
        video_path: Path,
        runner: CloneONNXCUDARuntime,
        seq_len: int = 120,
        img_size: int = 480,
        chunk_size: int = 24,
        device: str = "cuda:0",
        augmenter: Optional[DetectionAugmenter] = None,
        aug_seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Đọc nhanh video tuần tự, letterbox, tùy chọn áp dụng tăng cường dữ liệu (nếu có augmenter)
    với seed đồng bộ thời gian cho toàn bộ video clip, forward ONNX theo chunk và pool về 1D.
    Hỗ trợ cả định dạng .mp4, .avi, .mkv, .mov,...
    
    Returns:
        p3_tensor: [seq_len, 224] on CPU
        p4_tensor: [seq_len, 448] on CPU
        p5_tensor: [seq_len, 640] on CPU
        actual_frame_count: số khung hình thực tế trong video
    """
    frames_rgb, actual_frames = read_and_sample_video_frames(
        video_path=video_path,
        seq_len=seq_len,
        img_size=img_size
    )

    if augmenter is not None:
        frames_rgb, _, _, _ = augmenter.augment_video(
            frames=frames_rgb,
            boxes_list=None,
            labels_list=None,
            seed=aug_seed
        )

    p3_tensor, p4_tensor, p5_tensor = forward_video_chunks(
        frames_rgb=frames_rgb,
        runner=runner,
        chunk_size=chunk_size,
        device=device
    )

    return p3_tensor, p4_tensor, p5_tensor, actual_frames


# ==============================================================================
# 3. QUẢN LÝ PHÂN CHIA DATASET (PRE-SPLITTING)
# ==============================================================================
def prepare_dataset_split(
        dataset_dir: Path,
        split_json_path: Path,
        train_ratio: float = 0.8,
        seed: int = 42,
        video_exts: Tuple[str, ...] = (".mp4", ".avi", ".mkv", ".mov")
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """
    Quét video, phân chia Train/Val theo Stratified Split ở cấp độ Video ID trước khi trích xuất.
    Hỗ trợ đa định dạng video (.mp4, .avi, .mkv, .mov,...).
    """
    if split_json_path.exists():
        print(f"[+] Tìm thấy file cấu hình phân chia có sẵn: {split_json_path}")
        with open(split_json_path, "r", encoding="utf-8") as f:
            split_info = json.load(f)

        train_items = [(dataset_dir / item["name"], item["label"]) for item in split_info.get("train", [])]
        val_items = [(dataset_dir / item["name"], item["label"]) for item in split_info.get("val", [])]

        all_items = train_items + val_items
        if all_items and all(p.exists() for p, _ in all_items[:10]):
            print(f"[+] Đã tải phân chia: Train={len(train_items)} video, Val={len(val_items)} video.")
            return train_items, val_items
        else:
            print(
                f"[WARN] File cấu hình '{split_json_path}' không khớp với các video trong '{dataset_dir}'. Đang quét và phân chia lại...")

    print(f"[+] Quét toàn bộ video từ thư mục: {dataset_dir}...")
    normalized_exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in video_exts}

    video_files = [
        f for f in dataset_dir.iterdir()
        if f.is_file() and f.suffix.lower() in normalized_exts
    ]
    if not video_files:
        video_files = [
            f for f in dataset_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in normalized_exts
        ]
    video_files = sorted(video_files)

    if not video_files:
        raise FileNotFoundError(
            f"Không tìm thấy video nào thuộc các định dạng {sorted(list(normalized_exts))} trong: {dataset_dir}"
        )

    items = []
    for vf in video_files:
        stem = vf.stem
        parts = stem.split("-")
        label = None
        if len(parts) >= 2:
            label_str = parts[-2].lower()
            if label_str in ("driving", "alert", "normal"):
                label = 0
            elif label_str in ("drowsiness", "drowsy"):
                label = 1

        if label is None:
            name_lower = stem.lower()
            if "drowsiness" in name_lower or "drowsy" in name_lower:
                label = 1
            elif "driving" in name_lower or "alert" in name_lower or "normal" in name_lower:
                label = 0

        if label is not None:
            items.append((vf, label))

    print(f"[+] Tìm thấy tổng cộng {len(items)} video hợp lệ.")
    if not items:
        raise ValueError(f"Không thể trích xuất nhãn 'driving' hoặc 'drowsiness' từ video trong: {dataset_dir}")

    # Phân chia phân tầng (Stratified Split) thuần Python theo tỷ lệ train_ratio
    rng = random.Random(seed)
    class_0 = [item for item in items if item[1] == 0]
    class_1 = [item for item in items if item[1] == 1]

    rng.shuffle(class_0)
    rng.shuffle(class_1)

    split_0 = int(round(len(class_0) * train_ratio))
    split_1 = int(round(len(class_1) * train_ratio))

    train_items = class_0[:split_0] + class_1[:split_1]
    val_items = class_0[split_0:] + class_1[split_1:]

    rng.shuffle(train_items)
    rng.shuffle(val_items)

    # Lưu siêu dữ liệu phân bổ
    split_info = {
        "metadata": {
            "dataset_dir": str(dataset_dir),
            "seed": seed,
            "train_ratio": train_ratio,
            "total_videos": len(items),
            "train_count": len(train_items),
            "val_count": len(val_items),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "train": [
            {"name": p.relative_to(dataset_dir).as_posix() if p.is_relative_to(dataset_dir) else p.name, "label": lbl}
            for p, lbl in train_items],
        "val": [
            {"name": p.relative_to(dataset_dir).as_posix() if p.is_relative_to(dataset_dir) else p.name, "label": lbl}
            for p, lbl in val_items]
    }

    split_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_json_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=4, ensure_ascii=False)

    print(f"[+] Đã lưu cấu hình phân bổ độc lập tại: {split_json_path}")
    print(f"    - Tập Train: {len(train_items)} videos ({len(train_items) / len(items) * 100:.1f}%)")
    print(f"    - Tập Val:   {len(val_items)} videos ({len(val_items) / len(items) * 100:.1f}%)")

    return train_items, val_items


# ==============================================================================
# 4. QUY TRÌNH TRÍCH XUẤT CHÍNH CHO TỪNG TẬP
# ==============================================================================
def process_split_set(
        split_name: str,
        items: List[Tuple[Path, int]],
        runner: CloneONNXCUDARuntime,
        output_pt_path: Path,
        cache_dir: Path,
        seq_len: int = 120,
        img_size: int = 480,
        chunk_size: int = 24,
        use_fp16: bool = False,
        augmenter: Optional[DetectionAugmenter] = None,
        num_aug: int = 0,
        include_original: bool = True,
        base_seed: int = 42,
        force_recompute: bool = False
) -> None:
    """
    Trích xuất đặc trưng cho một phân tập (Train hoặc Val), hỗ trợ tăng cường dữ liệu và cache từng video.
    
    Args:
        split_name: Tên phân tập ("train" hoặc "val")
        items: Danh sách bộ đôi (video_path, label)
        runner: Đối tượng ONNX CUDA Runtime
        output_pt_path: Đường dẫn tệp .pt đầu ra
        cache_dir: Thư mục lưu cache các video đã trích xuất
        seq_len: Số khung hình chuẩn hóa mỗi video
        img_size: Kích thước resize letterbox
        chunk_size: Batch size khi forward ONNX
        use_fp16: Lưu đặc trưng dạng float16
        augmenter: Đối tượng DetectionAugmenter từ augment.py
        num_aug: Số lượng biến thể augment cần tạo cho mỗi video (chỉ áp dụng khi augmenter != None)
        include_original: Giữ lại video gốc bên cạnh các bản augment
        base_seed: Seed cơ sở để tạo chuỗi seed ngẫu nhiên nhưng có tính lặp lại (reproducible)
        force_recompute: Bắt buộc tính toán lại bỏ qua cache
    """
    is_augmented_split = (augmenter is not None and num_aug > 0)
    actual_include_original = include_original if is_augmented_split else True

    print(f"\n{'=' * 75}")
    print(f"[*] BẮT ĐẦU TRÍCH XUẤT ĐẶC TRƯNG CHO PHÂN TẬP: {split_name.upper()} ({len(items)} VIDEOS)")
    if is_augmented_split:
        aug_info = f"BẬT (Số bản augment={num_aug}, Giữ bản gốc={'Có' if actual_include_original else 'Không'})"
    else:
        aug_info = "TẮT (Chỉ trích xuất video gốc)"
    print(f"[*] Tăng cường dữ liệu (Augmentation): {aug_info}")
    print(f"{'=' * 75}")

    split_cache_dir = cache_dir / split_name
    split_cache_dir.mkdir(parents=True, exist_ok=True)

    p3_all, p4_all, p5_all = [], [], []
    labels_all, video_ids_all, seq_lens_all = [], [], []

    # Khởi động trước (Warm-up) mô hình
    print("[+] Khởi động (Warm-up) ONNX Execution Provider...")
    dummy = torch.zeros(chunk_size, 3, img_size, img_size, device=f"cuda:{runner.device_id}")
    runner.forward(dummy)
    print("[+] Khởi động hoàn tất!")

    start_time = time.time()
    for idx, (video_path, label) in enumerate(tqdm(items, desc=f"[{split_name.upper()}]")):
        video_id = video_path.stem

        # Xác định danh sách các biến thể cần xử lý cho video này
        targets = []
        if actual_include_original:
            targets.append({
                "sub_id": video_id,
                "is_aug": False,
                "aug_idx": 0,
                "seed": None,
            })

        if is_augmented_split:
            for k in range(1, num_aug + 1):
                aug_id = f"{video_id}_aug{k}"
                # Tạo seed xác định dựa trên base_seed và md5 hash của aug_id
                aug_seed = (base_seed + int(hashlib.md5(aug_id.encode("utf-8")).hexdigest()[:8], 16)) % (2 ** 31 - 1)
                targets.append({
                    "sub_id": aug_id,
                    "is_aug": True,
                    "aug_idx": k,
                    "seed": aug_seed,
                })

        # Kiểm tra xem những target nào chưa có trong cache
        needed_targets = []
        for t in targets:
            cache_file = split_cache_dir / f"{t['sub_id']}.pt"
            if force_recompute or not cache_file.exists():
                needed_targets.append(t)

        # Nếu cần trích xuất ít nhất một target, chỉ đọc và giải mã video một lần duy nhất
        if len(needed_targets) > 0:
            try:
                raw_frames, actual_frames = read_and_sample_video_frames(
                    video_path=video_path,
                    seq_len=seq_len,
                    img_size=img_size
                )
            except Exception as e:
                print(f"\n[ERROR] Lỗi khi đọc video {video_path.name}: {e}")
                continue

            for t in needed_targets:
                try:
                    if t["is_aug"]:
                        aug_frames, _, _, _ = augmenter.augment_video(
                            frames=raw_frames,
                            boxes_list=None,
                            labels_list=None,
                            seed=t["seed"]
                        )
                        p3, p4, p5 = forward_video_chunks(
                            frames_rgb=aug_frames,
                            runner=runner,
                            chunk_size=chunk_size,
                            device=f"cuda:{runner.device_id}"
                        )
                    else:
                        p3, p4, p5 = forward_video_chunks(
                            frames_rgb=raw_frames,
                            runner=runner,
                            chunk_size=chunk_size,
                            device=f"cuda:{runner.device_id}"
                        )

                    cache_file = split_cache_dir / f"{t['sub_id']}.pt"
                    torch.save({
                        "video_id": t["sub_id"],
                        "label": label,
                        "p3": p3,
                        "p4": p4,
                        "p5": p5,
                        "actual_frames": actual_frames,
                        "is_augmented": t["is_aug"],
                        "aug_seed": t["seed"]
                    }, cache_file)
                except Exception as e:
                    print(f"\n[ERROR] Lỗi khi xử lý mẫu {t['sub_id']}: {e}")

        # Nạp dữ liệu từ cache cho tất cả targets của video này
        for t in targets:
            cache_file = split_cache_dir / f"{t['sub_id']}.pt"
            if cache_file.exists():
                cached_data = torch.load(cache_file, map_location="cpu")
                p3_all.append(cached_data["p3"])
                p4_all.append(cached_data["p4"])
                p5_all.append(cached_data["p5"])
                labels_all.append(cached_data["label"])
                video_ids_all.append(cached_data["video_id"])
                seq_lens_all.append(cached_data.get("actual_frames", seq_len))

    total_duration = time.time() - start_time
    print(
        f"\n[+] Hoàn tất trích xuất {len(video_ids_all)} mẫu tập {split_name.upper()} trong {total_duration / 60:.2f} phút!")

    if len(video_ids_all) == 0:
        print(f"[WARN] Không có mẫu nào được trích xuất cho phân tập {split_name.upper()}!")
        return

    # Đóng gói toàn bộ tensor 3D
    print(f"[+] Đang đóng gói Tensor cho tập {split_name}...")
    final_p3 = torch.stack(p3_all, dim=0)  # [N, 120, 224]
    final_p4 = torch.stack(p4_all, dim=0)  # [N, 120, 448]
    final_p5 = torch.stack(p5_all, dim=0)  # [N, 120, 640]
    final_labels = torch.tensor(labels_all, dtype=torch.long)
    final_seq_lens = torch.tensor(seq_lens_all, dtype=torch.int32)

    if use_fp16:
        print("[+] Ép kiểu sang float16 để tối ưu 50% dung lượng lưu trữ...")
        final_p3 = final_p3.half()
        final_p4 = final_p4.half()
        final_p5 = final_p5.half()

    num_alerts = (final_labels == 0).sum().item()
    num_drowsy = (final_labels == 1).sum().item()
    num_orig = sum(1 for vid in video_ids_all if "_aug" not in vid)
    num_aug_count = sum(1 for vid in video_ids_all if "_aug" in vid)

    save_dict = {
        "p3": final_p3,
        "p4": final_p4,
        "p5": final_p5,
        "labels": final_labels,
        "video_ids": video_ids_all,
        "seq_lens": final_seq_lens,
        "dtype": "float16" if use_fp16 else "float32",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "augmented": is_augmented_split,
        "num_aug": num_aug if is_augmented_split else 0,
        "include_original": actual_include_original,
    }

    output_pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_dict, str(output_pt_path))

    file_size_mb = output_pt_path.stat().st_size / (1024 * 1024)
    print(f"[SUCCESS] Đã lưu thành công tập {split_name.upper()} vào: {output_pt_path.resolve()}")
    print(f"          - Kích thước tệp: {file_size_mb:.2f} MB")
    print(f"          - Tổng số mẫu:   {len(video_ids_all)} (Gốc: {num_orig}, Augment: {num_aug_count})")
    print(f"          - Phân bố nhãn:  Tỉnh táo (0) = {num_alerts}, Buồn ngủ (1) = {num_drowsy}")
    print(f"          - Tensor p3:     {list(final_p3.shape)}")
    print(f"          - Tensor p4:     {list(final_p4.shape)}")
    print(f"          - Tensor p5:     {list(final_p5.shape)}")
    print(f"          - Labels:        {list(final_labels.shape)}")


# ==============================================================================
# 5. ĐIỂM VÀO CHÍNH (MAIN ENTRY POINT)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Trích xuất đặc trưng video (.mp4, .avi, ...) ra file .pt")
    parser.add_argument("--data_dir", type=str, default=r"D:\Project\AI\dataset\SUST",
                        help="Thư mục chứa video dữ liệu")
    parser.add_argument("--video_exts", type=str, default=".mp4,.avi,.mkv,.mov",
                        help="Danh sách phần mở rộng video hỗ trợ, phân cách bởi dấu phẩy (mặc định: .mp4,.avi,.mkv,.mov)")
    parser.add_argument("--onnx_path", type=str,
                        default=r"outsrc\myCNN\checkpoints_ftCOCO\clone.onnx",
                        help="Đường dẫn file mô hình ONNX")
    parser.add_argument("--output_dir", type=str, default="extracted_features_pt",
                        help="Thư mục xuất kết quả")
    parser.add_argument("--train_name", type=str, default="features_sust_train.pt",
                        help="Tên file tensor train đầu ra (mặc định: features_sust_train.pt)")
    parser.add_argument("--val_name", type=str, default="features_sust_val.pt",
                        help="Tên file tensor val đầu ra (mặc định: features_sust_val.pt)")
    parser.add_argument("--seq_len", type=int, default=120, help="Số khung hình cố định mỗi video")
    parser.add_argument("--img_size", type=int, default=480, help="Kích thước resize letterbox")
    parser.add_argument("--chunk_size", type=int, default=24, help="Batch size khi forward ONNX")
    parser.add_argument("--device_id", type=int, default=0, help="CUDA device index")
    parser.add_argument("--limit", type=int, default=None,
                        help="Giới hạn số lượng video để chạy thử nghiệm (ví dụ: --limit 10)")
    parser.add_argument("--fp16", action="store_true", help="Lưu đặc trưng dạng float16 (giảm 50%% dung lượng)")

    # Tham số tăng cường dữ liệu (Data Augmentation) từ augment.py
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                        help="Bật/tắt tăng cường dữ liệu từ augment.py (mặc định: True)")
    parser.add_argument("--num_aug", type=int, default=1,
                        help="Số bản sao tăng cường cho mỗi video trong tập áp dụng augment (mặc định: 1)")
    parser.add_argument("--include_original", action=argparse.BooleanOptionalAction, default=True,
                        help="Giữ lại video gốc bên cạnh các bản sao tăng cường (mặc định: True)")
    parser.add_argument("--augment_splits", type=str, default="train",
                        help="Các phân tập áp dụng tăng cường, phân cách bởi dấu phẩy (mặc định: train)")
    parser.add_argument("--aug_seed", type=int, default=42,
                        help="Random seed cơ sở cho việc tăng cường dữ liệu (mặc định: 42)")
    parser.add_argument("--aug_config", type=str, default=None,
                        help="Đường dẫn file JSON cấu hình augment tùy chỉnh (mặc định: dùng config mặc định từ augment.py)")
    parser.add_argument("--force_recompute", action="store_true", default=False,
                        help="Bắt buộc trích xuất lại từ đầu, bỏ qua cache hiện có")
    args = parser.parse_args()

    dataset_dir = Path(args.data_dir)
    onnx_path = Path(args.onnx_path)
    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "cache"

    video_exts = tuple(ext.strip() for ext in args.video_exts.split(",") if ext.strip())
    augment_splits = [s.strip().lower() for s in args.augment_splits.split(",") if s.strip()]

    split_json_path = output_dir / "dataset_split.json"
    train_pt_path = output_dir / args.train_name
    val_pt_path = output_dir / args.val_name

    print("=" * 75)
    print("      HỆ THỐNG TRÍCH XUẤT ĐẶC TRƯNG LOCAL SANG PYTORCH TENSOR (.PT)      ")
    print("=" * 75)
    print(f"[*] Dataset Directory : {dataset_dir}")
    print(f"[*] Video Exts        : {video_exts}")
    print(f"[*] ONNX Model Path   : {onnx_path}")
    print(f"[*] Output Directory  : {output_dir}")
    print(f"[*] Sequence Length   : {args.seq_len}")
    print(f"[*] Chunk Size        : {args.chunk_size}")
    print(f"[*] Precision         : {'Float16' if args.fp16 else 'Float32'}")
    print(f"[*] Augmentation      : {'BẬT' if args.augment else 'TẮT'}")
    if args.augment:
        print(f"    - Augment Splits  : {augment_splits}")
        print(f"    - Num Aug Copies  : {args.num_aug}")
        print(f"    - Include Original: {args.include_original}")
        print(f"    - Base Seed       : {args.aug_seed}")
    if args.force_recompute:
        print(f"[*] Force Recompute   : TRUE (Bỏ qua cache hiện có)")
    if args.limit:
        print(f"[*] TEST MODE         : Giới hạn xử lý {args.limit} video!")
    print("=" * 75)

    # 1. Phân chia Train/Validation TRƯỚC
    train_items, val_items = prepare_dataset_split(
        dataset_dir=dataset_dir,
        split_json_path=split_json_path,
        train_ratio=0.8,
        seed=42,
        video_exts=video_exts
    )

    if args.limit:
        train_limit = max(1, int(args.limit * 0.8))
        val_limit = max(1, args.limit - train_limit)
        train_items = train_items[:train_limit]
        val_items = val_items[:val_limit]
        print(f"[*] Áp dụng giới hạn: Train={len(train_items)} video, Val={len(val_items)} video.")

    # 2. Khởi tạo ONNX Runtime
    runner = CloneONNXCUDARuntime(str(onnx_path), device_id=args.device_id)

    # 3. Khởi tạo DetectionAugmenter nếu bật augmentation
    augmenter = None
    if args.augment and args.num_aug > 0 and len(augment_splits) > 0:
        aug_cfg = dict(DEFAULT_AUG_CONFIG)
        if args.aug_config:
            aug_cfg_path = Path(args.aug_config)
            if aug_cfg_path.exists():
                with open(aug_cfg_path, "r", encoding="utf-8") as f:
                    custom_cfg = json.load(f)
                    aug_cfg.update(custom_cfg)
                print(f"[+] Đã nạp cấu hình augment tùy chỉnh từ: {aug_cfg_path}")
            else:
                print(f"[WARN] Không tìm thấy file {aug_cfg_path}, dùng cấu hình mặc định từ augment.py")
        augmenter = DetectionAugmenter(aug_cfg)
        print(f"[+] Đã khởi tạo DetectionAugmenter với cấu hình:\n    {aug_cfg}")

    # 4. Trích xuất tuần tự cho Train và Validation
    should_augment_train = args.augment and ("train" in augment_splits)
    process_split_set(
        split_name="train",
        items=train_items,
        runner=runner,
        output_pt_path=train_pt_path,
        cache_dir=cache_dir,
        seq_len=args.seq_len,
        img_size=args.img_size,
        chunk_size=args.chunk_size,
        use_fp16=args.fp16,
        augmenter=augmenter if should_augment_train else None,
        num_aug=args.num_aug if should_augment_train else 0,
        include_original=args.include_original,
        base_seed=args.aug_seed,
        force_recompute=args.force_recompute
    )

    should_augment_val = args.augment and ("val" in augment_splits)
    process_split_set(
        split_name="val",
        items=val_items,
        runner=runner,
        output_pt_path=val_pt_path,
        cache_dir=cache_dir,
        seq_len=args.seq_len,
        img_size=args.img_size,
        chunk_size=args.chunk_size,
        use_fp16=args.fp16,
        augmenter=augmenter if should_augment_val else None,
        num_aug=args.num_aug if should_augment_val else 0,
        include_original=args.include_original,
        base_seed=args.aug_seed,
        force_recompute=args.force_recompute
    )

    print("\n" + "=" * 75)
    print("      [HOÀN TẤT GIAI ĐOẠN 1] TRÍCH XUẤT ĐẶC TRƯNG THÀNH CÔNG!     ")
    print("=" * 75)
    print(f"1. Cấu hình phân bổ : {split_json_path.resolve()}")
    print(f"2. File Train Tensor: {train_pt_path.resolve()}")
    print(f"3. File Val Tensor  : {val_pt_path.resolve()}")
    print("=" * 75)


if __name__ == "__main__":
    main()
