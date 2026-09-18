#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
File: multi_gpu_worker.py
Mục đích:
    Worker độc lập phục vụ trích xuất đặc trưng song song đa GPU CUDA
    (Multi-GPU Parallel Feature Extraction).
    Hỗ trợ cả môi trường Windows (mp.spawn) và Linux / Kaggle / Colab.
"""

import os
import sys
import time
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# Đảm bảo console Windows hỗ trợ UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F
import numpy as np
import cv2

# Nạp CUDA DLL của PyTorch nếu chạy trên Windows
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.path.exists(torch_lib_path) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(torch_lib_path)
    except Exception:
        pass

import onnxruntime as ort

from augment import DetectionAugmenter, config as DEFAULT_AUG_CONFIG


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
    """Quản lý suy luận ONNX Runtime trên GPU CUDA với Zero-Copy I/O Binding."""

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

        self.session = ort.InferenceSession(
            self.onnx_model_path,
            sess_options=self.session_options,
            providers=providers,
        )

        active = self.session.get_providers()
        if "CUDAExecutionProvider" not in active:
            raise RuntimeError(f"CUDAExecutionProvider chưa được kích hoạt trên GPU {self.device_id}! Providers: {active}")

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


def read_and_sample_video_frames(
        video_path: Path,
        seq_len: int = 120,
        img_size: int = 480
) -> Tuple[List[np.ndarray], int]:
    """
    Đọc video tuần tự, letterbox sang kích thước cố định và định dạng RGB (uint8).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    if total_frames <= 0:
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
    """
    seq_len = len(frames_rgb)
    tensor_list = [
        torch.from_numpy(np.ascontiguousarray(f.transpose(2, 0, 1))).float() / 255.0
        for f in frames_rgb
    ]
    video_tensor = torch.stack(tensor_list, dim=0)

    p3_chunks, p4_chunks, p5_chunks = [], [], []
    for i in range(0, seq_len, chunk_size):
        chunk = video_tensor[i: i + chunk_size].to(device, non_blocking=True)
        outs = runner.forward(chunk)
        p3_chunks.append(F.adaptive_avg_pool2d(outs[0], (1, 1)).flatten(1).cpu())
        p4_chunks.append(F.adaptive_avg_pool2d(outs[1], (1, 1)).flatten(1).cpu())
        p5_chunks.append(F.adaptive_avg_pool2d(outs[2], (1, 1)).flatten(1).cpu())

    p3_tensor = torch.cat(p3_chunks, dim=0)
    p4_tensor = torch.cat(p4_chunks, dim=0)
    p5_tensor = torch.cat(p5_chunks, dim=0)
    return p3_tensor, p4_tensor, p5_tensor


def gpu_worker_loop(
        worker_id: int,
        gpu_id: int,
        task_queue: Any,
        result_queue: Any,
        onnx_model_path: str,
        split_cache_dir_str: str,
        seq_len: int = 120,
        img_size: int = 480,
        chunk_size: int = 24,
        augment_enabled: bool = False,
        aug_cfg: Optional[Dict[str, Any]] = None,
        num_aug: int = 0,
        include_original: bool = True,
        base_seed: int = 42,
        force_recompute: bool = False
):
    """
    Hàm thực thi của mỗi tiến trình Worker gắn với một GPU CUDA xác định.
    Khởi tạo ONNX Runtime & Augmenter một lần duy nhất, sau đó xử lý liên tục các video từ task_queue.
    """
    try:
        # Thiết lập CUDA device hiện tại cho tiến trình này
        torch.cuda.set_device(gpu_id)
        device_str = f"cuda:{gpu_id}"

        # 1. Khởi tạo ONNX InferenceSession trên GPU được chỉ định
        runner = CloneONNXCUDARuntime(onnx_model_path=onnx_model_path, device_id=gpu_id)

        # Khởi động (Warm-up) ONNX Engine trên GPU
        dummy = torch.zeros(chunk_size, 3, img_size, img_size, device=device_str)
        runner.forward(dummy)
        del dummy
        torch.cuda.empty_cache()

        # 2. Khởi tạo Augmenter nếu bật
        augmenter = None
        if augment_enabled and num_aug > 0:
            cfg = dict(DEFAULT_AUG_CONFIG)
            if aug_cfg:
                cfg.update(aug_cfg)
            augmenter = DetectionAugmenter(cfg)

        split_cache_dir = Path(split_cache_dir_str)
        split_cache_dir.mkdir(parents=True, exist_ok=True)

        is_augmented_split = (augmenter is not None and num_aug > 0)
        actual_include_original = include_original if is_augmented_split else True

        # Báo cáo worker đã sẵn sàng
        result_queue.put({
            "type": "ready",
            "worker_id": worker_id,
            "gpu_id": gpu_id
        })

        # 3. Vòng lặp lấy task từ Queue
        while True:
            task = task_queue.get()
            if task is None:
                # Tín hiệu dừng từ main process
                break

            video_path_str, label = task
            video_path = Path(video_path_str)
            video_id = video_path.stem

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
                    aug_seed = (base_seed + int(hashlib.md5(aug_id.encode("utf-8")).hexdigest()[:8], 16)) % (2 ** 31 - 1)
                    targets.append({
                        "sub_id": aug_id,
                        "is_aug": True,
                        "aug_idx": k,
                        "seed": aug_seed,
                    })

            # Kiểm tra cache
            needed_targets = []
            for t in targets:
                cache_file = split_cache_dir / f"{t['sub_id']}.pt"
                if force_recompute or not cache_file.exists():
                    needed_targets.append(t)

            if len(needed_targets) == 0:
                # Đã có đầy đủ trong cache
                result_queue.put({
                    "type": "done",
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "video_id": video_id,
                    "status": "cached",
                    "num_targets": len(targets)
                })
                continue

            # Đọc video và trích xuất
            try:
                raw_frames, actual_frames = read_and_sample_video_frames(
                    video_path=video_path,
                    seq_len=seq_len,
                    img_size=img_size
                )

                for t in needed_targets:
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
                            device=device_str
                        )
                    else:
                        p3, p4, p5 = forward_video_chunks(
                            frames_rgb=raw_frames,
                            runner=runner,
                            chunk_size=chunk_size,
                            device=device_str
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

                result_queue.put({
                    "type": "done",
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "video_id": video_id,
                    "status": "extracted",
                    "num_targets": len(targets)
                })

            except Exception as e:
                result_queue.put({
                    "type": "error",
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "video_id": video_id,
                    "error": str(e)
                })

    except Exception as fatal_e:
        result_queue.put({
            "type": "fatal_error",
            "worker_id": worker_id,
            "gpu_id": gpu_id,
            "error": str(fatal_e)
        })
