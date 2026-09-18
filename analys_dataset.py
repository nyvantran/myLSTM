#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
File: analys_dataset.py
Mục đích:
    Phân tích toàn diện và chuyên sâu về thời gian (duration, FPS, frame count, sampling rate,
    temporal resolution) của các video trong dataset phục vụ mô hình Deep LSTM (Drowsiness Detection).

Các chức năng chính:
    1. Trích xuất thông số thời gian từng video:
       - Thời lượng chính xác (giây & MM:SS.mmm).
       - Tốc độ khung hình (FPS) và chu kỳ giữa 2 khung hình liên tiếp (Δt tính bằng ms).
       - Tổng số khung hình (Total Frames).
       - Phân tích lấy mẫu theo seq_len (mặc định 120):
         + Tốc độ lấy mẫu hiệu dụng (Effective FPS = seq_len / duration).
         + Khoảng thời gian giữa 2 khung hình đưa vào LSTM (Sampling interval ms).
         + Hành động tương thích thời gian (Subsampling, Padding, Exact Match).
       - Tỷ lệ bitrate theo thời gian (kbps) và dung lượng tiêu thụ (MB/phút).
       - Nhãn (Driving / Drowsiness), Subject, Điều kiện ánh sáng (từ tên file).

    2. Thống kê cấp độ Dataset (Dataset-Level Temporal Statistics):
       - Tổng thời lượng dataset (giờ, phút, giây).
       - Thống kê phân tán: Min, Max, Mean, Std, Median, IQR, Variance, Skewness, Kurtosis.
       - Phân vị chi tiết: 1%, 5%, 10%, 25% (Q1), 50% (Median), 75% (Q3), 90%, 95%, 99%.
       - Phân bổ theo các dải thời gian (Duration Bins / Histogram Intervals).
       - Phân bố các mức FPS trong dataset.

    3. Phân tích tương quan Nhãn (Label-wise Temporal Discrepancy & Bias Detection):
       - Thống kê thời gian riêng cho nhãn Tỉnh táo (Driving) và Buồn ngủ (Drowsiness).
       - Kiểm định giả thuyết thống kê (Two-sample T-test & Mann-Whitney U test) để
         phát hiện rủi ro Shortcut Learning / Data Bias về thời lượng giữa 2 nhãn.

    4. Đánh giá độ phù hợp với Cửa sổ thời gian của LSTM (LSTM Temporal Window Suitability):
       - Thống kê số lượng video cần Padding (lặp frame) vs Subsampling (nén thời gian).
       - Khuyến nghị cấu hình seq_len hoặc sampling strategy tối ưu.

    5. Xuất báo cáo & Trực quan hóa:
       - Báo cáo console chi tiết, bảng biểu rõ ràng.
       - Xuất dữ liệu chi tiết từng video ra file CSV (video_time_analysis.csv).
       - Xuất bản tóm tắt thống kê ra file JSON (dataset_time_summary.json).
       - Xuất tổ hợp biểu đồ độ phân giải cao ra file PNG (dataset_time_analysis.png).
"""

import os
import sys
import time
import json
import math
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Sequence
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

# Đảm bảo console Windows hỗ trợ in UTF-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Nhập các thư viện khoa học dữ liệu
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

# Import cấu hình mặc định nếu có
try:
    from config import TrainConfig
    DEFAULT_DATASET_DIR = TrainConfig.dataset_dir
    DEFAULT_SEQ_LEN = TrainConfig.seq_len
    DEFAULT_VIDEO_EXTS = list(TrainConfig.video_exts)
except Exception:
    DEFAULT_DATASET_DIR = r"D:\Project\AI\dataset\SUST"
    DEFAULT_SEQ_LEN = 120
    DEFAULT_VIDEO_EXTS = [".avi", ".mp4", ".mkv", ".mov"]


# ==============================================================================
# 1. HÀM TRÍCH XUẤT THÔNG TIN TỪ TÊN FILE (METADATA PARSER)
# ==============================================================================
def parse_filename_metadata(video_path: Path) -> Dict[str, Any]:
    """
    Trích xuất siêu dữ liệu (nhãn, đối tượng, điều kiện ánh sáng) từ tên tệp video.
    Hỗ trợ cả định dạng SUST ('sust-d_1-drowsiness-1.mp4') và VBDDD ('subject0-littleBright-driving-1.avi').
    """
    stem = video_path.stem
    name_lower = stem.lower()
    parts = stem.split('-')

    label_str = "unknown"
    label_id = -1
    subject = "unknown"
    condition = "unknown"

    # 1. Kiểm tra nhãn chuẩn (thành phần thứ -2)
    if len(parts) >= 2:
        penultimate = parts[-2].lower()
        if penultimate in ("driving", "alert", "normal"):
            label_str = "driving"
            label_id = 0
        elif penultimate in ("drowsiness", "drowsy"):
            label_str = "drowsiness"
            label_id = 1

    # 2. Fallback nếu chưa tìm thấy nhãn
    if label_id == -1:
        if "drowsiness" in name_lower or "drowsy" in name_lower or stem.startswith("d_") or stem == "d":
            label_str = "drowsiness"
            label_id = 1
        elif "driving" in name_lower or "normal" in name_lower or "alert" in name_lower or stem.startswith("n_") or stem == "n":
            label_str = "driving"
            label_id = 0

    # 3. Phân tích subject và condition
    if len(parts) >= 4:
        # Ví dụ VBDDD: subject0-littleBright-driving-1 -> [subject0, littleBright, driving, 1]
        # Ví dụ SUST: sust-d_1-drowsiness-1 -> [sust, d_1, drowsiness, 1]
        if parts[0].lower() == "sust":
            subject = parts[1]
            condition = "sust_standard"
        else:
            subject = parts[0]
            condition = parts[1]
    elif len(parts) >= 2:
        subject = parts[0]
    else:
        subject = stem

    return {
        "label_str": label_str,
        "label_id": label_id,
        "subject": subject,
        "condition": condition
    }


# ==============================================================================
# 2. HÀM ĐỌC THÔNG TIN THỜI GIAN CỦA 1 VIDEO (PER-VIDEO TIME ANALYZER)
# ==============================================================================
def format_seconds_to_time_str(seconds: float) -> str:
    """Đổi số giây thành định dạng MM:SS.mmm hoặc HH:MM:SS.mmm."""
    if seconds < 0:
        return "00:00.000"
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:06.3f}"
    return f"{mins:02d}:{secs:06.3f}"


def analyze_single_video(args_tuple: Tuple[Path, int]) -> Optional[Dict[str, Any]]:
    """
    Phân tích toàn diện thuộc tính thời gian của một video bằng OpenCV VideoCapture.
    Args:
        args_tuple: (video_path, seq_len)
    Returns:
        Dict thông tin chi tiết hoặc None nếu file hỏng.
    """
    video_path, seq_len = args_tuple
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Kiểm tra tính hợp lệ của video
        if total_frames <= 0 or fps <= 0 or math.isnan(fps) or math.isinf(fps):
            # Thử đếm khung hình thủ công nếu header bị lỗi
            cap = cv2.VideoCapture(str(video_path))
            cnt = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                cnt += 1
            cap.release()
            total_frames = cnt
            if fps <= 0 or math.isnan(fps) or math.isinf(fps):
                fps = 30.0  # Giả định chuẩn 30fps nếu header không ghi

        if total_frames <= 0:
            return None

        duration_sec = total_frames / fps
        frame_interval_ms = (1000.0 / fps) if fps > 0 else 0.0  # Δt giữa 2 frame gốc

        # Tương thích với LSTM chuỗi cố định seq_len
        effective_sampling_fps = (seq_len / duration_sec) if duration_sec > 0 else 0.0
        sampling_interval_ms = (duration_sec * 1000.0 / seq_len) if seq_len > 0 else 0.0
        frame_stride = (total_frames / seq_len) if seq_len > 0 else 1.0

        if total_frames > seq_len:
            temporal_action = "Subsampling"  # Cần nén/bỏ bớt frame
            frame_diff = total_frames - seq_len
        elif total_frames < seq_len:
            temporal_action = "Padding"      # Cần lặp frame/chèn đệm
            frame_diff = seq_len - total_frames
        else:
            temporal_action = "Exact"        # Khớp chính xác 1:1
            frame_diff = 0

        # Thông tin dung lượng và Bitrate theo thời gian
        file_size_bytes = video_path.stat().st_size
        file_size_mb = file_size_bytes / (1024.0 * 1024.0)
        bitrate_kbps = (file_size_bytes * 8.0) / (duration_sec * 1000.0) if duration_sec > 0 else 0.0
        mb_per_minute = (file_size_mb / (duration_sec / 60.0)) if duration_sec > 0 else 0.0

        # Siêu dữ liệu từ tên file
        meta = parse_filename_metadata(video_path)

        return {
            "filename": video_path.name,
            "filepath": str(video_path.resolve()),
            "extension": video_path.suffix.lower(),
            "subject": meta["subject"],
            "condition": meta["condition"],
            "label_str": meta["label_str"],
            "label_id": meta["label_id"],
            "duration_sec": round(duration_sec, 4),
            "duration_formatted": format_seconds_to_time_str(duration_sec),
            "fps": round(fps, 3),
            "total_frames": total_frames,
            "frame_interval_ms": round(frame_interval_ms, 2),
            "effective_sampling_fps": round(effective_sampling_fps, 3),
            "sampling_interval_ms": round(sampling_interval_ms, 2),
            "frame_stride": round(frame_stride, 3),
            "temporal_action": temporal_action,
            "frame_diff_to_seq_len": frame_diff,
            "width": width,
            "height": height,
            "resolution": f"{width}x{height}",
            "aspect_ratio": round(width / height, 2) if height > 0 else 0.0,
            "file_size_mb": round(file_size_mb, 3),
            "bitrate_kbps": round(bitrate_kbps, 2),
            "mb_per_minute": round(mb_per_minute, 2)
        }
    except Exception as e:
        return None


# ==============================================================================
# 3. HÀM QUÉT TOÀN BỘ VIDEO & PHÂN TÍCH ĐA LUỒNG (BATCH SCANNER)
# ==============================================================================
def scan_and_analyze_dataset(
        dataset_dir: Path,
        seq_len: int = 120,
        video_exts: Sequence[str] = (".avi", ".mp4", ".mkv", ".mov"),
        recursive: bool = True,
        num_workers: int = 8
) -> List[Dict[str, Any]]:
    """Quét và phân tích toàn bộ video trong thư mục dataset với ThreadPoolExecutor."""
    normalized_exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in video_exts}

    print(f"\n[+] Đang quét danh sách file video từ: {dataset_dir}...")
    if recursive:
        video_paths = [
            f for f in dataset_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in normalized_exts
        ]
    else:
        video_paths = [
            f for f in dataset_dir.iterdir()
            if f.is_file() and f.suffix.lower() in normalized_exts
        ]

    video_paths = sorted(video_paths)
    total_found = len(video_paths)
    print(f"[+] Đã tìm thấy {total_found} tệp video hợp lệ thuộc các định dạng {sorted(list(normalized_exts))}.")

    if total_found == 0:
        raise FileNotFoundError(f"Không tìm thấy video nào trong thư mục: {dataset_dir}")

    # Chạy trích xuất song song qua ThreadPoolExecutor
    print(f"[+] Bắt đầu trích xuất thông số thời gian ({num_workers} luồng xử lý song song)...")
    tasks = [(p, seq_len) for p in video_paths]
    results: List[Dict[str, Any]] = []

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for res in tqdm(executor.map(analyze_single_video, tasks), total=len(tasks), desc="Phân tích Video", unit="vid"):
            if res is not None:
                results.append(res)

    elapsed = time.time() - t0
    valid_count = len(results)
    corrupted_count = total_found - valid_count
    print(f"[+] Hoàn thành phân tích trong {elapsed:.2f}s ({valid_count / max(elapsed, 0.001):.1f} video/giây)!")
    if corrupted_count > 0:
        print(f"[!] Cảnh báo: Có {corrupted_count} video bị lỗi định dạng hoặc không đọc được frame.")

    return results


# ==============================================================================
# 4. HÀM TỔNG HỢP VÀ THỐNG KÊ TOÀN DIỆN (DATASET TEMPORAL STATISTICS)
# ==============================================================================
def compute_comprehensive_statistics(records: List[Dict[str, Any]], seq_len: int) -> Dict[str, Any]:
    """Tính toán thống kê chi tiết cấp dataset, nhãn, fps, và tương thích LSTM."""
    df = pd.DataFrame(records)

    durations = df["duration_sec"].values
    frames = df["total_frames"].values
    fps_arr = df["fps"].values
    sampling_fps = df["effective_sampling_fps"].values
    sampling_intervals = df["sampling_interval_ms"].values
    frame_intervals = df["frame_interval_ms"].values
    bitrates = df["bitrate_kbps"].values

    total_videos = len(df)
    total_duration_sec = float(np.sum(durations))
    total_frames = int(np.sum(frames))
    total_size_mb = float(np.sum(df["file_size_mb"]))

    # Thời lượng đổi sang định dạng thân thiện
    total_hours = total_duration_sec / 3600.0
    total_minutes = total_duration_sec / 60.0

    # Phân vị thời lượng
    percentiles = {
        "p01": float(np.percentile(durations, 1)),
        "p05": float(np.percentile(durations, 5)),
        "p10": float(np.percentile(durations, 10)),
        "p25_Q1": float(np.percentile(durations, 25)),
        "p50_Median": float(np.percentile(durations, 50)),
        "p75_Q3": float(np.percentile(durations, 75)),
        "p90": float(np.percentile(durations, 90)),
        "p95": float(np.percentile(durations, 95)),
        "p99": float(np.percentile(durations, 99)),
    }
    iqr = percentiles["p75_Q3"] - percentiles["p25_Q1"]

    # Phân bố thời lượng theo các dải thời gian (Duration Bins)
    min_d = float(np.min(durations))
    max_d = float(np.max(durations))

    # Tự động tạo khoảng chia hợp lý
    bin_counts = {}
    if max_d <= 15:
        bins = [0, 3, 5, 8, 10, 12, 15, float('inf')]
        bin_labels = ["< 3s", "3s - 5s", "5s - 8s", "8s - 10s", "10s - 12s", "12s - 15s", "> 15s"]
    elif max_d <= 60:
        bins = [0, 5, 10, 20, 30, 45, 60, float('inf')]
        bin_labels = ["< 5s", "5s - 10s", "10s - 20s", "20s - 30s", "30s - 45s", "45s - 60s", "> 60s"]
    else:
        bins = [0, 10, 30, 60, 120, 300, float('inf')]
        bin_labels = ["< 10s", "10s - 30s", "30s - 1m", "1m - 2m", "2m - 5m", "> 5m"]

    binned = pd.cut(df["duration_sec"], bins=bins, labels=bin_labels, right=False)
    for bl in bin_labels:
        cnt = int((binned == bl).sum())
        pct = (cnt / total_videos) * 100.0 if total_videos > 0 else 0.0
        bin_counts[bl] = {"count": cnt, "percentage": round(pct, 2)}

    # Thống kê FPS
    fps_counts = df["fps"].value_counts().to_dict()
    fps_summary = {
        f"{k:.2f} FPS": {"count": int(v), "percentage": round((v / total_videos) * 100.0, 2)}
        for k, v in sorted(fps_counts.items(), key=lambda x: -x[1])
    }

    # Thống kê hành động khớp chuỗi thời gian (LSTM Alignment)
    action_counts = df["temporal_action"].value_counts().to_dict()
    temporal_actions_summary = {
        act: {
            "count": int(action_counts.get(act, 0)),
            "percentage": round((action_counts.get(act, 0) / total_videos) * 100.0, 2)
        }
        for act in ["Subsampling", "Padding", "Exact"]
    }

    # Phân tích tương quan theo Nhãn (Label-wise Time Analysis)
    label_stats = {}
    labels = df["label_str"].unique()
    for lbl in ["driving", "drowsiness"]:
        if lbl in labels:
            df_lbl = df[df["label_str"] == lbl]
            lbl_dur = df_lbl["duration_sec"].values
            lbl_frames = df_lbl["total_frames"].values
            lbl_fps = df_lbl["fps"].values

            label_stats[lbl] = {
                "count": len(df_lbl),
                "count_percentage": round((len(df_lbl) / total_videos) * 100.0, 2),
                "total_duration_sec": round(float(np.sum(lbl_dur)), 2),
                "total_duration_hours": round(float(np.sum(lbl_dur)) / 3600.0, 3),
                "duration_percentage": round((float(np.sum(lbl_dur)) / max(total_duration_sec, 0.001)) * 100.0, 2),
                "mean_sec": round(float(np.mean(lbl_dur)), 3),
                "std_sec": round(float(np.std(lbl_dur)), 3),
                "median_sec": round(float(np.median(lbl_dur)), 3),
                "min_sec": round(float(np.min(lbl_dur)), 3),
                "max_sec": round(float(np.max(lbl_dur)), 3),
                "mean_frames": round(float(np.mean(lbl_frames)), 1),
                "mean_fps": round(float(np.mean(lbl_fps)), 2),
            }

    # Kiểm định thống kê độ chênh lệch thời lượng giữa 2 nhãn (T-test & Mann-Whitney U test)
    statistical_tests = {}
    if "driving" in labels and "drowsiness" in labels:
        dur_dr = df[df["label_str"] == "driving"]["duration_sec"].values
        dur_dw = df[df["label_str"] == "drowsiness"]["duration_sec"].values

        if len(dur_dr) > 1 and len(dur_dw) > 1:
            # 1. Independent two-sample t-test (Welch's t-test)
            t_stat, t_pval = stats.ttest_ind(dur_dr, dur_dw, equal_var=False)
            # 2. Mann-Whitney U test (phi tham số)
            u_stat, u_pval = stats.mannwhitneyu(dur_dr, dur_dw, alternative='two-sided')

            diff_mean = float(np.mean(dur_dw) - np.mean(dur_dr))
            diff_median = float(np.median(dur_dw) - np.median(dur_dr))

            is_significant = bool((t_pval < 0.05) or (u_pval < 0.05))
            statistical_tests = {
                "t_statistic": round(float(t_stat), 4),
                "t_pvalue": float(t_pval),
                "mann_whitney_u": round(float(u_stat), 2),
                "mann_whitney_pvalue": float(u_pval),
                "difference_mean_drowsy_minus_driving": round(diff_mean, 3),
                "difference_median_drowsy_minus_driving": round(diff_median, 3),
                "statistically_significant_difference": is_significant,
                "assessment": (
                    "CẢNH BÁO NGUY HIỂM: Thời lượng giữa 2 nhãn có chênh lệch có ý nghĩa thống kê (p < 0.05). "
                    "Mô hình LSTM có thể học 'thời lượng video' như một lối tắt (Shortcut Bias) thay vì đặc trưng chuyển động khuôn mặt!"
                    if is_significant else
                    "TỐT: Không có sự chênh lệch thời lượng có ý nghĩa thống kê giữa 2 nhãn (p >= 0.05). "
                    "Dataset đảm bảo tính cân bằng thời gian (Temporal Equivalence)."
                )
            }

    # Tổng hợp toàn bộ chỉ số
    summary = {
        "dataset_overview": {
            "total_videos": total_videos,
            "total_duration_seconds": round(total_duration_sec, 2),
            "total_duration_minutes": round(total_minutes, 2),
            "total_duration_hours": round(total_hours, 3),
            "total_frames": total_frames,
            "total_size_gb": round(total_size_mb / 1024.0, 3),
            "mean_bitrate_kbps": round(float(np.mean(bitrates)), 2),
        },
        "duration_statistics": {
            "min_sec": round(min_d, 3),
            "max_sec": round(max_d, 3),
            "range_sec": round(max_d - min_d, 3),
            "mean_sec": round(float(np.mean(durations)), 3),
            "std_sec": round(float(np.std(durations)), 3),
            "median_sec": round(float(np.median(durations)), 3),
            "variance_sec": round(float(np.var(durations)), 3),
            "iqr_sec": round(iqr, 3),
            "skewness": round(float(stats.skew(durations)), 3),
            "kurtosis": round(float(stats.kurtosis(durations)), 3),
            "percentiles": percentiles
        },
        "duration_bins": bin_counts,
        "fps_statistics": {
            "min_fps": round(float(np.min(fps_arr)), 2),
            "max_fps": round(float(np.max(fps_arr)), 2),
            "mean_fps": round(float(np.mean(fps_arr)), 2),
            "median_fps": round(float(np.median(fps_arr)), 2),
            "frame_interval_ms_mean": round(float(np.mean(frame_intervals)), 2),
            "frame_interval_ms_min": round(float(np.min(frame_intervals)), 2),
            "frame_interval_ms_max": round(float(np.max(frame_intervals)), 2),
            "fps_breakdown": fps_summary
        },
        "lstm_seq_len_compatibility": {
            "target_seq_len": seq_len,
            "actions_breakdown": temporal_actions_summary,
            "effective_sampling_fps": {
                "min": round(float(np.min(sampling_fps)), 2),
                "max": round(float(np.max(sampling_fps)), 2),
                "mean": round(float(np.mean(sampling_fps)), 2),
                "median": round(float(np.median(sampling_fps)), 2)
            },
            "sampling_interval_ms": {
                "min": round(float(np.min(sampling_intervals)), 2),
                "max": round(float(np.max(sampling_intervals)), 2),
                "mean": round(float(np.mean(sampling_intervals)), 2),
                "median": round(float(np.median(sampling_intervals)), 2)
            },
            "stride_original_frames": {
                "min": round(float(np.min(df["frame_stride"])), 2),
                "max": round(float(np.max(df["frame_stride"])), 2),
                "mean": round(float(np.mean(df["frame_stride"])), 2),
            }
        },
        "label_time_breakdown": label_stats,
        "statistical_tests": statistical_tests
    }

    return summary


# ==============================================================================
# 5. HÀM IN BÁO CÁO CONSOLE ĐẸP MẮT (PRETTY TERMINAL REPORTER)
# ==============================================================================
def print_terminal_report(summary: Dict[str, Any], records: List[Dict[str, Any]], show_top: int = 5):
    """In bản báo cáo thống kê trực quan, rõ ràng trên terminal."""
    ov = summary["dataset_overview"]
    ds = summary["duration_statistics"]
    fps_s = summary["fps_statistics"]
    lstm_s = summary["lstm_seq_len_compatibility"]
    lbl_s = summary["label_time_breakdown"]
    stat_s = summary.get("statistical_tests", {})

    print("\n" + "=" * 80)
    print("      BÁO CÁO PHÂN TÍCH THỜI GIAN VIDEO DATASET CHO MÔ HÌNH DEEP LSTM      ")
    print("=" * 80)

    print("\n[1] TỔNG QUAN THỜI LƯỢNG TOÀN BỘ DATASET:")
    print(f"    • Tổng số video phân tích      : {ov['total_videos']:,} video")
    print(f"    • Tổng thời lượng video        : {ov['total_duration_hours']:.2f} giờ ({ov['total_duration_minutes']:.1f} phút / {ov['total_duration_seconds']:,.1f} giây)")
    print(f"    • Tổng số khung hình (Frames)  : {ov['total_frames']:,} frames")
    print(f"    • Tổng dung lượng lưu trữ      : {ov['total_size_gb']:.2f} GB")
    print(f"    • Tốc độ Bitrate trung bình    : {ov['mean_bitrate_kbps']:,.1f} kbps")

    print("\n[2] PHÂN BỐ VÀ ĐỘ PHÂN TÁN THỜI LƯỢNG MỖI VIDEO (DURATION DISTRIBUTION):")
    print(f"    • Thời lượng ngắn nhất (Min)   : {ds['min_sec']:.2f} giây")
    print(f"    • Thời lượng dài nhất (Max)    : {ds['max_sec']:.2f} giây")
    print(f"    • Khoảng biến thiên (Range)    : {ds['range_sec']:.2f} giây")
    print(f"    • Trung bình (Mean ± Std)      : {ds['mean_sec']:.2f} ± {ds['std_sec']:.2f} giây")
    print(f"    • Trung vị (Median / 50th)     : {ds['median_sec']:.2f} giây")
    print(f"    • Độ trải giữa (IQR: Q3 - Q1)  : {ds['iqr_sec']:.2f} giây")
    print(f"    • Hệ số lệch (Skewness)        : {ds['skewness']:+.3f}")
    print(f"    • Hệ số nhọn (Kurtosis)        : {ds['kurtosis']:+.3f}")
    print("    - Các mốc phân vị (Percentiles):")
    p = ds["percentiles"]
    print(f"      + P1%  : {p['p01']:.2f}s | P5%  : {p['p05']:.2f}s | P10% : {p['p10']:.2f}s")
    print(f"      + P25% (Q1): {p['p25_Q1']:.2f}s | P50% (Median): {p['p50_Median']:.2f}s | P75% (Q3): {p['p75_Q3']:.2f}s")
    print(f"      + P90% : {p['p90']:.2f}s | P95% : {p['p95']:.2f}s | P99% : {p['p99']:.2f}s")

    print("\n    - Bảng phân chia dải thời lượng (Duration Intervals):")
    for bin_name, info in summary["duration_bins"].items():
        bar_len = int(info["percentage"] / 2.5)
        bar_str = "█" * bar_len
        print(f"      {bin_name:<10}: {info['count']:>5} video ({info['percentage']:>6.2f}%)  {bar_str}")

    print("\n[3] THÔNG SỐ TỐC ĐỘ KHUNG HÌNH (FPS & TEMPORAL RESOLUTION):")
    print(f"    • Tốc độ FPS (Min - Max)       : {fps_s['min_fps']:.2f} - {fps_s['max_fps']:.2f} FPS (TB: {fps_s['mean_fps']:.2f} FPS)")
    print(f"    • Chu kỳ khung hình gốc (Δt)   : {fps_s['frame_interval_ms_mean']:.1f} ms (Min: {fps_s['frame_interval_ms_min']:.1f}ms, Max: {fps_s['frame_interval_ms_max']:.1f}ms)")
    print("    - Tỷ lệ các mức FPS xuất hiện:")
    for fps_name, info in fps_s["fps_breakdown"].items():
        print(f"      + {fps_name:<10}: {info['count']:>5} video ({info['percentage']:>5.1f}%)")

    print(f"\n[4] TƯƠNG THÍCH CỬA SỔ THỜI GIAN MÔ HÌNH DEEP LSTM (SEQ_LEN = {lstm_s['target_seq_len']}):")
    act = lstm_s["actions_breakdown"]
    print(f"    • Subsampling (Tổng frame > {lstm_s['target_seq_len']} -> Bỏ bớt frame) : {act['Subsampling']['count']:>5} video ({act['Subsampling']['percentage']:.1f}%)")
    print(f"    • Padding     (Tổng frame < {lstm_s['target_seq_len']} -> Lặp frame đệm) : {act['Padding']['count']:>5} video ({act['Padding']['percentage']:.1f}%)")
    print(f"    • Exact Match (Tổng frame = {lstm_s['target_seq_len']} -> Vừa khít 1:1)   : {act['Exact']['count']:>5} video ({act['Exact']['percentage']:.1f}%)")
    eff = lstm_s["effective_sampling_fps"]
    si = lstm_s["sampling_interval_ms"]
    st = lstm_s["stride_original_frames"]
    print(f"    • Tốc độ lấy mẫu hiệu dụng (Effective FPS) : {eff['mean']:.2f} FPS (Min: {eff['min']:.2f}, Max: {eff['max']:.2f})")
    print(f"    • Khoảng thời gian giữa 2 frame LSTM (Δt)  : {si['mean']:.1f} ms (Min: {si['min']:.1f}ms, Max: {si['max']:.1f}ms)")
    print(f"    • Bước nhảy lấy mẫu khung hình gốc (Stride): ~{st['mean']:.2f} frames/bước (Min: {st['min']:.2f}, Max: {st['max']:.2f})")

    print("\n[5] SO SÁNH THỜI GIAN THEO NHÃN & PHÁT HIỆN BIAS (LABEL-WISE COMPARISON):")
    if "driving" in lbl_s and "drowsiness" in lbl_s:
        dr = lbl_s["driving"]
        dw = lbl_s["drowsiness"]
        print(f"    {'-'*75}")
        print(f"    {'Chỉ số':<28} | {'Driving (0: Tỉnh táo)':<20} | {'Drowsiness (1: Buồn ngủ)':<20}")
        print(f"    {'-'*75}")
        print(f"    {'Số lượng video':<28} | {dr['count']:>10} ({dr['count_percentage']:>5.1f}%) | {dw['count']:>10} ({dw['count_percentage']:>5.1f}%)")
        print(f"    {'Tổng thời lượng (giờ)':<28} | {dr['total_duration_hours']:>10.2f}h ({dr['duration_percentage']:>5.1f}%) | {dw['total_duration_hours']:>10.2f}h ({dw['duration_percentage']:>5.1f}%)")
        print(f"    {'Thời lượng TB ± Std (giây)':<28} | {dr['mean_sec']:>8.2f} ± {dr['std_sec']:<5.2f}s | {dw['mean_sec']:>8.2f} ± {dw['std_sec']:<5.2f}s")
        print(f"    {'Trung vị (Median) (giây)':<28} | {dr['median_sec']:>10.2f}s         | {dw['median_sec']:>10.2f}s")
        print(f"    {'Min - Max (giây)':<28} | {dr['min_sec']:>5.1f}s - {dr['max_sec']:<6.1f}s    | {dw['min_sec']:>5.1f}s - {dw['max_sec']:<6.1f}s")
        print(f"    {'Số frames TB / video':<28} | {dr['mean_frames']:>10.1f}           | {dw['mean_frames']:>10.1f}")
        print(f"    {'FPS trung bình':<28} | {dr['mean_fps']:>10.2f}           | {dw['mean_fps']:>10.2f}")
        print(f"    {'-'*75}")

        if stat_s:
            print("\n    * Kết quả kiểm định thống kê (Statistical Hypothesis Testing):")
            print(f"      + Welch's t-test         : t = {stat_s['t_statistic']:+.4f}, p-value = {stat_s['t_pvalue']:.4e}")
            print(f"      + Mann-Whitney U test    : U = {stat_s['mann_whitney_u']}, p-value = {stat_s['mann_whitney_pvalue']:.4e}")
            print(f"      + Chênh lệch trung bình  : {stat_s['difference_mean_drowsy_minus_driving']:+.3f} giây (Drowsiness - Driving)")
            print(f"      + Đánh giá chuyên môn    : {stat_s['assessment']}")

    # Top video ngắn nhất và dài nhất
    sorted_by_dur = sorted(records, key=lambda x: x["duration_sec"])
    print(f"\n[6] TOP {show_top} VIDEO CÓ THỜI LƯỢNG NGẮN NHẤT:")
    for idx, item in enumerate(sorted_by_dur[:show_top], 1):
        print(f"    {idx}. {item['filename']} | Thời lượng: {item['duration_sec']:.2f}s ({item['total_frames']} frames @ {item['fps']} fps) | Nhãn: {item['label_str']}")

    print(f"\n[7] TOP {show_top} VIDEO CÓ THỜI LƯỢNG DÀI NHẤT:")
    for idx, item in enumerate(reversed(sorted_by_dur[-show_top:]), 1):
        print(f"    {idx}. {item['filename']} | Thời lượng: {item['duration_sec']:.2f}s ({item['total_frames']} frames @ {item['fps']} fps) | Nhãn: {item['label_str']}")

    print("=" * 80 + "\n")


# ==============================================================================
# 6. HÀM VẼ BIỂU ĐỒ TRỰC QUAN HÓA (VISUALIZATION GENERATOR)
# ==============================================================================
def generate_temporal_visualizations(
        records: List[Dict[str, Any]],
        summary: Dict[str, Any],
        output_png_path: Path
):
    """Vẽ tổ hợp 6 biểu đồ chuyên sâu về thời gian video trong dataset."""
    df = pd.DataFrame(records)
    seq_len = summary["lstm_seq_len_compatibility"]["target_seq_len"]

    # Cấu hình thẩm mỹ cho matplotlib
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('BÁO CÁO TRỰC QUAN PHÂN TÍCH THỜI GIAN DATASET VIDEO CHO DEEP LSTM', fontsize=16, fontweight='bold', y=0.98)

    color_dr = '#2ecc71'
    color_dw = '#e74c3c'
    color_primary = '#3498db'
    color_purple = '#9b59b6'

    # --- SUBPLOT 1: Phân bố thời lượng video (Histogram + KDE) ---
    ax1 = axes[0, 0]
    durations = df["duration_sec"].values
    mean_dur = float(np.mean(durations))
    median_dur = float(np.median(durations))

    n_bins = min(30, max(10, int(len(np.unique(durations)))))
    ax1.hist(durations, bins=n_bins, color=color_primary, edgecolor='black', alpha=0.75, density=False)
    ax1.axvline(mean_dur, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_dur:.2f}s')
    ax1.axvline(median_dur, color='green', linestyle='-', linewidth=2, label=f'Median: {median_dur:.2f}s')
    ax1.set_title('1. Phân bố Thời lượng Video (Duration Distribution)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Thời lượng (giây)')
    ax1.set_ylabel('Số lượng Video')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- SUBPLOT 2: So sánh thời lượng theo Nhãn (Boxplot) ---
    ax2 = axes[0, 1]
    lbl_data = []
    lbl_names = []
    lbl_colors = []
    for lbl, name, col in [("driving", "Driving\n(0: Tỉnh táo)", color_dr), ("drowsiness", "Drowsiness\n(1: Buồn ngủ)", color_dw)]:
        sub_d = df[df["label_str"] == lbl]["duration_sec"].values
        if len(sub_d) > 0:
            lbl_data.append(sub_d)
            lbl_names.append(name)
            lbl_colors.append(col)

    if lbl_data:
        bp = ax2.boxplot(lbl_data, tick_labels=lbl_names, patch_artist=True, widths=0.45,
                         medianprops=dict(color="black", linewidth=2.5))
        for patch, color in zip(bp['boxes'], lbl_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        ax2.set_title('2. So sánh Thời lượng: Driving vs Drowsiness', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Thời lượng (giây)')
        ax2.grid(True, linestyle='--', alpha=0.5)
    else:
        ax2.text(0.5, 0.5, "Không phân tách được nhãn", ha='center', va='center')

    # --- SUBPLOT 3: Phân bố số khung hình và ranh giới seq_len ---
    ax3 = axes[0, 2]
    frames = df["total_frames"].values
    ax3.hist(frames, bins=min(30, max(10, len(np.unique(frames)))), color=color_purple, edgecolor='black', alpha=0.75)
    ax3.axvline(seq_len, color='red', linestyle='-', linewidth=2.5, label=f'seq_len = {seq_len} frames')
    padding_pct = summary["lstm_seq_len_compatibility"]["actions_breakdown"]["Padding"]["percentage"]
    subsample_pct = summary["lstm_seq_len_compatibility"]["actions_breakdown"]["Subsampling"]["percentage"]
    ax3.set_title(f'3. Số Khung Hình vs Ngưỡng seq_len={seq_len}', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Số lượng Khung hình (Frames)')
    ax3.set_ylabel('Số lượng Video')
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.text(0.05, 0.90, f"Padding (<{seq_len}): {padding_pct}%\nSubsampling (>{seq_len}): {subsample_pct}%",
             transform=ax3.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)

    # --- SUBPLOT 4: Phân bố các mức FPS trong dataset ---
    ax4 = axes[1, 0]
    fps_counts = df["fps"].round(2).value_counts().sort_index()
    fps_labels = [f"{k} FPS" for k in fps_counts.index]
    bars = ax4.bar(fps_labels, fps_counts.values, color='#f39c12', edgecolor='black', width=0.5, alpha=0.85)
    ax4.set_title('4. Phân bố Tốc độ Khung hình (FPS Distribution)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Tốc độ Khung hình')
    ax4.set_ylabel('Số lượng Video')
    ax4.grid(True, linestyle='--', alpha=0.5)
    for b in bars:
        h = b.get_height()
        ax4.text(b.get_x() + b.get_width() / 2.0, h + 5, f"{h}\n({h / len(df) * 100:.1f}%)", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # --- SUBPLOT 5: Tốc độ lấy mẫu hiệu dụng (Effective Sampling FPS) ---
    ax5 = axes[1, 1]
    eff_fps = df["effective_sampling_fps"].values
    ax5.hist(eff_fps, bins=min(25, max(8, len(np.unique(eff_fps)))), color='#16a085', edgecolor='black', alpha=0.75)
    ax5.axvline(np.mean(eff_fps), color='red', linestyle='--', linewidth=2, label=f'TB: {np.mean(eff_fps):.2f} FPS')
    ax5.set_title(f'5. Tốc độ Lấy Mẫu Hiệu Dụng khi vào LSTM (seq_len={seq_len})', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Effective FPS (seq_len / duration)')
    ax5.set_ylabel('Số lượng Video')
    ax5.legend(loc='upper right')
    ax5.grid(True, linestyle='--', alpha=0.5)

    # --- SUBPLOT 6: Tỷ lệ tổng thời lượng giữa Driving vs Drowsiness (Donut Chart) ---
    ax6 = axes[1, 2]
    lbl_s = summary["label_time_breakdown"]
    if "driving" in lbl_s and "drowsiness" in lbl_s:
        sizes = [lbl_s["driving"]["total_duration_sec"], lbl_s["drowsiness"]["total_duration_sec"]]
        labels_donut = [f"Driving\n{lbl_s['driving']['total_duration_hours']:.1f}h ({lbl_s['driving']['duration_percentage']}%)",
                        f"Drowsiness\n{lbl_s['drowsiness']['total_duration_hours']:.1f}h ({lbl_s['drowsiness']['duration_percentage']}%)"]
        wedges, texts = ax6.pie(sizes, labels=labels_donut, colors=[color_dr, color_dw],
                                startangle=140, wedgeprops=dict(width=0.45, edgecolor='black'))
        for t in texts:
            t.set_fontsize(10)
            t.set_fontweight('bold')
        ax6.set_title('6. Tỷ Lệ Tổng Thời Lượng Theo Nhãn', fontsize=12, fontweight='bold')
    else:
        ax6.text(0.5, 0.5, "Không đủ nhãn để vẽ tỷ lệ", ha='center', va='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_png_path), dpi=300)
    plt.close(fig)
    print(f"[+] Đã lưu biểu đồ phân tích trực quan tại: {output_png_path.resolve()}")


# ==============================================================================
# 7. HÀM XUẤT BÁO CÁO CSV VÀ JSON (EXPORTERS)
# ==============================================================================
def export_results(
        records: List[Dict[str, Any]],
        summary: Dict[str, Any],
        output_dir: Path
) -> Tuple[Path, Path]:
    """Xuất bảng chi tiết ra CSV và bảng tóm tắt ra JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "video_time_analysis.csv"
    json_path = output_dir / "dataset_time_summary.json"

    df = pd.DataFrame(records)
    # Sắp xếp các cột logic
    ordered_cols = [
        "filename", "label_str", "label_id", "subject", "condition",
        "duration_sec", "duration_formatted", "fps", "total_frames",
        "frame_interval_ms", "effective_sampling_fps", "sampling_interval_ms",
        "frame_stride", "temporal_action", "frame_diff_to_seq_len",
        "resolution", "aspect_ratio", "file_size_mb", "bitrate_kbps", "mb_per_minute",
        "filepath"
    ]
    existing_cols = [c for c in ordered_cols if c in df.columns]
    df = df[existing_cols]
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[+] Đã xuất báo cáo chi tiết từng video ({len(df)} hàng) ra CSV: {csv_path.resolve()}")

    def _json_serializable(o):
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        if isinstance(o, (np.integer, np.int64, np.int32)):
            return int(o)
        if isinstance(o, (np.floating, np.float64, np.float32)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, default=_json_serializable, ensure_ascii=False, indent=4)
    print(f"[+] Đã xuất dữ liệu tóm tắt phân tích ra JSON: {json_path.resolve()}")

    return csv_path, json_path


# ==============================================================================
# 8. HÀM MAIN & CLI ENTRYPOINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Phân tích chi tiết thời gian và đặc trưng thời gian của các video trong dataset cho Deep LSTM."
    )
    parser.add_argument(
        "--dataset_dir", "-d",
        type=str,
        default=DEFAULT_DATASET_DIR,
        help=f"Đường dẫn thư mục chứa video dataset (Mặc định: {DEFAULT_DATASET_DIR})"
    )
    parser.add_argument(
        "--seq_len", "-s",
        type=int,
        default=DEFAULT_SEQ_LEN,
        help=f"Độ dài chuỗi khung hình cố định cho LSTM (Mặc định: {DEFAULT_SEQ_LEN})"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="evaluation_results/time_analysis",
        help="Thư mục lưu các tệp báo cáo CSV, JSON và ảnh PNG (Mặc định: evaluation_results/time_analysis)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=min(16, max(4, (os.cpu_count() or 4))),
        help="Số luồng đọc file video song song (Mặc định: min(16, cpu_count))"
    )
    parser.add_argument(
        "--no_recursive",
        action="store_true",
        help="Không quét đệ quy các thư mục con"
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Tắt chức năng vẽ biểu đồ PNG"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Số video dài nhất/ngắn nhất hiển thị trong terminal (Mặc định: 5)"
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        print(f"[!] Lỗi: Thư mục dataset không tồn tại: {dataset_path.resolve()}")
        sys.exit(1)

    output_path = Path(args.output_dir)

    # 1. Quét và phân tích toàn bộ video
    records = scan_and_analyze_dataset(
        dataset_dir=dataset_path,
        seq_len=args.seq_len,
        video_exts=DEFAULT_VIDEO_EXTS,
        recursive=not args.no_recursive,
        num_workers=args.workers
    )

    if not records:
        print("[!] Không có dữ liệu video hợp lệ để phân tích.")
        sys.exit(1)

    # 2. Tính toán thống kê toàn diện
    summary = compute_comprehensive_statistics(records, seq_len=args.seq_len)

    # 3. In kết quả console
    print_terminal_report(summary, records, show_top=args.top_k)

    # 4. Xuất file CSV và JSON
    csv_file, json_file = export_results(records, summary, output_path)

    # 5. Sinh biểu đồ nếu được bật
    if not args.no_plot:
        png_path = output_path / "dataset_time_analysis.png"
        generate_temporal_visualizations(records, summary, png_path)

    print("\n[+] Hoàn tất toàn bộ phân tích thời gian dataset!")


if __name__ == "__main__":
    main()
