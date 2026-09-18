#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
File: pre_process_data.py
Mục đích:
    Tiền xử lý và phân loại tập dữ liệu video dựa trên ngưỡng thời gian (Duration Threshold):
    - Chia dataset thành 2 thư mục riêng biệt:
        1. Thư mục chứa các video NẰM TRONG ngưỡng thời gian (In-Threshold / Valid Duration).
        2. Thư mục chứa các video KHÔNG NẰM TRONG ngưỡng thời gian (Out-Threshold / Invalid Duration).
    - Hỗ trợ các phương thức: Sao chép an toàn (copy), Di chuyển (move), hoặc Tạo liên kết cứng (hardlink - không tốn dung lượng ổ đĩa).
    - Xuất tệp manifest CSV & JSON chi tiết phục vụ kiểm tra và truy vết dữ liệu.
"""

import os
import sys
import time
import shutil
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Sequence
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

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Nạp cấu hình mặc định từ config.py nếu có
try:
    from config import TrainConfig
    DEFAULT_DATASET_DIR = TrainConfig.dataset_dir
    DEFAULT_VIDEO_EXTS = list(TrainConfig.video_exts)
except Exception:
    DEFAULT_DATASET_DIR = r"D:\Project\AI\dataset\SUST"
    DEFAULT_VIDEO_EXTS = [".avi", ".mp4", ".mkv", ".mov"]


# ==============================================================================
# 1. HÀM ĐỌC THÔNG TIN THỜI LƯỢNG VIDEO
# ==============================================================================
def get_video_duration_info(video_path: Path) -> Dict[str, Any]:
    """
    Đọc nhanh thông số thời gian của video thông qua OpenCV VideoCapture.
    Returns:
        Dict chứa: duration_sec, fps, total_frames, is_valid, error_msg.
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {
                "duration_sec": 0.0,
                "fps": 0.0,
                "total_frames": 0,
                "is_valid": False,
                "error_msg": "Không thể mở tệp video qua OpenCV"
            }

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        # Fallback nếu header không ghi số khung hình hoặc FPS không hợp lệ
        if total_frames <= 0 or fps <= 0 or np.isnan(fps) or np.isinf(fps):
            cap = cv2.VideoCapture(str(video_path))
            cnt = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                cnt += 1
            cap.release()
            total_frames = cnt
            if fps <= 0 or np.isnan(fps) or np.isinf(fps):
                fps = 30.0  # Mặc định 30 FPS nếu không trích xuất được

        if total_frames <= 0:
            return {
                "duration_sec": 0.0,
                "fps": fps,
                "total_frames": 0,
                "is_valid": False,
                "error_msg": "Video rỗng (0 frames)"
            }

        duration_sec = total_frames / fps
        return {
            "duration_sec": round(duration_sec, 4),
            "fps": round(fps, 3),
            "total_frames": total_frames,
            "is_valid": True,
            "error_msg": ""
        }
    except Exception as e:
        return {
            "duration_sec": 0.0,
            "fps": 0.0,
            "total_frames": 0,
            "is_valid": False,
            "error_msg": str(e)
        }


# ==============================================================================
# 2. HÀM KIỂM TRA ĐIỀU KIỆN NGƯỠNG THỜI GIAN
# ==============================================================================
def check_duration_threshold(
        duration: float,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        exact_threshold: Optional[float] = None,
        mode: str = "within"
) -> Tuple[bool, str]:
    """
    Kiểm tra thời lượng video có thỏa mãn ngưỡng thời gian hay không.
    Args:
        duration: Thời lượng tính bằng giây.
        min_duration: Ngưỡng tối thiểu (giây).
        max_duration: Ngưỡng tối đa (giây).
        exact_threshold: Ngưỡng đơn lẻ.
        mode: Chế độ kiểm tra ('within', 'gte', 'lte', 'eq').
    Returns:
        (is_in_threshold, reason_str)
    """
    if mode == "gte":
        thresh = exact_threshold if exact_threshold is not None else min_duration
        if thresh is None:
            return True, "Không cấu hình ngưỡng"
        if duration >= thresh:
            return True, f"Thời lượng ({duration:.2f}s) >= ngưỡng ({thresh:.2f}s)"
        return False, f"Thời lượng ({duration:.2f}s) < ngưỡng ({thresh:.2f}s)"

    elif mode == "lte":
        thresh = exact_threshold if exact_threshold is not None else max_duration
        if thresh is None:
            return True, "Không cấu hình ngưỡng"
        if duration <= thresh:
            return True, f"Thời lượng ({duration:.2f}s) <= ngưỡng ({thresh:.2f}s)"
        return False, f"Thời lượng ({duration:.2f}s) > ngưỡng ({thresh:.2f}s)"

    # Mặc định mode = "within" (khoảng [min_duration, max_duration])
    min_val = min_duration if min_duration is not None else (exact_threshold if exact_threshold is not None else 0.0)
    max_val = max_duration if max_duration is not None else float("inf")

    if min_val <= duration <= max_val:
        return True, f"Thời lượng ({duration:.2f}s) nằm trong khoảng [{min_val:.2f}s, {max_val:.2f}s]"
    else:
        if duration < min_val:
            return False, f"Thời lượng ({duration:.2f}s) nhỏ hơn ngưỡng tối thiểu ({min_val:.2f}s)"
        return False, f"Thời lượng ({duration:.2f}s) vượt quá ngưỡng tối đa ({max_val:.2f}s)"


# ==============================================================================
# 3. HÀM THỰC THI CHUYỂN / SAO CHÉP TỆP (FILE TRANSFER WORKER)
# ==============================================================================
def transfer_file(
        src_path: Path,
        dst_path: Path,
        action: str = "copy"
) -> Tuple[bool, str]:
    """
    Thực hiện sao chép, di chuyển, hoặc tạo hardlink an toàn từ src_path sang dst_path.
    """
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # Tránh ghi đè nếu tệp đích đã tồn tại
        final_dst = dst_path
        counter = 1
        while final_dst.exists():
            final_dst = dst_path.parent / f"{dst_path.stem}_{counter}{dst_path.suffix}"
            counter += 1

        if action == "copy":
            shutil.copy2(str(src_path), str(final_dst))
        elif action == "move":
            shutil.move(str(src_path), str(final_dst))
        elif action == "hardlink":
            os.link(str(src_path), str(final_dst))
        else:
            return False, f"Hành động không hợp lệ: {action}"

        return True, str(final_dst)
    except Exception as e:
        return False, str(e)


# ==============================================================================
# 4. HÀM CHÍNH THỰC HIỆN CHIA DATASET (PIPELINE)
# ==============================================================================
def process_dataset_split(
        dataset_dir: Path,
        output_dir: Path,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        exact_threshold: Optional[float] = None,
        mode: str = "within",
        action: str = "copy",
        in_folder_name: str = "in_threshold",
        out_folder_name: str = "out_threshold",
        video_exts: Sequence[str] = (".avi", ".mp4", ".mkv", ".mov"),
        recursive: bool = True,
        num_workers: int = 8,
        dry_run: bool = False
) -> Dict[str, Any]:
    """
    Quét toàn bộ video, trích xuất thời lượng và chia vào 2 thư mục tương ứng.
    """
    normalized_exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in video_exts}

    print("\n" + "=" * 80)
    print("      TIỀN XỬ LÝ & CHIA DATASET THEO NGƯỠNG THỜI GIAN VIDEO      ")
    print("=" * 80)
    print(f"[+] Thư mục nguồn (Input Dataset)   : {dataset_dir.resolve()}")
    print(f"[+] Thư mục đích (Output Root)     : {output_dir.resolve()}")
    print(f"    • Folder trong ngưỡng (In)      : {in_folder_name}/")
    print(f"    • Folder ngoài ngưỡng (Out)     : {out_folder_name}/")
    print(f"[+] Cấu hình ngưỡng thời lượng      :")
    if mode == "within":
        print(f"    • Chế độ: Nằm trong khoảng [{min_duration if min_duration is not None else 0.0}s, {max_duration if max_duration is not None else 'inf'}s]")
    elif mode == "gte":
        thresh = exact_threshold if exact_threshold is not None else min_duration
        print(f"    • Chế độ: Lớn hơn hoặc bằng >= {thresh}s")
    elif mode == "lte":
        thresh = exact_threshold if exact_threshold is not None else max_duration
        print(f"    • Chế độ: Nhỏ hơn hoặc bằng <= {thresh}s")
    print(f"[+] Phương thức xử lý tệp          : {action.upper()} {'(MÔ PHỎNG - DRY RUN)' if dry_run else ''}")
    print(f"[+] Số luồng xử lý song song       : {num_workers}")
    print("=" * 80 + "\n")

    # 1. Quét danh sách file
    print(f"[+] Đang quét tệp video từ: {dataset_dir}...")
    if recursive:
        video_files = [f for f in dataset_dir.rglob("*") if f.is_file() and f.suffix.lower() in normalized_exts]
    else:
        video_files = [f for f in dataset_dir.iterdir() if f.is_file() and f.suffix.lower() in normalized_exts]

    video_files = sorted(video_files)
    total_videos = len(video_files)
    print(f"[+] Đã tìm thấy tổng cộng {total_videos} video hợp lệ.")

    if total_videos == 0:
        raise FileNotFoundError(f"Không tìm thấy video nào trong: {dataset_dir}")

    # 2. Đo thời lượng từng video bằng đa luồng
    print(f"[+] Đang đọc thông số thời lượng các video ({num_workers} luồng)...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        duration_results = list(tqdm(executor.map(get_video_duration_info, video_files),
                                     total=total_videos, desc="Phân tích thời lượng", unit="vid"))
    read_elapsed = time.time() - t0
    print(f"[+] Đã phân tích xong {total_videos} video trong {read_elapsed:.2f}s ({total_videos / max(read_elapsed, 0.001):.1f} vid/s)!")

    # 3. Phân loại theo ngưỡng
    in_dir = output_dir / in_folder_name
    out_dir = output_dir / out_folder_name

    records = []
    in_records = []
    out_records = []
    corrupted_records = []

    for vid_path, dur_info in zip(video_files, duration_results):
        if not dur_info["is_valid"]:
            is_in = False
            reason = f"Lỗi đọc video: {dur_info['error_msg']}"
            category = "corrupted_or_invalid"
            target_folder = out_dir
        else:
            is_in, reason = check_duration_threshold(
                duration=dur_info["duration_sec"],
                min_duration=min_duration,
                max_duration=max_duration,
                exact_threshold=exact_threshold,
                mode=mode
            )
            category = "in_threshold" if is_in else "out_threshold"
            target_folder = in_dir if is_in else out_dir

        dest_file = target_folder / vid_path.name
        record = {
            "filename": vid_path.name,
            "original_path": str(vid_path.resolve()),
            "destination_path": str(dest_file.resolve()),
            "category": category,
            "is_in_threshold": is_in,
            "duration_sec": dur_info["duration_sec"],
            "fps": dur_info["fps"],
            "total_frames": dur_info["total_frames"],
            "file_size_mb": round(vid_path.stat().st_size / (1024.0 * 1024.0), 3),
            "reason": reason,
            "transfer_status": "dry_run" if dry_run else "pending"
        }
        records.append(record)

        if not dur_info["is_valid"]:
            corrupted_records.append(record)
        elif is_in:
            in_records.append(record)
        else:
            out_records.append(record)

    in_count = len(in_records)
    out_count = len(out_records)
    corrupted_count = len(corrupted_records)

    print("\n" + "-" * 60)
    print("           KẾT QUẢ PHÂN LOẠI THEO NGƯỠNG")
    print("-" * 60)
    print(f"  • Video NẰM TRONG ngưỡng ({in_folder_name})  : {in_count:>5} video ({in_count / total_videos * 100:>5.1f}%)")
    print(f"  • Video NGOÀI ngưỡng ({out_folder_name})     : {out_count:>5} video ({out_count / total_videos * 100:>5.1f}%)")
    if corrupted_count > 0:
        print(f"  • Video lỗi/hỏng (chuyển vào out)           : {corrupted_count:>5} video ({corrupted_count / total_videos * 100:>5.1f}%)")
    print("-" * 60)

    # 4. Thực thi Sao chép / Di chuyển / Hardlink
    if not dry_run:
        print(f"\n[+] Bắt đầu thực hiện hành động '{action.upper()}' vào các thư mục tương ứng...")
        transfer_tasks = [
            (Path(r["original_path"]), Path(r["destination_path"]), action)
            for r in records
        ]

        def _exec_transfer(task):
            src, dst, act = task
            ok, msg = transfer_file(src, dst, act)
            return ok, msg

        t_transfer_0 = time.time()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            transfer_results = list(tqdm(executor.map(_exec_transfer, transfer_tasks),
                                         total=len(transfer_tasks), desc=f"Đang {action}", unit="file"))
        t_transfer_elapsed = time.time() - t_transfer_0

        # Cập nhật kết quả transfer vào record
        success_count = 0
        fail_count = 0
        for r, (ok, msg) in zip(records, transfer_results):
            if ok:
                r["destination_path"] = msg
                r["transfer_status"] = "success"
                success_count += 1
            else:
                r["transfer_status"] = f"failed: {msg}"
                fail_count += 1

        print(f"[+] Hoàn tất xử lý tệp trong {t_transfer_elapsed:.2f}s ({success_count} thành công, {fail_count} thất bại)!")
    else:
        print("\n[*] Chế độ DRY RUN: Không tạo hay sao chép tệp thực tế.")

    # 5. Xuất báo cáo Manifest CSV & Summary JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = output_dir / "split_manifest.csv"
    summary_json = output_dir / "split_summary.json"

    df_manifest = pd.DataFrame(records)
    df_manifest.to_csv(manifest_csv, index=False, encoding="utf-8-sig")
    print(f"[+] Đã lưu danh sách chi tiết (Manifest CSV): {manifest_csv.resolve()}")

    in_durations = [r["duration_sec"] for r in in_records]
    out_durations = [r["duration_sec"] for r in out_records]

    summary_data = {
        "metadata": {
            "source_dir": str(dataset_dir.resolve()),
            "output_dir": str(output_dir.resolve()),
            "in_folder": in_folder_name,
            "out_folder": out_folder_name,
            "action": action,
            "dry_run": dry_run,
            "threshold_config": {
                "mode": mode,
                "min_duration": min_duration,
                "max_duration": max_duration,
                "exact_threshold": exact_threshold
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "statistics": {
            "total_videos": total_videos,
            "in_threshold_count": in_count,
            "in_threshold_percentage": round(in_count / total_videos * 100.0, 2),
            "in_total_duration_sec": round(float(np.sum(in_durations)), 2) if in_durations else 0.0,
            "in_mean_duration_sec": round(float(np.mean(in_durations)), 3) if in_durations else 0.0,
            "in_min_duration_sec": round(float(np.min(in_durations)), 3) if in_durations else 0.0,
            "in_max_duration_sec": round(float(np.max(in_durations)), 3) if in_durations else 0.0,
            "out_threshold_count": out_count,
            "out_threshold_percentage": round(out_count / total_videos * 100.0, 2),
            "out_total_duration_sec": round(float(np.sum(out_durations)), 2) if out_durations else 0.0,
            "out_mean_duration_sec": round(float(np.mean(out_durations)), 3) if out_durations else 0.0,
            "out_min_duration_sec": round(float(np.min(out_durations)), 3) if out_durations else 0.0,
            "out_max_duration_sec": round(float(np.max(out_durations)), 3) if out_durations else 0.0,
            "corrupted_count": corrupted_count
        }
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=4)
    print(f"[+] Đã lưu tổng hợp kết quả (Summary JSON) : {summary_json.resolve()}")

    print("\n[+] Hoàn tất toàn bộ quy trình tiền xử lý chia dataset!\n")
    return summary_data


# ==============================================================================
# 5. CLI ENTRYPOINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Tiền xử lý và chia dataset video thành 2 thư mục dựa trên ngưỡng thời gian."
    )
    parser.add_argument(
        "--dataset_dir", "-i",
        type=str,
        default=DEFAULT_DATASET_DIR,
        help=f"Đường dẫn thư mục chứa video dataset nguồn (Mặc định: {DEFAULT_DATASET_DIR})"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default=r"D:\Project\AI\dataset\filtered_by_time",
        help="Thư mục gốc lưu 2 folder kết quả (Mặc định: D:\\Project\\AI\\dataset\\filtered_by_time)"
    )
    parser.add_argument(
        "--min_duration", "--min_time",
        type=float,
        default=None,
        help="Ngưỡng thời lượng tối thiểu (giây). Các video < min_duration sẽ bị xếp vào ngoài ngưỡng."
    )
    parser.add_argument(
        "--max_duration", "--max_time",
        type=float,
        default=None,
        help="Ngưỡng thời lượng tối đa (giây). Các video > max_duration sẽ bị xếp vào ngoài ngưỡng."
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=None,
        help="Ngưỡng thời gian đơn lẻ (kết hợp với --mode 'gte', 'lte', hoặc 'within')."
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["within", "gte", "lte"],
        default="within",
        help="Chế độ kiểm tra: 'within' (nằm trong khoảng [min, max]), 'gte' (>= ngưỡng), 'lte' (<= ngưỡng). Mặc định: within."
    )
    parser.add_argument(
        "--action", "-a",
        type=str,
        choices=["copy", "move", "hardlink"],
        default="copy",
        help="Phương thức xử lý tệp: 'copy' (sao chép an toàn), 'move' (di chuyển), 'hardlink' (liên kết cứng tức thì, 0 tốn dung lượng). Mặc định: copy."
    )
    parser.add_argument(
        "--in_folder",
        type=str,
        default="in_threshold",
        help="Tên thư mục chứa video trong ngưỡng (Mặc định: in_threshold)."
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        default="out_threshold",
        help="Tên thư mục chứa video ngoài ngưỡng (Mặc định: out_threshold)."
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=min(16, max(4, (os.cpu_count() or 4))),
        help="Số luồng đọc và xử lý song song (Mặc định: min(16, cpu_count))."
    )
    parser.add_argument(
        "--no_recursive",
        action="store_true",
        help="Không quét đệ quy các thư mục con."
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Chạy thử mô phỏng kiểm tra số lượng và thời lượng video mà KHÔNG copy/move tệp thật."
    )

    args = parser.parse_args()

    # Kiểm tra tính hợp lệ của ngưỡng
    if args.min_duration is None and args.max_duration is None and args.threshold is None:
        # Nếu người dùng không nhập tham số ngưỡng, gợi ý giá trị mặc định chuẩn theo LSTM seq_len
        print("[!] Không có tham số ngưỡng được chỉ định. Tự động áp dụng ngưỡng mặc định: [4.0s, 15.0s]")
        min_dur = 4.0
        max_dur = 15.0
    else:
        min_dur = args.min_duration
        max_dur = args.max_duration

    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        print(f"[!] Lỗi: Thư mục dataset nguồn không tồn tại: {dataset_path.resolve()}")
        sys.exit(1)

    output_path = Path(args.output_dir)

    process_dataset_split(
        dataset_dir=dataset_path,
        output_dir=output_path,
        min_duration=min_dur,
        max_duration=max_dur,
        exact_threshold=args.threshold,
        mode=args.mode,
        action=args.action,
        in_folder_name=args.in_folder,
        out_folder_name=args.out_folder,
        video_exts=DEFAULT_VIDEO_EXTS,
        recursive=not args.no_recursive,
        num_workers=args.workers,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
