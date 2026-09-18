# ----------------------------- Augmentation ----------------------------- #
import inspect
import os
import sys
import random
from typing import List, Tuple, Optional

# Đảm bảo console Windows hỗ trợ in tiếng Việt UTF-8 không bị lỗi charmap
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ngăn chặn xung đột OpenMP runtime trên Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import albumentations as A
import cv2
import numpy as np


def _make_gauss_noise(var_limit, p):
    params = inspect.signature(A.GaussNoise.__init__).parameters
    if "var_limit" in params:
        return A.GaussNoise(var_limit=var_limit, p=p)
    if "std_range" in params:
        var_min, var_max = var_limit
        std_min = max(0.0, min(1.0, (var_min ** 0.5) / 255.0))
        std_max = max(0.0, min(1.0, (var_max ** 0.5) / 255.0))
        return A.GaussNoise(std_range=(std_min, std_max), p=p)
    raise RuntimeError("Phiên bản albumentations hiện tại không hỗ trợ tham số GaussNoise đã biết.")


def _make_shift_scale_rotate(shift_limit, scale_limit, rotate_limit, p, fill_color=(114, 114, 114)):
    # Ưu tiên A.Affine trên Albumentations phiên bản mới để tránh cảnh báo UserWarning
    if hasattr(A, "Affine"):
        scale = (1.0 - scale_limit, 1.0 + scale_limit) if isinstance(scale_limit, (int, float)) else (
            1.0 + scale_limit[0], 1.0 + scale_limit[1])
        translate = (-shift_limit, shift_limit) if isinstance(shift_limit, (int, float)) else shift_limit
        rotate = (-rotate_limit, rotate_limit) if isinstance(rotate_limit, (int, float)) else rotate_limit
        return A.Affine(scale=scale, translate_percent=translate, rotate=rotate,
                        border_mode=cv2.BORDER_CONSTANT, fill=fill_color, p=p)

    params = inspect.signature(A.ShiftScaleRotate.__init__).parameters
    kwargs = dict(shift_limit=shift_limit, scale_limit=scale_limit, rotate_limit=rotate_limit,
                  border_mode=cv2.BORDER_CONSTANT, p=p)
    kwargs["value" if "value" in params else "fill"] = fill_color
    return A.ShiftScaleRotate(**kwargs)


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


class DetectionAugmenter:
    """
    Pipeline tăng cường dữ liệu cho Object Detection và Video.
    Hỗ trợ tính nhất quán theo thời gian (temporal consistency) cho chuỗi video
    bằng cách dùng chung 1 random seed cho tất cả các frames trong cùng một video clip.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        shift_limit, scale_limit, rotate_limit, ssr_p = cfg["shiftScaleRotate"]
        hue_shift, sat_shift, val_shift, hsv_p = cfg["hueSaturationValue"]
        var_min, var_max, gn_p = cfg["gaussNoise"]
        blur_limit, blur_p = cfg["blur"]
        p = cfg["p"]

        self.transform: A.Compose = A.Compose(
            [
                A.HorizontalFlip(p=cfg["horizontalFlip"]),
                _make_shift_scale_rotate(shift_limit, scale_limit, rotate_limit, ssr_p),
                A.RandomBrightnessContrast(p=cfg["randomBrightnessContrast"]),
                A.HueSaturationValue(hue_shift_limit=hue_shift, sat_shift_limit=sat_shift,
                                     val_shift_limit=val_shift, p=hsv_p),
                _make_gauss_noise((var_min, var_max), p=gn_p),
                A.Blur(blur_limit=blur_limit, p=blur_p),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"], min_visibility=0.4),
            p=p
        )

        self.video_seed: Optional[int] = None

    def set_seed(self, seed: int):
        """Thiết lập random seed đồng bộ cho Python random, NumPy và Albumentations."""
        random.seed(seed)
        np.random.seed(seed)
        if hasattr(self.transform, "set_random_seed"):
            self.transform.set_random_seed(seed)

    def set_video_seed(self, seed: Optional[int] = None) -> int:
        """
        Cố định một random seed cho toàn bộ video tiếp theo.
        Nếu seed=None, tự động sinh ngẫu nhiên một seed mới.
        """
        if seed is None:
            seed = random.randint(0, 2 ** 31 - 1)
        self.video_seed = seed
        return self.video_seed

    def reset_video_seed(self):
        """Hủy seed video cố định để quay lại chế độ augment ngẫu nhiên độc lập từng frame."""
        self.video_seed = None

    def __call__(self, image: np.ndarray, boxes=None, labels=None, seed: Optional[int] = None):
        """
        Áp dụng transform cho 1 frame.
        Nếu seed (hoặc self.video_seed) được chỉ định, áp dụng seed này trước khi biến đổi.
        """
        effective_seed = seed if seed is not None else self.video_seed
        if effective_seed is not None:
            self.set_seed(effective_seed)

        if boxes is None:
            boxes = []
        if labels is None:
            labels = []

        boxes = np.array(boxes, dtype=np.float32).tolist() if len(boxes) > 0 else []
        labels = np.array(labels, dtype=np.int64).tolist() if len(labels) > 0 else []

        try:
            out = self.transform(image=image, bboxes=boxes, category_ids=labels)
            out_boxes = out["bboxes"]
            out_labels = [int(c) for c in out["category_ids"]]
            return out["image"], out_boxes, out_labels
        except Exception as e:
            print(f"[Augmenter][Warning] Bỏ qua augment do lỗi: {e}")
            return image, boxes, labels

    def augment_video(self, frames: List[np.ndarray],
                      boxes_list: Optional[List] = None,
                      labels_list: Optional[List] = None,
                      seed: Optional[int] = None) -> Tuple[List[np.ndarray], List, List, int]:
        """
        Tăng cường toàn bộ chuỗi frames của một video bằng CHUNG 1 RANDOM SEED.
        Mọi khung hình trong video sẽ được áp dụng chính xác cùng một góc xoay, độ co giãn,
        lật ngang, thay đổi màu sắc và độ sáng.

        Args:
            frames: Danh sách các ảnh frame (RGB np.ndarray).
            boxes_list: Danh sách bboxes cho từng frame (tùy chọn).
            labels_list: Danh sách nhãn category_ids cho từng frame (tùy chọn).
            seed: Seed dùng chung (nếu None sẽ tự sinh ngẫu nhiên).

        Returns:
            aug_frames: Danh sách frames sau khi augment.
            aug_boxes_list: Danh sách bboxes sau khi augment.
            aug_labels_list: Danh sách labels sau khi augment.
            seed: Seed đã được sử dụng cho video.
        """
        if seed is None:
            seed = random.randint(0, 2 ** 31 - 1)

        aug_frames = []
        aug_boxes_list = []
        aug_labels_list = []

        for idx, frame in enumerate(frames):
            b = boxes_list[idx] if boxes_list is not None and idx < len(boxes_list) else []
            l = labels_list[idx] if labels_list is not None and idx < len(labels_list) else []

            # Đặt lại cùng seed trước mỗi frame của video
            aug_f, aug_b, aug_l = self(frame, b, l, seed=seed)

            aug_frames.append(aug_f)
            if boxes_list is not None:
                aug_boxes_list.append(aug_b)
            if labels_list is not None:
                aug_labels_list.append(aug_l)

        return aug_frames, aug_boxes_list, aug_labels_list, seed


# ---- Cấu hình mặc định ----
config = {
    "horizontalFlip": 0.5,
    "shiftScaleRotate": (0.08, 0.10, 15, 0.8),
    "randomBrightnessContrast": 0.4,
    "hueSaturationValue": (15, 20, 15, 0.3),
    "gaussNoise": (5.0, 25.0, 0.25),
    "blur": (5, 0.15),
    "p": 0.5
}

# ==============================================================================
# DEMO CHẠY THỬ VỚI 1 VIDEO TRONG TẬP DATASET SUST
# ==============================================================================
if __name__ == "__main__":
    sust_dir = r"D:\Project\AI\dataset\SUST"
    if not os.path.exists(sust_dir):
        raise FileNotFoundError(f"Thư mục dataset SUST không tồn tại: {sust_dir}")

    # Tìm video đầu tiên trong thư mục SUST
    video_files = [f for f in os.listdir(sust_dir) if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))]
    if not video_files:
        raise FileNotFoundError(f"Không tìm thấy video nào trong: {sust_dir}")

    sample_video_path = os.path.join(sust_dir, video_files[0])
    print(f"[1/4] Đang nạp video mẫu: {sample_video_path}")

    cap = cv2.VideoCapture(sample_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở video: {sample_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"      Thông số video: {total_frames_in_video} frames, {fps:.1f} FPS")

    # Đọc tối đa 90 frames (khoảng 3 giây) để làm demo trực quan
    max_demo_frames = 300
    target_img_size = 480
    raw_frames_bgr = []

    while len(raw_frames_bgr) < max_demo_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_lb = letterbox(frame, target_img_size)
        raw_frames_bgr.append(frame_lb)

    cap.release()
    print(f"[2/4] Đã đọc và letterbox ({target_img_size}x{target_img_size}) xong {len(raw_frames_bgr)} frames.")

    # Chuyển đổi sang không gian màu RGB cho Albumentations
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in raw_frames_bgr]

    # Tạo bounding box giả định ở vùng giữa ảnh (vùng khuôn mặt/người lái) để demo biến đổi box
    # Format pascal_voc: [xmin, ymin, xmax, ymax]
    dummy_boxes_list = []
    dummy_labels_list = []

    # Khởi tạo DetectionAugmenter và áp dụng chung 1 random seed cho toàn bộ video
    augmenter = DetectionAugmenter(config)
    selected_seed = None  # Bạn có thể đổi sang số khác hoặc để None để ngẫu nhiên
    print(f"[3/4] Đang augment toàn bộ video với chung random seed = {selected_seed}...")

    aug_frames_rgb, aug_boxes_list, aug_labels_list, used_seed = augmenter.augment_video(
        frames_rgb,
        boxes_list=dummy_boxes_list,
        labels_list=dummy_labels_list,
        seed=selected_seed
    )

    # Chuẩn bị lưu video so sánh Side-by-Side (Original | Augmented)
    output_demo_path = r"./demo_sust_augmented.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_width = target_img_size * 2
    video_height = target_img_size
    writer = cv2.VideoWriter(output_demo_path, fourcc, fps, (video_width, video_height))

    print(f"[4/4] Đang xuất video so sánh ra: {output_demo_path}")
    print("      (Nhấn phím 'q' hoặc ESC trên cửa sổ hiển thị để dừng xem trước)")

    window_name = "SUST Video Augmentation: [Left: Original | Right: Augmented (Shared Seed)]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    for i in range(len(raw_frames_bgr)):
        # Khung hình gốc
        orig_vis = raw_frames_bgr[i].copy()
        # if dummy_boxes_list[i]:
        #     bx = [int(v) for v in dummy_boxes_list[i][0]]
        #     cv2.rectangle(orig_vis, (bx[0], bx[1]), (bx[2], bx[3]), (0, 255, 0), 2)
        cv2.putText(orig_vis, f"Original #{i + 1}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Khung hình đã augment
        aug_bgr = cv2.cvtColor(aug_frames_rgb[i], cv2.COLOR_RGB2BGR)
        aug_vis = aug_bgr.copy()
        if aug_boxes_list[i]:
            abx = [int(v) for v in aug_boxes_list[i][0]]
            cv2.rectangle(aug_vis, (abx[0], abx[1]), (abx[2], abx[3]), (0, 255, 255), 2)
        cv2.putText(aug_vis, f"Augmented (Seed: {used_seed})", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Ghép 2 khung hình ngang nhau
        combined = np.hstack([orig_vis, aug_vis])
        writer.write(combined)

        # Hiển thị live preview
        cv2.imshow(window_name, combined)
        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key in [ord('q'), 27]:  # Bấm 'q' hoặc ESC để thoát
            break

    writer.release()
    cv2.destroyAllWindows()
    print(f"\n[Hoàn thành] Đã tạo thành công video demo tại: {os.path.abspath(output_demo_path)}")
