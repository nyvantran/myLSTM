import os
import time
from pathlib import Path
from typing import Optional, Callable, Tuple, Sequence, Any, Union, List

import cv2
import torch
from torch.utils.data import Dataset

from config import TrainConfig


class PreloadedTensorDataset(Dataset):
    """
    Dataset tải toàn bộ Tensor đặc trưng .pt vào RAM một lần duy nhất.
    Được tối ưu hóa theo CELL 6 của datn4ni2.ipynb:
    - Thời gian nạp chỉ < 0.5s cho toàn bộ video.
    - Không chịu gánh nặng I/O giải mã video hay forward CNN lặp đi lặp lại.
    - Hỗ trợ cả tensor lưu dạng float32 và float16, tự động ép sang float32 khi đưa vào mô hình.
    """

    def __init__(self, pt_path: Union[str, Path], is_train: bool = True):
        super(PreloadedTensorDataset, self).__init__()
        path = Path(pt_path)

        if not path.exists():
            print(f"[PreloadedTensorDataset][WARN] Không tìm thấy tệp: {path.resolve()}.")
            print(f"    -> Đang sinh tập dữ liệu mô phỏng ngẫu nhiên để kiểm thử pipeline...")
            num_samples = 64 if is_train else 16
            self.p3 = torch.randn(num_samples, 120, 224, dtype=torch.float32)
            self.p4 = torch.randn(num_samples, 120, 448, dtype=torch.float32)
            self.p5 = torch.randn(num_samples, 120, 640, dtype=torch.float32)
            self.labels = torch.randint(0, 2, (num_samples,), dtype=torch.long)
            self.video_ids = [f"dummy_video_{i:04d}" for i in range(num_samples)]
            print(f"[+] Đã tạo {num_samples} mẫu dữ liệu mô phỏng.")
            return

        t0 = time.time()
        print(f"[PreloadedTensorDataset] Đang nạp dữ liệu từ: {path.resolve()}...")
        data = torch.load(str(path), map_location="cpu")

        # Ép kiểu float32 để đảm bảo tương thích tính toán trên mọi nền tảng
        self.p3 = data["p3"].float()       # [N, 120, 224]
        self.p4 = data["p4"].float()       # [N, 120, 448]
        self.p5 = data["p5"].float()       # [N, 120, 640]
        self.labels = data["labels"].long() # [N]
        self.video_ids = data.get("video_ids", [f"video_{i:04d}" for i in range(len(self.labels))])

        load_sec = time.time() - t0
        print(f"    - Nạp hoàn tất {len(self.labels)} video vào RAM trong {load_sec:.2f}s!")
        print(f"    - Phân bố nhãn: Tỉnh táo (0) = {(self.labels == 0).sum().item()}, Buồn ngủ (1) = {(self.labels == 1).sum().item()}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Trả về:
            ((p3, p4, p5), label): Bộ 3 đặc trưng không gian đa tỷ lệ và nhãn Ground Truth
        """
        return (self.p3[idx], self.p4[idx], self.p5[idx]), self.labels[idx]


class MyLSTMDataset(Dataset):
    """
    Dataset tùy chỉnh PyTorch dùng để tải chuỗi khung hình video và nhãn cho mô hình Deep LSTM (Legacy Mode).

    Quy tắc gán nhãn:
    - Đọc tập tin video từ đường dẫn thư mục dataset.
    - Tách tên file theo ký tự '-': Thành phần thứ -2 xác định trạng thái của người lái xe:
      + 'driving'    -> Gán nhãn 0 (Tỉnh táo)
      + 'drowsiness' -> Gán nhãn 1 (Buồn ngủ)
    """

    def __init__(
            self,
            dataset_dir: str = TrainConfig.dataset_dir,
            seq_len: int = TrainConfig.seq_len,
            image_size: Tuple[int, int] = TrainConfig.image_size,
            transform: Optional[Callable] = None,
            video_exts: Sequence[str] = TrainConfig.video_exts
    ):
        super(MyLSTMDataset, self).__init__()

        self.dataset_dir = Path(dataset_dir)
        self.seq_len = seq_len
        self.image_size = image_size
        self.transform = transform

        self.label_map = {
            "driving": 0,    # 0: Tỉnh táo
            "drowsiness": 1  # 1: Buồn ngủ
        }

        self.video_paths = []
        if self.dataset_dir.exists():
            for ext in video_exts:
                self.video_paths.extend(list(self.dataset_dir.glob(f"*{ext}")))

        if len(self.video_paths) == 0:
            print(f"[MyLSTMDataset][WARN] Không tìm thấy file video nào trong thư mục: {dataset_dir}")
        else:
            print(f"[MyLSTMDataset] Tìm thấy {len(self.video_paths)} file video trong {dataset_dir}")

    @classmethod
    def from_config(cls, config: Any, transform: Optional[Callable] = None) -> "MyLSTMDataset":
        """Khởi tạo MyLSTMDataset trực tiếp từ đối tượng TrainConfig."""
        return cls(
            dataset_dir=config.dataset_dir,
            seq_len=config.seq_len,
            image_size=config.image_size,
            transform=transform,
            video_exts=list(config.video_exts)
        )

    def __len__(self) -> int:
        return len(self.video_paths)

    def _extract_label(self, video_path: Path) -> int:
        """
        Trích xuất nhãn từ tên file video theo quy tắc thành phần thứ -2.
        Ví dụ: 'subject0-littleBright-driving-1.avi' -> 'driving' -> Nhãn 0.
        """
        filename_stem = video_path.stem
        parts = filename_stem.split('-')

        if len(parts) >= 2:
            label_str = parts[-2].lower()
        else:
            raise ValueError(f"Tên file {video_path.name} không đúng định dạng chứa dấu '-'")

        if label_str in self.label_map:
            return self.label_map[label_str]
        else:
            raise KeyError(
                f"Nhãn '{label_str}' trong {video_path.name} không nằm trong {list(self.label_map.keys())}")

    def _load_video_frames(self, video_path: Path) -> torch.Tensor:
        """Đọc và trích xuất đều seq_len khung hình từ tập tin video qua OpenCV."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Không thể mở file video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        if total_frames <= 0:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            cap.release()
            total_frames = len(frames)
            if total_frames == 0:
                raise ValueError(f"Video không chứa khung hình nào: {video_path}")

            indices = torch.linspace(0, total_frames - 1, self.seq_len).long().tolist()
            frames = [frames[i] for i in indices]
        else:
            indices = torch.linspace(0, total_frames - 1, self.seq_len).long().tolist()

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                else:
                    blank_frame = torch.zeros((self.image_size[0], self.image_size[1], 3), dtype=torch.uint8).numpy()
                    frames.append(blank_frame)
            cap.release()

        while len(frames) < self.seq_len:
            frames.append(frames[-1] if len(frames) > 0 else torch.zeros((self.image_size[0], self.image_size[1], 3),
                                                                         dtype=torch.uint8).numpy())

        frames = frames[:self.seq_len]

        processed_frames = []
        for frame in frames:
            if self.transform is not None:
                frame = self.transform(frame)[0]
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            processed_frames.append(frame_tensor)

        return torch.stack(processed_frames, dim=0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path = self.video_paths[idx]
        label = self._extract_label(video_path)
        video_tensor = self._load_video_frames(video_path)
        return video_tensor, label


def visualize_video_demo(video_tensor: torch.Tensor, label: int, video_name: str = "Demo Video", fps: int = 30):
    """Hàm hiển thị video trực quan từ PyTorch Tensor trả về từ MyLSTMDataset."""
    seq_len, channels, height, width = video_tensor.shape
    label_text = "0: Tinh tao (Driving)" if label == 0 else "1: Buon ngu (Drowsiness)"
    text_color = (0, 255, 0) if label == 0 else (0, 0, 255)

    print(f"\n[+] Đang phát video demo: {video_name}")
    print(f"    - Kích thước: {width}x{height} | Số khung hình: {seq_len} | Nhãn: {label_text}")

    window_name = f"Demo Video - MyLSTMDataset [{video_name}]"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 480, 480)
        delay_ms = int(1000 / fps)

        for t in range(seq_len):
            frame_tensor = video_tensor[t]
            frame_np = (frame_tensor.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype('uint8')
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

            overlay = frame_bgr.copy()
            cv2.rectangle(overlay, (0, 0), (width, 45), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame_bgr, 0.4, 0, frame_bgr)

            info_str = f"Khung: {t + 1}/{seq_len} | Nhan: {label_text}"
            cv2.putText(frame_bgr, info_str, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2, cv2.LINE_AA)
            cv2.imshow(window_name, frame_bgr)

            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord('q') or key == 27:
                break

        cv2.destroyWindow(window_name)
    except Exception as e:
        print(f"    [!] Không thể mở cửa sổ hiển thị OpenCV GUI ({e}).")


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    print("==========================================================================")
    print("=== KIỂM TRA LỚP PRELOADEDTENSORDATASET ===")
    print("==========================================================================")
    
    cfg = TrainConfig()
    ds = PreloadedTensorDataset(cfg.val_pt, is_train=False)
    print(f"[+] Kích thước tập dữ liệu: {len(ds)} mẫu")
    (p3, p4, p5), label = ds[0]
    print(f"[+] Mẫu 0: p3 shape={list(p3.shape)}, p4={list(p4.shape)}, p5={list(p5.shape)}, label={label}")
    print("==========================================================================")
