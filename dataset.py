from pathlib import Path
from typing import Optional, Callable, Tuple, List, Sequence, Any

import cv2
import torch
from torch.utils.data import Dataset

from config import TrainConfig


class MyLSTMDataset(Dataset):
    """
    Dataset tùy chỉnh PyTorch dùng để tải chuỗi khung hình video và nhãn cho mô hình Deep LSTM.

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
            "driving": 0,  # 0: Tỉnh táo
            "drowsiness": 1  # 1: Buồn ngủ
        }

        self.video_paths = []
        for ext in video_exts:
            self.video_paths.extend(list(self.dataset_dir.glob(f"*{ext}")))

        if len(self.video_paths) == 0:
            raise FileNotFoundError(f"Không tìm thấy file video nào trong thư mục: {dataset_dir}")

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
        """
        Đọc và trích xuất đều seq_len khung hình từ tập tin video qua OpenCV.

        Returns:
            torch.Tensor: Tensor chuỗi khung hình có kích thước [Seq_Len, 3, Height, Width]
        """
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
        """
        Lấy ra 1 mẫu dữ liệu video và nhãn tương ứng.

        Args:
            idx (int): Chỉ số index của video trong Dataset.

        Returns:
            Tuple[torch.Tensor, int]: 
                - video_tensor: Tensor chuỗi khung hình [Seq_Len, 3, Height, Width]
                - label: Nhãn số nguyên (0: Tỉnh táo, 1: Buồn ngủ)
        """
        video_path = self.video_paths[idx]
        label = self._extract_label(video_path)
        video_tensor = self._load_video_frames(video_path)
        return video_tensor, label


def visualize_video_demo(video_tensor: torch.Tensor, label: int, video_name: str = "Demo Video", fps: int = 30):
    """
    Hàm hiển thị video trực quan từ PyTorch Tensor trả về từ MyLSTMDataset.
    """
    seq_len, channels, height, width = video_tensor.shape
    label_text = "0: Tinh tao (Driving)" if label == 0 else "1: Buon ngu (Drowsiness)"
    text_color = (0, 255, 0) if label == 0 else (0, 0, 255)

    print(f"\n[+] Đang phát video demo: {video_name}")
    print(f"    - Kích thước: {width}x{height} | Số khung hình: {seq_len} | Nhãn: {label_text}")
    print("    - Nhấn phím 'q' hoặc 'ESC' trên cửa sổ video để dừng hiển thị.\n")

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
            cv2.putText(
                frame_bgr,
                info_str,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                text_color,
                2,
                cv2.LINE_AA
            )

            cv2.imshow(window_name, frame_bgr)

            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord('q') or key == 27:
                print("    -> Người dùng đã chủ động dừng phát video.")
                break

        cv2.destroyWindow(window_name)
        print("    -> Đã hoàn thành phát video demo thành công.")

    except Exception as e:
        print(
            f"    [!] Không thể mở cửa sổ hiển thị OpenCV GUI ({e}). Khung hình video đã được load thành công dạng Tensor.")


if __name__ == "__main__":
    import sys
    from functools import partial
    from config import TrainConfig
    from myCNN.src.runtime.infer import letterbox

    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    config = TrainConfig()
    transforms = partial(letterbox, new_size=480)

    dataset = MyLSTMDataset(transform=transforms)

    print(f"\n[+] Tổng số video load thành công: {len(dataset)}")

    video_tensor, label = dataset[10]
    first_video_name = dataset.video_paths[10].name

    visualize_video_demo(video_tensor, label, video_name=first_video_name, fps=30)
