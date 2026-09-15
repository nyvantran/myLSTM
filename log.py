import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, Sequence
import torch

# Đảm bảo console Windows hỗ trợ in UTF-8 không bị lỗi charmap
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Kiếm tra xem viện tensorboard đã được cài đặt chưa
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class TensorBoardLogger:
    """
    Trình quản lý và ghi log TensorBoard (TensorBoard Logger Engine).
    Tách biệt logic ghi log số liệu huấn luyện (Loss, Accuracy, Learning Rate, Graph, Histograms) 
    khỏi vòng lặp huấn luyện chính.
    """
    def __init__(self, log_dir: str = "runs", experiment_name: str = "lstm_drowsiness"):
        """
        Khởi tạo Logger Engine.

        Args:
            log_dir (str): Thư mục gốc lưu trữ log (mặc định "runs").
            experiment_name (str): Tên bài thí nghiệm (mặc định "lstm_drowsiness").
        """
        self.enabled = TENSORBOARD_AVAILABLE
        
        if not self.enabled:
            print("[TensorBoardLogger] Cảnh báo: Chưa cài đặt thư viện `tensorboard`. Logic ghi log sẽ bị tắt.")
            self.writer = None
            self.log_path = None
            return

        # 1. Tạo tên thư mục log độc nhất dựa trên timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = Path(log_dir) / f"{experiment_name}_{timestamp}"
        self.log_path.mkdir(parents=True, exist_ok=True)

        # 2. Khởi tạo SummaryWriter
        self.writer = SummaryWriter(log_dir=str(self.log_path))
        print(f"[TensorBoardLogger] Đã khởi tạo log engine tại thư mục: {self.log_path}")

    def log_train_batch(self, loss: float, accuracy: float, step: Optional[int] = None, global_step: Optional[int] = None):
        """
        Ghi log chỉ số Loss và Accuracy tại từng Batch huấn luyện (Batch-level).
        Hỗ trợ cả tham số step (datn4ni2.ipynb) và global_step.
        """
        if not self.enabled or self.writer is None:
            return
        
        curr_step = step if step is not None else (global_step if global_step is not None else 0)
        self.writer.add_scalar("Train/Batch_Loss", loss, curr_step)
        self.writer.add_scalar("Train/Batch_Accuracy", accuracy, curr_step)

    def log_train_epoch(self, epoch_loss: float, epoch_acc: float, learning_rate: float, epoch: int):
        """
        Ghi log các chỉ số tổng kết sau mỗi Epoch tập Train (Epoch-level).

        Args:
            epoch_loss (float): Loss trung bình của cả epoch.
            epoch_acc (float): Accuracy trung bình của cả epoch.
            learning_rate (float): Tốc độ học (Learning Rate) hiện tại.
            epoch (int): Số thứ tự epoch.
        """
        if not self.enabled or self.writer is None:
            return
        
        self.writer.add_scalar("Train/Epoch_Loss", epoch_loss, epoch)
        self.writer.add_scalar("Train/Epoch_Accuracy", epoch_acc, epoch)
        self.writer.add_scalar("Train/Learning_Rate", learning_rate, epoch)

    def log_val_epoch(self, val_loss: float, val_acc: float, epoch: int):
        """
        Ghi log các chỉ số tổng kết sau mỗi Epoch tập Validation.

        Args:
            val_loss (float): Loss trung bình tập Validation.
            val_acc (float): Accuracy trung bình tập Validation.
            epoch (int): Số thứ tự epoch.
        """
        if not self.enabled or self.writer is None:
            return
        
        self.writer.add_scalar("Val/Epoch_Loss", val_loss, epoch)
        self.writer.add_scalar("Val/Epoch_Accuracy", val_acc, epoch)

    def log_custom_scalars(self, metrics: Dict[str, float], step: int, main_tag: str = "Metrics"):
        """
        Ghi log từ điển các chỉ số tùy biến (Custom Metrics Dictionary).

        Args:
            metrics (Dict[str, float]): Từ điển dạng {"precision": 0.92, "recall": 0.88, ...}
            step (int): Bước tính toán hoặc epoch hiện tại.
            main_tag (str): Phân nhóm thẻ chính.
        """
        if not self.enabled or self.writer is None:
            return
        
        for key, val in metrics.items():
            self.writer.add_scalar(f"{main_tag}/{key}", val, step)

    def log_metrics(self, metrics: Dict[str, float], epoch: int):
        """
        Ghi nhận các chỉ số chuyên biệt: Precision, Recall, F1-Score lên TensorBoard.
        Đồng bộ với CELL 3 của datn4ni2.ipynb.
        """
        self.log_custom_scalars(metrics, step=epoch, main_tag="Metrics")

    def log_confusion_matrix(self, cm: Any, epoch: int, class_names: Tuple[str, str] = ("Tỉnh táo", "Buồn ngủ")):
        """
        Vẽ và đẩy ảnh heatmap ma trận nhầm lẫn (Confusion Matrix) lên TensorBoard.
        Đồng bộ với CELL 3 của datn4ni2.ipynb.
        """
        if not self.enabled or self.writer is None:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
            ax.set_xlabel("Dự đoán (Predicted)")
            ax.set_ylabel("Thực tế (Actual)")
            ax.set_title(f"Confusion Matrix (Epoch {epoch})")
            plt.tight_layout()
            self.writer.add_figure("Images/Confusion_Matrix", fig, epoch)
            plt.close(fig)
        except Exception as e:
            print(f"[TensorBoardLogger][WARN] Không thể log confusion matrix: {e}")

    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        """
        Ghi log biểu đồ phân bố trọng số / gradient (Histograms).

        Args:
            tag (str): Tên thẻ biểu đồ.
            values (torch.Tensor): Tensor chứa trọng số hoặc gradient.
            step (int): Bước tính toán hoặc epoch hiện tại.
        """
        if not self.enabled or self.writer is None:
            return
        
        self.writer.add_histogram(tag, values, step)

    def log_model_graph(self, model: torch.nn.Module, input_sample: Any):
        """
        Ghi log luồng ma trận kiến trúc mô hình (Computational Graph).
        Hỗ trợ cả input_sample là torch.Tensor hoặc tuple (p3, p4, p5).

        Args:
            model (torch.nn.Module): Mô hình PyTorch.
            input_sample (Any): Tensor mẫu hoặc tuple các tensor đầu vào.
        """
        if not self.enabled or self.writer is None:
            return
        
        try:
            self.writer.add_graph(model, input_sample)
            print("[TensorBoardLogger] Đã lưu sơ đồ ma trận kiến trúc mô hình (Computational Graph).")
        except Exception as e:
            print(f"[TensorBoardLogger][WARN] Không thể lưu sơ đồ kiến trúc mô hình: {e}")

    # Alias đồng bộ với datn4ni2.ipynb
    log_graph = log_model_graph

    def close(self):
        """Đóng và hoàn tất tiến trình ghi log."""
        if self.enabled and self.writer is not None:
            self.writer.close()
            print(f"[TensorBoardLogger] Đã hoàn tất và đóng TensorBoard Logger tại: {self.log_path}")


if __name__ == "__main__":
    print("==========================================================================")
    print("=== KIỂM TRA LỚP TENSORBOARDLOGGER TRONG LOG.PY ===")
    print("==========================================================================")
    
    # Khởi tạo logger
    logger = TensorBoardLogger(log_dir="runs", experiment_name="test_experiment")
    
    # Giả lập ghi log 3 epochs
    for epoch in range(1, 4):
        # Ghi log train
        logger.log_train_epoch(epoch_loss=0.5 / epoch, epoch_acc=0.7 + (epoch * 0.08), learning_rate=0.001, epoch=epoch)
        # Ghi log val
        logger.log_val_epoch(val_loss=0.6 / epoch, val_acc=0.65 + (epoch * 0.08), epoch=epoch)
        
    logger.close()
