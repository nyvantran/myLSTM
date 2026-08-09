import time
from pathlib import Path
from typing import Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TrainConfig
from dataloader import get_dataloaders
from dataset import MyLSTMDataset
from loss import DrowsinessBCELoss
from log import TensorBoardLogger
from model import DeepLSTMClassifier


class TrainLstm:
    """
    Trình quản lý và thực thi huấn luyện mô hình Deep LSTM cho bài toán phát hiện buồn ngủ (Drowsiness Detection).

    Tích hợp:
    - Quản lý cấu hình toàn hệ thống qua TrainConfig.
    - Nạp dữ liệu qua MyLSTMDataset và DataLoader với collate_fn trích xuất đặc trưng CNN.
    - Đếm và hiển thị tiến trình huấn luyện chi tiết từng Epoch với tqdm.
    - Ghi log giám sát đầy đủ các chỉ số (Loss, Accuracy, Learning Rate) qua TensorBoardLogger trong log.py.
    - Quản lý lưu trữ và khôi phục (Resume) Checkpoints tự động sau mỗi epoch.
    """

    def __init__(
        self,
        config: Optional[TrainConfig] = None,
        model: Optional[DeepLSTMClassifier] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ):
        self.config = config if config is not None else TrainConfig()

        # 1. Thiết lập Thiết bị tính toán (GPU / CPU)
        if torch.cuda.is_available() and self.config.device == "cuda":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"[TrainLstm] Sử dụng thiết bị tính toán: {self.device}")

        # 2. Khởi tạo Mô hình DeepLSTMClassifier
        if model is not None:
            self.model = model
        else:
            self.model = DeepLSTMClassifier.from_config(self.config)
        self.model.to(self.device)

        # 3. Khởi tạo DataLoaders (Train & Validation)
        if train_loader is not None and val_loader is not None:
            self.train_loader = train_loader
            self.val_loader = val_loader
        else:
            print("[TrainLstm] Đang khởi tạo DataLoader từ TrainConfig...")
            self.train_loader, self.val_loader = get_dataloaders(self.config, use_cnn_collate=True)

        # 4. Khởi tạo Hàm mất mát (Loss Function)
        if criterion is not None:
            self.criterion = criterion
        else:
            self.criterion = DrowsinessBCELoss.from_config(self.config)
        self.criterion.to(self.device)

        # 5. Khởi tạo Bộ tối ưu hóa (Optimizer)
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            opt_name = self.config.optimizer.lower()
            if opt_name == "adamw":
                self.optimizer = optim.AdamW(
                    self.model.parameters(),
                    lr=self.config.lr0,
                    weight_decay=self.config.weight_decay,
                    betas=self.config.betas
                )
            elif opt_name == "adam":
                self.optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=self.config.lr0,
                    weight_decay=self.config.weight_decay,
                    betas=self.config.betas
                )
            elif opt_name == "sgd":
                self.optimizer = optim.SGD(
                    self.model.parameters(),
                    lr=self.config.lr0,
                    momentum=self.config.momentum,
                    weight_decay=self.config.weight_decay
                )
            else:
                raise ValueError(f"Optimizer không hợp lệ: {self.config.optimizer}")

        # 6. Khởi tạo Learning Rate Scheduler
        if scheduler is not None:
            self.scheduler = scheduler
        elif self.config.use_scheduler:
            lr_min = self.config.lr0 * self.config.lr_min_factor
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=lr_min
            )
        else:
            self.scheduler = None

        # 7. Khởi tạo TensorBoard Logger từ log.py
        self.logger = TensorBoardLogger(
            log_dir=self.config.tb_log_dir,
            experiment_name=self.config.experiment_name
        )

        # Khởi tạo trạng thái theo dõi huấn luyện
        self.start_epoch = 1
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0

        # Nếu cấu hình yêu cầu khôi phục checkpoint
        if self.config.resume:
            self.load_checkpoint(self.config.resume)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Khôi phục trạng thái mô hình, optimizer, scheduler và epoch từ file checkpoint (.pth/.pt).
        """
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            print(f"[TrainLstm][Warning] Không tìm thấy file checkpoint để resume: {checkpoint_path}")
            return

        ckpt = torch.load(ckpt_path, map_location=self.device)

        if "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
        elif "state_dict" in ckpt:
            self.model.load_state_dict(ckpt["state_dict"])
        else:
            self.model.load_state_dict(ckpt)

        if "optimizer_state_dict" in ckpt and self.optimizer is not None:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if "scheduler_state_dict" in ckpt and self.scheduler is not None and ckpt["scheduler_state_dict"] is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.global_step = ckpt.get("global_step", self.global_step)
        self.best_val_loss = ckpt.get("best_val_loss", self.best_val_loss)
        self.best_val_acc = ckpt.get("best_val_acc", self.best_val_acc)

        print(f"[TrainLstm] Đã khôi phục checkpoint thành công từ: {ckpt_path.resolve()}")
        print(f"            - Epoch tiếp theo: {self.start_epoch}")
        print(f"            - Best Val Loss hiện tại: {self.best_val_loss:.6f} | Best Val Acc: {self.best_val_acc * 100:.2f}%")

    def save_checkpoint(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        is_best: bool = False
    ) -> None:
        """
        Lưu trạng thái checkpoint mô hình sau mỗi epoch.
        """
        save_dir = Path(self.config.checkpoint_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_dict = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "config": self.config.to_dict(),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc
        }

        # Lưu checkpoint định kỳ theo số epoch
        if epoch % self.config.save_ckpt_interval_epochs == 0 and not self.config.save_best_only:
            ckpt_filename = save_dir / f"checkpoint_epoch_{epoch:03d}.pth"
            torch.save(checkpoint_dict, ckpt_filename)

        # Lưu checkpoint mới nhất (last.pth)
        last_ckpt = save_dir / "last.pth"
        torch.save(checkpoint_dict, last_ckpt)

        # Lưu checkpoint tốt nhất (best.pth)
        if is_best:
            best_ckpt = save_dir / "best.pth"
            torch.save(checkpoint_dict, best_ckpt)
            print(f"[TrainLstm] -> Đã lưu BEST checkpoint mới tại: {best_ckpt.resolve()}")

    def _compute_accuracy(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        """Tính độ chính xác Accuracy (0.0 -> 1.0) cho batch."""
        pred_labels = preds.argmax(dim=-1)
        if targets.dim() < pred_labels.dim():
            targets = targets.unsqueeze(-1).expand_as(pred_labels)
        correct = (pred_labels == targets).sum().item()
        total = pred_labels.numel()
        return correct / total if total > 0 else 0.0

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Huấn luyện mô hình 1 epoch."""
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(
            enumerate(self.train_loader, start=1),
            total=num_batches,
            desc=f"Epoch {epoch:02d}/{self.config.epochs:02d} [Train]",
            leave=False
        )

        for batch_idx, (features, targets) in pbar:
            self.global_step += 1
            features = features.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Lan truyền tiến (Forward pass)
            preds = self.model(features)
            loss = self.criterion(preds, targets)

            # Lan truyền ngược (Backward pass) & Grad Clipping
            loss.backward()
            if self.config.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            self.optimizer.step()

            # Tính toán chỉ số
            batch_loss = loss.item()
            batch_acc = self._compute_accuracy(preds, targets)

            total_loss += batch_loss
            total_acc += batch_acc

            # Ghi log chỉ số Batch lên TensorBoard
            self.logger.log_train_batch(batch_loss, batch_acc, self.global_step)

            current_lr = self.optimizer.param_groups[0]["lr"]
            pbar.set_postfix({
                "loss": f"{batch_loss:.4f}",
                "acc": f"{batch_acc * 100:.1f}%",
                "lr": f"{current_lr:.6f}"
            })

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_acc = total_acc / num_batches if num_batches > 0 else 0.0
        return avg_loss, avg_acc

    def val_epoch(self, epoch: int) -> Tuple[float, float]:
        """Đánh giá mô hình trên tập Validation trong 1 epoch."""
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = len(self.val_loader)

        pbar = tqdm(
            enumerate(self.val_loader, start=1),
            total=num_batches,
            desc=f"Epoch {epoch:02d}/{self.config.epochs:02d} [ Val ]",
            leave=False
        )

        with torch.no_grad():
            for batch_idx, (features, targets) in pbar:
                features = features.to(self.device)
                targets = targets.to(self.device)

                preds = self.model(features)
                loss = self.criterion(preds, targets)

                batch_loss = loss.item()
                batch_acc = self._compute_accuracy(preds, targets)

                total_loss += batch_loss
                total_acc += batch_acc

                pbar.set_postfix({
                    "val_loss": f"{batch_loss:.4f}",
                    "val_acc": f"{batch_acc * 100:.1f}%"
                })

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_acc = total_acc / num_batches if num_batches > 0 else 0.0
        return avg_loss, avg_acc

    def fit(self) -> None:
        """Bắt đầu toàn bộ quy trình huấn luyện mô hình."""
        print("\n" + "=" * 70)
        print(f"=== BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH DEEP LSTM ({self.config.epochs} EPOCHS) ===")
        print("=" * 70)

        for epoch in range(self.start_epoch, self.config.epochs + 1):
            start_time = time.time()

            # 1. Huấn luyện Epoch
            train_loss, train_acc = self.train_epoch(epoch)

            # 2. Đánh giá Validation Epoch
            val_loss, val_acc = self.val_epoch(epoch)

            # 3. Cập nhật Learning Rate Scheduler
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                self.scheduler.step()

            # 4. Ghi log TensorBoard qua log.py
            self.logger.log_train_epoch(train_loss, train_acc, current_lr, epoch)
            self.logger.log_val_epoch(val_loss, val_acc, epoch)

            # 5. Kiểm tra Best Checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc

            # 6. Lưu Checkpoint
            self.save_checkpoint(epoch, train_loss, train_acc, val_loss, val_acc, is_best=is_best)

            elapsed = time.time() - start_time
            print(f"Epoch [{epoch:02d}/{self.config.epochs:02d}] ({elapsed:.1f}s) | "
                  f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc * 100:.2f}% | "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc * 100:.2f}% | "
                  f"LR: {current_lr:.6f}" + (" [BEST]" if is_best else ""))

        print("\n" + "=" * 70)
        print(f"[TrainLstm] ĐÃ HOÀN THÀNH HUẤN LUYỆN!")
        print(f"            - Best Val Loss: {self.best_val_loss:.6f}")
        print(f"            - Best Val Acc: {self.best_val_acc * 100:.2f}%")
        print("=" * 70)

        self.logger.close()


# Các tên Alias đồng nhất
TrainLtsm = TrainLstm
TrainLSTM = TrainLstm


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    print("=== TEST DEMO CLASS TRAINLSTM TRONG TRAIN.PY ===")

    # Khởi tạo config giả lập 2 epoch với batch_size nhỏ để chạy thử nghiệm nhanh
    cfg = TrainConfig(
        epochs=2,
        batch_size=2,
        seq_len=5,
        tb_log_dir="runs_test",
        checkpoint_dir="./checkpoints_test"
    )

    trainer = TrainLstm(config=cfg)
    trainer.fit()
