import os
import sys
import time
import zipfile
import argparse
from pathlib import Path
from typing import Optional, Tuple, Any, Dict, List

# Đảm bảo console UTF-8 trên Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Tối ưu hóa phân bổ bộ nhớ PyTorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from config import TrainConfig
from dataloader import get_dataloaders
from loss import get_loss_function, DrowsinessLoss
from log import TensorBoardLogger
from model import DeepLSTMClassifier


def seed_everything(seed: int = 42) -> None:
    """Cố định seed ngẫu nhiên đảm bảo tính tái lập 100% (Reproducibility)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_autocast(device: torch.device, enabled: bool = True):
    """Tạo context manager autocast tương thích PyTorch 1.x và 2.x."""
    is_cuda = "cuda" in str(device)
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda" if is_cuda else "cpu", enabled=(enabled and is_cuda))
    return torch.cuda.amp.autocast(enabled=(enabled and is_cuda))


class TrainLstm:
    """
    Trình quản lý và thực thi huấn luyện mô hình Deep LSTM (Drowsiness Detection).
    Được tối ưu hóa toàn diện theo CELL 7 của datn4ni2.ipynb:
    - Nạp trực tiếp PyTorch Tensor (.pt) siêu tốc vào RAM qua PreloadedTensorDataset (< 0.5s).
    - Tích hợp Automatic Mixed Precision (AMP FP16) đạt tốc độ ~1.2s/epoch trên GPU Tesla T4/RTX.
    - Bộ lập lịch CosineAnnealingLR & Gradient Clipping.
    - Giám sát toàn diện qua TensorBoard: Batch Loss, Epoch Loss, Accuracy, F1, Confusion Matrix Heatmap.
    - Tự động lưu trữ và đóng gói Best Checkpoint (best_lstm.pth) và Last Checkpoint (last_lstm.pth).
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
        seed_everything(self.config.seed)

        # 1. Thiết bị tính toán (GPU / CPU)
        if torch.cuda.is_available() and "cuda" in self.config.device:
            self.device = torch.device(self.config.device)
            gpu_name = torch.cuda.get_device_name(self.device)
            vram_gb = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)
            print(f"[TrainLstm] Sử dụng GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")
        else:
            self.device = torch.device("cpu")
            print(f"[TrainLstm] Sử dụng CPU tính toán.")

        # 2. Khởi tạo Mô hình DeepLSTMClassifier
        if model is not None:
            self.model = model
        else:
            self.model = DeepLSTMClassifier.from_config(self.config)
        self.model.to(self.device)

        # 3. Khởi tạo DataLoaders (Mặc định nạp Tensor .pt siêu tốc)
        if train_loader is not None and val_loader is not None:
            self.train_loader = train_loader
            self.val_loader = val_loader
        else:
            print("[TrainLstm] Đang nạp tập dữ liệu huấn luyện và kiểm định...")
            self.train_loader, self.val_loader = get_dataloaders(self.config)

        # 4. Khởi tạo Hàm mất mát (Loss Function)
        if criterion is not None:
            self.criterion = criterion
        else:
            self.criterion = get_loss_function(self.config)
        self.criterion.to(self.device)

        # 5. Khởi tạo Optimizer
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

        # 6. Khởi tạo Learning Rate Scheduler (Cosine Annealing)
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

        # 7. Khởi tạo Automatic Mixed Precision (AMP FP16 GradScaler)
        is_cuda = "cuda" in str(self.device)
        scaler_enabled = self.config.amp and is_cuda
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

        # 8. Khởi tạo TensorBoard Logger và đăng ký Computational Graph
        self.logger = TensorBoardLogger(
            log_dir=self.config.tb_log_dir,
            experiment_name=self.config.experiment_name
        )
        try:
            dummy_graph_input = (
                torch.zeros(1, self.config.seq_len, 224, device=self.device),
                torch.zeros(1, self.config.seq_len, 448, device=self.device),
                torch.zeros(1, self.config.seq_len, 640, device=self.device)
            )
            self.logger.log_graph(self.model, dummy_graph_input)
        except Exception as e:
            print(f"[TrainLstm][WARN] Chưa thể ghi computational graph: {e}")

        # Khởi tạo biến trạng thái
        self.start_epoch = 1
        self.global_step = 0
        self.best_val_acc = 0.0
        self.best_f1_score = 0.0

        # Lịch sử huấn luyện
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "precision": [], "recall": [], "f1": [],
            "lr": []
        }

        # Khôi phục nếu có checkpoint
        if self.config.resume:
            self.load_checkpoint(self.config.resume)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Khôi phục trạng thái mô hình và optimizer từ checkpoint .pth."""
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            print(f"[TrainLstm][Warning] Không tìm thấy file checkpoint để resume: {checkpoint_path}")
            return

        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        self.model.load_state_dict(state_dict, strict=False)

        if "optimizer_state_dict" in ckpt and self.optimizer is not None:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt and self.scheduler is not None and ckpt["scheduler_state_dict"] is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.global_step = ckpt.get("global_step", self.global_step)
        self.best_val_acc = ckpt.get("val_acc", ckpt.get("best_val_acc", self.best_val_acc))
        self.best_f1_score = ckpt.get("f1_score", 0.0)

        print(f"[TrainLstm] Đã khôi phục checkpoint thành công từ: {ckpt_path.resolve()}")
        print(f"            - Epoch tiếp theo: {self.start_epoch} | Best Val Acc: {self.best_val_acc * 100:.2f}%")

    def save_checkpoint(self, epoch: int, val_acc: float, f1: float, is_best: bool = False) -> None:
        """Lưu checkpoint best_lstm.pth và last_lstm.pth chuẩn khớp datn4ni2.ipynb."""
        save_dir = Path(self.config.checkpoint_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        ckpt_data = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "val_acc": val_acc,
            "f1_score": f1,
            "config": {
                "input_dim": self.config.input_dim,
                "hidden_dim": self.config.hidden_dim,
                "num_layers": self.config.num_layers,
                "num_classes": self.config.num_classes
            }
        }

        # 1. Lưu checkpoint mới nhất
        last_ckpt_path = save_dir / "last_lstm.pth"
        torch.save(ckpt_data, last_ckpt_path)

        # 2. Lưu checkpoint định kỳ nếu cấu hình yêu cầu
        if epoch % self.config.save_ckpt_interval_epochs == 0 and not self.config.save_best_only:
            periodic_path = save_dir / f"checkpoint_epoch_{epoch:03d}.pth"
            torch.save(ckpt_data, periodic_path)

        # 3. Lưu best checkpoint
        if is_best:
            best_ckpt_path = save_dir / "best_lstm.pth"
            torch.save(ckpt_data, best_ckpt_path)
            print(f"[TrainLstm] -> ĐÃ LƯU BEST CHECKPOINT MỚI: {best_ckpt_path.resolve()} (Val Acc: {val_acc*100:.2f}%)")

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Vòng lặp huấn luyện 1 Epoch với Automatic Mixed Precision (AMP FP16)."""
        self.model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        num_batches = len(self.train_loader)

        pbar = tqdm(
            enumerate(self.train_loader, start=1),
            total=num_batches,
            desc=f"Epoch {epoch:02d}/{self.config.epochs:02d} [Train]",
            leave=False
        )

        for batch_idx, (features, targets) in pbar:
            self.global_step += 1
            if isinstance(features, (tuple, list)):
                p3, p4, p5 = features
                p3 = p3.to(self.device, non_blocking=True)
                p4 = p4.to(self.device, non_blocking=True)
                p5 = p5.to(self.device, non_blocking=True)
                features_input = (p3, p4, p5)
            else:
                features_input = features.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # Mixed Precision Forward & Loss
            with get_autocast(self.device, self.config.amp):
                logits = self.model(features_input, return_sequence=True)
                loss = self.criterion(logits, targets)

            # Scaled Backward & Gradient Clipping
            self.scaler.scale(loss).backward()
            if self.config.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Tính Accuracy theo khung hình quyết định cuối
            if logits.dim() == 3:
                preds = logits[:, -1, :].argmax(dim=-1)
            else:
                preds = logits.argmax(dim=-1)

            batch_correct = (preds == targets).sum().item()
            batch_total = len(targets)
            batch_loss = loss.item()
            batch_acc = batch_correct / batch_total

            total_loss += batch_loss * batch_total
            total_correct += batch_correct
            total_samples += batch_total

            # Ghi log Batch-level
            self.logger.log_train_batch(loss=batch_loss, accuracy=batch_acc, step=self.global_step)

            pbar.set_postfix({"loss": f"{batch_loss:.4f}", "acc": f"{batch_acc*100:.1f}%"})

        epoch_loss = total_loss / total_samples if total_samples > 0 else 0.0
        epoch_acc  = total_correct / total_samples if total_samples > 0 else 0.0
        return epoch_loss, epoch_acc

    def val_epoch(self, epoch: int) -> Tuple[float, float, float, float, float]:
        """Vòng lặp đánh giá trên tập Validation, tính Loss, Acc, Precision, Recall, F1."""
        self.model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for features, targets in self.val_loader:
                if isinstance(features, (tuple, list)):
                    p3, p4, p5 = features
                    p3 = p3.to(self.device, non_blocking=True)
                    p4 = p4.to(self.device, non_blocking=True)
                    p5 = p5.to(self.device, non_blocking=True)
                    features_input = (p3, p4, p5)
                else:
                    features_input = features.to(self.device, non_blocking=True)
                targets_dev = targets.to(self.device, non_blocking=True)

                with get_autocast(self.device, self.config.amp):
                    logits = self.model(features_input, return_sequence=True)
                    loss = self.criterion(logits, targets_dev)

                total_loss += loss.item() * len(targets)
                if logits.dim() == 3:
                    preds = logits[:, -1, :].argmax(dim=-1)
                else:
                    preds = logits.argmax(dim=-1)

                total_correct += (preds == targets_dev).sum().item()
                total_samples += len(targets)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(targets.numpy().tolist())

        epoch_loss = total_loss / total_samples if total_samples > 0 else 0.0
        epoch_acc  = total_correct / total_samples if total_samples > 0 else 0.0

        # Tính Precision, Recall, F1 của lớp Buồn ngủ (1)
        prec = precision_score(all_targets, all_preds, pos_label=1, zero_division=0)
        rec  = recall_score(all_targets, all_preds, pos_label=1, zero_division=0)
        f1   = f1_score(all_targets, all_preds, pos_label=1, zero_division=0)

        # Ghi log ảnh Confusion Matrix mỗi 5 epoch và ở epoch cuối
        if epoch % 5 == 0 or epoch == self.config.epochs:
            cm = confusion_matrix(all_targets, all_preds)
            self.logger.log_confusion_matrix(cm, epoch)

        return epoch_loss, epoch_acc, prec, rec, f1

    def fit(self) -> None:
        """Thực thi toàn bộ quy trình huấn luyện và giám sát."""
        print("\n" + "=" * 80)
        print(f"[*] BẮT ĐẦU HUẤN LUYỆN DEEP LSTM ({self.config.epochs} EPOCHS | AMP FP16 = {self.config.amp})")
        print(f"[*] Thiết bị: {self.device} | Batch Size: {self.config.batch_size} | LR: {self.config.lr0}")
        print("=" * 80)

        start_total_time = time.time()

        for epoch in range(self.start_epoch, self.config.epochs + 1):
            t_start = time.time()

            # 1. Huấn luyện
            train_loss, train_acc = self.train_epoch(epoch)

            # 2. Cập nhật Learning Rate Scheduler
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                self.scheduler.step()

            # 3. Ghi log Train Epoch
            self.logger.log_train_epoch(train_loss, train_acc, current_lr, epoch)

            # 4. Đánh giá Validation
            val_loss, val_acc, prec, rec, f1 = self.val_epoch(epoch)
            self.logger.log_val_epoch(val_loss, val_acc, epoch)
            self.logger.log_metrics({"Precision": prec, "Recall": rec, "F1_Score": f1}, epoch)

            # 5. Lưu lịch sử
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["precision"].append(prec)
            self.history["recall"].append(rec)
            self.history["f1"].append(f1)
            self.history["lr"].append(current_lr)

            # 6. Kiểm tra Best Checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_f1_score = f1

            # 7. Lưu Checkpoint
            self.save_checkpoint(epoch, val_acc, f1, is_best=is_best)

            elapsed = time.time() - t_start
            print(f"Epoch {epoch:02d}/{self.config.epochs:02d} [{elapsed:.2f}s] | "
                  f"Train Loss: {train_loss:.4f} - Acc: {train_acc*100:.2f}% | "
                  f"Val Loss: {val_loss:.4f} - Acc: {val_acc*100:.2f}% | "
                  f"F1: {f1:.4f}" + (" [BEST]" if is_best else ""))

        total_time_min = (time.time() - start_total_time) / 60
        print("\n" + "=" * 80)
        print(f"[SUCCESS] HUẤN LUYỆN HOÀN TẤT TRONG {total_time_min:.2f} PHÚT!")
        print(f"[+] Kỷ lục Best Val Accuracy : {self.best_val_acc * 100:.2f}%")
        print(f"[+] Best F1-Score            : {self.best_f1_score:.4f}")
        print(f"[+] Thư mục Checkpoint      : {self.config.checkpoint_dir}")
        print("=" * 80)

        self.logger.close()

    def export_results_zip(self, zip_filename: str = "lstm_experiment_results.zip") -> str:
        """Đóng gói tự động toàn bộ log TensorBoard và checkpoint thành file zip."""
        zip_path = Path(zip_filename)
        print(f"[+] Đang đóng gói kết quả thí nghiệm vào: {zip_path.resolve()}...")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # 1. Nén toàn bộ log TensorBoard
            log_dir = Path(self.config.tb_log_dir)
            if log_dir.exists():
                for f in log_dir.rglob("*"):
                    if f.is_file():
                        zipf.write(f, f.relative_to(log_dir.parent))
            # 2. Nén checkpoints
            ckpt_dir = Path(self.config.checkpoint_dir)
            if ckpt_dir.exists():
                for f in ckpt_dir.rglob("*"):
                    if f.is_file():
                        zipf.write(f, f.relative_to(ckpt_dir.parent))

        size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"[SUCCESS] Đã tạo gói kết quả thành công ({size_mb:.2f} MB): {zip_path.resolve()}")
        return str(zip_path)


# Alias tương thích
TrainLtsm = TrainLstm
TrainLSTM = TrainLstm


def parse_args():
    """Hỗ trợ chạy huấn luyện từ terminal với các tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Deep LSTM (SUST Dataset)")
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs, help="Số lượng epochs")
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size, help="Batch size")
    parser.add_argument("--lr", type=float, default=TrainConfig.lr0, help="Learning rate khởi tạo")
    parser.add_argument("--device", type=str, default=TrainConfig.device, help="Thiết bị ('cuda' hoặc 'cpu')")
    parser.add_argument("--amp", action="store_true", default=TrainConfig.amp, help="Bật AMP FP16")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="Tắt AMP")
    parser.add_argument("--train-pt", type=str, default=TrainConfig.train_pt, help="Đường dẫn file train .pt")
    parser.add_argument("--val-pt", type=str, default=TrainConfig.val_pt, help="Đường dẫn file val .pt")
    parser.add_argument("--resume", type=str, default="", help="Đường dẫn checkpoint để resume")
    parser.add_argument("--export-zip", action="store_true", default=False, help="Đóng gói kết quả ra file zip")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr0=args.lr,
        device=args.device,
        amp=args.amp,
        train_pt=args.train_pt,
        val_pt=args.val_pt,
        resume=args.resume
    )

    trainer = TrainLstm(config=cfg)
    trainer.fit()

    if args.export_zip:
        trainer.export_results_zip()
