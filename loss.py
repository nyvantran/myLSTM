from typing import Optional, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TrainConfig


class DrowsinessLoss(nn.Module):
    """
    Hàm mất mát Cross-Entropy Loss tiêu chuẩn cho mô hình nhận diện buồn ngủ Deep LSTM.
    Đồng bộ 100% với CELL 7 trong `datn4ni2.ipynb`.
    
    Hỗ trợ cả 2 trường hợp:
    1. Giám sát toàn bộ chuỗi thời gian (Sequence-level supervision): 
       Logits có shape [B, T, C=2], targets có shape [B].
       Tự động reshape logits thành [B * T, C] và mở rộng targets thành [B * T] để tính loss từng frame.
    2. Giám sát tại khung hình cuối (Clip-level / Final frame supervision):
       Logits có shape [B, C=2], targets có shape [B].
    """

    def __init__(self, weight: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super(DrowsinessLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    @classmethod
    def from_config(cls, config: Any) -> "DrowsinessLoss":
        """Khởi tạo DrowsinessLoss từ TrainConfig."""
        pos_w = getattr(config, "pos_weight", None)
        weight = None
        if pos_w is not None and pos_w > 0:
            weight = torch.tensor([1.0, float(pos_w)], dtype=torch.float32)
        return cls(weight=weight, reduction="mean")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Tính toán Cross-Entropy Loss.

        Args:
            logits (torch.Tensor): Raw logits đầu ra từ mô hình [B, T, C] hoặc [B, C].
            targets (torch.Tensor): Nhãn nguyên bản Ground Truth [B] hoặc [B, T].

        Returns:
            torch.Tensor: Giá trị loss vô hướng (scalar).
        """
        if logits.dim() == 3:
            b, t, c = logits.shape
            logits_flat = logits.reshape(b * t, c)
            if targets.dim() == 1:
                # Mở rộng nhãn [B] thành [B * T] cho toàn bộ các frame trong chuỗi
                targets_expanded = targets.unsqueeze(1).expand(b, t).reshape(b * t)
            else:
                targets_expanded = targets.reshape(b * t)
            return self.criterion(logits_flat, targets_expanded)

        return self.criterion(logits, targets)


class DrowsinessBCELoss(nn.Module):
    """
    Hàm mất mát Binary Cross Entropy (BCE) tùy chỉnh dành cho mô hình Deep LSTM
    khi đầu ra đã được đưa qua hàm kích hoạt Softmax (trả về xác suất 2 lớp [Tỉnh táo, Buồn ngủ]).

    Hỗ trợ cả nhãn dạng chỉ số (0/1) và dạng One-hot, tự động kẹp (clamp) xác suất 
    trong khoảng [eps, 1 - eps] để tránh lỗi log(0).
    """

    def __init__(self, eps: float = 1e-7, reduction: str = 'mean', pos_weight: Optional[float] = None):
        super(DrowsinessBCELoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.pos_weight = pos_weight

    @classmethod
    def from_config(cls, config: Any) -> "DrowsinessBCELoss":
        """Khởi tạo DrowsinessBCELoss trực tiếp từ đối tượng TrainConfig."""
        return cls(
            eps=getattr(config, "bce_eps", 1e-7),
            reduction=getattr(config, "bce_reduction", "mean"),
            pos_weight=getattr(config, "pos_weight", None)
        )

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Tính toán BCE Loss.

        Args:
            preds (torch.Tensor): Đầu ra Softmax [B, T, 2] hoặc [B, 2].
            targets (torch.Tensor): Nhãn thực tế dạng index [B, T]/[B] hoặc One-hot [B, T, 2]/[B, 2].

        Returns:
            torch.Tensor: Giá trị loss thu được theo reduction ('mean', 'sum', hoặc 'none').
        """
        preds_clamped = torch.clamp(preds, min=self.eps, max=1.0 - self.eps)

        if targets.dim() == preds.dim():
            p1 = preds_clamped[..., 1]
            y1 = targets.float()[..., 1]
        else:
            p1 = preds_clamped[..., 1]
            y1 = targets.float()
            if y1.dim() < p1.dim():
                y1 = y1.unsqueeze(-1)

        loss_pos = y1 * torch.log(p1)
        if self.pos_weight is not None:
            loss_pos = loss_pos * self.pos_weight

        loss_neg = (1.0 - y1) * torch.log(1.0 - p1)
        bce_elementwise = -(loss_pos + loss_neg)

        if self.reduction == 'mean':
            return torch.mean(bce_elementwise)
        elif self.reduction == 'sum':
            return torch.sum(bce_elementwise)
        else:
            return bce_elementwise


def get_loss_function(config: Optional[TrainConfig] = None) -> nn.Module:
    """
    Factory function khởi tạo hàm mất mát dựa trên config.loss_type:
    - 'ce' hoặc 'crossentropy': DrowsinessLoss (Mặc định chuẩn theo datn4ni2.ipynb)
    - 'bce': DrowsinessBCELoss
    """
    if config is None:
        config = TrainConfig()
    
    loss_name = getattr(config, "loss_type", "ce").lower()
    if loss_name in ("ce", "crossentropy", "cross_entropy"):
        return DrowsinessLoss.from_config(config)
    elif loss_name == "bce":
        return DrowsinessBCELoss.from_config(config)
    else:
        print(f"[Loss][Warning] Không nhận diện loss_type='{loss_name}'. Sử dụng DrowsinessLoss mặc định.")
        return DrowsinessLoss.from_config(config)


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    print("==========================================================================")
    print("=== KIỂM TRA CÁC HÀM MẤT MÁT (LOSS FUNCTIONS) ===")
    print("==========================================================================")

    # 1. Kiểm thử DrowsinessLoss (Cross-Entropy chuẩn từ datn4ni2.ipynb)
    ce_crit = DrowsinessLoss()
    b, t, c = 4, 120, 2
    dummy_logits_seq = torch.randn(b, t, c)
    dummy_logits_clip = torch.randn(b, c)
    dummy_targets = torch.randint(0, 2, (b,))

    loss_seq = ce_crit(dummy_logits_seq, dummy_targets)
    loss_clip = ce_crit(dummy_logits_clip, dummy_targets)
    print(f"[+] DrowsinessLoss - Sequence Input [4, 120, 2]: {loss_seq.item():.4f}")
    print(f"[+] DrowsinessLoss - Clip Input     [4, 2]      : {loss_clip.item():.4f}")

    # 2. Kiểm thử DrowsinessBCELoss (Legacy)
    bce_crit = DrowsinessBCELoss()
    probs_seq = F.softmax(dummy_logits_seq, dim=-1)
    loss_bce = bce_crit(probs_seq, dummy_targets)
    print(f"[+] DrowsinessBCELoss - Sequence Probabilities  : {loss_bce.item():.4f}")

    # 3. Kiểm thử factory function
    crit_factory = get_loss_function(TrainConfig(loss_type="ce"))
    print(f"[+] get_loss_function(loss_type='ce')           : {type(crit_factory).__name__}")
    print("==========================================================================")
