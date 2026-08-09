from typing import Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


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
            eps=config.bce_eps,
            reduction=config.bce_reduction,
            pos_weight=config.pos_weight
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
        # Kẹp giá trị xác suất tránh log(0)
        preds_clamped = torch.clamp(preds, min=self.eps, max=1.0 - self.eps)

        # Lấy xác suất và nhãn tương ứng cho lớp 1 (Buồn ngủ)
        if targets.dim() == preds.dim():
            p1 = preds_clamped[..., 1]
            y1 = targets.float()[..., 1]
        else:
            p1 = preds_clamped[..., 1]
            y1 = targets.float()
            if y1.dim() < p1.dim():
                y1 = y1.unsqueeze(-1)

        # Tính BCE Loss
        loss_pos = y1 * torch.log(p1)
        if self.pos_weight is not None:
            loss_pos = loss_pos * self.pos_weight

        loss_neg = (1.0 - y1) * torch.log(1.0 - p1)
        bce_elementwise = -(loss_pos + loss_neg)

        # Gom nhóm kết quả (reduction)
        if self.reduction == 'mean':
            return torch.mean(bce_elementwise)
        elif self.reduction == 'sum':
            return torch.sum(bce_elementwise)
        else:
            return bce_elementwise


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    print("=== KIỂM TRA HÀM BINARY CROSS ENTROPY LOSS ===")

    criterion = DrowsinessBCELoss(eps=1e-7, reduction='mean')
    batch_size, seq_len = 4, 60

    # 1. Dự đoán chuỗi [B, T, 2]
    raw_logits = torch.randn(batch_size, seq_len, 2)
    preds_seq = F.softmax(raw_logits, dim=-1)
    targets_seq = torch.randint(0, 2, (batch_size, seq_len))
    loss_seq = criterion(preds_seq, targets_seq)
    print(f"1. Sequence Loss [4, 60, 2]: {loss_seq.item():.6f}")

    # 2. Dự đoán đơn [B, 2]
    preds_clip = F.softmax(torch.randn(batch_size, 2), dim=-1)
    targets_clip = torch.randint(0, 2, (batch_size,))
    loss_clip = criterion(preds_clip, targets_clip)
    print(f"2. Clip Loss [4, 2]: {loss_clip.item():.6f}")

    # 3. Target dạng One-Hot [B, T, 2]
    targets_onehot = F.one_hot(targets_seq, num_classes=2)
    loss_onehot = criterion(preds_seq, targets_onehot)
    print(f"3. One-Hot Loss [4, 60, 2]: {loss_onehot.item():.6f}")
