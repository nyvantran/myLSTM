from typing import Optional, Union, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TrainConfig


class SpatialFeatureAdapter(nn.Module):
    """
    Bộ chuyển đổi / nén đặc trưng không gian (Spatial Feature Adapter).
    Mục đích: Chuyển đổi tensor bản đồ đặc trưng 5D sinh ra từ MCNN [Batch, Seq_Len, 640, 15, 15] 
    thành chuỗi vector đặc trưng x^t 256 chiều [Batch, Seq_Len, 256] trước khi đưa vào LSTM.
    """

    def __init__(
            self,
            in_channels: int = TrainConfig.cnn_out_channels,
            out_dim: int = TrainConfig.input_dim
    ):
        super(SpatialFeatureAdapter, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(in_channels, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:

            b, t, c, h, w = x.shape

            x_reshaped = x.view(b * t, c, h, w)

            pooled = self.pool(x_reshaped).view(b * t, c)

            proj = self.projection(pooled)

            return proj.view(b, t, -1)

        elif x.dim() == 4:
            b, c, h, w = x.shape

            pooled = self.pool(x).view(b, c)

            return self.projection(pooled)

        else:
            return x


class DeepLSTMClassifier(nn.Module):
    """
    Mô hình LSTM sâu (Deep LSTM Classifier - 3 lớp xếp chồng) trích xuất mối quan hệ phụ thuộc 
    thời gian dài hạn (long-term temporal dependencies) dựa theo tài liệu kientruc.md.

    Cấu trúc mạng chi tiết:
    - Input: Chuỗi vector đặc trưng x^t kích thước 256 (hoặc tensor không gian [B, T, 640, 15, 15]).
    - Deep LSTM: 3 lớp LSTM xếp chồng (num_layers=3), mỗi khối chứa 256 đơn vị ẩn (hidden_units=256).
    - Memory Step (Độ dài bộ nhớ): 60 khung hình (FI-DDD) hoặc 120 khung hình (NTHU-DDD).
    - FC Output (Lớp kết nối đầy đủ W^R, b^R): Chiếu từ 256 chiều ẩn xuống 2 chiều (Buồn ngủ / Bình thường).
    - Softmax: Tính xác suất của 2 danh mục phân loại.
    """

    def __init__(
            self,
            input_dim: int = TrainConfig.input_dim,
            hidden_dim: int = TrainConfig.hidden_dim,
            num_layers: int = TrainConfig.num_layers,
            num_classes: int = TrainConfig.num_classes,
            spatial_in_channels: Optional[int] = TrainConfig.cnn_out_channels,
            dropout: float = TrainConfig.dropout
    ):
        super(DeepLSTMClassifier, self).__init__()

        self.input_dim = input_dim  # Giữ giá trị input_dim = 256
        self.hidden_dim = hidden_dim  # Giữ giá trị hidden_dim = 256
        self.num_layers = num_layers  # Giữ giá trị num_layers = 3
        self.num_classes = num_classes  # Giữ giá trị num_classes = 2

        # Khởi tạo bộ chuyển đổi không gian nếu đầu vào là feature map MCNN [B, T, 640, 15, 15]
        if spatial_in_channels is not None:
            # Khởi tạo SpatialFeatureAdapter (biến đổi từ 640 channels -> 256 dimensions)
            self.spatial_adapter = SpatialFeatureAdapter(in_channels=spatial_in_channels, out_dim=input_dim)
        else:
            self.spatial_adapter = None

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc_out = nn.Linear(hidden_dim, num_classes)

    @classmethod
    def from_config(cls, config: Any) -> "DeepLSTMClassifier":
        """Khởi tạo DeepLSTMClassifier trực tiếp từ đối tượng TrainConfig."""
        return cls(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_classes=config.num_classes,
            spatial_in_channels=config.cnn_out_channels,
            dropout=config.dropout
        )

    def forward(
            self,
            x: torch.Tensor,
            hc: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            return_sequence: bool = True,
            apply_softmax: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Quá trình lan truyền tiến (Forward pass).

        Giải thích kích thước dữ liệu qua từng bước:
        """
        if x.dim() == 5:
            if self.spatial_adapter is not None:
                x = self.spatial_adapter(x)
            else:
                b, t, c, h, w = x.shape  # Trích xuất kích thước b, t, c, h, w: [B, T, 640, 15, 15]
                x = F.adaptive_avg_pool2d(x.view(b * t, c, h, w), (1, 1)).view(b, t, c)

        lstm_out, (h_n, c_n) = self.lstm(x, hc)

        if return_sequence:

            logits = self.fc_out(lstm_out)
        else:

            h3_last = lstm_out[:, -1, :]

            logits = self.fc_out(h3_last)

        if apply_softmax:

            output = F.softmax(logits, dim=-1)
        else:

            output = logits

        return output


if __name__ == "__main__":
    pass
