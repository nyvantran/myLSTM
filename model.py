from pathlib import Path
from typing import Optional, Union, Tuple, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TrainConfig


class SpatialFeatureAdapter(nn.Module):
    """
    Bộ chuyển đổi / nén đặc trưng không gian đa mức (Spatial Feature Adapter).
    Mục đích: Chuyển đổi các bản đồ đặc trưng không gian đa mức (p3, p4, p5) sinh ra từ Backbone và Neck
    của mô hình CNN (với các kích thước thực tế từ PAFPN:
      - p3: [Batch, Seq_Len, 224, 60, 60] hoặc [bz, 224, 60, 60]
      - p4: [Batch, Seq_Len, 448, 30, 30] hoặc [bz, 448, 30, 30]
      - p5: [Batch, Seq_Len, 640, 15, 15] hoặc [bz, 640, 15, 15])
    thành chuỗi vector đặc trưng x^t 256 chiều ([Batch, Seq_Len, 256] hoặc [bz, 256]) trước khi đưa vào LSTM.

    Hỗ trợ linh hoạt các định dạng đầu vào:
    1. Tuple/List: adapter((p3, p4, p5)) hoặc adapter([p3, p4, p5])
    2. Nhiều tham số vị trí: adapter(p3, p4, p5)
    3. Tensor đơn lẻ:
       - Tensor 5D đã ghép kênh: [Batch, Seq_Len, 1312, 15, 15] hoặc tensor 5D đơn [Batch, Seq_Len, 640, 15, 15]
       - Tensor 4D đã ghép kênh: [bz, 1312, 15, 15] hoặc tensor 4D đơn [bz, 640, 15, 15]
       - Tensor 6D xếp chồng: [Batch, Seq_Len, 3, 640, 15, 15]
    """

    def __init__(
            self,
            in_channels: Union[int, Tuple[int, ...], List[int]] = TrainConfig.cnn_neck_channels,
            out_dim: int = TrainConfig.input_dim,
            num_features: int = TrainConfig.cnn_num_features,
            fusion: str = TrainConfig.spatial_fusion,
            spatial_size: Tuple[int, int] = TrainConfig.cnn_spatial_size,
            dropout: float = TrainConfig.adapter_dropout,
            use_norm: bool = getattr(TrainConfig, "use_norm", True)
    ):
        """
        Khởi tạo SpatialFeatureAdapter.

        Args:
            in_channels (Union[int, Tuple[int, ...], List[int]]): Cấu hình kênh của các tầng Neck (mặc định: (224, 448, 640)).
            out_dim (int): Số chiều vector đầu ra cho LSTM (mặc định: 256).
            num_features (int): Số lượng feature maps đầu vào (p3, p4, p5 -> 3).
            fusion (str): Chiến lược kết hợp đặc trưng: 'concat' | 'sum' | 'mean' | 'conv' | 'attention'.
            spatial_size (Tuple[int, int]): Kích thước không gian mục tiêu khi căn chỉnh 2D (15, 15).
            dropout (float): Tỷ lệ Dropout sau projection.
        """
        super(SpatialFeatureAdapter, self).__init__()
        self.out_dim = out_dim
        self.fusion = fusion.lower()
        self.spatial_size = spatial_size

        # Xác định danh sách số kênh cho từng mức p3, p4, p5
        if isinstance(in_channels, (tuple, list)):
            self.neck_channels = tuple(in_channels)
            self.num_features = len(self.neck_channels)
            self.total_in_channels = sum(self.neck_channels)
        elif in_channels == 1312:
            self.neck_channels = (224, 448, 640)
            self.num_features = 3
            self.total_in_channels = 1312
        else:
            self.num_features = num_features
            self.neck_channels = (in_channels,) * num_features
            self.total_in_channels = in_channels * num_features if self.fusion == "concat" else in_channels

        # Bộ nén không gian Adaptive Pooling về kích thước (1, 1) cho từng feature map
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Thiết lập cấu trúc tầng theo chiến lược Fusion
        if self.fusion == "concat":
            # Ghép nối các vector sau pooling: 224 + 448 + 640 = 1312 -> out_dim (256)
            if use_norm:
                self.projection = nn.Sequential(
                    nn.Linear(self.total_in_channels, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.ReLU(inplace=True)
                )
            else:
                self.projection = nn.Linear(self.total_in_channels, out_dim)
            self.level_projections = None
            self.conv_fusion = None

        elif self.fusion in ("sum", "mean"):
            # Chiếu riêng từng level về out_dim rồi cộng dồn / lấy trung bình
            self.level_projections = nn.ModuleList([
                nn.Linear(c, out_dim) for c in self.neck_channels
            ])
            self.projection = None
            self.conv_fusion = None

        elif self.fusion == "attention":
            # Chiếu riêng từng level về out_dim kết hợp trọng số Softmax học được
            self.level_projections = nn.ModuleList([
                nn.Linear(c, out_dim) for c in self.neck_channels
            ])
            self.level_weights = nn.Parameter(torch.ones(self.num_features))
            self.projection = None
            self.conv_fusion = None

        elif self.fusion == "conv":
            # Đồng nhất không gian 2D về spatial_size (15, 15) rồi qua 1x1 Conv
            self.spatial_align_pool = nn.AdaptiveAvgPool2d(spatial_size)
            self.conv_fusion = nn.Sequential(
                nn.Conv2d(self.total_in_channels, out_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            )
            self.projection = nn.Linear(out_dim, out_dim)
            self.level_projections = None

        else:
            raise ValueError(f"Phương thức fusion không hợp lệ: '{fusion}'. "
                             f"Chọn một trong: 'concat', 'sum', 'mean', 'conv', 'attention'")

        # Projection dự phòng khi nhận duy nhất 1 tầng đơn lẻ (vd chỉ p5 = 640)
        self.single_projections = nn.ModuleDict({
            str(c): nn.Linear(c, out_dim) for c in set(self.neck_channels + (224, 448, 640, 1312, 1920))
        })

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def _pool_single_tensor(self, feat: torch.Tensor) -> Tuple[torch.Tensor, Optional[Tuple[int, int]]]:
        """
        Nén không gian cho 1 tensor đặc trưng bất kỳ (tự thích ứng với kích thước H x W bất kỳ: 60x60, 30x30, 15x15).
        - 5D [B, T, C, H, W] -> pooled [B, T, C] và trả về (B, T)
        - 4D [B, C, H, W] -> pooled [B, C] và trả về None
        - 3D/2D -> giữ nguyên
        """
        if feat.dim() == 5:
            b, t, c, h, w = feat.shape
            feat_flat = feat.reshape(b * t, c, h, w)
            pooled = self.pool(feat_flat).view(b * t, c)
            return pooled.view(b, t, c), (b, t)
        elif feat.dim() == 4:
            b, c, h, w = feat.shape
            pooled = self.pool(feat).view(b, c)
            return pooled, None
        elif feat.dim() in (2, 3):
            return feat, (feat.shape[0], feat.shape[1]) if feat.dim() == 3 else None
        else:
            raise ValueError(f"Kích thước tensor không hợp lệ: {feat.shape} (yêu cầu 4D hoặc 5D)")

    def forward(
            self,
            x: Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]],
            *args: torch.Tensor
    ) -> torch.Tensor:
        """
        Quá trình lan truyền tiến (Forward pass).

        Args:
            x: Tensor đặc trưng hoặc Tuple/List chứa (p3, p4, p5).
            *args: Các tensor đặc trưng p4, p5 bổ sung nếu truyền dạng adapter(p3, p4, p5).

        Returns:
            torch.Tensor: Chuỗi vector đặc trưng kích thước [B, T, out_dim] hoặc [bz, out_dim].
        """
        # 1. Gom nhóm danh sách các feature map đầu vào
        if len(args) > 0:
            features = [x, *args]
        elif isinstance(x, (tuple, list)):
            features = list(x)
        elif isinstance(x, torch.Tensor) and x.dim() == 6:
            # Dạng [B, T, 3, C, H, W]
            if x.shape[2] == self.num_features:
                features = [x[:, :, i, ...] for i in range(self.num_features)]
            elif x.shape[1] == self.num_features:
                features = [x[:, i, ...] for i in range(self.num_features)]
            else:
                features = [f.squeeze(0) for f in torch.chunk(x, chunks=x.shape[0], dim=0)]
        else:
            features = [x]

        # 2. Xử lý trường hợp đầu vào là 1 tensor đơn lẻ
        if len(features) == 1:
            single_feat = features[0]
            if not isinstance(single_feat, torch.Tensor):
                raise TypeError(f"Đầu vào phải là torch.Tensor, nhận được {type(single_feat)}")

            c_dim = 2 if single_feat.dim() == 5 else (1 if single_feat.dim() == 4 else -1)
            num_c = single_feat.shape[c_dim] if single_feat.dim() in (4, 5) else single_feat.shape[-1]

            # Nếu tensor đã được ghép 1312 kênh từ trước
            if num_c == self.total_in_channels and self.projection is not None:
                pooled, _ = self._pool_single_tensor(single_feat)
                out = self.projection(pooled)
                return self.dropout(out)

            elif str(num_c) in self.single_projections:
                pooled, _ = self._pool_single_tensor(single_feat)
                out = self.single_projections[str(num_c)](pooled)
                return self.dropout(out)

            elif single_feat.dim() in (2, 3) and single_feat.shape[-1] == self.out_dim:
                return self.dropout(single_feat)

            else:
                pooled, _ = self._pool_single_tensor(single_feat)
                if self.projection is not None and pooled.shape[-1] == self.total_in_channels:
                    out = self.projection(pooled)
                else:
                    out = F.linear(pooled, self.projection.weight[:, :pooled.shape[-1]] if self.projection is not None else None)
                return self.dropout(out)

        # 3. Xử lý trường hợp đầu vào gồm nhiều feature maps (p3, p4, p5)
        # TH 3.1: Fusion 'conv' (căn chỉnh không gian 2D về spatial_size trước khi ghép)
        if self.fusion == "conv":
            is_5d = features[0].dim() == 5
            if is_5d:
                b, t = features[0].shape[0], features[0].shape[1]
                aligned_feats = [
                    self.spatial_align_pool(f.reshape(b * t, f.shape[2], f.shape[3], f.shape[4]))
                    for f in features
                ]
                cat_spatial = torch.cat(aligned_feats, dim=1)  # [B*T, 1312, 15, 15]
                conv_out = self.conv_fusion(cat_spatial)       # [B*T, 256, 15, 15]
                pooled = self.pool(conv_out).view(b, t, -1)     # [B, T, 256]
            else:
                aligned_feats = [self.spatial_align_pool(f) for f in features]
                cat_spatial = torch.cat(aligned_feats, dim=1)  # [bz, 1312, 15, 15]
                conv_out = self.conv_fusion(cat_spatial)       # [bz, 256, 15, 15]
                pooled = self.pool(conv_out).view(cat_spatial.shape[0], -1)  # [bz, 256]
            out = self.projection(pooled)
            return self.dropout(out)

        # TH 3.2: Các phương thức nén dựa trên Adaptive Average Pooling từng tầng
        pooled_feats = []
        for feat in features:
            pooled_f, _ = self._pool_single_tensor(feat)
            pooled_feats.append(pooled_f)

        if self.fusion == "concat":
            # Ghép nối dọc theo chiều kênh: [B, T, 224 + 448 + 640] = [B, T, 1312] hoặc [bz, 1312]
            fused = torch.cat(pooled_feats, dim=-1)
            out = self.projection(fused)

        elif self.fusion == "sum":
            # Chiếu từng tầng về out_dim rồi cộng dồn: [B, T, 256] hoặc [bz, 256]
            proj_feats = [proj(f) for proj, f in zip(self.level_projections, pooled_feats)]
            out = torch.stack(proj_feats, dim=0).sum(dim=0)

        elif self.fusion == "mean":
            # Chiếu từng tầng về out_dim rồi lấy trung bình: [B, T, 256] hoặc [bz, 256]
            proj_feats = [proj(f) for proj, f in zip(self.level_projections, pooled_feats)]
            out = torch.stack(proj_feats, dim=0).mean(dim=0)

        elif self.fusion == "attention":
            # Chiếu từng tầng về out_dim rồi nhân trọng số Softmax
            proj_feats = [proj(f) for proj, f in zip(self.level_projections, pooled_feats)]
            stacked = torch.stack(proj_feats, dim=0)  # [num_features, B, T, out_dim]
            weights = F.softmax(self.level_weights, dim=0)
            w_shape = [self.num_features] + [1] * (stacked.dim() - 1)
            out = (stacked * weights.view(*w_shape)).sum(dim=0)

        return self.dropout(out)


class DeepLSTMClassifier(nn.Module):
    """
    Mô hình LSTM sâu (Deep LSTM Classifier - 3 lớp xếp chồng) trích xuất mối quan hệ phụ thuộc 
    thời gian dài hạn (long-term temporal dependencies) dựa theo tài liệu kientruc.md.

    Cấu trúc mạng chi tiết:
    - Input: Chuỗi vector đặc trưng x^t kích thước 256 (hoặc các feature maps p3, p4, p5 [B, T, C, H, W] / [bz, C, H, W]).
    - Spatial Feature Adapter: Nén không gian đa mức p3 (224x60x60), p4 (448x30x30), p5 (640x15x15) từ CNN Neck thành chuỗi vector 256 chiều.
    - Deep LSTM: 3 lớp LSTM xếp chồng (num_layers=3), mỗi khối chứa 256 đơn vị ẩn (hidden_units=256).
    - Memory Step (Độ dài bộ nhớ): 60 khung hình (FI-DDD / VBDDD) hoặc 120 khung hình (NTHU-DDD).
    - FC Output (Lớp kết nối đầy đủ W^R, b^R): Chiếu từ 256 chiều ẩn xuống 2 chiều (Buồn ngủ / Bình thường).
    - Softmax: Tính xác suất của 2 danh mục phân loại.
    """

    def __init__(
            self,
            input_dim: int = TrainConfig.input_dim,
            hidden_dim: int = TrainConfig.hidden_dim,
            num_layers: int = TrainConfig.num_layers,
            num_classes: int = TrainConfig.num_classes,
            spatial_in_channels: Optional[Union[int, Tuple[int, ...], List[int]]] = TrainConfig.cnn_neck_channels,
            num_features: int = TrainConfig.cnn_num_features,
            fusion: str = TrainConfig.spatial_fusion,
            adapter_dropout: float = TrainConfig.adapter_dropout,
            dropout: float = TrainConfig.dropout,
            use_norm: bool = getattr(TrainConfig, "use_norm", True)
    ):
        super(DeepLSTMClassifier, self).__init__()

        self.input_dim = input_dim  # Giữ giá trị input_dim = 256
        self.hidden_dim = hidden_dim  # Giữ giá trị hidden_dim = 256
        self.num_layers = num_layers  # Giữ giá trị num_layers = 3
        self.num_classes = num_classes  # Giữ giá trị num_classes = 2
        self.use_norm = use_norm

        # Khởi tạo bộ chuyển đổi không gian nếu đầu vào là feature map MCNN (p3, p4, p5)
        if spatial_in_channels is not None:
            self.spatial_adapter = SpatialFeatureAdapter(
                in_channels=spatial_in_channels,
                out_dim=input_dim,
                num_features=num_features,
                fusion=fusion,
                dropout=adapter_dropout,
                use_norm=use_norm
            )
        else:
            self.spatial_adapter = None

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        if use_norm:
            self.fc_out = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.fc_out = nn.Linear(hidden_dim, num_classes)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Union[str, Path], map_location: str = "cpu") -> "DeepLSTMClassifier":
        """
        Khởi tạo DeepLSTMClassifier và nạp trọng số trực tiếp từ file checkpoint (.pth/.pt).
        Tự động nhận diện cấu hình lưu trong checkpoint và nạp khớp chuẩn 100% với datn4ni2.ipynb.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file checkpoint: {path}")

        ckpt = torch.load(str(path), map_location=map_location)
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

        # Tự động nhận diện kiến trúc
        use_norm = ("spatial_adapter.projection.0.weight" in state_dict) or ("fc_out.1.weight" in state_dict)
        cfg_dict = ckpt.get("config", {})

        input_dim = int(cfg_dict.get("input_dim", 256))
        hidden_dim = int(cfg_dict.get("hidden_dim", 256))
        num_layers = int(cfg_dict.get("num_layers", 3))
        num_classes = int(cfg_dict.get("num_classes", 2))

        model = cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            use_norm=use_norm
        )
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model

    @classmethod
    def from_config(cls, config: Any) -> "DeepLSTMClassifier":
        """Khởi tạo DeepLSTMClassifier trực tiếp từ đối tượng TrainConfig."""
        spatial_in_channels = getattr(config, "cnn_neck_channels", getattr(config, "cnn_out_channels", (224, 448, 640)))
        num_features = getattr(config, "cnn_num_features", getattr(config, "num_features", 3))
        fusion = getattr(config, "spatial_fusion", getattr(config, "fusion", "concat"))
        adapter_dropout = getattr(config, "adapter_dropout", 0.1)
        dropout = getattr(config, "dropout", 0.2)
        use_norm = getattr(config, "use_norm", True)

        return cls(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_classes=config.num_classes,
            spatial_in_channels=spatial_in_channels,
            num_features=num_features,
            fusion=fusion,
            adapter_dropout=adapter_dropout,
            dropout=dropout,
            use_norm=use_norm
        )

    def forward(
            self,
            x: Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]],
            *args: torch.Tensor,
            hc: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            return_sequence: bool = True,
            apply_softmax: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Quá trình lan truyền tiến (Forward pass).

        Args:
            x: Feature maps đa tầng (p3, p4, p5), feature map 5D [B, T, C, H, W], 4D [bz, C, H, W],
               chuỗi vector [B, T, D], hoặc Tuple/List/nhiều đối số chứa (p3, p4, p5).
            *args: Các tensor đặc trưng p4, p5 nếu truyền dạng model(p3, p4, p5).
            hc: Trạng thái ẩn (h_0, c_0) ban đầu của LSTM.
            return_sequence: True -> trả về kết quả cho toàn bộ chuỗi [B, T, num_classes].
                             False -> chỉ trả về kết quả ở frame cuối cùng [B, num_classes].
            apply_softmax: True -> áp dụng Softmax xác suất. False -> trả về raw logits.
        """
        # Nếu truyền dạng model(p3, p4, p5)
        if len(args) > 0:
            x = (x, *args)

        # Xử lý nén đặc trưng không gian qua SpatialFeatureAdapter nếu có
        if self.spatial_adapter is not None:
            x = self.spatial_adapter(x)
        else:
            if isinstance(x, (tuple, list)):
                x = torch.cat([
                    F.adaptive_avg_pool2d(
                        f.reshape(-1, f.shape[-3], f.shape[-2], f.shape[-1]), (1, 1)
                    ).view(f.shape[0], f.shape[1] if f.dim() == 5 else 1, -1)
                    for f in x
                ], dim=-1)
            elif isinstance(x, torch.Tensor):
                if x.dim() == 6:
                    b, t, n, c, h, w = x.shape
                    pooled = F.adaptive_avg_pool2d(x.reshape(b * t * n, c, h, w), (1, 1)).view(b, t, n * c)
                    x = pooled
                elif x.dim() == 5:
                    b, t, c, h, w = x.shape
                    x = F.adaptive_avg_pool2d(x.reshape(b * t, c, h, w), (1, 1)).view(b, t, c)
                elif x.dim() == 4:
                    b, c, h, w = x.shape
                    x = F.adaptive_avg_pool2d(x, (1, 1)).view(b, 1, c)

        # Đảm bảo x có dạng [Batch, Seq_Len, Input_Dim] cho LSTM
        if isinstance(x, torch.Tensor) and x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, input_dim]

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
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=== Kiểm tra SpatialFeatureAdapter & DeepLSTMClassifier (Neck Dimensions) ===")
    
    # 1. Kích thước thực tế từ CNN Neck (PAFPN):
    # p3: [Batch=2, Seq_Len=60, Channels=224, Height=60, Width=60]
    # p4: [Batch=2, Seq_Len=60, Channels=448, Height=30, Width=30]
    # p5: [Batch=2, Seq_Len=60, Channels=640, Height=15, Width=15]
    bz, seqlen = 2, 60
    p3 = torch.randn(bz, seqlen, 224, 60, 60)
    p4 = torch.randn(bz, seqlen, 448, 30, 30)
    p5 = torch.randn(bz, seqlen, 640, 15, 15)

    adapter = SpatialFeatureAdapter()
    model = DeepLSTMClassifier()

    out_tuple = adapter((p3, p4, p5))
    out_args = adapter(p3, p4, p5)
    out_model = model((p3, p4, p5))
    print(f"[5D Multi-scale] Adapter out (tuple): {out_tuple.shape} -> Mong đợi [2, 60, 256]")
    print(f"[5D Multi-scale] Adapter out (args):  {out_args.shape} -> Mong đợi [2, 60, 256]")
    print(f"[5D Multi-scale] Model out:           {out_model.shape} -> Mong đợi [2, 60, 2]")
    assert out_tuple.shape == (2, 60, 256)
    assert out_args.shape == (2, 60, 256)
    assert out_model.shape == (2, 60, 2)

    # 2. Kiểm tra với tensor 4D [bz=16]
    p3_4d = torch.randn(16, 224, 60, 60)
    p4_4d = torch.randn(16, 448, 30, 30)
    p5_4d = torch.randn(16, 640, 15, 15)
    out_4d = adapter((p3_4d, p4_4d, p5_4d))
    print(f"[4D Multi-scale] Adapter out:         {out_4d.shape} -> Mong đợi [16, 256]")
    assert out_4d.shape == (16, 256)

    # 3. Kiểm tra các phương thức Fusion:
    for fusion_mode in ["sum", "mean", "conv", "attention"]:
        ad = SpatialFeatureAdapter(fusion=fusion_mode)
        md = DeepLSTMClassifier(fusion=fusion_mode)
        o_ad = ad((p3, p4, p5))
        o_md = md((p3, p4, p5))
        print(f"[Fusion: {fusion_mode:<9}] Adapter: {o_ad.shape}, Model: {o_md.shape}")
        assert o_ad.shape == (2, 60, 256)
        assert o_md.shape == (2, 60, 2)

    # 4. Kiểm tra tương thích ngược với Single Tensor p5 (640 channels)
    out_p5 = adapter(p5)
    print(f"[Backward Compat] Single p5 out:     {out_p5.shape} -> Mong đợi [2, 60, 256]")
    assert out_p5.shape == (2, 60, 256)

    print("=== TẤT CẢ KIỂM TRA ĐỀU HOÀN THÀNH CHÍNH XÁC ===")


