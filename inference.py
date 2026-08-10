import os
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, Union
from pathlib import Path

# Import NMSFreeDetector từ myCNN và DeepLSTMClassifier từ root project
from myCNN.src.model import NMSFreeDetector
from model import DeepLSTMClassifier
from config import TrainConfig

# Đường dẫn mặc định tới checkpoint manifest và trọng số của NMSFreeDetector
MANIFEST_PATH = Path(__file__).parent / "myCNN" / "checkpoints_ftCOCO" / "model_mainfest.json"
WEIGHTS_PATH = Path(__file__).parent / "myCNN" / "checkpoints_ftCOCO" / "ft_step00091000.pt"


class FinalModel(nn.Module):
    """
    Mô hình kết hợp End-to-End (FinalModel) gom trọn toàn bộ Pipeline:
    1. CNN Feature Extractor (NMSFreeDetector): Trích xuất đặc trưng không gian (spatial feature map) [B*T, 640, H, W] từ các khung hình thô.
    2. Deep LSTM Classifier (DeepLSTMClassifier): Nhận chuỗi đặc trưng không gian [B, T, 640, 15, 15] 
       và đưa ra dự đoán phân loại trạng thái buồn ngủ (Drowsiness Detection) [B, 2] hoặc [B, T, 2].
    """

    def __init__(
            self,
            cnn_model: Optional[NMSFreeDetector] = None,
            lstm_model: Optional[DeepLSTMClassifier] = None,
            cnn_manifest_path: Optional[str] = str(MANIFEST_PATH),
            cnn_weights_path: Optional[str] = str(WEIGHTS_PATH),
            lstm_config: Optional[TrainConfig] = None,
            freeze_cnn: bool = True,
            alpha: float = 0.5,
            beta: float = 0.5
    ):
        """
        Khởi tạo FinalModel.

        Args:
            cnn_model (NMSFreeDetector, optional): Instance mô hình CNN. Nếu None sẽ khởi tạo từ cnn_manifest_path.
            lstm_model (DeepLSTMClassifier, optional): Instance mô hình LSTM. Nếu None sẽ khởi tạo từ lstm_config.
            cnn_manifest_path (str, optional): Đường dẫn file manifest JSON để khởi tạo NMSFreeDetector.
            cnn_weights_path (str, optional): Đường dẫn checkpoint (.pt/.pth) chứa trọng số cho CNN.
            lstm_config (TrainConfig, optional): Cấu hình TrainConfig cho DeepLSTMClassifier.
            freeze_cnn (bool): Đóng bằng trọng số của CNN khi suy luận / huấn luyện LSTM.
        """
        super().__init__()

        # 1. Khởi tạo CNN Feature Extractor (NMSFreeDetector)
        if cnn_model is not None:
            self.cnn = cnn_model
        elif cnn_manifest_path and os.path.exists(cnn_manifest_path):
            self.cnn = NMSFreeDetector.from_config(cnn_manifest_path)
        else:
            self.cnn = NMSFreeDetector()

        # Nạp trọng số pretrained cho CNN nếu có file checkpoint
        if cnn_weights_path and os.path.exists(cnn_weights_path):
            try:
                ckpt = torch.load(cnn_weights_path, map_location="cpu")
                if "model" in ckpt:
                    self.cnn.load_state_dict(ckpt["model"], strict=False)
                elif "backbone" in ckpt:
                    self.cnn.load_trunk(ckpt, strict=False)
                else:
                    self.cnn.load_state_dict(ckpt, strict=False)
                print(f"[FinalModel] Đã nạp thành công trọng số CNN từ: {cnn_weights_path}")
            except Exception as e:
                print(f"[FinalModel][Warning] Không thể nạp trọng số CNN từ {cnn_weights_path}: {e}")

        # Đóng bằng trọng số của CNN nếu freeze_cnn=True
        if freeze_cnn:
            self.cnn.eval()
            for param in self.cnn.parameters():
                param.requires_grad = False

        # 2. Khởi tạo Deep LSTM Classifier
        if lstm_model is not None:
            self.lstm = lstm_model
        elif lstm_config is not None:
            self.lstm = DeepLSTMClassifier.from_config(lstm_config)
        else:
            default_config = TrainConfig()
            self.lstm = DeepLSTMClassifier.from_config(default_config)

        self.alpha = alpha
        self.beta = beta

    def load_dict(
            self,
            checkpoint_path: str,
            strict: bool = True,
            map_location: str = "cpu"
    ):
        """
        Nạp trọng số cho toàn bộ FinalModel từ file checkpoint (.pth/.pt).
        """
        ckpt = torch.load(checkpoint_path, map_location=map_location)
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        self.load_state_dict(state_dict, strict=strict)
        print(f"[FinalModel] Đã nạp state_dict thành công từ: {checkpoint_path}")

    def load_lstm_weights(
            self,
            checkpoint_path: str,
            strict: bool = True,
            map_location: str = "cpu"
    ):
        """
        Nạp trọng số riêng cho nhánh mô hình DeepLSTMClassifier từ file checkpoint.
        """
        ckpt = torch.load(checkpoint_path, map_location=map_location)
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        self.lstm.load_state_dict(state_dict, strict=strict)
        print(f"[FinalModel] Đã nạp trọng số riêng cho LSTM từ: {checkpoint_path}")

    def forward(
            self,
            x: torch.Tensor,

    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:  # chưa xong
        """
        Quá trình lan chạy model thực tế.

        Args:
            x (torch.Tensor): Tensor chuỗi khung hình video:
                - Dạng 5D khung hình thô: [Batch (B), Seq_Len (T), Channels=3, Height, Width]
        Returns:
            - lstm_out: Tensor xác suất dự đoán (0: Tỉnh táo, 1: Buồn ngủ) [B, 2] hoặc [B, T, 2].
        """

        bs, sl, c, w, h = x.shape
        x_flat = x.contiguous().view(-1, c, w, h)
        p3, p4, p5 = self.cnn.backbone(x_flat)  # p5 [bs*sl, 640, 15, 15]
        print(f"p3 shape: ")
        p3, p4, p5 = self.cnn.neck(p3, p4, p5)

        # Nhánh 1: neck và head (Detection)
        # results_cnn = self.head([p3, p4, p5])

        # Nhánh 2: Deep LSTM Classifier
        p3_5d = p3.view(bs, sl, *p3.shape[1:])
        p4_5d = p4.view(bs, sl, *p4.shape[1:])
        p5_5d = p5.view(bs, sl, *p5.shape[1:])
        results_lstm = self.lstm((p3_5d, p4_5d, p5_5d))
        return results_lstm
        # results = results_cnn * self.alpha + results_lstm * self.beta
        # return results


if __name__ == "__main__":
    import sys
    from dataset import MyLSTMDataset
    from myCNN.src.runtime.infer import letterbox
    from functools import partial

    dataset_path = r"D:\Project\AI\dataset\VBDDD-dataset"
    tranforms = partial(letterbox, new_size=480)
    # Tạo Dataset với seq_len = 60 khung hình và resize khung hình 224x224
    dataset = MyLSTMDataset(
        dataset_dir=dataset_path,
        seq_len=120,
        transform=tranforms
    )
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    final_model = FinalModel(freeze_cnn=True)

    video_tensor, label = dataset[0]

    video_tensor = video_tensor.unsqueeze(0)

    result = final_model(video_tensor)
    print(f"results = {result}")
