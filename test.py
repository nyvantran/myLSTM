#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: test.py
Mục đích:
    Kiểm thử và đánh giá toàn diện mô hình Deep LSTM đã huấn luyện (kế thừa từ test_model.ipynb).
    Cung cấp giao diện dòng lệnh (CLI) độc lập và xuất báo cáo kết quả chi tiết:
    - Bảng thông số định lượng: Accuracy, Balanced Acc, F1, Precision, Recall, ROC-AUC, PR-AUC, MCC, Kappa, Brier.
    - Ma trận nhầm lẫn chi tiết: TP, TN, FP, FN, FPR, FNR.
    - Biểu đồ trực quan hóa (PNG): Confusion Matrix, ROC, PR, Histogram, Threshold scan, Temporal sequence.
    - Xuất dữ liệu: test_evaluation_report.json và test_predictions.csv.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Đảm bảo console UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    cohen_kappa_score,
    brier_score_loss,
    log_loss
)

from config import TrainConfig
from dataset import PreloadedTensorDataset
from model import DeepLSTMClassifier


def evaluate_model(
    checkpoint_path: str = "lstm_experiment_results/checkpoints/best_lstm.pth",
    test_pt_path: str = "extracted_features_pt/features_sust_val.pt",
    output_dir: str = "./evaluation_results",
    batch_size: int = 64,
    device_str: str = "cuda",
    decision_threshold: float = 0.5,
    class_names: Tuple[str, str] = ("Tỉnh táo (Alert)", "Buồn ngủ (Drowsy)")
) -> Dict[str, Any]:
    """Chạy đánh giá toàn diện mô hình và xuất báo cáo + biểu đồ."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Thiết bị tính toán
    if torch.cuda.is_available() and "cuda" in device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cpu")

    print("=" * 85)
    print(f"[*] KHỞI ĐỘNG KIỂM THỬ MÔ HÌNH DEEP LSTM TRÊN TẬP DỮ LIỆU CHUẨN")
    print(f"[*] Checkpoint : {checkpoint_path}")
    print(f"[*] Dữ liệu    : {test_pt_path}")
    print(f"[*] Thiết bị   : {device}")
    print(f"[*] Ngưỡng (T) : {decision_threshold}")
    print("=" * 85)

    # 2. Nạp mô hình từ checkpoint
    model = DeepLSTMClassifier.from_checkpoint(checkpoint_path, map_location=device)
    model.to(device)
    model.eval()

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_metadata = {
        "epoch": ckpt.get("epoch", "N/A"),
        "val_acc": ckpt.get("val_acc", 0.0),
        "f1_score": ckpt.get("f1_score", 0.0),
        "total_params": sum(p.numel() for p in model.parameters())
    }

    # 3. Nạp tập dữ liệu
    test_dataset = PreloadedTensorDataset(test_pt_path, is_train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True if "cuda" in str(device) else False
    )

    # 4. Thực thi suy luận & đo đạc tốc độ
    all_video_ids = list(test_dataset.video_ids)
    all_targets = test_dataset.labels.numpy()

    all_probs_drowsy: List[float] = []
    all_preds_video: List[int] = []
    all_probs_seq: List[np.ndarray] = []

    if torch.cuda.is_available() and "cuda" in str(device):
        torch.cuda.synchronize()
    t0 = time.time()

    with torch.no_grad():
        for (p3, p4, p5), _ in test_loader:
            p3 = p3.to(device, non_blocking=True)
            p4 = p4.to(device, non_blocking=True)
            p5 = p5.to(device, non_blocking=True)

            logits_seq = model((p3, p4, p5), return_sequence=True)
            probs_seq = torch.softmax(logits_seq, dim=-1)[:, :, 1]

            logits_last = logits_seq[:, -1, :]
            probs_drowsy = torch.softmax(logits_last, dim=-1)[:, 1]
            preds = (probs_drowsy >= decision_threshold).long()

            all_probs_drowsy.extend(probs_drowsy.cpu().numpy().tolist())
            all_preds_video.extend(preds.cpu().numpy().tolist())
            all_probs_seq.append(probs_seq.cpu().numpy())

    if torch.cuda.is_available() and "cuda" in str(device):
        torch.cuda.synchronize()
    total_infer_time = time.time() - t0

    all_preds_video = np.array(all_preds_video)
    all_probs_drowsy = np.array(all_probs_drowsy)
    all_probs_seq = np.concatenate(all_probs_seq, axis=0)

    total_samples = len(all_targets)
    latency_ms = (total_infer_time / total_samples) * 1000
    throughput = total_samples / total_infer_time
    fps = (total_samples * 120) / total_infer_time

    # 5. Tính toán các chỉ số định lượng
    cm = confusion_matrix(all_targets, all_preds_video)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    acc = accuracy_score(all_targets, all_preds_video)
    bal_acc = balanced_accuracy_score(all_targets, all_preds_video)
    prec_macro = precision_score(all_targets, all_preds_video, average="macro", zero_division=0)
    rec_macro  = recall_score(all_targets, all_preds_video, average="macro", zero_division=0)
    f1_macro   = f1_score(all_targets, all_preds_video, average="macro", zero_division=0)

    prec_drowsy = precision_score(all_targets, all_preds_video, pos_label=1, zero_division=0)
    rec_drowsy  = recall_score(all_targets, all_preds_video, pos_label=1, zero_division=0)
    f1_drowsy   = f1_score(all_targets, all_preds_video, pos_label=1, zero_division=0)

    prec_alert = precision_score(all_targets, all_preds_video, pos_label=0, zero_division=0)
    rec_alert  = recall_score(all_targets, all_preds_video, pos_label=0, zero_division=0)
    f1_alert   = f1_score(all_targets, all_preds_video, pos_label=0, zero_division=0)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    mcc = matthews_corrcoef(all_targets, all_preds_video)
    kappa = cohen_kappa_score(all_targets, all_preds_video)

    ce_loss = log_loss(all_targets, np.stack([1 - all_probs_drowsy, all_probs_drowsy], axis=-1))
    brier = brier_score_loss(all_targets, all_probs_drowsy)

    roc_auc = roc_auc_score(all_targets, all_probs_drowsy)
    pr_auc = average_precision_score(all_targets, all_probs_drowsy)

    # 6. Khảo sát ngưỡng tối ưu (Threshold Tuning)
    threshold_scan = np.linspace(0.05, 0.95, 37)
    best_f1_score = -1.0
    best_f1_threshold = 0.5
    safety_threshold = None

    for t in threshold_scan:
        p_t = (all_probs_drowsy >= t).astype(int)
        r = recall_score(all_targets, p_t, pos_label=1, zero_division=0)
        f = f1_score(all_targets, p_t, pos_label=1, zero_division=0)
        if f > best_f1_score:
            best_f1_score = f
            best_f1_threshold = t
        if r >= 0.90 and safety_threshold is None:
            safety_threshold = t

    # In báo cáo tóm tắt
    print("\n" + "=" * 85)
    print("                     KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (EXECUTIVE REPORT)")
    print("=" * 85)
    print(f"[*] Độ chính xác tổng thể (Accuracy)       : {acc * 100:.2f} %")
    print(f"[*] Độ chính xác cân bằng (Balanced Acc)   : {bal_acc * 100:.2f} %")
    print(f"[*] F1-Score Lớp Buồn ngủ (Class 1)        : {f1_drowsy:.4f} ({f1_drowsy * 100:.2f} %)")
    print(f"[*] F1-Score Lớp Tỉnh táo (Class 0)        : {f1_alert:.4f} ({f1_alert * 100:.2f} %)")
    print(f"[*] Macro F1-Score                         : {f1_macro:.4f}")
    print(f"[*] Diện tích dưới đường cong ROC-AUC      : {roc_auc:.4f} ({roc_auc * 100:.2f} %)")
    print(f"[*] Diện tích dưới đường cong PR-AUC (AP)  : {pr_auc:.4f} ({pr_auc * 100:.2f} %)")
    print(f"[*] Hệ số tương quan Matthews (MCC)        : {mcc:.4f}")
    print(f"[*] Hệ số thống nhất Cohen's Kappa         : {kappa:.4f}")
    print(f"[*] Độ nhạy bắt trúng (Recall/Sensitivity) : {sensitivity * 100:.2f} %")
    print(f"[*] Độ đặc hiệu (Specificity/TNR)          : {specificity * 100:.2f} %")
    print(f"[*] Tỷ lệ báo động giả (FPR)               : {fpr * 100:.2f} %")
    print(f"[*] Tỷ lệ bỏ sót nguy hiểm (FNR)           : {fnr * 100:.2f} %")
    print(f"[*] Độ trễ suy luận (Latency / Video)      : {latency_ms:.2f} ms")
    print(f"[*] Tốc độ xử lý tương đương               : {fps:.2f} FPS")
    print(f"[*] Ngưỡng đạt Best F1                     : T = {best_f1_threshold:.2f} (F1 = {best_f1_score:.4f})")
    print("=" * 85)

    # 7. Vẽ và lưu 4 biểu đồ tổng hợp
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    annot_labels = np.array([
        [f"TN: {tn}\n({cm_norm[0,0]*100:.1f}%)", f"FP: {fp}\n({cm_norm[0,1]*100:.1f}%)"],
        [f"FN: {fn}\n({cm_norm[1,0]*100:.1f}%)", f"TP: {tp}\n({cm_norm[1,1]*100:.1f}%)"]
    ])
    sns.heatmap(cm, annot=annot_labels, fmt="", cmap="Blues", cbar=True,
                xticklabels=class_names, yticklabels=class_names, ax=axes[0, 0])
    axes[0, 0].set_title(f"1. Ma Trận Nhầm Lẫn (Accuracy: {acc*100:.2f}%)", fontweight="bold")
    axes[0, 0].set_xlabel("Dự đoán", fontweight="bold")
    axes[0, 0].set_ylabel("Thực tế", fontweight="bold")

    fpr_curve, tpr_curve, _ = roc_curve(all_targets, all_probs_drowsy)
    axes[0, 1].plot(fpr_curve, tpr_curve, color="darkorange", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    axes[0, 1].plot([0, 1], [0, 1], "navy", linestyle="--", label="Ngẫu nhiên")
    axes[0, 1].set_title("2. Đường Cong ROC", fontweight="bold")
    axes[0, 1].set_xlabel("FPR", fontweight="bold")
    axes[0, 1].set_ylabel("TPR (Recall)", fontweight="bold")
    axes[0, 1].legend(loc="lower right")

    prec_c, rec_c, _ = precision_recall_curve(all_targets, all_probs_drowsy)
    axes[1, 0].plot(rec_c, prec_c, color="forestgreen", lw=2.5, label=f"PR Curve (AP = {pr_auc:.4f})")
    axes[1, 0].axhline(y=(all_targets == 1).mean(), color="gray", linestyle="--", label="Tỷ lệ lớp dương")
    axes[1, 0].set_title("3. Đường Cong Precision-Recall", fontweight="bold")
    axes[1, 0].set_xlabel("Recall", fontweight="bold")
    axes[1, 0].set_ylabel("Precision", fontweight="bold")
    axes[1, 0].legend(loc="lower left")

    axes[1, 1].hist(all_probs_drowsy[all_targets == 0], bins=25, alpha=0.6, color="dodgerblue", label="Tỉnh táo (0)", density=True)
    axes[1, 1].hist(all_probs_drowsy[all_targets == 1], bins=25, alpha=0.6, color="crimson", label="Buồn ngủ (1)", density=True)
    axes[1, 1].axvline(x=decision_threshold, color="black", linestyle="--", label=f"Ngưỡng T={decision_threshold}")
    axes[1, 1].set_title("4. Phân Bố Xác Suất Dự Đoán (Separability)", fontweight="bold")
    axes[1, 1].set_xlabel("P(Buồn ngủ)", fontweight="bold")
    axes[1, 1].set_ylabel("Mật độ", fontweight="bold")
    axes[1, 1].legend()

    plt.tight_layout()
    chart_path = out_dir / "evaluation_summary_charts.png"
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Đã lưu biểu đồ tổng quan vào: {chart_path.resolve()}")

    # 8. Xuất file test_predictions.csv
    df_results = pd.DataFrame({
        "video_id": all_video_ids,
        "ground_truth_label": all_targets,
        "ground_truth_name": [class_names[t] for t in all_targets],
        "predicted_label": all_preds_video,
        "predicted_name": [class_names[p] for p in all_preds_video],
        "prob_drowsy": np.round(all_probs_drowsy, 4),
        "prob_alert": np.round(1 - all_probs_drowsy, 4),
        "is_correct": (all_targets == all_preds_video)
    })
    csv_path = out_dir / "test_predictions.csv"
    df_results.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[+] Đã xuất danh sách dự đoán ({len(df_results)} video) vào: {csv_path.resolve()}")

    # 9. Xuất file test_evaluation_report.json
    report_dict = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_checkpoint": checkpoint_path,
        "test_dataset": test_pt_path,
        "model_metadata": model_metadata,
        "test_samples_total": int(total_samples),
        "decision_threshold": float(decision_threshold),
        "best_f1_threshold": float(best_f1_threshold),
        "best_f1_score": float(best_f1_score),
        "metrics": {
            "overall_accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "cross_entropy_loss": float(ce_loss),
            "brier_score_loss": float(brier),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "matthews_corrcoef": float(mcc),
            "cohen_kappa": float(kappa),
            "macro_f1": float(f1_macro),
            "drowsy_class": {
                "precision": float(prec_drowsy),
                "recall_sensitivity": float(sensitivity),
                "f1_score": float(f1_drowsy),
                "support": int((all_targets == 1).sum())
            },
            "alert_class": {
                "precision": float(prec_alert),
                "recall_specificity": float(specificity),
                "f1_score": float(f1_alert),
                "support": int((all_targets == 0).sum())
            }
        },
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
            "false_positive_rate": float(fpr),
            "false_negative_rate": float(fnr)
        },
        "inference_speed": {
            "total_time_seconds": float(total_infer_time),
            "latency_ms_per_video": float(latency_ms),
            "throughput_videos_per_sec": float(throughput),
            "equivalent_fps": float(fps)
        }
    }
    json_path = out_dir / "test_evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=4)
    print(f"[+] Đã xuất báo cáo tổng hợp JSON vào: {json_path.resolve()}")

    return report_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Kiểm thử và đánh giá mô hình Deep LSTM (SUST Dataset)")
    parser.add_argument("--checkpoint", type=str, default="lstm_experiment_results/checkpoints/best_lstm.pth", help="Đường dẫn file checkpoint .pth")
    parser.add_argument("--data", type=str, default="extracted_features_pt/features_sust_val.pt", help="Đường dẫn tệp tensor .pt")
    parser.add_argument("--output-dir", type=str, default="./evaluation_results", help="Thư mục xuất kết quả báo cáo")
    parser.add_argument("--batch-size", type=int, default=64, help="Kích thước batch")
    parser.add_argument("--device", type=str, default="cuda", help="Thiết bị ('cuda' hoặc 'cpu')")
    parser.add_argument("--threshold", type=float, default=0.5, help="Ngưỡng quyết định nhị phân")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(
        checkpoint_path=args.checkpoint,
        test_pt_path=args.data,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device_str=args.device,
        decision_threshold=args.threshold
    )
