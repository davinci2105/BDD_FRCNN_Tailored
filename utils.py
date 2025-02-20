import torch
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchmetrics

# ================== COMPUTE EVALUATION METRICS ==================
def compute_evaluation_metrics(predictions, targets, iou_thresholds=[0.5, 0.75, 0.95]):
    """
    Computes mAP, AR (Average Recall), F1 Score, and IOI (Intersection-over-Intersection) at different IoU thresholds.
    Assumes that predictions and targets are lists of dictionaries (one per image) with:
      - predictions: {"boxes": tensor of shape [N, 4], "scores": tensor of shape [N], "labels": tensor of shape [N]}
      - targets: {"boxes": tensor of shape [M, 4], "labels": tensor of shape [M]}
    """
    metrics = {}

    # Helper function: compute IoU for two boxes (each box is a list or tensor: [x1, y1, x2, y2])
    def compute_iou(box1, box2):
        # Ensure boxes are lists of floats.
        if isinstance(box1, torch.Tensor): box1 = box1.tolist()
        if isinstance(box2, torch.Tensor): box2 = box2.tolist()
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_width = max(0, x2 - x1)
        inter_height = max(0, y2 - y1)
        intersection = inter_width * inter_height
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0

    for iou_thresh in iou_thresholds:
        # Initialize torchmetrics function for mAP using the current IoU threshold.
        map_metric = torchmetrics.detection.MeanAveragePrecision(iou_thresholds=[iou_thresh]).cuda()
        
        formatted_preds = []
        for pred in predictions:
            # Convert any list fields to tensors and ensure they are on GPU.
            boxes = pred["boxes"]
            scores = pred["scores"]
            labels = pred["labels"]
            if isinstance(boxes, list):
                boxes = torch.tensor(boxes, dtype=torch.float32, device="cuda")
            if isinstance(scores, list):
                scores = torch.tensor(scores, dtype=torch.float32, device="cuda")
            if isinstance(labels, list):
                labels = torch.tensor(labels, dtype=torch.int64, device="cuda")
            formatted_preds.append({
                "boxes": boxes, 
                "scores": scores, 
                "labels": labels
            })

        formatted_targets = []
        for target in targets:
            boxes = target["boxes"]
            labels = target["labels"]
            if isinstance(boxes, list):
                boxes = torch.tensor(boxes, dtype=torch.float32, device="cuda")
            if isinstance(labels, list):
                labels = torch.tensor(labels, dtype=torch.int64, device="cuda")
            formatted_targets.append({
                "boxes": boxes, 
                "labels": labels
            })

        # Compute mAP & AR using torchmetrics.
        try:
            map_result = map_metric(formatted_preds, formatted_targets)
            mean_ap = map_result["map"].item()     # Mean Average Precision
            mean_ar = map_result["mar_100"].item()   # Mean Average Recall (for 100 detections)
        except Exception as e:
            print(f"Error computing mAP at IoU {iou_thresh}: {e}")
            mean_ap, mean_ar = 0.0, 0.0

        # Compute F1 Score from precision (mAP) and recall (AR)
        precision = mean_ap
        recall = mean_ar
        f1_score = (2 * precision * recall) / (precision + recall + 1e-8)

        # Compute IOI (Intersection-over-Intersection) for each sample.
        ioi_list = []
        for pred, target in zip(formatted_preds, formatted_targets):
            pred_boxes = pred["boxes"]
            target_boxes = target["boxes"]
            sample_ioi = []
            if pred_boxes.shape[0] > 0 and target_boxes.shape[0] > 0:
                for pb in pred_boxes:
                    # For each predicted box, compute IoU with each target box
                    best_iou = 0.0
                    for tb in target_boxes:
                        best_iou = max(best_iou, compute_iou(pb, tb))
                    sample_ioi.append(best_iou)
                if sample_ioi:
                    ioi_list.append(sum(sample_ioi) / len(sample_ioi))
        ioi_score = sum(ioi_list) / len(ioi_list) if ioi_list else 0.0

        # Store computed metrics for this IoU threshold.
        metrics[iou_thresh] = {
            "mAP": mean_ap,
            "AR": mean_ar,
            "F1": f1_score,
            "IOI": ioi_score
        }

    return metrics


# ================== PREDICTION VISUALIZATION ==================
import torch
import torchvision.ops as ops

def visualize_predictions(predictions_list, targets, device, CLASS_TO_IDX, score_threshold=0.3, nms_threshold=0.3):
    """
    Processes raw model outputs into two representations:
      1. Concatenated tensors for loss computation.
      2. Per-image dictionaries for metric computation.
    
    Applies a score threshold and non-maximum suppression (NMS) to filter predictions.
    
    Returns:
      concat_preds: tuple (pred_scores, pred_boxes, target_labels, target_boxes)
      per_image_preds: list of dicts {"boxes", "scores", "labels"} per image.
      per_image_targets: list of dicts {"boxes", "labels"} per image.
    """
    per_image_preds = []
    for pred in predictions_list:
        if "instances" in pred:
            inst = pred["instances"]
            boxes = inst.pred_boxes.tensor.to(device).float()
            scores = inst.scores.to(device).float()
            if hasattr(inst, "pred_classes"):
                labels = inst.pred_classes.to(device).long()
            else:
                labels = torch.zeros(len(scores), dtype=torch.long, device=device)
            # Apply score threshold filtering.
            keep = scores > score_threshold
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            # Apply NMS if any boxes remain.
            if boxes.shape[0] > 0:
                keep_indices = ops.nms(boxes, scores, nms_threshold)
                boxes = boxes[keep_indices]
                scores = scores[keep_indices]
                labels = labels[keep_indices]
        else:
            boxes = torch.empty((0, 4), device=device)
            scores = torch.empty((0,), device=device)
            labels = torch.empty((0,), device=device)
        per_image_preds.append({
            "boxes": boxes,
            "scores": scores,
            "labels": labels
        })
    
    per_image_targets = []
    for target in targets:
        if not isinstance(target["boxes"], torch.Tensor):
            boxes = torch.tensor(target["boxes"], dtype=torch.float32, device=device)
        else:
            boxes = target["boxes"].to(device).float()
        if not isinstance(target["labels"], torch.Tensor):
            labels = torch.tensor(target["labels"], dtype=torch.int64, device=device)
        else:
            labels = target["labels"].to(device).long()
        # Optionally remap target labels using CLASS_TO_IDX.
        remapped = torch.tensor([CLASS_TO_IDX.get(int(lbl.item()), 0) for lbl in labels],
                                dtype=torch.long, device=device)
        per_image_targets.append({
            "boxes": boxes,
            "labels": remapped
        })
    
    # Concatenate predictions and targets for loss computation.
    if len(per_image_preds) > 0:
        concat_scores = torch.cat([p["scores"].unsqueeze(1) for p in per_image_preds], dim=0)
        concat_boxes = torch.cat([p["boxes"] for p in per_image_preds], dim=0)
    else:
        concat_scores = torch.empty((0, 1), device=device)
        concat_boxes = torch.empty((0, 4), device=device)
    if len(per_image_targets) > 0:
        concat_target_labels = torch.cat([t["labels"] for t in per_image_targets], dim=0)
        concat_target_boxes = torch.cat([t["boxes"] for t in per_image_targets], dim=0)
    else:
        concat_target_labels = torch.empty((0,), device=device)
        concat_target_boxes = torch.empty((0, 4), device=device)
    
    concat_preds = (concat_scores, concat_boxes, concat_target_labels, concat_target_boxes)
    return concat_preds, per_image_preds, per_image_targets




# ================== CHECKPOINT MANAGEMENT ==================
def save_checkpoint(model, optimizer, val_loss, epoch, checkpoint_path):
    """
    Saves model checkpoint.
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"🔹 Model checkpoint saved at {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"🔹 Loaded checkpoint from {checkpoint_path}, starting at epoch {checkpoint['epoch']+1}")
        return checkpoint["val_loss"], checkpoint["epoch"] + 1
    return float("inf"), 0  


# ================== LOSS PLOTTING ==================
def plot_loss_curve(train_losses, val_losses, save_path="training_loss_curve.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='s')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()
    print(f"🔹 Loss curve saved as {save_path}")


# ================== TENSORBOARD LOGGING ==================
def log_tensorboard(writer, epoch, loss, mode="train"):
    writer.add_scalar(f"Loss/{mode}", loss, epoch)
