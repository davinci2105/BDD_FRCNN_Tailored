import torch
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchmetrics

# ================== COMPUTE EVALUATION METRICS ==================
def compute_evaluation_metrics(predictions, targets, iou_thresholds=[0.5, 0.75, 0.95]):
    """
    Computes mAP, AR (Average Recall), F1 Score, and IOI (Intersection-over-Intersection) at different IoU thresholds.
    """
    metrics = {}
    for iou in iou_thresholds:
        # Initialize torchmetrics function for mAP
        map_metric = torchmetrics.detection.MeanAveragePrecision(iou_thresholds=[iou]).cuda()
        
        formatted_preds = []
        for pred in predictions:
            if isinstance(pred["scores"], list):  # Convert list to tensor
                pred["scores"] = torch.tensor(pred["scores"], dtype=torch.float32, device="cuda")
            if isinstance(pred["boxes"], list):  # Convert list to tensor
                pred["boxes"] = torch.tensor(pred["boxes"], dtype=torch.float32, device="cuda")
            if isinstance(pred["labels"], list):  # Convert list to tensor
                pred["labels"] = torch.tensor(pred["labels"], dtype=torch.int64, device="cuda")

            formatted_preds.append({"boxes": pred["boxes"], "scores": pred["scores"], "labels": pred["labels"]})

        formatted_targets = []
        for target in targets:
            if isinstance(target["boxes"], list):
                target["boxes"] = torch.tensor(target["boxes"], dtype=torch.float32, device="cuda")
            if isinstance(target["labels"], list):
                target["labels"] = torch.tensor(target["labels"], dtype=torch.int64, device="cuda")

            formatted_targets.append({"boxes": target["boxes"], "labels": target["labels"]})

        # Compute mAP & AR
        try:
            map_result = map_metric(formatted_preds, formatted_targets)
            mean_ap = map_result["map"].item()  # Mean Average Precision
            mean_ar = map_result["mar_100"].item()  # Mean Average Recall (for 100 detections)
        except Exception as e:
            print(f"Error computing mAP at IoU {iou}: {e}")
            mean_ap, mean_ar = 0.0, 0.0

        # Compute F1 Score
        precision = mean_ap  # Precision from mAP
        recall = mean_ar  # Recall from mAP
        f1_score = (2 * precision * recall) / (precision + recall + 1e-8)  # Avoid division by zero

        # Compute IOI (Intersection-over-Intersection)
        ioi_scores = []
        for pred, target in zip(formatted_preds, formatted_targets):
            if len(pred["boxes"]) > 0 and len(target["boxes"]) > 0:
                intersection = torch.min(pred["boxes"], target["boxes"]).sum(dim=1)
                union = torch.max(pred["boxes"], target["boxes"]).sum(dim=1)
                ioi_scores.append((intersection / (union + 1e-8)).mean().item())

        ioi_score = sum(ioi_scores) / len(ioi_scores) if ioi_scores else 0.0

        # Store results
        metrics[iou] = {
            "mAP": mean_ap,
            "AR": mean_ar,
            "F1": f1_score,
            "IOI": ioi_score
        }

    return metrics


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

# ================== PREDICTION VISUALIZATION ==================
def visualize_predictions(predictions_list, targets, device, CLASS_TO_IDX):
    
    pred_logits, pred_boxes, target_labels, target_boxes = [], [], [], []

    for pred, target in zip(predictions_list, targets):
        if "instances" in pred:
            pred_instances = pred["instances"]
            pred_logits.append(pred_instances.scores.unsqueeze(1))  # Convert to column tensor
            pred_boxes.append(pred_instances.pred_boxes.tensor)

        # Process target labels & boxes
        target_labels.append(target["labels"])
        target_boxes.append(target["boxes"])

    # Convert lists to tensors
    pred_logits = torch.cat(pred_logits, dim=0).float() if pred_logits else torch.empty((0, 1), device=device)
    pred_boxes = torch.cat(pred_boxes, dim=0).float() if pred_boxes else torch.empty((0, 4), device=device)
    target_labels = torch.cat(target_labels, dim=0).long() if target_labels else torch.empty((0,), device=device)
    target_boxes = torch.cat(target_boxes, dim=0).float() if target_boxes else torch.empty((0, 4), device=device)

    # Remap target labels to class indices
    target_labels = torch.tensor([CLASS_TO_IDX.get(int(lbl.item()), 0) for lbl in target_labels], dtype=torch.long, device=device)

    return pred_logits, pred_boxes, target_labels, target_boxes
