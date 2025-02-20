import torch
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
import torchvision.ops as ops

# --------------------------------------------------
# Function: compute_evaluation_metrics
# Description: Calculates evaluation metrics such as Mean Average Precision (mAP),
#              Average Recall (AR), F1 Score, and Intersection-over-Intersection (IOI)
#              for given predictions and targets at specified IoU thresholds.
#
# Parameters:
#   predictions - List of dicts, each containing:
#                 "boxes": tensor of shape [N, 4],
#                 "scores": tensor of shape [N],
#                 "labels": tensor of shape [N]
#   targets     - List of dicts, each containing:
#                 "boxes": tensor of shape [M, 4],
#                 "labels": tensor of shape [M]
#   iou_thresholds - List of IoU thresholds to evaluate metrics on (default [0.5, 0.75, 0.95])
#
# Returns:
#   metrics - Dictionary mapping each IoU threshold to a dict with keys "mAP", "AR", "F1", and "IOI".
# --------------------------------------------------
def compute_evaluation_metrics(predictions, targets, iou_thresholds=[0.5, 0.75, 0.95]):
    def compute_iou(box1, box2):
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

    metrics = {}
    for iou_thresh in iou_thresholds:
        map_metric = torchmetrics.detection.MeanAveragePrecision(iou_thresholds=[iou_thresh]).cuda()
        formatted_preds = []
        for pred in predictions:
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

        try:
            map_result = map_metric(formatted_preds, formatted_targets)
            mean_ap = map_result["map"].item()
            mean_ar = map_result["mar_100"].item()
        except Exception as e:
            print(f"Error computing mAP at IoU {iou_thresh}: {e}")
            mean_ap, mean_ar = 0.0, 0.0

        precision = mean_ap
        recall = mean_ar
        f1_score = (2 * precision * recall) / (precision + recall + 1e-8)

        ioi_list = []
        for pred, target in zip(formatted_preds, formatted_targets):
            pred_boxes = pred["boxes"]
            target_boxes = target["boxes"]
            sample_ioi = []
            if pred_boxes.shape[0] > 0 and target_boxes.shape[0] > 0:
                for pb in pred_boxes:
                    best_iou = 0.0
                    for tb in target_boxes:
                        best_iou = max(best_iou, compute_iou(pb, tb))
                    sample_ioi.append(best_iou)
                if sample_ioi:
                    ioi_list.append(sum(sample_ioi) / len(sample_ioi))
        ioi_score = sum(ioi_list) / len(ioi_list) if ioi_list else 0.0

        metrics[iou_thresh] = {
            "mAP": mean_ap,
            "AR": mean_ar,
            "F1": f1_score,
            "IOI": ioi_score
        }
    return metrics


# --------------------------------------------------
# Function: visualize_predictions
# Description: Processes raw model predictions and target annotations.
#              It filters predictions using a score threshold and applies Non-Maximum Suppression (NMS).
#              It returns both concatenated tensors for loss computation and per-image dicts for metric evaluation.
#
# Parameters:
#   predictions_list - List of prediction outputs (each may contain an "instances" dict)
#   targets          - List of target annotations; each is a dict with "boxes" and "labels"
#   device           - The device to perform computations on (e.g., "cuda")
#   CLASS_TO_IDX     - Mapping from class labels to indices
#   score_threshold  - Minimum score required to keep a prediction (default 0.3)
#   nms_threshold    - IoU threshold for applying NMS (default 0.3)
#
# Returns:
#   concat_preds     - Tuple containing concatenated scores, boxes, target labels, and target boxes
#   per_image_preds  - List of dicts with keys "boxes", "scores", "labels" per image
#   per_image_targets- List of dicts with keys "boxes" and "labels" per image
# --------------------------------------------------
def visualize_predictions(predictions_list, targets, device, CLASS_TO_IDX, score_threshold=0.3, nms_threshold=0.3):
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
            keep = scores > score_threshold
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
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
        remapped = torch.tensor([CLASS_TO_IDX.get(int(lbl.item()), 0) for lbl in labels],
                                dtype=torch.long, device=device)
        per_image_targets.append({
            "boxes": boxes,
            "labels": remapped
        })
    
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


# --------------------------------------------------
# Function: save_checkpoint
# Description: Saves the current state of the model and optimizer into a checkpoint file.
#
# Parameters:
#   model         - The PyTorch model to save
#   optimizer     - The optimizer associated with the model
#   val_loss      - Validation loss value at the current epoch
#   epoch         - Current epoch number
#   checkpoint_dir- Directory where the checkpoint file will be stored (default "checkpoints")
#
# Returns:
#   None; saves checkpoint file to disk.
# --------------------------------------------------
def save_checkpoint(model, optimizer, val_loss, epoch, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"faster_rcnn_epoch_{epoch}.pth")
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"🔹 Checkpoint saved at {checkpoint_path}")


# --------------------------------------------------
# Function: load_checkpoint
# Description: Loads a checkpoint from disk. If a specific epoch is provided, it loads that checkpoint;
#              otherwise, it loads the checkpoint corresponding to the latest epoch.
#
# Parameters:
#   model         - The PyTorch model to load state into
#   optimizer     - The optimizer to load state into
#   checkpoint_dir- Directory where checkpoint files are stored (default "checkpoints")
#   epoch         - Specific epoch checkpoint to load (default None, meaning load latest)
#
# Returns:
#   Tuple containing the best validation loss and the next epoch number.
# --------------------------------------------------
def load_checkpoint(model, optimizer, checkpoint_dir="checkpoints", epoch=None):
    if epoch is not None:
        checkpoint_path = os.path.join(checkpoint_dir, f"faster_rcnn_epoch_{epoch}.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"🔹 Loaded checkpoint from {checkpoint_path}, resuming at epoch {checkpoint['epoch'] + 1}")
            return checkpoint["val_loss"], checkpoint["epoch"] + 1
        else:
            print(f"🔹 No checkpoint found for epoch {epoch}. Starting from scratch.")
            return float("inf"), 0
    else:
        if not os.path.exists(checkpoint_dir):
            return float("inf"), 0
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("faster_rcnn_epoch_") and f.endswith(".pth")]
        if not checkpoint_files:
            return float("inf"), 0
        epochs = [int(f.split("_")[-1].split(".pth")[0]) for f in checkpoint_files]
        latest_epoch = max(epochs)
        checkpoint_path = os.path.join(checkpoint_dir, f"faster_rcnn_epoch_{latest_epoch}.pth")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"🔹 Loaded checkpoint from {checkpoint_path}, resuming at epoch {checkpoint['epoch'] + 1}")
        return checkpoint["val_loss"], checkpoint["epoch"] + 1


# --------------------------------------------------
# Function: plot_loss_curve
# Description: Generates and saves a plot of training and validation loss curves.
#
# Parameters:
#   train_losses - List of training loss values per epoch
#   val_losses   - List of validation loss values per epoch
#   save_path    - Path where the loss curve image will be saved (default "training_loss_curve.png")
#
# Returns:
#   None; displays and saves the loss curve plot.
# --------------------------------------------------
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


# --------------------------------------------------
# Function: log_tensorboard
# Description: Logs the loss value to TensorBoard under a specified mode (e.g., training or validation).
#
# Parameters:
#   writer - TensorBoard SummaryWriter object
#   epoch  - Current epoch number
#   loss   - Loss value to log
#   mode   - Mode of operation ("train" or "val", default "train")
#
# Returns:
#   None; logs scalar value to TensorBoard.
# --------------------------------------------------
def log_tensorboard(writer, epoch, loss, mode="train"):
    writer.add_scalar(f"Loss/{mode}", loss, epoch)
