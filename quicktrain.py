import torch
import torch.optim as optim
from model import get_model
from dataloader import get_dataloader
from losses import HybridLoss
from utils import compute_evaluation_metrics  # Import the metrics function
import tqdm

# Define quick training parameters
QUICK_HYPERPARAMS = {
    "batch_size": 4,
    "learning_rate": 0.0001,
    "epochs": 3,  # Just a few epochs to validate pipeline
    "max_train_samples": 10,  # Only use 10 training images
    "max_val_samples": 10,  # Only use 10 validation images
    "iou_thresholds": [0.5, 0.75, 0.95]  # IoU thresholds for metrics
}

# File paths
TRAIN_LABELS = "bdd_dataset/labels/bdd100k_labels_images_train.json"
TRAIN_IMAGES = "bdd_dataset/100k/train"
VAL_LABELS = "bdd_dataset/labels/bdd100k_labels_images_val.json"
VAL_IMAGES = "bdd_dataset/100k/val"

# Load dataset
train_loader = get_dataloader(TRAIN_LABELS, TRAIN_IMAGES, batch_size=QUICK_HYPERPARAMS["batch_size"], is_train=True)
val_loader = get_dataloader(VAL_LABELS, VAL_IMAGES, batch_size=QUICK_HYPERPARAMS["batch_size"], is_train=False)

# Initialize model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model().to(device)
optimizer = optim.Adam(model.parameters(), lr=QUICK_HYPERPARAMS["learning_rate"])
criterion = HybridLoss()

# Define class mapping
SELECTED_CLASSES = ['car', 'truck', 'bus', 'motor', 'bike', 'train', 
                    'traffic light', 'traffic sign', 'rider', 'person']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(SELECTED_CLASSES)}  # Background = 0

# Training loop for quick validation
for epoch in range(QUICK_HYPERPARAMS["epochs"]):
    model.train()
    running_loss = 0.0
    train_samples_count = 0

    for images, targets in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{QUICK_HYPERPARAMS['epochs']} Quick Training"):
        if train_samples_count >= QUICK_HYPERPARAMS["max_train_samples"]:
            break

        images = torch.stack(images).to(device).float() / 255.0
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        predictions_list = model(images)

        # Extract predictions
        pred_logits = [p["instances"].scores.unsqueeze(1) for p in predictions_list] if predictions_list else []
        pred_boxes = [p["instances"].pred_boxes.tensor for p in predictions_list] if predictions_list else []

        # Stack tensors
        pred_logits = torch.cat(pred_logits, dim=0).float() if pred_logits else torch.empty((0, 1), device=device, dtype=torch.float32)
        pred_boxes = torch.cat(pred_boxes, dim=0).float() if pred_boxes else torch.empty((0, 4), device=device, dtype=torch.float32)

        # Stack ground truth
        target_labels = torch.cat([t["labels"] for t in targets], dim=0).long() if targets else torch.empty((0,), device=device, dtype=torch.long)
        target_boxes = torch.cat([t["boxes"] for t in targets], dim=0).float() if targets else torch.empty((0, 4), device=device, dtype=torch.float32)

        # Map labels
        target_labels = torch.tensor([CLASS_TO_IDX.get(int(lbl.item()), 0) for lbl in target_labels], dtype=torch.long, device=device)

        # Fix mismatch issue
        min_size = min(pred_logits.shape[0], target_labels.shape[0], pred_boxes.shape[0], target_boxes.shape[0])

        if min_size > 0:
            pred_logits = pred_logits[:min_size]
            target_labels = target_labels[:min_size]
            pred_boxes = pred_boxes[:min_size]
            target_boxes = target_boxes[:min_size]

            # Compute loss
            loss = criterion(pred_logits, target_labels, pred_boxes, target_boxes)
            if not loss.requires_grad:
                loss = loss.clone().detach().requires_grad_(True)

            if loss.isnan() or loss.isinf():
                continue

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_samples_count += len(images)

    avg_train_loss = running_loss / max(1, train_samples_count)
    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_samples_count = 0
    all_predictions, all_targets = [], []

    with torch.no_grad():
        for images, targets in val_loader:
            if val_samples_count >= QUICK_HYPERPARAMS["max_val_samples"]:
                break

            images = torch.stack(images).to(device).float() / 255.0
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            predictions_list = model(images)

            # Extract predictions
            pred_logits = [p["instances"].scores.unsqueeze(1) for p in predictions_list] if predictions_list else []
            pred_boxes = [p["instances"].pred_boxes.tensor for p in predictions_list] if predictions_list else []

            pred_logits = torch.cat(pred_logits, dim=0).float() if pred_logits else torch.empty((0, 1), device=device, dtype=torch.float32)
            pred_boxes = torch.cat(pred_boxes, dim=0).float() if pred_boxes else torch.empty((0, 4), device=device, dtype=torch.float32)

            # Stack ground truth
            target_labels = torch.cat([t["labels"] for t in targets], dim=0).long() if targets else torch.empty((0,), device=device, dtype=torch.long)
            target_boxes = torch.cat([t["boxes"] for t in targets], dim=0).float() if targets else torch.empty((0, 4), device=device, dtype=torch.float32)

            # Map labels
            target_labels = torch.tensor([CLASS_TO_IDX.get(int(lbl.item()), 0) for lbl in target_labels], dtype=torch.long, device=device)

            # Fix size mismatch
            min_size = min(pred_logits.shape[0], target_labels.shape[0], pred_boxes.shape[0], target_boxes.shape[0])

            if min_size > 0:
                pred_logits = pred_logits[:min_size]
                target_labels = target_labels[:min_size]
                pred_boxes = pred_boxes[:min_size]
                target_boxes = target_boxes[:min_size]

                # Compute loss
                loss = criterion(pred_logits, target_labels, pred_boxes, target_boxes)
                val_loss += loss.item()
                val_samples_count += len(images)

                # Store predictions & targets for evaluation
                all_predictions.append({"boxes": pred_boxes, "scores": pred_logits, "labels": target_labels})
                all_targets.append({"boxes": target_boxes, "labels": target_labels})

    avg_val_loss = val_loss / max(1, val_samples_count)
    print(f"Epoch {epoch+1}: Validation Loss = {avg_val_loss:.4f}")

    # Compute and print evaluation metrics
    metrics = compute_evaluation_metrics(all_predictions, all_targets, iou_thresholds=QUICK_HYPERPARAMS["iou_thresholds"])
    for iou, metric_values in metrics.items():
        print(f"IoU {iou}: mAP={metric_values['mAP']:.3f}, AR={metric_values['AR']:.3f}, F1={metric_values['F1']:.3f}, IOI={metric_values['IOI']:.3f}")

print("Quick training validation complete!")
