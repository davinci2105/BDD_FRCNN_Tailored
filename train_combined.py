import torch
import torch.optim as optim
import tqdm
from model import get_model
from dataloader import get_dataloader
from losses import HybridLoss
from utils import save_checkpoint, load_checkpoint, plot_loss_curve, log_tensorboard, visualize_predictions, compute_evaluation_metrics
import os
from torch.utils.tensorboard import SummaryWriter

# ================== HYPERPARAMETERS ==================
HYPERPARAMS = {
    "batch_size": 16,
    "learning_rate": 0.0001,
    "epochs": 15,
    "patience": 3,  
    "checkpoint_path": "checkpoints/faster_rcnn_best.pth",
    "tensorboard_log_dir": "logs",
    "use_regularization": True, 
    "weight_decay": 5e-5, 
    "iou_thresholds": [0.5, 0.75, 0.95],  
    "train_mode": "full",  # "quick" -> Few images, "full" -> Complete dataset
    "max_train_samples": 10, 
    "max_val_samples": 10,  
}

# ================== DATASET PATHS ==================
TRAIN_LABELS = "bdd_dataset/labels/bdd100k_labels_images_train.json"
TRAIN_IMAGES = "bdd_dataset/100k/train"
VAL_LABELS = "bdd_dataset/labels/bdd100k_labels_images_val.json"
VAL_IMAGES = "bdd_dataset/100k/val"

SELECTED_CLASSES = ['car', 'truck', 'bus', 'motor', 'bike', 'train', 
                    'traffic light', 'traffic sign', 'rider', 'person']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(SELECTED_CLASSES)}

# ================== SETUP MODEL & OPTIMIZER ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model().to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=HYPERPARAMS["learning_rate"],
    weight_decay=HYPERPARAMS["weight_decay"] if HYPERPARAMS["use_regularization"] else 0.0
)

criterion = HybridLoss()
writer = SummaryWriter(HYPERPARAMS["tensorboard_log_dir"])

# Load Checkpoint if exists
best_val_loss, start_epoch = load_checkpoint(model, optimizer, HYPERPARAMS["checkpoint_path"])

print(f"📌 Resuming training from epoch {start_epoch}...")

# ================== DATALOADERS ==================
train_loader = get_dataloader(TRAIN_LABELS, TRAIN_IMAGES, batch_size=HYPERPARAMS["batch_size"], is_train=True)
val_loader = get_dataloader(VAL_LABELS, VAL_IMAGES, batch_size=HYPERPARAMS["batch_size"], is_train=False)

train_losses, val_losses = [], []
patience_counter = 0

# ================== TRAINING LOOP ==================
for epoch in range(start_epoch, HYPERPARAMS["epochs"]):
    model.train()
    running_loss = 0.0
    train_samples_count = 0

    print(f"🔹 Training Mode: {HYPERPARAMS['train_mode']}")

    for batch_idx, (images, targets) in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{HYPERPARAMS['epochs']} Training")):
        images = torch.stack(images).to(device).float() / 255.0  # Normalize
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        predictions_list = model(images)

        pred_logits, pred_boxes, target_labels, target_boxes = visualize_predictions(predictions_list, targets, device, CLASS_TO_IDX)

        min_size = min(pred_logits.shape[0], target_labels.shape[0], pred_boxes.shape[0], target_boxes.shape[0])
        if min_size > 0:
            pred_logits, target_labels, pred_boxes, target_boxes = pred_logits[:min_size], target_labels[:min_size], pred_boxes[:min_size], target_boxes[:min_size]

            # Compute loss
            loss = criterion(pred_logits, target_labels, pred_boxes, target_boxes)
            
            # Ensure loss requires grad
            if not loss.requires_grad:
                loss = loss.clone().detach().requires_grad_(True)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_samples_count += len(images)

    avg_train_loss = running_loss / max(1, train_samples_count)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")
    log_tensorboard(writer, epoch, avg_train_loss, mode="train")

    # ================== VALIDATION ==================
    model.eval()
    val_loss = 0.0
    val_samples_count = 0
    all_predictions, all_targets = [], []

    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            images = torch.stack(images).to(device).float() / 255.0
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    
            predictions_list = model(images)
    
            pred_logits, pred_boxes, target_labels, target_boxes = visualize_predictions(predictions_list, targets, device, CLASS_TO_IDX)
    
            if len(pred_boxes) == 0:
                print(f" No predictions for batch {i}")
    
            all_predictions.append({"boxes": pred_boxes, "scores": pred_logits, "labels": target_labels})
            all_targets.append({"boxes": target_boxes, "labels": target_labels})
    
            min_size = min(pred_logits.shape[0], target_labels.shape[0], pred_boxes.shape[0], target_boxes.shape[0])
            if min_size > 0:
                pred_logits, target_labels, pred_boxes, target_boxes = pred_logits[:min_size], target_labels[:min_size], pred_boxes[:min_size], target_boxes[:min_size]
                loss = criterion(pred_logits, target_labels, pred_boxes, target_boxes)
                
                if not loss.requires_grad:
                    loss = loss.clone().detach().requires_grad_(True)
                val_loss += loss.item()
                val_samples_count += len(images)

    avg_val_loss = val_loss / max(1, val_samples_count)
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch+1}: Validation Loss = {avg_val_loss:.4f}")

    # Compute & Print Evaluation Metrics
    metrics = compute_evaluation_metrics(all_predictions, all_targets, iou_thresholds=HYPERPARAMS["iou_thresholds"])
    for iou, metric_values in metrics.items():
        print(f"IoU {iou}: mAP={metric_values['mAP']:.3f}, AR={metric_values['AR']:.3f}")

    # ================== CHECKPOINT & EARLY STOPPING ==================
    if avg_val_loss < best_val_loss:
        save_checkpoint(model, optimizer, avg_val_loss, epoch, HYPERPARAMS["checkpoint_path"])
        best_val_loss = avg_val_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= HYPERPARAMS["patience"]:
        break

plot_loss_curve(train_losses, val_losses)
print("Training Complete!")
writer.close()
