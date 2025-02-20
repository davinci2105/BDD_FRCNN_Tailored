import torch
import torch.optim as optim
import tqdm
from model import get_model
from dataloader import get_dataloader
from losses import HybridLoss
from utils import save_checkpoint, load_checkpoint, plot_loss_curve, log_tensorboard, visualize_predictions, compute_evaluation_metrics
import os
from torch.utils.tensorboard import SummaryWriter

# Main Training Script
# Description:
#   This script sets up and trains an object detection model using a custom training loop.
#   It configures hyperparameters, data loaders, model, optimizer, scheduler, and loss criterion.
#   During training, it computes the training loss, validation loss, and evaluation metrics.
#   The script supports checkpointing, early stopping, and TensorBoard logging.
#
# Variables:
#   HYPERPARAMS         - Dictionary containing training hyperparameters and configuration settings.
#   TRAIN_LABELS        - Path to the training labels JSON file.
#   TRAIN_IMAGES        - Path to the training images directory.
#   VAL_LABELS          - Path to the validation labels JSON file.
#   VAL_IMAGES          - Path to the validation images directory.
#   SELECTED_CLASSES    - List of selected classes for detection.
#   CLASS_TO_IDX        - Dictionary mapping class names to integer indices.

HYPERPARAMS = {
    "batch_size": 16,
    "learning_rate": 0.0001,
    "epochs": 25,
    "patience": 5,
    "checkpoint_dir": "checkpoints",  # Directory to store checkpoint files
    "tensorboard_log_dir": "logs",
    "use_regularization": False,
    "weight_decay": 5e-5,
    "iou_thresholds": [0.5, 0.75, 0.95],
    "train_mode": "quick",  # "quick" for a few images, "full" for the complete dataset
    "max_train_samples": 10,
    "max_val_samples": 10,
    "step_size": 7,    # Number of epochs between LR decay
    "lr_gamma": 0.1,   # Factor for LR decay
}

TRAIN_LABELS = "bdd_dataset/labels/bdd100k_labels_images_train.json"
TRAIN_IMAGES = "bdd_dataset/100k/train"
VAL_LABELS = "bdd_dataset/labels/bdd100k_labels_images_val.json"
VAL_IMAGES = "bdd_dataset/100k/val"

SELECTED_CLASSES = ['car', 'truck', 'bus', 'motor', 'bike', 'train', 
                    'traffic light', 'traffic sign', 'rider', 'person']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(SELECTED_CLASSES)}

# Device setup, model initialization, optimizer and scheduler configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model().to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=HYPERPARAMS["learning_rate"],
    weight_decay=HYPERPARAMS["weight_decay"] if HYPERPARAMS["use_regularization"] else 0.0
)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=HYPERPARAMS["step_size"], gamma=HYPERPARAMS["lr_gamma"])

criterion = HybridLoss()
writer = SummaryWriter(HYPERPARAMS["tensorboard_log_dir"])

# Load latest checkpoint if available and resume training
best_val_loss, start_epoch = load_checkpoint(model, optimizer, HYPERPARAMS["checkpoint_dir"])
print(f"📌 Resuming training from epoch {start_epoch}...")

# Setup data loaders based on training mode
if HYPERPARAMS["train_mode"] == "quick":
    train_loader = get_dataloader(
        TRAIN_LABELS, TRAIN_IMAGES, 
        batch_size=HYPERPARAMS["batch_size"], 
        shuffle=True, 
        is_train=True, 
        max_samples=HYPERPARAMS["max_train_samples"]
    )
    val_loader = get_dataloader(
        VAL_LABELS, VAL_IMAGES, 
        batch_size=HYPERPARAMS["batch_size"], 
        shuffle=False, 
        is_train=False, 
        max_samples=HYPERPARAMS["max_val_samples"]
    )
    print(f"Quick mode active: Using {len(train_loader.dataset)} training samples and {len(val_loader.dataset)} validation samples.")
else:
    train_loader = get_dataloader(TRAIN_LABELS, TRAIN_IMAGES, batch_size=HYPERPARAMS["batch_size"], shuffle=True, is_train=True)
    val_loader = get_dataloader(VAL_LABELS, VAL_IMAGES, batch_size=HYPERPARAMS["batch_size"], shuffle=False, is_train=False)

train_losses, val_losses = [], []
patience_counter = 0

# Training and validation loop
for epoch in range(start_epoch, HYPERPARAMS["epochs"]):
    model.train()
    running_loss = 0.0
    train_samples_count = 0

    print(f"🔹 Training Mode: {HYPERPARAMS['train_mode']}")

    for batch_idx, (images, targets) in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{HYPERPARAMS['epochs']} Training")):
        images = torch.stack(images).to(device).float() / 255.0  # Normalize images to [0,1]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        predictions_list = model(images)

        # Process predictions and targets for loss computation and metric evaluation
        concat_preds, per_image_preds, per_image_targets = visualize_predictions(
            predictions_list, targets, device, CLASS_TO_IDX, score_threshold=0.05)
        pred_logits, pred_boxes, target_labels, target_boxes = concat_preds

        # Align tensor lengths if necessary
        min_size = min(pred_logits.shape[0], target_labels.shape[0], pred_boxes.shape[0], target_boxes.shape[0])
        if min_size > 0:
            pred_logits = pred_logits[:min_size]
            target_labels = target_labels[:min_size]
            pred_boxes = pred_boxes[:min_size]
            target_boxes = target_boxes[:min_size]

            loss = criterion(pred_logits, target_labels, pred_boxes, target_boxes)
            
            if not loss.requires_grad:
                loss = loss.clone().detach().requires_grad_(True)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_samples_count += len(images)

    avg_train_loss = running_loss / max(1, train_samples_count)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")
    log_tensorboard(writer, epoch, avg_train_loss, mode="train")

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_samples_count = 0
    all_per_image_preds = []
    all_per_image_targets = []

    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            images = torch.stack(images).to(device).float() / 255.0
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    
            predictions_list = model(images)
    
            concat_preds, per_image_preds, per_image_targets = visualize_predictions(
                predictions_list, targets, device, CLASS_TO_IDX, score_threshold=0.05)
            pred_logits, pred_boxes, target_labels, target_boxes = concat_preds
    
            if pred_boxes.shape[0] == 0:
                print(f" No predictions for batch {i}")
    
            all_per_image_preds.extend(per_image_preds)
            all_per_image_targets.extend(per_image_targets)
    
            min_size = min(pred_logits.shape[0], target_labels.shape[0], pred_boxes.shape[0], target_boxes.shape[0])
            if min_size > 0:
                pred_logits = pred_logits[:min_size]
                target_labels = target_labels[:min_size]
                pred_boxes = pred_boxes[:min_size]
                target_boxes = target_boxes[:min_size]
                loss = criterion(pred_logits, target_labels, pred_boxes, target_boxes)
                
                if not loss.requires_grad:
                    loss = loss.clone().detach().requires_grad_(True)
                val_loss += loss.item()
                val_samples_count += len(images)

    avg_val_loss = val_loss / max(1, val_samples_count)
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch+1}: Validation Loss = {avg_val_loss:.4f}")

    # Compute and display evaluation metrics based on validation predictions
    metrics = compute_evaluation_metrics(all_per_image_preds, all_per_image_targets, iou_thresholds=HYPERPARAMS["iou_thresholds"])
    for iou, metric_values in metrics.items():
        print(f"IoU {iou}: mAP={metric_values['mAP']:.3f}, AR={metric_values['AR']:.3f}")

    # Save checkpoint and check for early stopping criteria
    save_checkpoint(model, optimizer, avg_val_loss, epoch, HYPERPARAMS["checkpoint_dir"])
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
    else:
        patience_counter += 1

    scheduler.step()

    if patience_counter >= HYPERPARAMS["patience"]:
        break

plot_loss_curve(train_losses, val_losses)
print("Training Complete!")
writer.close()
