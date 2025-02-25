import torch
import torch.optim as optim
import tqdm
import os
from torch.utils.tensorboard import SummaryWriter

# Optional imports for distributed training.
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader

from model import get_model
from dataloader import get_dataloader
from losses import HybridLoss
from utils import save_checkpoint, load_checkpoint, plot_loss_curve, log_tensorboard, visualize_predictions, compute_evaluation_metrics

# ================== HYPERPARAMETERS ==================
# Set "distributed" to True when using multi-GPU DDP training.
HYPERPARAMS = {
    "distributed": True,           # Toggle distributed training here.
    "batch_size": 64,               # Base batch size (will be adjusted per GPU if distributed)
    "learning_rate": 0.0002,
    "epochs": 25,
    "patience": 5,
    "checkpoint_dir": "checkpoints",
    "tensorboard_log_dir": "logs",
    "use_regularization": False,
    "weight_decay": 5e-5,
    "iou_thresholds": [0.5, 0.75, 0.95],
    "train_mode": "full",          # "quick" for few images, "full" for complete dataset.
    "max_train_samples": 10,
    "max_val_samples": 10,
    "step_size": 5,
    "lr_gamma": 0.1,
    "num_workers": 8,
    "optimizer": "adamw"            # Choose "adamw" or "adam"
}

# ================== CUSTOM COLLATE FUNCTION (for DDP) ==================
def custom_collate_fn(batch):
    images, targets = zip(*batch)  # Unzip batch
    return list(images), list(targets)  # Return lists instead of stacked tensors

# ================== DATASET PATHS ==================
TRAIN_LABELS = "bdd_dataset/labels/bdd100k_labels_images_train.json"
TRAIN_IMAGES = "bdd_dataset/100k/train"
VAL_LABELS = "bdd_dataset/labels/bdd100k_labels_images_val.json"
VAL_IMAGES = "bdd_dataset/100k/val"

SELECTED_CLASSES = ['car', 'truck', 'bus', 'motor', 'bike', 'train', 
                    'traffic light', 'traffic sign', 'rider', 'person']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(SELECTED_CLASSES)}

# ================== DEVICE & DISTRIBUTED SETUP ==================
if HYPERPARAMS["distributed"]:
    def setup_ddp():
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
        return rank, world_size
    rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{rank}")
    # Adjust batch size per GPU.
    HYPERPARAMS["batch_size"] = HYPERPARAMS["batch_size"] // world_size
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = 0  # For non-distributed training, we treat rank as 0.

# ================== MODEL & OPTIMIZER SETUP ==================
model = get_model().to(device)
if HYPERPARAMS["distributed"]:
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

for param in model.parameters():
    param.requires_grad = True

if HYPERPARAMS["optimizer"] == "adamw":
    optimizer = optim.AdamW(
        model.parameters(),
        lr=HYPERPARAMS["learning_rate"],
        weight_decay=HYPERPARAMS["weight_decay"] if HYPERPARAMS["use_regularization"] else 0.0
    )
else:
    optimizer = optim.Adam(
        model.parameters(),
        lr=HYPERPARAMS["learning_rate"],
        weight_decay=HYPERPARAMS["weight_decay"] if HYPERPARAMS["use_regularization"] else 0.0
    )

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=HYPERPARAMS["step_size"], gamma=HYPERPARAMS["lr_gamma"])
criterion = HybridLoss()
writer = SummaryWriter(HYPERPARAMS["tensorboard_log_dir"]) if (not HYPERPARAMS["distributed"] or rank == 0) else None

best_val_loss, start_epoch = load_checkpoint(model, optimizer, HYPERPARAMS["checkpoint_dir"])
print(f"📌 Resuming training from epoch {start_epoch}...")

# ================== DATALOADERS SETUP ==================
if HYPERPARAMS["distributed"]:
    # Use quick mode if specified.
    if HYPERPARAMS["train_mode"] == "quick":
        train_loader_tmp = get_dataloader(
            TRAIN_LABELS, TRAIN_IMAGES,
            batch_size=HYPERPARAMS["batch_size"],
            shuffle=True,
            is_train=True,
            max_samples=HYPERPARAMS["max_train_samples"]
        )
        val_loader_tmp = get_dataloader(
            VAL_LABELS, VAL_IMAGES,
            batch_size=HYPERPARAMS["batch_size"],
            shuffle=False,
            is_train=False,
            max_samples=HYPERPARAMS["max_val_samples"]
        )
        print(f"Quick mode active: Using {len(train_loader_tmp.dataset)} training samples and {len(val_loader_tmp.dataset)} validation samples.")
    else:
        train_loader_tmp = get_dataloader(
            TRAIN_LABELS, TRAIN_IMAGES,
            batch_size=HYPERPARAMS["batch_size"],
            shuffle=True,
            is_train=True
        )
        val_loader_tmp = get_dataloader(
            VAL_LABELS, VAL_IMAGES,
            batch_size=HYPERPARAMS["batch_size"],
            shuffle=False,
            is_train=False
        )
    # Extract datasets from the temporary loaders.
    train_dataset = train_loader_tmp.dataset
    val_dataset = val_loader_tmp.dataset

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=HYPERPARAMS["batch_size"], shuffle=False,
        sampler=train_sampler, num_workers=HYPERPARAMS["num_workers"],
        pin_memory=True, prefetch_factor=2, collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=HYPERPARAMS["batch_size"], shuffle=False,
        sampler=val_sampler, num_workers=HYPERPARAMS["num_workers"],
        pin_memory=True, prefetch_factor=2, collate_fn=custom_collate_fn
    )
else:
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

# For accumulating per-image predictions during validation.
all_per_image_preds = []
all_per_image_targets = []

# ================== TRAINING & VALIDATION LOOP ==================
for epoch in range(start_epoch, HYPERPARAMS["epochs"]):
    model.train()
    if HYPERPARAMS["distributed"]:
        train_sampler.set_epoch(epoch)
    running_loss = 0.0
    train_samples_count = 0

    if not HYPERPARAMS["distributed"]:
        print(f"🔹 Training Mode: {HYPERPARAMS['train_mode']}")

    for batch_idx, (images, targets) in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{HYPERPARAMS['epochs']} Training", 
                                                                  disable=(HYPERPARAMS["distributed"] and rank != 0))):
        images = torch.stack(images).to(device).float() / 255.0  # Normalize images
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        predictions_list = model(images)

        concat_preds, per_image_preds, per_image_targets = visualize_predictions(
            predictions_list, targets, device, CLASS_TO_IDX, score_threshold=0.05
        )
        pred_logits, pred_boxes, target_labels, target_boxes = concat_preds

        min_size = min(pred_logits.shape[0], target_labels.shape[0], pred_boxes.shape[0], target_boxes.shape[0])
        if min_size > 0:
            loss = criterion(
                pred_logits[:min_size], target_labels[:min_size],
                pred_boxes[:min_size], target_boxes[:min_size]
            )
            if not loss.requires_grad:
                loss = loss.clone().detach().requires_grad_(True)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_samples_count += len(images)

    avg_train_loss = running_loss / max(1, train_samples_count)
    if not HYPERPARAMS["distributed"] or rank == 0:
        train_losses.append(avg_train_loss)
        log_tensorboard(writer, epoch, avg_train_loss, mode="train")
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")

    # ------------------ VALIDATION ------------------
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
                predictions_list, targets, device, CLASS_TO_IDX, score_threshold=0.05
            )
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
    if not HYPERPARAMS["distributed"] or rank == 0:
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}: Validation Loss = {avg_val_loss:.4f}")

        metrics = compute_evaluation_metrics(all_per_image_preds, all_per_image_targets, iou_thresholds=HYPERPARAMS["iou_thresholds"])
        for iou, metric_values in metrics.items():
            print(f"IoU {iou}: mAP={metric_values['mAP']:.3f}, AR={metric_values['AR']:.3f}")

    # ------------------ CHECKPOINT & EARLY STOPPING ------------------
    if not HYPERPARAMS["distributed"] or rank == 0:
        save_checkpoint(model, optimizer, avg_val_loss, epoch, HYPERPARAMS["checkpoint_dir"])
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

    scheduler.step()

    if patience_counter >= HYPERPARAMS["patience"]:
        if not HYPERPARAMS["distributed"] or rank == 0:
            print("Early stopping triggered.")
        break

# ------------------ FINALIZE ------------------
if not HYPERPARAMS["distributed"] or rank == 0:
    plot_loss_curve(train_losses, val_losses)
    print("Training Complete!")
    writer.close()

if HYPERPARAMS["distributed"]:
    dist.destroy_process_group()
