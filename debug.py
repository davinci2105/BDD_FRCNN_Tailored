import torch
import os
import cv2
import numpy as np
import json
from torch.utils.data import DataLoader, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from model import get_model  # Your model.py (defines get_model, UNet, LowLightEnhancement, etc.)
from dataloader import BDDDataset  # Your dataset class
from utils import visualize_predictions  # Updated visualization function

# ======= Debug Settings =======
DEBUG_BATCH = 1  # Number of batches to debug
BATCH_SIZE = 2   # Small batch size for debugging

# ======= Define Data Augmentations (same as in your dataloader) =======
train_transform = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.RandomGamma(p=0.3),
    A.ToGray(p=0.1),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.2))

# ======= Debug Dataloader Function =======
def get_debug_dataloader(label_file, image_dir, batch_size, max_samples=None):
    dataset = BDDDataset(label_file, image_dir, transform=train_transform)
    if max_samples is not None:
        indices = list(range(min(len(dataset), max_samples)))
        dataset = Subset(dataset, indices)
    # Use a simple collate_fn that returns a tuple (images, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# ======= Paths (Update these paths as necessary) =======
TRAIN_LABELS = "bdd_dataset/labels/bdd100k_labels_images_train.json"
TRAIN_IMAGES = "bdd_dataset/100k/train"

# ======= Create Debug Dataloader =======
debug_loader = get_debug_dataloader(TRAIN_LABELS, TRAIN_IMAGES, batch_size=BATCH_SIZE, max_samples=10)

# ======= Initialize Model =======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model().to(device)
model.eval()  # Set model to evaluation mode for debugging

# ======= CLASS MAPPING =======
SELECTED_CLASSES = ['car', 'truck', 'bus', 'motor', 'bike', 'train', 
                    'traffic light', 'traffic sign', 'rider', 'person']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(SELECTED_CLASSES)}

# ======= Debug Loop =======
print("Starting Debug Loop...")

for batch_idx, (images, targets) in enumerate(debug_loader):
    if batch_idx >= DEBUG_BATCH:
        break

    print(f"\n--- Batch {batch_idx+1} ---")
    print(f"Number of images in batch: {len(images)}")
    
    # Print individual image shapes
    for i, img in enumerate(images):
        print(f"Image {i} shape: {img.shape}")
    
    # Print target details for each image
    print("\nTargets:")
    for i, target in enumerate(targets):
        print(f"Target {i} - boxes shape: {target['boxes'].shape}, labels shape: {target['labels'].shape}")
    
    # Prepare images: stack into tensor; note your model does normalization internally,
    # so here we simply convert to float.
    images_tensor = torch.stack(images).to(device).float()
    
    # Run the model to obtain predictions (one prediction per image)
    predictions_list = model(images_tensor)
    print(f"\nModel returned {len(predictions_list)} predictions (one per image).")
    
    # Use the updated visualization function with a score threshold.
    # This returns three outputs:
    #   1. concat_preds: concatenated (scores, boxes, target_labels, target_boxes) for loss computation,
    #   2. per_image_preds: list of dictionaries for each image's predictions,
    #   3. per_image_targets: list of dictionaries for each image's ground truth.
    concat_preds, per_image_preds, per_image_targets = visualize_predictions(
        predictions_list, targets, device, CLASS_TO_IDX, score_threshold=0.05
    )
    pred_logits, pred_boxes, target_labels, target_boxes = concat_preds

    # Debug print concatenated outputs shapes
    print("\nConcatenated Predictions for Loss Computation:")
    print(f"pred_logits shape: {pred_logits.shape}")
    print(f"pred_boxes shape: {pred_boxes.shape}")
    print("Ground Truth (Concatenated):")
    print(f"target_labels shape: {target_labels.shape}")
    print(f"target_boxes shape: {target_boxes.shape}")

    # Print sample prediction details
    print("\nSample Prediction Details:")
    if pred_boxes.shape[0] > 0:
        print(f"First predicted box: {pred_boxes[0].cpu().detach().numpy()}")
        print(f"First predicted score: {pred_logits[0].cpu().detach().numpy()}")
        # Note: pred_logits here holds scores; if you want to print the predicted label, you can index per_image_preds.
        # For example, print the first predicted label of the first image:
        first_image_pred = per_image_preds[0]
        if first_image_pred["labels"].shape[0] > 0:
            print(f"First predicted label: {first_image_pred['labels'][0].cpu().detach().numpy()}")
        else:
            print("No predicted labels found in first image.")
    else:
        print("No predicted boxes found.")

    # Print sample ground truth details
    print("\nSample Ground Truth Details:")
    if target_boxes.shape[0] > 0:
        print(f"First ground truth box: {target_boxes[0].cpu().detach().numpy()}")
        print(f"First ground truth label: {target_labels[0].cpu().detach().numpy()}")
    else:
        print("No ground truth boxes found.")

    # Optionally, you can also print the per-image dictionaries in more detail:
    print("\nPer-Image Predictions:")
    for idx, pred in enumerate(per_image_preds):
        print(f"Image {idx}:")
        print(f"  Boxes shape: {pred['boxes'].shape}")
        print(f"  Scores shape: {pred['scores'].shape}")
        print(f"  Labels shape: {pred['labels'].shape}")
    
    print("\nPer-Image Ground Truth:")
    for idx, tgt in enumerate(per_image_targets):
        print(f"Image {idx}:")
        print(f"  Boxes shape: {tgt['boxes'].shape}")
        print(f"  Labels shape: {tgt['labels'].shape}")

    # Break after processing one batch for debugging.
    break

print("\nDebugging complete.")
