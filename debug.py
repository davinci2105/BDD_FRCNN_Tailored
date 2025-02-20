import torch
import os
import cv2
import numpy as np
import json
from torch.utils.data import DataLoader, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from model import get_model  # Model architecture and initialization
from dataloader import BDDDataset  # Custom dataset class for BDD
from utils import visualize_predictions  # Function to process predictions and ground truth for visualization

# --------------------------------------------------
# Debug Settings
# Description:
#   Settings for debugging the dataloader and model prediction pipeline.
# Variables:
#   DEBUG_BATCH - Number of batches to process for debugging.
#   BATCH_SIZE  - Number of samples per batch during debugging.
# --------------------------------------------------
DEBUG_BATCH = 1  # Number of batches to debug
BATCH_SIZE = 2   # Small batch size for debugging

# --------------------------------------------------
# Data Augmentations for Training
# Description:
#   Define a set of augmentations to be applied to training images and their bounding boxes.
# Variables:
#   train_transform - Albumentations Compose object that applies resizing, flipping,
#                     brightness/contrast adjustments, gamma corrections, grayscale conversion,
#                     and finally converts the image to a tensor.
# --------------------------------------------------
train_transform = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.RandomGamma(p=0.3),
    A.ToGray(p=0.1),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.2))

# --------------------------------------------------
# Function: get_debug_dataloader
# Description:
#   Creates a DataLoader for debugging purposes using a subset of the dataset.
# Parameters:
#   label_file - Path to the JSON file containing image labels.
#   image_dir  - Directory containing image files.
#   batch_size - Number of samples per batch.
#   max_samples- Optional maximum number of samples to use for debugging.
# Returns:
#   DataLoader object configured for debugging.
# --------------------------------------------------
def get_debug_dataloader(label_file, image_dir, batch_size, max_samples=None):
    dataset = BDDDataset(label_file, image_dir, transform=train_transform)
    if max_samples is not None:
        indices = list(range(min(len(dataset), max_samples)))
        dataset = Subset(dataset, indices)
    # Collate function that groups images and targets into tuples
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# --------------------------------------------------
# Paths for Dataset
# Description:
#   Define the paths to the training labels and images for debugging.
# Variables:
#   TRAIN_LABELS - Path to the training labels JSON file.
#   TRAIN_IMAGES - Directory path to training images.
# --------------------------------------------------
TRAIN_LABELS = "bdd_dataset/labels/bdd100k_labels_images_train.json"
TRAIN_IMAGES = "bdd_dataset/100k/train"

# --------------------------------------------------
# Create Debug Dataloader
# Description:
#   Initialize a dataloader for a small subset of the dataset for debugging purposes.
# --------------------------------------------------
debug_loader = get_debug_dataloader(TRAIN_LABELS, TRAIN_IMAGES, batch_size=BATCH_SIZE, max_samples=10)

# --------------------------------------------------
# Model Initialization
# Description:
#   Initialize the model, set it to evaluation mode, and move it to the appropriate device.
# Variables:
#   device - torch.device indicating whether to use 'cuda' or 'cpu'.
#   model  - Initialized model ready for inference.
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model().to(device)
model.eval()  # Set model to evaluation mode for debugging

# --------------------------------------------------
# Class Mapping for Predictions
# Description:
#   Define the list of selected classes and create a mapping from class names to indices.
# Variables:
#   SELECTED_CLASSES - List of class names to be detected.
#   CLASS_TO_IDX     - Dictionary mapping each class name to an integer index.
# --------------------------------------------------
SELECTED_CLASSES = ['car', 'truck', 'bus', 'motor', 'bike', 'train', 
                    'traffic light', 'traffic sign', 'rider', 'person']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(SELECTED_CLASSES)}

# --------------------------------------------------
# Debug Loop
# Description:
#   Iterate over a batch of data from the debug loader to inspect image shapes,
#   target details, and model predictions. Also processes predictions using the
#   visualize_predictions function.
# --------------------------------------------------
print("Starting Debug Loop...")

for batch_idx, (images, targets) in enumerate(debug_loader):
    if batch_idx >= DEBUG_BATCH:
        break

    print(f"\n--- Batch {batch_idx+1} ---")
    print(f"Number of images in batch: {len(images)}")
    
    # Print the shape of each image in the batch
    for i, img in enumerate(images):
        print(f"Image {i} shape: {img.shape}")
    
    # Print details about targets (bounding boxes and labels)
    print("\nTargets:")
    for i, target in enumerate(targets):
        print(f"Target {i} - boxes shape: {target['boxes'].shape}, labels shape: {target['labels'].shape}")
    
    # Prepare images for model input by stacking and converting to float
    images_tensor = torch.stack(images).to(device).float()
    
    # Run the model to obtain predictions for each image in the batch
    predictions_list = model(images_tensor)
    print(f"\nModel returned {len(predictions_list)} predictions (one per image).")
    
    # Process predictions and ground truth using the visualization function.
    # Returns:
    #   concat_preds      - Concatenated predictions for loss computation.
    #   per_image_preds   - List of dictionaries with predictions per image.
    #   per_image_targets - List of dictionaries with ground truth per image.
    concat_preds, per_image_preds, per_image_targets = visualize_predictions(
        predictions_list, targets, device, CLASS_TO_IDX, score_threshold=0.05
    )
    pred_logits, pred_boxes, target_labels, target_boxes = concat_preds

    # Print shapes of concatenated prediction outputs for debugging
    print("\nConcatenated Predictions for Loss Computation:")
    print(f"pred_logits shape: {pred_logits.shape}")
    print(f"pred_boxes shape: {pred_boxes.shape}")
    print("Ground Truth (Concatenated):")
    print(f"target_labels shape: {target_labels.shape}")
    print(f"target_boxes shape: {target_boxes.shape}")

    # Print details of sample predictions if available
    print("\nSample Prediction Details:")
    if pred_boxes.shape[0] > 0:
        print(f"First predicted box: {pred_boxes[0].cpu().detach().numpy()}")
        print(f"First predicted score: {pred_logits[0].cpu().detach().numpy()}")
        first_image_pred = per_image_preds[0]
        if first_image_pred["labels"].shape[0] > 0:
            print(f"First predicted label: {first_image_pred['labels'][0].cpu().detach().numpy()}")
        else:
            print("No predicted labels found in first image.")
    else:
        print("No predicted boxes found.")

    # Print details of sample ground truth if available
    print("\nSample Ground Truth Details:")
    if target_boxes.shape[0] > 0:
        print(f"First ground truth box: {target_boxes[0].cpu().detach().numpy()}")
        print(f"First ground truth label: {target_labels[0].cpu().detach().numpy()}")
    else:
        print("No ground truth boxes found.")

    # Print per-image predictions for further inspection
    print("\nPer-Image Predictions:")
    for idx, pred in enumerate(per_image_preds):
        print(f"Image {idx}:")
        print(f"  Boxes shape: {pred['boxes'].shape}")
        print(f"  Scores shape: {pred['scores'].shape}")
        print(f"  Labels shape: {pred['labels'].shape}")
    
    # Print per-image ground truth details
    print("\nPer-Image Ground Truth:")
    for idx, tgt in enumerate(per_image_targets):
        print(f"Image {idx}:")
        print(f"  Boxes shape: {tgt['boxes'].shape}")
        print(f"  Labels shape: {tgt['labels'].shape}")

    # Process only one batch for debugging
    break

print("\nDebugging complete.")
