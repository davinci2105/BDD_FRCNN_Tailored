import torch
from torch.utils.data import Dataset, DataLoader, Subset
import json
import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from PIL import Image
from model import UNet, LowLightEnhancement

# --------------------------------------------------
# Global Class Definitions
# Description:
#   SELECTED_CLASSES: List of classes to be detected.
#   CLASS_TO_IDX: Dictionary mapping each selected class to a unique integer index (background is 0).
# --------------------------------------------------
SELECTED_CLASSES = ['car', 'truck', 'bus', 'motor', 'bike', 'train',
                    'traffic light', 'traffic sign', 'rider', 'person']
CLASS_TO_IDX = {cls: idx+1 for idx, cls in enumerate(SELECTED_CLASSES)}  # Background = 0

# --------------------------------------------------
# Class: BDDDataset
# Description:
#   Custom Dataset class for handling BDD (Berkeley DeepDrive) dataset.
#   It loads images and corresponding annotations, filters out annotations not in SELECTED_CLASSES,
#   applies optional preprocessing using UNet and LowLightEnhancement, and finally applies augmentations.
#
# Parameters:
#   label_file: Path to the JSON file containing annotations.
#   image_dir: Directory containing the image files.
#   transform: Albumentations transformation pipeline to apply.
#   preprocess: Boolean flag to control whether to run the image through preprocessing models.
#
# Returns:
#   A tuple (image, target) where:
#       image: Transformed image tensor.
#       target: Dictionary containing 'boxes' (bounding boxes tensor) and 'labels' (class labels tensor).
# --------------------------------------------------
class BDDDataset(Dataset):
    def __init__(self, label_file, image_dir, transform=None, preprocess=True):
        self.image_dir = image_dir
        self.preprocess = preprocess
        self.transform = transform

        # Load annotations from JSON file.
        with open(label_file, 'r') as f:
            self.annotations = json.load(f)

        # Filter annotations to include only images with selected classes.
        self.filtered_annotations = []
        for ann in self.annotations:
            valid_objs = [obj for obj in ann['labels'] if obj['category'] in SELECTED_CLASSES]
            if valid_objs:
                self.filtered_annotations.append((ann['name'], valid_objs))

        # Initialize preprocessing models if enabled.
        if self.preprocess:
            self.unet = UNet().eval()
            self.low_light_enhancer = LowLightEnhancement().eval()

    def __len__(self):
        return len(self.filtered_annotations)

    def __getitem__(self, idx):
        # Retrieve image name and associated objects.
        img_name, objs = self.filtered_annotations[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape  

        # Convert bounding boxes and labels to normalized coordinates.
        bboxes = []
        labels = []
        for obj in objs:
            x1, y1, x2, y2 = obj['box2d']['x1'], obj['box2d']['y1'], obj['box2d']['x2'], obj['box2d']['y2']
            label = CLASS_TO_IDX[obj['category']]
            # Normalize bounding box coordinates.
            x1 /= width
            x2 /= width
            y1 /= height
            y2 /= height
            
            bboxes.append([x1, y1, x2, y2])
            labels.append(label)

        bboxes = np.array(bboxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        # Convert image to tensor for preprocessing.
        image_tensor = transforms.ToTensor()(image).unsqueeze(0)

        # Apply preprocessing using UNet and LowLightEnhancement if enabled.
        if self.preprocess:
            with torch.no_grad():
                image_tensor = self.unet(image_tensor)
                image_tensor = self.low_light_enhancer(image_tensor)

        # Convert the processed tensor back to a numpy image.
        image = image_tensor.squeeze(0).permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)

        # Apply data augmentations.
        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
            image, bboxes, labels = transformed["image"], transformed["bboxes"], transformed["labels"]

        target = {
            "boxes": torch.tensor(bboxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }
        return image, target

# --------------------------------------------------
# Data Augmentations using Albumentations
# Description:
#   train_transform: Augmentation pipeline for training images.
#   val_transform: Augmentation pipeline for validation images.
# --------------------------------------------------
train_transform = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.RandomGamma(p=0.3),
    A.ToGray(p=0.1),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.2))

val_transform = A.Compose([
    A.Resize(512, 512),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.2))

# --------------------------------------------------
# Function: get_dataloader
# Description:
#   Returns a DataLoader for the BDDDataset.
#
# Parameters:
#   label_file: Path to the JSON file containing annotations.
#   image_dir: Directory containing image files.
#   batch_size: Number of samples per batch (default is 8).
#   shuffle: Boolean flag to indicate whether to shuffle the dataset (default is True).
#   is_train: Boolean flag to indicate training mode (applies train_transform if True, else val_transform).
#   max_samples: Optional integer to limit the number of samples (useful for quick debugging).
#
# Returns:
#   A DataLoader object configured with the specified parameters.
# --------------------------------------------------
def get_dataloader(label_file, image_dir, batch_size=8, shuffle=True, is_train=True, max_samples=None):
    transform = train_transform if is_train else val_transform
    dataset = BDDDataset(label_file, image_dir, transform=transform)
    # Use a subset of the dataset if max_samples is provided.
    if max_samples is not None:
        indices = list(range(min(len(dataset), max_samples)))
        dataset = Subset(dataset, indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: tuple(zip(*x)))
