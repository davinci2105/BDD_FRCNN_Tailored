import torch
import torch.nn as nn
import detectron2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.engine import DefaultPredictor
import torchvision.models as models
import torch.nn.functional as F
import numpy as np

# --------------------------------------------------
# Global Definitions
# Description:
#   SELECTED_CLASSES: List of classes for detection.
#   NUM_CLASSES: Total number of classes (including background).
# --------------------------------------------------
SELECTED_CLASSES = ['car', 'truck', 'bus', 'motor', 'bike', 'train',
                    'traffic light', 'traffic sign', 'rider', 'person']
NUM_CLASSES = len(SELECTED_CLASSES) + 1  # Including background

# --------------------------------------------------
# Class: LowLightEnhancement
# Description:
#   A simple CNN-based module for enhancing images captured in low-light conditions.
# --------------------------------------------------
class LowLightEnhancement(nn.Module):
    def __init__(self):
        super(LowLightEnhancement, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x):
        return torch.relu(self.conv(x))

# --------------------------------------------------
# Class: UNet
# Description:
#   A U-Net based architecture for removing glare and flare from images.
#   Uses a pretrained ResNet18 as the encoder.
# --------------------------------------------------
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder: Pretrained ResNet18 (removing last two layers)
        self.encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        # Decoder: Upsampling layers to reconstruct a 3-channel image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # Normalize output between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# --------------------------------------------------
# Class: CustomFasterRCNN
# Description:
#   Custom Faster R-CNN model with integrated image enhancement modules.
#   The model uses Detectron2's Faster R-CNN (ResNet + FPN) as the base detector.
#   Enhancement modules (UNet and LowLightEnhancement) are applied before detection.
#
# Usage:
#   model = CustomFasterRCNN()
#   predictions = model(images)
# --------------------------------------------------
class CustomFasterRCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CustomFasterRCNN, self).__init__()
        # Initialize Detectron2 config and load Faster R-CNN model from model zoo
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        
        self.model = build_model(self.cfg)
        self.predictor = DefaultPredictor(self.cfg)

        # Adjust the classifier head to match the number of selected classes
        with torch.no_grad():
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor.cls_score = nn.Linear(in_features, num_classes)
            self.model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features, num_classes * 4)

        # Initialize enhancement modules
        self.unet = UNet()
        self.low_light_enhancer = LowLightEnhancement()

    def forward(self, images):
        """
        Process input images through enhancement modules, then perform detection.
        Parameters:
            images: Tensor of shape (B, C, H, W) in range [0, 255]
        Returns:
            predictions: List of detection outputs (one per image)
        """
        images = images.float() / 255.0  # Normalize images to [0,1]
        images = self.unet(images)  # Remove glare/flare
        images = self.low_light_enhancer(images)  # Enhance low-light conditions

        # Convert each image tensor to a NumPy array for Detectron2
        processed_images = []
        for img in images:
            img = img.detach().permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
            img = (img * 255).astype(np.uint8)
            processed_images.append(img)

        # Generate predictions using Detectron2's predictor
        predictions = [self.predictor(img) for img in processed_images]
        return predictions

# --------------------------------------------------
# Function: get_model
# Description:
#   Helper function to instantiate the custom Faster R-CNN model with enhancement modules.
#
# Returns:
#   An instance of CustomFasterRCNN.
# --------------------------------------------------
def get_model():
    return CustomFasterRCNN()
