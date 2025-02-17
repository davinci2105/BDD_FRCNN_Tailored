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

SELECTED_CLASSES = ['car', 'truck', 'bus', 'motor', 'bike', 'train',
                    'traffic light', 'traffic sign', 'rider', 'person']
NUM_CLASSES = len(SELECTED_CLASSES) + 1  

class LowLightEnhancement(nn.Module):
    """Low-light enhancement using a small CNN"""
    def __init__(self):
        super(LowLightEnhancement, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x):
        return torch.relu(self.conv(x))

class UNet(nn.Module):
    """A U-Net module for glare and flare removal"""
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder|| Pretrained ResNet18 as feature extractor
        self.encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])

        # Decoder 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class CustomFasterRCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CustomFasterRCNN, self).__init__()

        # Load Faster R-CNN with ResNet + FPN
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        
        self.model = build_model(self.cfg)
        self.predictor = DefaultPredictor(self.cfg)

        # Fix classifier head (to match selected class count)
        with torch.no_grad():
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor.cls_score = nn.Linear(in_features, num_classes)
            self.model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features, num_classes * 4)

        # Enhancement layers
        self.unet = UNet()  
        self.low_light_enhancer = LowLightEnhancement()  

    def forward(self, images):
        """Pass through enhancement layers, then Faster R-CNN"""
        images = images.float() / 255.0  # Normalize images to [0,1]
        images = self.unet(images)  # Apply U-Net for glare & flare removal
        images = self.low_light_enhancer(images)  # Apply low-light enhancement

        # Convert each image in the batch from Tensor to NumPy (H, W, C)
        processed_images = []
        for img in images:
            img = img.detach().permute(1, 2, 0).cpu().numpy()  # Convert (C, H, W) -> (H, W, C)
            img = (img * 255).astype(np.uint8)  # Convert back to uint8
            processed_images.append(img)

        # Detectron2 expects a list of images
        predictions = [self.predictor(img) for img in processed_images]

        return predictions

def get_model():
    return CustomFasterRCNN()
