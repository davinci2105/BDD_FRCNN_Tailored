import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --------------------------------------------------
# Class: FocalLoss
# Description:
#   Implements Focal Loss to address class imbalance.
#   The loss focuses on hard-to-classify examples by down-weighting easy examples.
#
# Parameters:
#   alpha: Weighting factor for the loss (default: 0.25)
#   gamma: Focusing parameter to modulate the loss (default: 2.0)
#
# Usage:
#   loss = FocalLoss()(inputs, targets)
# --------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# --------------------------------------------------
# Class: L1Loss
# Description:
#   Implements the L1 Loss (Mean Absolute Error) for bounding box regression.
#
# Usage:
#   loss = L1Loss()(predicted_boxes, target_boxes)
# --------------------------------------------------
class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, pred, target):
        return F.l1_loss(pred, target, reduction='mean')

# --------------------------------------------------
# Class: CIoULoss
# Description:
#   Implements the Complete Intersection over Union (CIoU) Loss for bounding box regression.
#   CIoU considers not only the overlap area but also the distance between box centers and aspect ratio.
#
# Usage:
#   loss = CIoULoss()(predicted_boxes, target_boxes)
# --------------------------------------------------
class CIoULoss(nn.Module):
    def __init__(self):
        super(CIoULoss, self).__init__()

    def forward(self, pred_boxes, target_boxes):
        """
        pred_boxes: Tensor of shape [N, 4] in the format [x1, y1, x2, y2]
        target_boxes: Tensor of shape [N, 4] in the same format
        """
        # Calculate width and height of predicted and target boxes
        pred_w = pred_boxes[:, 2] - pred_boxes[:, 0]
        pred_h = pred_boxes[:, 3] - pred_boxes[:, 1]
        target_w = target_boxes[:, 2] - target_boxes[:, 0]
        target_h = target_boxes[:, 3] - target_boxes[:, 1]

        # Compute areas of predicted and target boxes
        pred_area = pred_w * pred_h
        target_area = target_w * target_h

        # Compute intersection dimensions
        inter_w = torch.min(pred_boxes[:, 2], target_boxes[:, 2]) - torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        inter_h = torch.min(pred_boxes[:, 3], target_boxes[:, 3]) - torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        inter_w = torch.clamp(inter_w, min=0)
        inter_h = torch.clamp(inter_h, min=0)

        # Compute intersection and union areas
        inter_area = inter_w * inter_h
        union_area = pred_area + target_area - inter_area

        # Compute IoU
        iou = inter_area / (union_area + 1e-6)

        # Compute center distance between predicted and target boxes
        center_distance = (pred_boxes[:, 0] + pred_boxes[:, 2] - target_boxes[:, 0] - target_boxes[:, 2]) ** 2 + \
                          (pred_boxes[:, 1] + pred_boxes[:, 3] - target_boxes[:, 1] - target_boxes[:, 3]) ** 2

        # Compute the diagonal length of the smallest enclosing box
        diag_distance = (torch.max(pred_boxes[:, 2], target_boxes[:, 2]) - torch.min(pred_boxes[:, 0], target_boxes[:, 0])) ** 2 + \
                        (torch.max(pred_boxes[:, 3], target_boxes[:, 3]) - torch.min(pred_boxes[:, 1], target_boxes[:, 1])) ** 2

        # Compute CIoU
        ciou = iou - center_distance / (diag_distance + 1e-6)
        return (1 - ciou).mean()

# --------------------------------------------------
# Class: HybridLoss
# Description:
#   Combines Focal Loss, L1 Loss, and CIoU Loss into a single loss function.
#   The total loss is a weighted sum of the three losses.
#
# Parameters:
#   alpha: Weight for Focal Loss (default: 1.0)
#   beta: Weight for L1 Loss (default: 1.0)
#   gamma: Weight for CIoU Loss (default: 1.0)
#
# Usage:
#   loss = HybridLoss()(class_preds, class_targets, bbox_preds, bbox_targets)
# --------------------------------------------------
class HybridLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        super(HybridLoss, self).__init__()
        self.focal_loss = FocalLoss()
        self.l1_loss = L1Loss()
        self.ciou_loss = CIoULoss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, class_preds, class_targets, bbox_preds, bbox_targets):
        focal = self.focal_loss(class_preds, class_targets)
        l1 = self.l1_loss(bbox_preds, bbox_targets)
        ciou = self.ciou_loss(bbox_preds, bbox_targets)
        total_loss = self.alpha * focal + self.beta * l1 + self.gamma * ciou
        return total_loss
