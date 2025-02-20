import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance handling.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class L1Loss(nn.Module):
    """
    L1 Loss for bounding box regression.
    """
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, pred, target):
        return F.l1_loss(pred, target, reduction='mean')

class CIoULoss(nn.Module):
    """
    Complete Intersection over Union (CIoU) Loss for bounding box regression.
    """
    def __init__(self):
        super(CIoULoss, self).__init__()

    def forward(self, pred_boxes, target_boxes):
        """
        pred_boxes: [x1, y1, x2, y2]
        target_boxes: [x1, y1, x2, y2]
        """
        pred_w = pred_boxes[:, 2] - pred_boxes[:, 0]
        pred_h = pred_boxes[:, 3] - pred_boxes[:, 1]
        target_w = target_boxes[:, 2] - target_boxes[:, 0]
        target_h = target_boxes[:, 3] - target_boxes[:, 1]

        pred_area = pred_w * pred_h
        target_area = target_w * target_h

        inter_w = torch.min(pred_boxes[:, 2], target_boxes[:, 2]) - torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        inter_h = torch.min(pred_boxes[:, 3], target_boxes[:, 3]) - torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        inter_w = torch.clamp(inter_w, min=0)
        inter_h = torch.clamp(inter_h, min=0)

        inter_area = inter_w * inter_h
        union_area = pred_area + target_area - inter_area

        iou = inter_area / (union_area + 1e-6)

        center_distance = (pred_boxes[:, 0] + pred_boxes[:, 2] - target_boxes[:, 0] - target_boxes[:, 2]) ** 2 + \
                          (pred_boxes[:, 1] + pred_boxes[:, 3] - target_boxes[:, 1] - target_boxes[:, 3]) ** 2

        diag_distance = (torch.max(pred_boxes[:, 2], target_boxes[:, 2]) - torch.min(pred_boxes[:, 0], target_boxes[:, 0])) ** 2 + \
                        (torch.max(pred_boxes[:, 3], target_boxes[:, 3]) - torch.min(pred_boxes[:, 1], target_boxes[:, 1])) ** 2

        ciou = iou - center_distance / (diag_distance + 1e-6)
        return (1 - ciou).mean()

class HybridLoss(nn.Module):
    """
    Hybrid loss function combining Focal Loss, L1 Loss, and CIoU Loss.
    """
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