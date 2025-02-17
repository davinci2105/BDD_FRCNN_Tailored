import torch
import torchmetrics
import os
import json
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torchvision.transforms import functional as F
from model import get_model
from dataloader import get_dataloader
from utils import compute_evaluation_metrics, visualize_predictions
import cv2
from sklearn.metrics import confusion_matrix


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_LABELS = "/mnt/d/BDD-ObjectDetection/bdd_dataset/labels/bdd100k_labels_images_val.json"  
TEST_IMAGES = "/mnt/d/BDD-ObjectDetection/bdd_dataset/100k/val"
CHECKPOINT_PATH = "/mnt/d/BDD-ObjectDetection/faster_rcnn_best.pth"
RESULTS_DIR = "evaluation_results"

# Results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_SAMPLES = 20  # None for full Dataset

model = get_model().to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

test_loader = get_dataloader(TEST_LABELS, TEST_IMAGES, batch_size=4, is_train=False)

SELECTED_CLASSES = ['car', 'truck', 'bus', 'motor', 'bike', 'train', 
                    'traffic light', 'traffic sign', 'rider', 'person']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(SELECTED_CLASSES)}
NUM_CLASSES = len(SELECTED_CLASSES)

# ===================== QUANTITATIVE METRICS =====================
print("\n Running Evaluation on Test Set...\n")

all_predictions = []
all_targets = []
image_count = 0

with torch.no_grad():
    for images, targets in tqdm.tqdm(test_loader, desc="Evaluating"):
        if MAX_SAMPLES is not None and image_count >= MAX_SAMPLES:
            break  

        images = torch.stack(images).to(DEVICE).float() / 255.0
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        predictions_list = model(images)
        
        pred_logits, pred_boxes, target_labels, target_boxes = visualize_predictions(
            predictions_list, targets, DEVICE, CLASS_TO_IDX
        )

        all_predictions.append({
            "boxes": pred_boxes.cpu(),
            "scores": pred_logits.cpu().squeeze(),
            "labels": target_labels.cpu()
        })
        all_targets.append({
            "boxes": target_boxes.cpu(),
            "labels": target_labels.cpu()
        })

        image_count += len(images) 

iou_thresholds = [0.5, 0.75, 0.95]
metrics = compute_evaluation_metrics(all_predictions, all_targets, iou_thresholds)


metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)
print(f"🔹 Evaluation metrics saved at {metrics_path}")


for iou, metric_vals in metrics.items():
    print(f"\n🔹 **Metrics at IoU {iou}:**")
    for metric, value in metric_vals.items():
        print(f"   {metric}: {value:.4f}")

# ===================== CONFUSION MATRIX =====================
def plot_confusion_matrix(pred_labels, true_labels, class_names, save_path):
    

    cm = confusion_matrix(true_labels, pred_labels, labels=range(len(class_names)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()
    print(f"🔹 Confusion matrix saved at {save_path}")

predicted_labels = [p["labels"].tolist() for p in all_predictions]
true_labels = [t["labels"].tolist() for t in all_targets]
predicted_labels = sum(predicted_labels, []) 
true_labels = sum(true_labels, [])  

cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
plot_confusion_matrix(predicted_labels, true_labels, SELECTED_CLASSES, cm_path)

# ===================== QUALITATIVE ANALYSIS =====================
FAILURE_DIR = os.path.join(RESULTS_DIR, "failure_cases")
os.makedirs(FAILURE_DIR, exist_ok=True)

def visualize_failures(images, predictions, targets, save_dir):
    """
    Images showing failed detections.

    """
    for i, (img, pred, gt) in enumerate(zip(images, predictions, targets)):
        img = img.permute(1, 2, 0).cpu().numpy() * 255
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Draw Ground Truth Boxes (Green)
        for box in gt["boxes"]:
            x1, y1, x2, y2 = box.int().tolist()
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw Predicted Boxes (Red)
        for box in pred["boxes"]:
            x1, y1, x2, y2 = box.int().tolist()
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        save_path = os.path.join(save_dir, f"failure_{i}.png")
        cv2.imwrite(save_path, img)

    print(f"🔹 Failure visualizations saved in {save_dir}")

# Select a few images for failure analysis
visualize_failures(images[:10], all_predictions[:10], all_targets[:10], FAILURE_DIR)

# ===================== ERROR CLUSTERING =====================
def analyze_failure_modes(pred_labels, true_labels, save_path):
    """
    Identifies patterns in model errors and saves as JSON.
    """
    from collections import Counter

    errors = [t for p, t in zip(pred_labels, true_labels) if p != t]
    error_counts = Counter(errors)

    error_data = {
        "most_misclassified_classes": [
            {"class": SELECTED_CLASSES[cls], "count": count} for cls, count in error_counts.most_common(5)
        ]
    }

    with open(save_path, "w") as f:
        json.dump(error_data, f, indent=4)

    print("\n🔹 **Error Breakdown (Most Misclassified Classes):**")
    for entry in error_data["most_misclassified_classes"]:
        print(f"   {entry['class']}: {entry['count']} misclassifications")

error_json_path = os.path.join(RESULTS_DIR, "error_analysis.json")
analyze_failure_modes(predicted_labels, true_labels, error_json_path)

# ===================== SUMMARY & NEXT STEPS =====================
summary_path = os.path.join(RESULTS_DIR, "evaluation_summary.txt")
with open(summary_path, "w") as f:
    f.write(f" **Evaluation Summary (First {MAX_SAMPLES if MAX_SAMPLES else 'All'} Images):**\n")
    for iou, metric_vals in metrics.items():
        f.write(f"\n Metrics at IoU {iou}:\n")
        for metric, value in metric_vals.items():
            f.write(f"   {metric}: {value:.4f}\n")
    f.write("\n Confusion matrix and failure cases saved.\n")

print(f"\n **Full evaluation results saved in `{RESULTS_DIR}`**")
