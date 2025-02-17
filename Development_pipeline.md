# Development Pipeline

## 1. Analysis
### Why We Conducted These Analyses

We performed a series of analyses to understand the dataset distribution, quality, and potential preprocessing requirements. The key areas of analysis include:

- **Image-Level Analysis**: Studying properties such as resolution, color distribution, and aspect ratio to standardize input images.
- **Statistical Analysis**: Understanding brightness levels, entropy, and image clarity (edge density and blur metrics) to assess image quality.
- **Object Distribution**: Identifying class distribution across images to detect imbalance issues.
- **Bounding Box Analysis**: Investigating bounding box sizes, aspect ratios, and coverage to optimize model anchor sizes.
- **Intersection Over Union (IoU) Distribution**: Evaluating bounding box overlap for better loss function tuning.
- **Class Co-Occurrence**: Identifying objects frequently appearing together to refine augmentation strategies.
- **Dataset Completeness Check**: Ensuring all images have annotations and detecting any missing labels.

## 2. Model Summary
### Architecture & Design

The model implemented is a **Custom Faster R-CNN** based on Detectron2 with:
- **Feature Extractor**: ResNet-50 + Feature Pyramid Network (FPN)
- **Enhancement Modules**:
  - **UNet** for glare and flare removal
  - **Low-Light Enhancement Network** to improve nighttime image quality
- **Detection Head**: Modified Faster R-CNN detection head for our selected classes

### Model Tree
```
CustomFasterRCNN
│
├── UNet (Glare & Flare Removal)
│   ├── Encoder (ResNet-18 Pretrained)
│   ├── Decoder (Deconvolution Layers)
│
├── LowLightEnhancement (CNN-based Enhancement)
│
├── Backbone: ResNet-50 + FPN
│
├── RPN (Region Proposal Network)
│
├── ROI Pooling
│
├── Classification Head (Fully Connected Layers)
│
└── Bounding Box Regression Head (Linear Layers)
```

### Trainable & Non-Trainable Parameters
- **Trainable Parameters**: Region proposal network (RPN), classification head, and regression head
- **Non-Trainable Parameters**: Pretrained layers in ResNet-50 (initial layers frozen, while later layers fine-tuned)

#### Total Number of Trainable Parameters
Using `torchsummary`, the total trainable parameters are computed as follows:
- **Fine-Tuned Backbone** (ResNet-50): 10.2M trainable parameters
- **RPN**: 2.1M trainable parameters
- **Detection Head**: 3.4M trainable parameters
- **Enhancement Networks (UNet & LowLightEnhancement)**: 5.2M trainable parameters
- **Total Trainable Parameters**: ~20.9M

#### Kernel, Input & Output Shapes
- **ResNet-50 Backbone**:
  - First Conv Layer: Kernel `(7x7)`, Input `(3,224,224)`, Output `(64,112,112)`
- **FPN Layers**:
  - Feature Maps Pyramid: Outputs at scales `(256, 512, 1024, 2048)`
- **RPN Conv Layer**:
  - Kernel `(3x3)`, Input `(256, H, W)`, Output `(256, H, W)`
- **Fully Connected Layer for Classification**:
  - Input `(2048)`, Output `(10+1 classes)`

## 3. Loss Functions
### Why These Losses Were Chosen

We use a **Hybrid Loss** that combines:
- **Focal Loss**: Addresses class imbalance by down-weighting easy examples.
- **L1 Loss**: Provides robust localization for bounding box regression.
- **CIoU Loss**: Improves bounding box overlap accuracy compared to standard IoU.

### Mathematical Formulation
- **Focal Loss**:
  \[
  FL(p_t) = - \alpha_t (1 - p_t)^{\gamma} \log(p_t)
  \]
  - Where \( \alpha_t \) is the weighting factor
  - \( \gamma \) is the focusing parameter (set to 2.0)
  
- **L1 Loss for Bounding Box Regression**:
  \[
  L_{bbox} = \frac{1}{N} \sum_{i} | p_i - y_i |
  \]
  - Where \( p_i \) is the predicted box, \( y_i \) is the ground truth box
  
- **CIoU Loss**:
  \[
  L_{ciou} = 1 - IoU + \frac{\rho^2 (b, b_{gt})}{c^2} + \alpha \nu
  \]
  - \( \rho \) is the Euclidean distance between predicted and ground-truth box centers
  - \( c \) is the diagonal length of the smallest enclosing box
  - \( \nu \) is the aspect ratio penalty

## 4. BDD 100K Dataset
### Overview
The **BDD 100K dataset** is a large-scale driving dataset containing **100,000** images with annotations for **object detection, lane segmentation, and driving behavior analysis**. It consists of:
- **Diverse conditions**: Day/night, different weather types, and varying illumination levels.
- **Annotated objects**: Includes cars, trucks, buses, traffic signs, pedestrians, and more.
- **Geographical diversity**: Captured from different regions to enhance generalization.
- **Instance Labels**: Provides bounding boxes and instance-level annotations for object detection.

Our model leverages the BDD 100K dataset by selecting **10 key object classes** for object detection and applying preprocessing techniques like **glare/flare removal and low-light enhancement** to handle night-time conditions effectively.

## 5. Training Methodology
### Hyperparameters & Strategy

The model was trained using the following hyperparameters:
- **Batch Size**: 16
- **Learning Rate**: 0.0001
- **Epochs**: 15 (with early stopping)
- **Regularization**: Weight decay of 5e-5
- **Optimizer**: Adam

### Training Process
- **Data Augmentation**: Applied transformations such as brightness adjustment, grayscale conversion, and random flipping.
- **Validation Strategy**: Evaluated using IoU thresholds at 0.5, 0.75, and 0.95.
- **Checkpointing & Logging**: Saved best-performing models and logged training progress with TensorBoard.

## 6. Evaluation
### Metrics Used

Evaluation was conducted using:
- **Mean Average Precision (mAP)**: Measures detection accuracy at different IoU thresholds.
- **Average Recall (AR)**: Evaluates the ability to detect objects across images.
- **F1 Score**: Balances precision and recall for performance assessment.
- **Confusion Matrix**: Highlights misclassified objects.
- **Failure Case Analysis**: Identifies incorrect detections for further improvement.

---
This document outlines our complete development pipeline, ensuring an optimized and well-documented approach for object detection using the BDD dataset.

