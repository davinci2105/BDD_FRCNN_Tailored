# <p align="center">BDD-ObjectDetection: Object Detection using Faster R-CNN on BDD100K Dataset and Hybrid Weighted Loss</p>
BDD-ObjectDetection: Object Detection using Faster R-CNN on BDD100K Dataset and Hybrid Weighted Loss


This project implements an object detection system using the Faster R-CNN architecture on the BDD100K dataset. The system is evaluated using mAP (mean Average Precision) and other detection metrics.

## Introduction

The BDD100K dataset is one of the largest datasets for self-driving research, specifically designed for advanced driver assistance systems (ADAS). The dataset contains over 100,000 images with annotations for object detection, tracking, and segmentation. In this project, we focus on object detection using Faster R-CNN, a state-of-the-art model known for its speed and accuracy.

The objective of this project is to train a Faster R-CNN model on the BDD100K dataset, evaluate its performance, and provide recommendations for improvement based on the analysis.

## Dataset

The dataset consists of images collected from a variety of driving scenarios, with annotations for different object classes such as cars, trucks, pedestrians, traffic signs, and more. The annotations are stored in JSON format, which includes bounding box coordinates, object class labels, and other metadata.

### Dataset Structure

```plaintext
bdd_dataset/
├── 100k/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── bdd100k_labels_images_train.json
│   ├── bdd100k_labels_images_val.json 
```
The dataset is divided into training, validation, and testing sets. Each set contains images and corresponding annotations in the form of JSON files.

### Methodology

We use Faster R-CNN, an object detection model that combines a Region Proposal Network (RPN) with Fast R-CNN for end-to-end object detection. The model is trained using the training set, and its performance is evaluated using the validation and test sets.

Please find the detailed model report in https://github.com/davinci2105/BDD_FRCNN_Tailored/blob/main/Development_pipeline.md

### Model Architecture

The Faster R-CNN model is built using the following steps:

- **Feature Extraction:** A convolutional neural network (CNN) extracts feature maps from input images.
- **Region Proposal Network (RPN):** The RPN generates object proposals from the feature maps.
- **RoI Pooling:** Regions of interest (RoIs) are pooled to a fixed size.
- **Classification and Bounding Box Regression:** The RoIs are classified into object categories and their bounding boxes are refined.

### Training and Evaluation

The model is trained for 15 epochs using a learning rate of 0.0001 and a batch size of 16. The performance is evaluated using mean Average Precision (mAP) at different IoU thresholds (0.5, 0.75, 0.95). The model’s accuracy and recall are also evaluated to assess its performance in detecting different object categories.

## Results

### Training Loss and Validation Loss

The model’s performance was evaluated at each epoch. The training loss and validation loss were recorded, and the model was evaluated using mAP at multiple IoU thresholds. The following results were obtained:

![Training and Validation Loss Curve](training_loss_curve.png) To Be uploaded

### Evaluation Metrics

The evaluation metrics computed for different IoU thresholds are as follows:

| **IoU Threshold** | **mAP** | **AR** |
|-------------------|---------|--------|
| **IoU 0.5**       | TBA     | TBA    |
| **IoU 0.75**      | TBA     | TBA    |
| **IoU 0.95**      | TBA     | TBA    |

## Challenges and Limitations

TBA

## Future Work

To improve the model’s performance, we recommend the following:

TBA

## Additional Resources

The following resources are available for further exploration:

- **Analysis Report:** You can access the detailed analysis report of the project at the following link: [Analysis Report on Google Drive](https://drive.google.com/drive/folders/1DX3BQFpL6CaSEwz1uNX6BINg4WKnc7Ql?usp=sharing).
- **Data Analysis Container:** A pre-configured container for data analysis is available for download here: [Data Analysis Container on Google Drive](https://drive.google.com/file/d/1lac_4T1H480A2VuVBtLt2cSXsIotOFHp/view?usp=sharing).
- **Pre-trained Model:** The pre-trained model used in this project is available for download here: [Pre-trained Model on Google Drive](https://drive.google.com/file/d/1NZR54DFGz2_NOhlrtmQqDGzYpYBzVvi1/view?usp=sharing).

## Conclusion

This project uses Faster R-CNN with Unet and low light enhancement layers with hybrid loss for object detection on the BDD100K dataset. 

