# Dockerizing Data Analysis, Training, and Evaluation

This document provides a detailed guide on containerizing data analysis, training, and evaluation workflows using Docker. We will start by containerizing a data analysis script (`analysis.py`) and then discuss full model training and evaluation in a Dockerized environment.

## 📦 Dockerizing Data Analysis

### **1. Preparing the Dockerfile for Data Analysis**
To containerize `analysis.py`, create a `Dockerfile` with the following content:

```dockerfile
# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /workspace

# Install system dependencies for OpenCV
RUN apt update && apt install -y \
    python3-pip \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy analysis script into container
COPY analysis.py .

# Set default command to run analysis.py
CMD ["python", "analysis.py"]
```

### **2. Creating `requirements.txt`**
Ensure `requirements.txt` includes all necessary dependencies:

```txt
numpy
pandas
matplotlib
seaborn
pillow
opencv-python
scipy
```

### **3. Building and Running the Docker Container**
#### **Building the Image**
```bash
docker build -t analysis-container .
```

#### **Running the Container**
```bash
docker run --rm -v $(pwd):/workspace analysis-container
```

Explanation:
- `--rm`: Removes the container after execution.
- `-v $(pwd):/workspace`: Mounts your current directory inside the container.

---

## 🔥 Dockerizing Model Training and Evaluation

Once data analysis is complete, we can move to **training and evaluating** a model using Docker.

### **1. Dockerfile for Training and Evaluation**
Create a new `Dockerfile` optimized for machine learning tasks:

```dockerfile
# Use PyTorch base image with CUDA for GPU support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set the working directory
WORKDIR /workspace

# Install system dependencies
RUN apt update && apt install -y \
    python3-pip \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training and evaluation scripts
COPY train_combined.py .
COPY evaluate.py .

# Set default command (bash shell to allow manual execution of scripts)
CMD ["/bin/bash"]
```

### **2. Updating `requirements.txt` for Training and Evaluation**
Ensure `requirements.txt` contains all required dependencies:

```txt
torch
torchvision
torchmetrics
detectron2
numpy
pandas
matplotlib
opencv-python
scipy
```

### **3. Building and Running the Container**
#### **Building the Image**
```bash
docker build -t training-container .
```

#### **Running the Container with GPU Access**
```bash
docker run -it --rm --gpus all -v $(pwd):/workspace training-container
```

### **4. Training the Model Inside the Container**
Once inside the container, start training:
```bash
python train_combined.py
```

### **5. Evaluating the Model**
After training is complete, evaluate the model:
```bash
python evaluate.py
```

---

## 🏆 Full Dockerized Workflow
For a fully containerized workflow covering **data analysis, training, and evaluation**, create **separate Dockerfiles** for each stage, or use a **multi-stage build**.

### **Multi-Stage Dockerfile (Complete Workflow)**
```dockerfile
# Base image for analysis
FROM python:3.9-slim AS analysis
WORKDIR /workspace
RUN apt update && apt install -y python3-pip libglib2.0-0 libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY analysis.py .
CMD ["python", "analysis.py"]

# Base image for training and evaluation
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel AS training
WORKDIR /workspace
RUN apt update && apt install -y python3-pip libglib2.0-0 libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY train_combined.py evaluate.py .
CMD ["/bin/bash"]
```

### **Building and Running Multi-Stage Containers**
#### **Build Images**
```bash
docker build --target analysis -t analysis-container .
docker build --target training -t training-container .
```

#### **Run Containers**
```bash
docker run --rm -v $(pwd):/workspace analysis-container
```
```bash
docker run -it --rm --gpus all -v $(pwd):/workspace training-container
```

---

## ✅ **Summary**
### **1. Data Analysis**
- **Dockerfile**: Lightweight Python image (`python:3.9-slim`)
- **Run**: `docker run --rm -v $(pwd):/workspace analysis-container`

### **2. Model Training & Evaluation**
- **Dockerfile**: PyTorch image with CUDA 12.1
- **Run Training**: `python train_combined.py`
- **Run Evaluation**: `python evaluate.py`

### **3. Full Containerized Workflow**
- **Multi-stage Docker build**
- **Separate containers for analysis and training**

🚀 **Now you have a fully containerized workflow for data analysis, model training, and evaluation!** 🚀

