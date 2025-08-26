# Crop and Weed Detection using YOLOv8

This project presents a **deep learning-based solution** for real-time **crop and weed detection** in agricultural fields.  
By leveraging the **YOLOv8 object detection model**, the system can accurately identify and differentiate between sesame crops and invasive weeds from image data.

The primary goal is to enable **precision agriculture technologies**, such as targeted pesticide spraying systems. By only targeting weeds, we can significantly reduce herbicide usage, lower operational costs for farmers, and minimize environmental impact — leading to healthier crops and higher yields.

---

## Key Features
- **High Accuracy**: Achieves a *Mean Average Precision (mAP50)* of **90.1%** on the validation dataset.  
- **Real-Time Performance**: Built on the lightweight and fast **YOLOv8n** architecture, suitable for edge devices (drones, robotic sprayers).  
- **Specific Detection**: Trained to distinguish between **two classes** → *crop* and *weed*.  
- **Scalable**: Training pipeline easily adaptable for more crops/weeds with additional data.  

---

## Technology Stack
- **Framework**: PyTorch  
- **Model**: YOLOv8 (Ultralytics)  
- **Language**: Python 3.10+  
- **Libraries**: OpenCV, NumPy, Matplotlib  

---

## Project Structure
```
crop-weed-detector/
│
├── datasets/
│   └── crop-weed-data/         
│       ├── images/
│       │   ├── train/          
│       │   └── val/            
│       ├── labels/
│       │   ├── train/          
│       │   └── val/            
│       └── data.yaml           
│
├── runs/
│   └── detect/
│       └── yolov8n_crop_weed_model/
│           └── weights/
│               └── best.pt     
│
├── explore_data.py             # Visualize data and labels
├── train.py                    # Train YOLOv8 model
├── predict.py                  # Run inference on new images
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```

---

## Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/crop-weed-detector.git
cd crop-weed-detector
```

### 2. Create & Activate a Virtual Environment
```bash
# Create venv
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> ⚠️ For GPU training/inference, install a CUDA-enabled version of PyTorch.

---

## Usage

### 1. Explore the Dataset
Visualize images and labels:
```bash
python explore_data.py
```

### 2. Train the Model
```bash
python train.py
```
- The best weights will be saved to:
  ```
  runs/detect/yolov8n_crop_weed_model/weights/best.pt
  ```

### 3. Run Predictions
Update the `IMAGE_PATH` inside **predict.py**, then run:
```bash
python predict.py
```

A window will display the image with **detected crops & weeds**.

---

## Model Performance
Trained for **50 epochs** on the custom dataset:

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| all   | 260    | 369       | 0.846     | 0.860  | 0.901 | 0.616    |
| crop  | 131    | 208       | 0.797     | 0.856  | 0.900 | 0.644    |
| weed  | 129    | 161       | 0.895     | 0.863  | 0.901 | 0.587    |

Indicates **high reliability** for field use.

---

## Future Improvements
- **Optimization**: Convert to **ONNX / TensorRT** for faster inference on edge devices.  
- **Dataset Expansion**: Add more images (lighting conditions, growth stages, different farms).  
- **Web App**: Build a **Flask/Streamlit UI** for uploading and detecting crops/weeds.  
- **Live Video**: Extend prediction to **real-time video stream**.  

---

## Demo

**A screenshot of predict.py script in action** ⬇️

![Demo Screenshot](assets/demo_prediction.png)

---
