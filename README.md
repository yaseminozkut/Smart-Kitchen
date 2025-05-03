# 🧠 Smart Kitchen Assistant - Object Detection Model

Live multi-class YOLOv8 object detection model implementation to detect grocery items using the [Freiburg Groceries Dataset](http://aisdatasets.informatik.uni-freiburg.de/freiburg_groceries_dataset/). The goal of this project is to help users identify market items visually through a smart interface.

---

## 📁 Dataset

This project uses the **Freiburg Groceries Dataset**, which contains over 5,000 images across 25 grocery categories (e.g., coffee, spices, juice, etc.).

- Dataset GitHub: [PhilJd/freiburg_groceries_dataset](https://github.com/PhilJd/freiburg_groceries_dataset)
- Download link: [Freiburg Dataset](http://aisdatasets.informatik.uni-freiburg.de/freiburg_groceries_dataset/)

However, since the dataset was large to manually annotate by myself in a short time, I have annotated 1235 images amoung 5 categories (flour, coffee, tea, water, and milk) using Roboflow. Link to the annotated dataset: [Freiburg Dataset Annotated Subset](https://universe.roboflow.com/yasemin-ozkut-1/freiburg-groceries-dataset-1)

---

## 🎯 Objective

To build an image classification model that can:

- Accurately detect grocery items from images taken in real-world kitchen environments.
- Serve as the backend for future smart kitchen tools like automated grocery tracking or dietary analytics.

---

## 🛠️ Implementation Overview

### 🔹 Preprocessing and Data Augmentation

- Images resized and auto-oriented.
- Augmentations: horizontal and vertical flips, rotations, shear, cutouts, bounding box transformations (flip, rotate, shear).
- Total number of images after augmentation: 3503.
- Dataset split into train/validation/test (%92/%4/%4).

### 🔹 Model Training

- Architecture: **YOLOv8** (YOLOv8m).
- Optimizer: SGD
- Epochs: 120
- Batch size: 16
- Patience: 50

Training command:
```bash
!yolo task=detect mode=train model=yolov8m.pt data={dataset.location}/data.yaml epochs=120 imgsz=244 plots=True
```

### 🔹 Evaluation Metrics

- Accuracy
- Confusion Matrix
- Precision & Recall % F1

---

## 📊 Results

- Final model accuracy: **~87%** on the test set.
- Best performance on categories coffee, tea, and water.
- Worst performance on category milk.

Validating runs/detect/train/weights/best.pt...
Ultralytics YOLOv8.0.20 🚀 Python-3.10.12 torch-2.0.1+cu118 CUDA:0 (Tesla T4, 15102MiB)
Model summary (fused): 218 layers, 25843813 parameters, 0 gradients, 78.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 4/4 [00:04<00:00,  1.22s/it]
                   all        126        326      0.816      0.823      0.872      0.671
         Kaffee-Coffee        126         79      0.947      0.909       0.94      0.721
            Mehl-flour        126         18       0.74      0.791      0.834      0.683
            Milch-Milk        126         36       0.69      0.694      0.765      0.545
                Person        126         39      0.793      0.787       0.83      0.625
               Tee-Tea        126         63      0.847      0.889      0.945      0.762
          Wasser-Water        126         91      0.877      0.868       0.92       0.69

---

## 🧪 Demo - Live Detection

> A prototype script is provided to test the model on custom input images. Please replace the model path in the script with the path to your best.pt. Try placing a product in front of a camera and running:
```bash
python main.py
```

---

## 🚀 Future Work

- Integrate object detection to recognize multiple items in one frame.
- Add a voice assistant for hands-free interaction.
- Extend model with nutrition facts database for calorie tracking.


## 🤝 Acknowledgements

- [Freiburg Groceries Dataset](http://aisdatasets.informatik.uni-freiburg.de/freiburg_groceries_dataset/)
- Internship Project – Summer 2023
- Special thanks to mentors and peers who supported this work.
