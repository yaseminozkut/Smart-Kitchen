# 🧠 Smart Kitchen Assistant - Object Detection Model

Live multi-class YOLOv8 object detection model implementation to detect grocery items using the [Freiburg Groceries Dataset](http://aisdatasets.informatik.uni-freiburg.de/freiburg_groceries_dataset/). The goal of this project is to help users identify market items visually through a smart interface.

![YOLO Detection](https://drive.google.com/file/d/195JFyTQ19XukYvHLrFkgPz5q2LmWvyFZ/view?usp=sharing)
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

## 🧪 Demo - Live Detection
Demo Video can be found [here](https://drive.google.com/file/d/18QQ4yGtweis7_0Gm4agsRYPkUsXxOvCI/view?usp=sharing)
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
