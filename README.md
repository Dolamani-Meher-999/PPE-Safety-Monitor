# 🦺 PPE Safety Monitor

An AI-based PPE safety monitoring system for detecting motorcycle riders,
helmets, and helmet violations using YOLO11 and Streamlit.

## 🚀 Features

- Helmet detection
- No-helmet detection
- Motorcycle rider detection
- Image-based inference
- Real-time webcam monitoring
- Safety violation alerts
- Violation screenshot logging
- Streamlit web interface
- YOLO11 object detection
- Model evaluation using mAP, precision and recall

## 🧠 Classes

The model detects three classes:

| Class ID | Class |
|----------|-------|
| 0 | BikeWithRider |
| 1 | NoHelmet |
| 2 | Helmet |

## 📊 Model Performance

Test dataset results:

- mAP@50: 87.0%
- mAP@50-95: 60.2%
- Precision: 84.1%
- Recall: 81.3%

## 🛠️ Technologies

- Python
- YOLO11
- Ultralytics
- OpenCV
- Streamlit
- PyTorch
- NumPy
- Pandas
- Pillow

## 📁 Project Structure

```text
PPE-Safety/
│
├── app.py
├── train.py
├── predict.py
├── webcam.py
├── helmet_alert.py
├── evaluate.py
├── prepare_dataset.py
├── verify_yolo_dataset.py
├── check_dataset.py
├── requirements.txt
├── .gitignore
└── README.md