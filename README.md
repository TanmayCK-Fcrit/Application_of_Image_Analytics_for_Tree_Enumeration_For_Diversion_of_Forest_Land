# 🌳 Application of Image Analytics for Tree Enumeration and Species Classification (Django Web App)

[![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-3.2%2B-092E20?style=for-the-badge&logo=django)](https://www.djangoproject.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow%20/Keras-2.x-orange?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-5CB85C?style=for-the-badge&logo=ultralytics)](https://ultralytics.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

A full-stack, multi-modal Deep Learning application built with **Django** for automatically assessing forest land. The system takes drone or satellite imagery and performs **Tree Enumeration (Counting)** and **Species Classification** to generate an official PDF report for land diversion analysis.

---

## ✨ Key Features

* **Multi-Modal Deep Learning Pipeline:** Integrates a two-stage deep learning workflow:
    1. **Object Detection (YOLOv8):** Accurately detects and localizes all individual trees in the input image (drone/satellite photo).
    2. **Image Classification (CNN):** Crops the detected trees using the YOLO coordinates and classifies each tree's species.
* **Species Classification Models:** Utilizes various CNN architectures (VGG, ResNet, EfficientNet, CNN) for robust species prediction, supporting four specific classes:
    * `Common_coconut`
    * `Common_mango`
    * `Common_neem`
    * `Economical_sandalwood`
* **Automated Reporting:** Generates a comprehensive, printable **PDF Report** containing the total tree count, individual cropped tree images, classification labels, and precise bounding box coordinates.
* **Web Application Interface:** Built on **Django** (Python) with **HTML/CSS** for easy image upload and report generation via a simple web browser interface.

---

## 🛠️ Tech Stack & Architecture

### Backend (Deep Learning & Framework)
| Component | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Web Framework** | **Django** | Handles requests, file uploads, and template rendering. |
| **Object Detection** | **YOLOv8** (via Ultralytics) | Detects trees and provides bounding box coordinates. |
| **Image Classification** | **Keras / TensorFlow** | Classifies cropped tree images into species. |
| **Image Processing** | **OpenCV (`cv2`)** | Reading images, cropping, and resizing. |
| **PDF Generation** | **ReportLab** | Dynamically creates the final classification report PDF. |
| **Deep Learning Models** | **VGG, ResNet, EfficientNet, CNN** | The backbone architectures used for the species classification model. |

### Frontend
* **HTML, CSS:** Basic web interface for image upload and displaying results.

---

## 📂 Repository Structure

The project follows a standard Django application structure, with critical deep learning components stored internally:
