import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model 
from django.conf import settings
import os
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import letter

# Load YOLOv8 model
yolo_model = YOLO(os.path.join(settings.BASE_DIR, "yolov8/best.pt"))

# Load CNN model
cnn_model = load_model(os.path.join(settings.BASE_DIR, "cnn_model/tree_species_cnn.h5"))

# Tree class labels
CLASSES = ["Common_coconut", "Common_mango", "Common_neem", "Economical_sandalwood"]

def process_image(image_path):
    """Detect trees, crop, classify, and return results."""
    img = cv2.imread(image_path)
    results = yolo_model(img)[0]  # Perform object detection

    processed_results = []
    crop_dir = os.path.join(settings.MEDIA_ROOT, "cropped_trees")
    os.makedirs(crop_dir, exist_ok=True)  # Create cropped images folder

    for i, box in enumerate(results.boxes.xyxy):
        x_min, y_min, x_max, y_max = map(int, box)

        # Crop image
        cropped_tree = img[y_min:y_max, x_min:x_max]
        cropped_tree = cv2.resize(cropped_tree, (224, 224))

        # Classify using CNN
        img_array = np.expand_dims(cropped_tree, axis=0) / 255.0
        prediction = cnn_model.predict(img_array)
        label = CLASSES[np.argmax(prediction)]

        # Save cropped image
        crop_path = os.path.join(crop_dir, f"tree_{i}.jpg")
        cv2.imwrite(crop_path, cropped_tree)

        processed_results.append({
            "image": crop_path,
            "coordinates": f"({x_min}, {y_min}), ({x_max}, {y_max})",
            "label": label
        })

    return processed_results

def generate_pdf(results, output_path):
    """Generate a PDF report with tree images and classifications."""
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(200, height - 50, "Tree Classification Report")

    y_position = height - 100
    c.setFont("Helvetica", 12)

    if not results:
        c.drawString(50, height - 100, "No trees detected.")
    else:
        for i, result in enumerate(results):
            if y_position < 150:
                c.showPage()
                c.setFont("Helvetica", 12)
                y_position = height - 50

            text = f"Tree {i+1}: {result['label']} | Coordinates: {result['coordinates']}"
            c.drawString(50, y_position, text)

            # Ensure the cropped image exists before adding it
            if os.path.exists(result["image"]):
                tree_image = ImageReader(result["image"])
                c.drawImage(tree_image, 50, y_position - 100, width=100, height=100)

            y_position -= 150

    c.save()
