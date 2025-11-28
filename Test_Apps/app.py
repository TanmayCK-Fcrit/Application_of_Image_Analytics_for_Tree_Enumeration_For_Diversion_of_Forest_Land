from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors

# Load Models
yolo_model_path = "best.pt"  # Path to YOLOv8 model for tree detection
classifier_model_path = "tree_species_cnn.h5"  # CNN classification model
yolo_model = YOLO(yolo_model_path)
classifier = load_model(classifier_model_path)

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
DETECTED_FOLDER = 'static/detected/'
CROPPED_FOLDER = 'static/cropped/'
REPORT_PDF = 'static/classification_report.pdf'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTED_FOLDER, exist_ok=True)
os.makedirs(CROPPED_FOLDER, exist_ok=True)

def detect_trees(image_path):
    """Detect trees in an image using YOLOv8 and save detection results."""
    img = cv2.imread(image_path)
    results = yolo_model(image_path)
    
    detected_trees = []
    for i, box in enumerate(results[0].boxes.xyxy):  
        x1, y1, x2, y2 = map(int, box.tolist())  
        crop = img[y1:y2, x1:x2]  
        crop_path = os.path.join(CROPPED_FOLDER, f'tree_{i}.jpg')
        cv2.imwrite(crop_path, crop)
        detected_trees.append((crop_path, (x1, y1, x2, y2)))

        # Draw bounding boxes on the image
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    detected_path = os.path.join(DETECTED_FOLDER, "detected.jpg")
    cv2.imwrite(detected_path, img)
    
    return detected_trees, detected_path, len(detected_trees)  

def classify_images():
    """Classify cropped tree images using CNN model."""
    species = ['Common_coconut', 'Common_mango', 'Common_neem', 'Economical_sandalwood']
    results = {}

    for file in os.listdir(CROPPED_FOLDER):
        img_path = os.path.join(CROPPED_FOLDER, file)
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = classifier.predict(img_array)
        predicted_class = species[np.argmax(prediction)]

        results[file] = {"species": predicted_class, "coordinates": (0, 0, 224, 224)}

    return results

def generate_pdf(results, tree_count):
    """Generate a classification report as a PDF."""
    doc = SimpleDocTemplate(REPORT_PDF, pagesize=letter)
    table_data = [["Tree Number", "Pixel Coordinates", "Classified Species"]]
    
    for i, (image_name, data) in enumerate(results.items(), start=1):
        table_data.append([i, str(data["coordinates"]), data["species"]])
    
    table_data.append(["Total Trees", "", tree_count])  

    table = Table(table_data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
    ]))

    doc.build([table])

@app.route('/', methods=['GET', 'POST'])
def index():
    """Upload image and process it for tree detection."""
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part!", 400  # Return error if no file is provided

        file = request.files['file']
        if file.filename == '':
            return "No selected file!", 400  # Return error if filename is empty

        if file:
            filename = secure_filename(file.filename)
            img_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(img_path)

            # Detect trees using YOLOv8
            cropped_images, detected_path, tree_count = detect_trees(img_path)
            return redirect(url_for('detection', tree_count=tree_count))
    
    return render_template('index.html')

@app.route('/detection', methods=['GET', 'POST'])
def detection():
    """Display detected trees."""
    detected_image = os.path.join(DETECTED_FOLDER, "detected.jpg")
    cropped_images = os.listdir(CROPPED_FOLDER)
    tree_count = request.args.get('tree_count', 0, type=int)
    
    return render_template('detection.html', detected_image=detected_image, cropped_images=cropped_images, tree_count=tree_count)

@app.route('/report', methods=['GET'])
def report():
    """Generate classification report."""
    results = classify_images()
    tree_count = len(results)
    generate_pdf(results, tree_count)
    
    return render_template('report.html', results=results, tree_count=tree_count)

@app.route('/download_pdf', methods=['GET'])
def download_pdf():
    """Download the generated PDF report."""
    return send_file(REPORT_PDF, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
