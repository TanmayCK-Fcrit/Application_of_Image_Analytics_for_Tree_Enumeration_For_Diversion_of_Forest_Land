import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os

def detect_bounding_boxes(image):
    # Convert to HSV and create mask for pure green bounding boxes
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([60, 255, 60])  # Pure green in HSV
    upper_green = np.array([60, 255, 120])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Find contours of bounding boxes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10:  # Avoiding small false detections
            bounding_boxes.append((x, y, w, h))
    return bounding_boxes

def crop_and_resize(image, bounding_boxes):
    cropped_images = []
    for (x, y, w, h) in bounding_boxes:
        cropped = image[y:y+h, x:x+w]
        resized = cv2.resize(cropped, (224, 224))
        cropped_images.append(resized)
    return cropped_images

def classify_images(model, cropped_images, class_labels):
    predictions = []
    for img in cropped_images:
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        class_index = np.argmax(pred)
        predictions.append(class_labels[class_index])
    return predictions

def main():
    st.title("Tree Species Classification")
    st.write("Upload an image with green bounding boxes, and the system will classify the detected trees.")
    
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Load CNN model
        model_path = "tree_species_cnn.h5"  # Ensure this file is in the project directory
        model = load_model(model_path)
        class_labels = ["Common_coconut", "Common_mango", "Common_neem", "Economical_sandalwood"]
        
        # Detect bounding boxes and crop images
        bounding_boxes = detect_bounding_boxes(image)
        if not bounding_boxes:
            st.write("No valid bounding boxes detected.")
            return
        
        cropped_images = crop_and_resize(image, bounding_boxes)
        
        # Classify each cropped image
        predictions = classify_images(model, cropped_images, class_labels)
        
        # Display results
        for i, (crop, label) in enumerate(zip(cropped_images, predictions)):
            st.image(crop, caption=f"Tree {i+1}: {label}", use_column_width=False, width=150)
            st.write(f"Tree {i+1} classified as: {label}")

if __name__ == "__main__":
    main()