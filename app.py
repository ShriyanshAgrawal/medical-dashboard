from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2
import base64
import io
import os
import random
import tensorflow as tf
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Create a directory for storing uploaded images
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Add a ping endpoint to test if server is running
@app.route('/ping', methods=['GET'])
def ping():
    logger.info("Ping endpoint called")
    return jsonify({"status": "ok", "message": "Server is running"}), 200

# Load the brain tumor model if it exists
try:
    logger.info("Loading model...")
    brain_model = tf.keras.models.load_model("model/mri.keras")
    # Class labels (must match model training order exactly)
    brain_classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    model_loaded = True
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model_loaded = False
    brain_classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    print("Warning: Brain tumor model not found. Using placeholder detection.")

# Define tumor type classifications based on medical literature
tumor_classification = {
    'glioma': 'Malignant',      # Gliomas are typically malignant
    'meningioma': 'Benign',     # Meningiomas are typically benign
    'notumor': 'No tumor',      # Not applicable
    'pituitary': 'Benign'       # Pituitary tumors are typically benign
}

# Added medical descriptions for each tumor type
tumor_descriptions = {
    'glioma': 'Gliomas arise from glial cells and tend to be malignant, often requiring aggressive treatment.',
    'meningioma': 'Meningiomas develop from the meninges covering the brain and are typically benign with slow growth.',
    'notumor': 'No abnormal tissue growth detected in the brain scan.',
    'pituitary': 'Pituitary tumors form in the pituitary gland and are usually benign, though they may affect hormone production.'
}

# Treatment recommendations based on tumor type
treatment_recommendations = {
    'glioma': 'May require surgery, radiation therapy, and/or chemotherapy depending on grade and location.',
    'meningioma': 'Often monitored with regular imaging if asymptomatic. Surgery is common for symptomatic cases.',
    'pituitary': 'Treatment depends on hormone function and may include surgery, medication to control hormone production, or radiation.',
    'notumor': 'No tumor-specific treatment needed. Continue with regular check-ups as advised by your healthcare provider.'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/brain')
def brain_tumor():
    return render_template('brain_tumor.html')

@app.route('/ecg')
def ecg():
    # Placeholder for ECG analysis page
    return redirect(url_for('index'))

@app.route('/eeg')
def eeg():
    # Placeholder for EEG analysis page
    return redirect(url_for('index'))

@app.route('/history')
def history():
    # Placeholder for history page
    return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        # Process the file here
        return jsonify({'message': 'File successfully processed'})

@app.route('/analyze/brain', methods=['POST'])
def analyze_brain():
    print("Analyze brain endpoint called")
    print("Request method:", request.method)
    print("Request form data:", request.form)
    print("Request files:", request.files)
    
    try:
        if 'file' not in request.files:
            print("No file part in request")
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        print(f"File received: {file.filename}")
        
        if file.filename == '':
            print("Empty filename")
            return jsonify({'error': 'No selected file'}), 400
        
        # Read the image using PIL
        try:
            img = Image.open(file.stream).convert("RGB")
            print(f"Image opened successfully: {img.size}")
        except Exception as e:
            print(f"Error opening image: {str(e)}")
            return jsonify({'error': f'Error opening image: {str(e)}'}), 400
        
        # If model is loaded, use it for prediction
        if model_loaded:
            try:
                # Resize image for the model
                img_for_model = img.resize((299, 299))
                img_array = np.asarray(img_for_model) / 255.0  # Normalize to [0, 1]
                img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
                
                # Predict
                predictions = brain_model.predict(img_array)[0]
                top_class = int(np.argmax(predictions))
                confidence = float(predictions[top_class])
                prediction_class = brain_classes[top_class]
                print(f"Prediction: {prediction_class}, confidence: {confidence}")
                
                # Calculate benign vs malignant probabilities
                benign_score = 0
                malignant_score = 0
                no_tumor_score = 0
                
                for i, cls in enumerate(brain_classes):
                    if cls == 'notumor':
                        no_tumor_score = float(predictions[i])
                    elif tumor_classification[cls] == 'Benign':
                        benign_score += float(predictions[i])
                    elif tumor_classification[cls] == 'Malignant':
                        malignant_score += float(predictions[i])
                
                # Determine malignancy status
                malignancy_status = tumor_classification[prediction_class]
            except Exception as e:
                print(f"Error during model prediction: {str(e)}")
                return jsonify({'error': f'Error during model prediction: {str(e)}'}), 500
            
            # Convert PIL Image to OpenCV format for visualization
            img_cv = np.array(img)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
            
            # Add prediction text to the image
            result_img = img_cv.copy()
            tumor_detected = prediction_class != 'notumor'
            
            # Add text with prediction
            text = f"Prediction: {prediction_class}"
            confidence_text = f"Confidence: {round(confidence * 100, 2)}%"
            status_text = f"Status: {malignancy_status}"
            
            # Put text on image
            text_color = (0, 0, 255) if tumor_detected and malignancy_status == 'Malignant' else (0, 128, 255) if tumor_detected else (0, 255, 0)
            cv2.putText(result_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            cv2.putText(result_img, confidence_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            if tumor_detected:
                cv2.putText(result_img, status_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            
            # If tumor detected, add highlight
            if tumor_detected:
                # Add a border to indicate tumor detection
                border_color = (0, 0, 255) if malignancy_status == 'Malignant' else (0, 128, 255)
                result_img = cv2.copyMakeBorder(result_img, 10, 10, 10, 10, 
                                             cv2.BORDER_CONSTANT, value=border_color)
            
            # Convert processed image back to base64 for sending to client
            try:
                _, buffer = cv2.imencode('.jpg', result_img)
                img_str = base64.b64encode(buffer).decode('utf-8')
            except Exception as e:
                print(f"Error encoding result image: {str(e)}")
                return jsonify({'error': f'Error encoding result image: {str(e)}'}), 500
            
            # Return detailed analysis result
            return jsonify({
                'image': img_str,
                'tumor_detected': tumor_detected,
                'prediction': prediction_class,
                'confidence': round(confidence * 100, 2),
                'probabilities': {
                    cls: round(float(prob) * 100, 2)
                    for cls, prob in zip(brain_classes, predictions)
                },
                # Add classification information
                'classification': {
                    'status': malignancy_status,
                    'description': tumor_descriptions[prediction_class],
                    'benign_probability': round(benign_score * 100, 2),
                    'malignant_probability': round(malignant_score * 100, 2),
                    'no_tumor_probability': round(no_tumor_score * 100, 2),
                    'treatment_approach': treatment_recommendations[prediction_class]
                }
            })
        
        else:
            # If model not loaded, use placeholder function
            img_cv = np.array(img)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
            processed_img, tumor_detected = process_brain_mri(img_cv)
            
            # Randomly choose a prediction for the placeholder
            prediction_class = random.choice(['glioma', 'meningioma', 'notumor', 'pituitary'])
            if not tumor_detected:
                prediction_class = 'notumor'
            
            confidence = random.randint(70, 95)
            
            # Add classification data
            malignancy_status = tumor_classification[prediction_class]
            
            # Convert processed image back to base64 for sending to client
            try:
                _, buffer = cv2.imencode('.jpg', processed_img)
                img_str = base64.b64encode(buffer).decode('utf-8')
            except Exception as e:
                print(f"Error encoding result image: {str(e)}")
                return jsonify({'error': f'Error encoding result image: {str(e)}'}), 500
            
            # Generate some random probability values
            probabilities = {}
            total = 0
            for cls in brain_classes:
                if cls == prediction_class:
                    prob = confidence
                else:
                    prob = random.randint(1, int((100 - confidence) / (len(brain_classes) - 1)))
                probabilities[cls] = prob
                total += prob
            
            # Normalize to ensure they sum to 100
            for cls in probabilities:
                probabilities[cls] = round(probabilities[cls] * 100 / total, 2)
            
            # Calculate benign vs malignant for placeholder
            benign_score = sum(probabilities[cls] for cls in brain_classes if tumor_classification[cls] == 'Benign') / 100
            malignant_score = sum(probabilities[cls] for cls in brain_classes if tumor_classification[cls] == 'Malignant') / 100
            no_tumor_score = probabilities['notumor'] / 100
            
            # Return the processed image and detection result
            return jsonify({
                'image': img_str,
                'tumor_detected': tumor_detected,
                'prediction': prediction_class,
                'confidence': confidence,
                'probabilities': probabilities,
                # Add classification information
                'classification': {
                    'status': malignancy_status,
                    'description': tumor_descriptions[prediction_class],
                    'benign_probability': round(benign_score * 100, 2),
                    'malignant_probability': round(malignant_score * 100, 2),
                    'no_tumor_probability': round(no_tumor_score * 100, 2),
                    'treatment_approach': treatment_recommendations[prediction_class]
                }
            })
    except Exception as e:
        print(f"Unexpected error in analyze_brain: {str(e)}")
        return jsonify({'error': str(e)}), 500

def process_brain_mri(image):
    """
    Process MRI scan to detect brain tumors.
    This is a placeholder function - replace with your actual model.
    """
    # Create a copy of the image to avoid modifying the original
    result_img = image.copy()
    
    # Simulate tumor detection - in a real application, this would use your ML model
    # For demo purposes, we'll just detect a random region
    
    # Determine if we "detect" a tumor (random for this demo)
    tumor_detected = random.choice([True, False])
    
    if tumor_detected:
        # Simulate tumor detection by highlighting a random area
        height, width = image.shape[:2]
        center_x = random.randint(width // 3, width * 2 // 3)
        center_y = random.randint(height // 3, height * 2 // 3)
        radius = random.randint(20, 50)
        
        # Draw a red outline around the "detected" tumor
        cv2.circle(result_img, (center_x, center_y), radius, (0, 0, 255), 2)
        
        # Add a label
        cv2.putText(result_img, "Tumor Detected", (center_x - 70, center_y - radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return result_img, tumor_detected

if __name__ == '__main__':
    app.run(debug=True) 