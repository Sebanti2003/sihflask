# import os
# from flask import Flask, jsonify, request
# import torch
# import cv2
# from ultralytics import YOLO

# app = Flask(__name__)

# # Load the YOLO model
# model = YOLO('sih.pt')

# @app.route("/predict", methods=["GET"])
# def predict():
#     # Get the image path from the query parameters
#     image_path = request.args.get('image_path')
    
#     if not image_path or not os.path.exists(image_path):
#         return jsonify({"error": "Image path not provided or invalid"}), 400

#     # Load the image
#     img = cv2.imread(image_path)
    
#     # Perform detection
#     detections = model(img)

#     # Extract detailed information from the detections
#     result = detections[0]  # Get the first result (if there are multiple images)
#     predictions = []
    
#     for detection in result.boxes:
#         box = detection.xyxy[0]  # Get bounding box [x1, y1, x2, y2]
#         class_id = int(detection.cls[0])  # Get the class ID
#         confidence = float(detection.conf[0])  # Get the confidence score
        
#         predictions.append({
#             'class_id': class_id,
#             'class_name': result.names[class_id],  # Class name (e.g., "person")
#             'confidence': confidence,
#             'box': {
#                 'x1': float(box[0]),
#                 'y1': float(box[1]),
#                 'x2': float(box[2]),
#                 'y2': float(box[3])
#             }
#         })

#     # Return the extracted information as JSON
#     return jsonify(predictions)

# # Main function to run the Flask app
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)
# # import os
# from flask import Flask, jsonify, request
# import torch
# import cv2
# from ultralytics import YOLO

# app = Flask(__name__)

# # Load the YOLO model
# model = YOLO('sih.pt')

# @app.route("/predict", methods=["GET"])
# def predict():
#     # Get the image path from the query parameters
#     image_path = request.args.get('image_path')
    
#     if not image_path or not os.path.exists(image_path):
#         return jsonify({"error": "Image path not provided or invalid"}), 400

#     # Load the image
#     img = cv2.imread(image_path)
    
#     # Perform detection
#     detections = model(img)

#     # Extract detailed information from the detections
#     result = detections[0]  # Get the first result (if there are multiple images)
#     predictions = []
    
#     for detection in result.boxes:
#         box = detection.xyxy[0]  # Get bounding box [x1, y1, x2, y2]
#         class_id = int(detection.cls[0])  # Get the class ID
#         confidence = float(detection.conf[0])  # Get the confidence score
        
#         predictions.append({
#             'class_id': class_id,
#             'class_name': result.names[class_id],  # Class name (e.g., "person")
#             'confidence': confidence,
#             'box': {
#                 'x1': float(box[0]),
#                 'y1': float(box[1]),
#                 'x2': float(box[2]),
#                 'y2': float(box[3])
#             }
#         })

#     # Return the extracted information as JSON
#     return jsonify(predictions)

# # Main function to run the Flask app
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)
import os
from flask import Flask, jsonify, request
import torch
import cv2
import requests
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Load the YOLO model
model = YOLO('sih.pt')

@app.route("/predict", methods=["POST"])
def predict():
    # Get the image URL from the POST request body
    data = request.get_json()
    image_url = data.get('image_url')
    
    if not image_url:
        return jsonify({"error": "Image URL not provided"}), 400

    # Download the image from the URL
    try:
        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify({"error": "Failed to download image from the provided URL"}), 400

        # Convert the image content to a numpy array and decode it
        image_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Perform detection
    detections = model(img)

    # Extract detailed information from the detections
    result = detections[0]  # Get the first result (if there are multiple images)
    predictions = []
    
    for detection in result.boxes:
        box = detection.xyxy[0]  # Get bounding box [x1, y1, x2, y2]
        class_id = int(detection.cls[0])  # Get the class ID
        confidence = float(detection.conf[0])  # Get the confidence score
        
        predictions.append({
            'class_id': class_id,
            'class_name': result.names[class_id],  # Class name (e.g., "person")
            'confidence': confidence,
            'box': {
                'x1': float(box[0]),
                'y1': float(box[1]),
                'x2': float(box[2]),
                'y2': float(box[3])
            }
        })

    # Return the extracted information as JSON
    return jsonify(predictions)

# Main function to run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
