from flask import Flask, request, jsonify
import mediapipe as mp
import cv2
import numpy as np
import base64
import io
from PIL import Image
import requests

app = Flask(__name__)

# Set the maximum content length (e.g., 10 MB)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB

# Initialize MediaPipe Face Detection model
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)
mp_drawing = mp.solutions.drawing_utils

# Function to generate a fingerprint (face embedding) from the image data
def generate_fingerprint(image_data):
    # Convert byte data to an image
    image = Image.open(io.BytesIO(image_data))
    # Convert the image to RGB
    image = np.array(image.convert('RGB'))
    
    # Convert image to RGB and process it with MediaPipe
    results = face_detection.process(image)
    
    if results.detections:
        # Extract face landmarks
        for detection in results.detections:
            # Get bounding box information
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            face_image = image[y:y+h, x:x+w]  # Crop the face area

            # Encode the cropped face image into a fingerprint (encoding)
            fingerprint = np.array(face_image).flatten().tolist()  # Simple flattening for example
            
            return fingerprint
    else:
        return None

# Route to process the image
@app.route('/process-image', methods=['POST'])
def process_image():
    try:
        # Parse the JSON data from the request
        data = request.json
        # Decode the base64 image data
        image_data = base64.b64decode(data['image'].split(',')[1])
        # Generate the fingerprint
        fingerprint = generate_fingerprint(image_data)

        # If a face is detected, send data to the Node.js backend
        if fingerprint:
            payload = {
                "real_name": data['real_name'],
                "unique_code": data['unique_code'],
                "face_embedding": fingerprint,
            }
            # Send the data to the Node.js backend
            node_response = requests.post(
                'https://faceecho-back.onrender.com/register',
                json=payload
            )
            # Return the response from the Node.js backend
            return jsonify(node_response.json()), node_response.status_code
        else:
            return jsonify({'error': 'No face detected'}), 400
    except Exception as e:
        # Handle unexpected errors
        return jsonify({'error': 'An error occurred', 'details': str(e)}), 500

# Run the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
