from flask import Flask, request, jsonify
import mediapipe as mp
import numpy as np
import base64
import io
from PIL import Image
import requests

app = Flask(__name__)

# Initialize MediaPipe face detection and face mesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# Function to generate a fingerprint (face embedding) using MediaPipe
def generate_fingerprint(image_data):
    # Load the image
    image = Image.open(io.BytesIO(image_data))
    image = np.array(image)

    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        # Convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        # If a face is detected, generate an embedding based on facial landmarks
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            face_embedding = [landmark.x for landmark in landmarks] + [landmark.y for landmark in landmarks] + [landmark.z for landmark in landmarks]
            return np.array(face_embedding).tolist()
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
        # Generate the fingerprint (embedding)
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
    app.run(debug=True)
