from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import base64
import io
from PIL import Image
import requests

app = Flask(__name__)

# Function to generate a fingerprint (face embedding) from the image data
def generate_fingerprint(image_data):
    # Load the image
    image = face_recognition.load_image_file(io.BytesIO(image_data))
    # Generate face encodings
    face_encodings = face_recognition.face_encodings(image)
    # Return the first face encoding if a face is detected
    if len(face_encodings) > 0:
        return np.array(face_encodings[0]).tolist()
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
    app.run(debug=True)
