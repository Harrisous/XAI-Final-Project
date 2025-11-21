import json
import base64
import random
import time
from flask import Flask, request, jsonify
from flask_cors import CORS # Cross-Origin Resource Sharing module, allowing frontend access

# ----------------------------------------------------------------------
# 1. Configuration and Mock Data
# ----------------------------------------------------------------------

# Assume your ResNet50 model is loaded here
# import tensorflow as tf
# model = tf.keras.models.load_model('your_resnet50_model.h5')

app = Flask(__name__)
# Enable CORS, allowing all origins to access localhost:5000, so the frontend HTML file can send requests
CORS(app) 

# Hardcoded GradCAM SVG Base64 URI.
# In a real application, this would be the Base64 encoding of the model-generated heatmap image, matched to the original size
MOCK_GRADCAM_URI = 'data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxMDAgMTAwIj48ZGVmcz48bGluZWFyR3JhZGllbnQgaWQ9ImdyYWQiIHgxPSIwJSIgeTE9IjAlIiB4Mj0iMTAwJSIgeTI9IjEwMCUiPjxzdG9wIG9mZnNldD0iMCUiIHN0eWxlPSJzdG9wLWNvbG9yOnJnYigyNTUsMCwwKTtzdG9wLW9wYWNpdHk6MC44IiAvPjxzdG9wIG9mZnNldD0iMTAwJSIgc3R5bGU9InN0b3AtY29sb3I6cmdiKDI1NSwyNTUsMCk7c3RvcC1vcGFjaXR5OjAuNiIgLz48L2xpbmVhckdyYWRpZW50PjwvZGVmcz48cmVjdCB4PSIwIiIgeT0iMCIgd2lkdGg9IjEwMCIgaGVpZ2h0PSIxMDAiIGZpbGw9InVybCgjZ3JhZCkiLz48L3N2Zz4='

# ----------------------------------------------------------------------
# 2. Mock Detection Logic (Simulating Core Functionality)
# ----------------------------------------------------------------------

def run_model_prediction_and_gradcam(image_data_b64):
    """
    Simulates ResNet50 Deepfake detection and GradCAM generation.
    """
    
    # Simulate prediction time (0.1 to 0.5 seconds)
    time.sleep(random.uniform(0.1, 0.5))
    
    # Randomly generate a Deepfake confidence (0.0 to 1.0)
    confidence = random.uniform(0.05, 0.95)
    
    return {
        "confidence": round(confidence, 4),
        "gradcamImage": MOCK_GRADCAM_URI
    }

# ----------------------------------------------------------------------
# 3. API Route (API Interface)
# ----------------------------------------------------------------------

@app.route('/detect_batch', methods=['POST'])
def detect_batch():
    """
    Handles batch image detection requests from the frontend.
    """
    try:
        data = request.get_json()
        if not data or 'images' not in data or not isinstance(data['images'], list):
            return jsonify({"error": "Invalid request format. Expected 'images' list."}), 400

        results = []
        for image_item in data['images']:
            image_id = image_item.get('id', 'unknown_id')
            base64_data = image_item.get('base64Data')
            
            if not base64_data or not base64_data.startswith('data:image'):
                results.append({
                    "id": image_id,
                    "status": "error",
                    "message": "Missing or invalid Base64 data URI."
                })
                continue
            
            # Run model prediction simulation
            detection_result = run_model_prediction_and_gradcam(base64_data)
            
            results.append({
                "id": image_id,
                "status": "success",
                "confidence": detection_result['confidence'],
                "gradcamImage": detection_result['gradcamImage']
            })

        return jsonify({"results": results}), 200

    except Exception as e:
        app.logger.error(f"Detection error: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == '__main__':
    # Run the service on local port 5000
    app.run(debug=True, port=5000)