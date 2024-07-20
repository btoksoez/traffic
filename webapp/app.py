from flask import Flask, request, render_template
import os
import cv2
import numpy as np
import tensorflow as tf
import json

# Constants for your model and image processing
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43  # Ensure this matches your model's output size

# Load your trained model
model = tf.keras.models.load_model('./model.h5')  # Replace with your actual model filename

# Initialize Flask application
app = Flask(__name__)

# Define a dictionary for traffic sign names
traffic_sign_names = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons"
}

# Define route for home page
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Get the file from the POST request
        file = request.files['file']

        # Check if the file is valid
        if file and allowed_file(file.filename):
            # Save the file to a temporary location or memory
            image = process_image(file)

            # Make prediction
            predicted_label, probability = predict_traffic_sign(image)

            # Get the name of the traffic sign based on prediction
            traffic_sign_name = traffic_sign_names.get(predicted_label, "Unknown sign")

            # Render the prediction result in the HTML template
            return render_template('result.html', traffic_sign_name=traffic_sign_name, probability=probability)

    # Render the upload form on GET request
    return render_template('upload.html')

def allowed_file(filename):
    # Check if the file extension is in the set of allowed extensions
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

def process_image(file):
    # Read the image using OpenCV (cv2)
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Resize the image to match model's expected sizing
    resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

    # Convert the resized image to a numpy array
    resized_array = np.array(resized)

    processed_img = np.expand_dims(resized_array, axis=0)

    return processed_img

def predict_traffic_sign(image):
    # Predict with the model
    predictions = model.predict(image)
    predicted_label = np.argmax(predictions[0])  # Get index of the highest probability
    probability = round(np.max(predictions[0]) * 100, 2)

    print(f"predicted: {predicted_label}, probability: {probability}")

    return predicted_label, probability

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
