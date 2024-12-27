from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import cv2

# Creating the app
app = Flask(__name__)

# Loading the model (using Method 1 for example)
model = load_model(r"model/skin_disorder_classifier_EfficientNetB2.h5")

# Function to check if the file is an allowed image type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

# Function to detect skin color
def is_skin(img):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # Define range of skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    # Create a binary mask of skin color pixels
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    # Count the number of skin color pixels
    skin_pixels = np.sum(mask > 0)
    # Calculate the percentage of skin color pixels in the image
    skin_percent = skin_pixels / (img.shape[0] * img.shape[1]) * 100
    # Return True if skin percentage is above a threshold, else False
    return skin_percent > 5

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for the prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    file = request.files['file']

    # Check if the file is an image
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Only image files are allowed'})

    # Open the image using PIL
    image = Image.open(file)

    # Check if the image contains human skin
    if not is_skin(np.array(image)):
        return jsonify({'error': 'The uploaded image could not be processed. Please ensure that the image contains skin and try again.'})

    # Preprocess the image
    img = image.resize((100, 100))  # Ensure the image is resized to the correct dimensions
    img_array = img_to_array(img)
    img = img_array / 255.0
    img = np.expand_dims(img, axis=0)

    # Make prediction
    pred = model.predict(img)
    class_idx = np.argmax(pred)

    # Classes
    classes = ["Acne", "Impetigo", "Melanoma", "Rosacea"]  # Your 4 classes

    # Predicted class
    pred_class = classes[class_idx]

    # Probability of prediction
    prob = pred[0][class_idx]

    # Set probability threshold
    threshold = 0.6

    # Check if probability is above threshold
    if prob < threshold:
        return jsonify({'error': 'Inconclusive result. Please consult a healthcare professional for an accurate diagnosis.'})

    # Render the results page with the prediction
    return jsonify({
        'prediction': pred_class,
        'probability': float(prob)
    })

# Run the application
if __name__ == '__main__':
    app.run(debug=True)
