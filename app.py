from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt



mnist = fetch_openml('mnist_784', version=1)
mnist.keys()
X, y = mnist["data"], mnist["target"]

knn_model = joblib.load("models/sgd_clf.pkl")
knn_model.fit(X, y)

# Create Flask app
app = Flask(__name__)
CORS(app)

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    
 
    image = Image.open(file.stream).convert("L").resize((28, 28))
    
    image_array = np.array(image)
    
    threshold_value = 150  # You can adjust this threshold
    binary_image_array = np.where(image_array > threshold_value, image_array, 0)
    
    # Invert the image (255 - pixel_value)
    inverted_image_array = 255 - binary_image_array

    # Flatten the image to a 1D array for model prediction
    inverted_image_array = inverted_image_array.flatten()
    
    print(inverted_image_array)
    
    # plt.imshow(binary_image_array.reshape(28, 28), cmap="gray")
    # plt.show()
    
    # Make prediction
    prediction = knn_model.predict([inverted_image_array])
    
    predicted_number = prediction[0]

    print("Prediction:", prediction)
    print("Predicted number:", predicted_number)
    # Send response
    return jsonify({"number_prediction": predicted_number})

if __name__ == '__main__':
    app.run(debug=True, port=8080)