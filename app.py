import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename 
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

DOGS_API_ENDPOINT = "https://api.api-ninjas.com/v1/dogs"

HEADERS = {
    "X-Api-Key": os.getenv("X-API-KEY")
}


app = Flask(__name__)

# Load the trained model
model = load_model('fine_tuned_dog_breed_model.h5')

with open('class_labels.txt', 'r') as f:
    class_labels = f.read().splitlines()

# Function to preprocess image and predict breed
def predict_breed(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array)
    breed_index = np.argmax(predictions)
    return class_labels[breed_index]

@app.route('/')
def index():
    return render_template('index.html')

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        breed = predict_breed(filepath).replace('_', ' ')

        # Make API request with the predicted breed
        params = {"name": breed}
        response = requests.get(url=DOGS_API_ENDPOINT, params=params, headers=HEADERS)
        breed_info = response.json()[0]

        return render_template('result.html', breed=breed, img_path=filename, breed_info=breed_info)
    else:
        flash('File not allowed')
        return redirect(url_for('index'))

    

if __name__ == '__main__':
    app.run(debug=True)
