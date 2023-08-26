import os
import pandas as pd
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename 
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model('fine_tuned_dog_breed_model.h5')

# Extract unique breeds from labels.csv
labels_df = pd.read_csv('./dataset/dog-breed-identification/labels.csv')
class_labels = labels_df['breed'].unique().tolist()

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

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        breed = predict_breed(filepath)
        return breed

if __name__ == '__main__':
    app.run(debug=True)
