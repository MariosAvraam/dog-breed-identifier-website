import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename 
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

# Constants
DOGS_API_ENDPOINT = "https://api.api-ninjas.com/v1/dogs"
DOGTIME_LINK = "https://dogtime.com/dog-breeds/"
HEADERS = {"X-Api-Key": os.getenv("X-API-KEY")}
EXPECTED_BREED_KEYS = [
    "name", "min_life_expectancy", "max_life_expectancy", 
    "min_height_male", "max_height_male", 
    "min_height_female", "max_height_female", 
    "min_weight_male", "max_weight_male", 
    "min_weight_female", "max_weight_female", 
    "energy", "trainability", "good_with_children", 
    "good_with_strangers", "shedding", "grooming", "barking"
]

app = Flask(__name__)

# Load the trained model
model = load_model('fine_tuned_dog_breed_model.h5')

# Load class labels
with open('class_labels.txt', 'r') as f:
    class_labels = f.read().splitlines()

def get_description(breed):
    """
    Fetches the description of a dog breed from dogtime.com.
    
    Args:
    - breed (str): The breed name.
    
    Returns:
    - str: A description of the breed.
    """
    try:
        response = requests.get(url=DOGTIME_LINK + breed)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        description_element = soup.find(class_="entry-content")
        return description_element.find("p").text
    except:
        return "No description found for this breed."

def predict_breed(img_path):
    """
    Predicts the breed of a dog from an image.
    
    Args:
    - img_path (str): Path to the image.
    
    Returns:
    - str: Predicted breed name.
    """
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array)
    return class_labels[np.argmax(predictions)]

@app.route('/')
def index():
    """Renders the index page."""
    return render_template('index.html')

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles the file upload and breed prediction."""
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

        try:
            params = {"name": breed}
            response = requests.get(url=DOGS_API_ENDPOINT, params=params, headers=HEADERS)
            response.raise_for_status()
            breed_info = response.json()[0]
        except:
            breed_info = {key: "Unknown" for key in EXPECTED_BREED_KEYS[1:]}
            breed_info["name"] = breed

        breed_description = get_description(breed.replace(' ', '-'))
        return render_template('result.html', breed=breed, img_path=filename, breed_info=breed_info, breed_description=breed_description)
    
    flash('File not allowed')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
