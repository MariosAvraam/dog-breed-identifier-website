# Dog Breed Identifier

## Introduction
This repository contains a Flask web application that allows users to upload images of dogs and get predictions on the breed of the dog, along with a description and some information about the breed. The application uses a deep learning model trained on a dataset of various dog breeds.

![Screenshot of website dog short text](static/images/dog_short_text_image.png)

![Screenshot of website dog general information](static/images/dog_general_info_image.png)

## Prerequisites
- Python 3.x
- Flask
- Keras
- TensorFlow
- Other dependencies listed in `requirements.txt`

## Setup & Installation

1. **Clone the Repository**
```
git clone https://github.com/MariosAvraam/dog-breed-identifier-website.git
```

2. **Navigate to the project directory**
```
cd dog-breed-identifier-website
```

3. **Setting up a Virtual Environment (Optional but Recommended)**
```
python -m venv venv
source venv/bin/activate # On Windows, use venv\Scripts\activate
```

4. **Install Dependencies (This may take a while)**
```
pip install -r requirements.txt
```


5. **Obtain the Dog Breed Identification Dataset from Kaggle**

   - **Set Up a Kaggle Account**
     - If you don't already have a Kaggle account, sign up at [Kaggle](https://www.kaggle.com/).

   - **Navigate to the Dataset Page**
     - Go to the `dog-breed-identification` dataset page on Kaggle: [Dog Breed Identification Dataset](https://www.kaggle.com/c/dog-breed-identification/data).

   - **Download the Dataset**
     - Click on the `Download` button. This will download a zip file named `dog-breed-identification.zip`.

   - **Extract the Dataset**
     - Extract the `dog-breed-identification.zip` file to get a `train` folder and a `labels.csv` file.

   - **Place the Dataset in the Project Directory**
     - Create a `dataset` directory in the project root thas has a subfolder called `dog-breed-identification`.
     - Move the `train` folder and `labels.csv` file into `dataset/dog-breed-identification`.

6. **Train the Model**
   - Train the model using `model_train.py` with the dataset in `dataset/dog-breed-identification` (This can take a long time. I suggest you use an online GPU cloud service instead).
   ```
   python model_train.py
   ```
   - After training, the model, `fine_tuned_dog_breed_model.h5`, will be saved. Move this to the project root.

7. **API Key Setup**
   - Register at [API Ninjas](https://api-ninjas.com/) for an account.
   - Navigate to your account and generate an API key.
   - In the project root, create a `.env` file.
   - Add to `.env`: `X-API-KEY=YOUR_API_KEY`, replacing `YOUR_API_KEY` with your key from API Ninjas.

8. **Run the Application**
```
python app.py
```

9. **Access the Web Application**
- Open a web browser and navigate to `http://127.0.0.1:5000/`.
- Upload a dog image and get the breed prediction!


## About `class_labels.txt`
The `class_labels.txt` file contains the names of the dog breeds that the model has been trained on. Each line in the file represents a unique breed. This file is used by the application to map the model's predictions to the corresponding breed names.

## Contributing
Feel free to fork this repository, make changes, and submit pull requests. Any contributions are welcome!

## License
This project is licensed under the MIT License.

## Acknowledgements
Thanks to [Kaggle](https://www.kaggle.com/c/dog-breed-identification) for the dog breed dataset and [DogsAPI](https://api-ninjas.com/api/dogs) for the breed information API.