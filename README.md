# Dog Breed Identifier

## Introduction
This repository contains a Flask web application that allows users to upload images of dogs and get predictions on the breed of the dog. The application uses a deep learning model trained on a dataset of various dog breeds.

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

4. **Install Dependencies**
```
pip install -r requirements.txt

```

5. **Train the Model**
- The application requires a trained model to make predictions. Due to the large size of the model, it's not included in the repository. You'll need to train it yourself.
- Use the `model_train.py` script to train the model. Ensure you have the required dataset in the `dataset/` directory.
- Once trained, the model will be saved as `fine_tuned_dog_breed_model.h5`. Copy this file to the root directory of the project.

6. **Run the Application**
```
python app.py
```

7. **Access the Web Application**
- Open a web browser and navigate to `http://127.0.0.1:5000/`.
- Upload a dog image and get the breed prediction!

## API Key Setup
To fetch additional information about the predicted dog breed, the application uses an external API. You'll need to obtain an API key for this service.

1. Visit [API Ninjas](https://api-ninjas.com/) and sign up for an account.
2. Once registered, navigate to the API section and look for the 'Dogs' API.
3. Generate an API key for the Dogs API.
4. Create a `.env` file in the root directory of the project.
5. Add the following line to the `.env` file:
```
X-API-KEY=YOUR_API_KEY
```
Replace `YOUR_API_KEY` with the key you obtained from API Ninjas.


## Contributing
Feel free to fork this repository, make changes, and submit pull requests. Any contributions are welcome!

## License
This project is licensed under the MIT License.

## Acknowledgements
Thanks to [Kaggle](https://www.kaggle.com/c/dog-breed-identification) for the dog breed dataset and [DogsAPI](https://api-ninjas.com/api/dogs) for the breed information API."


