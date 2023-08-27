# Dog Breed Identifier

This web application allows users to upload an image of a dog and get a prediction of its breed. It uses a trained deep learning model to make the prediction and provides additional information about the predicted breed.

## Features
- Upload an image of a dog.
- Predict the breed of the dog using a trained model.
- Display additional information about the predicted breed, such as height, weight, life expectancy, and more.
- Responsive design for both desktop and mobile devices.

## Technologies Used
- Flask: Backend web framework.
- Keras: Deep learning framework used to train the model.
- Bootstrap: Frontend framework for styling.
- BeautifulSoup: For web scraping breed descriptions.

## Files Overview
1. `styles.css`: Contains global styles and specific styles for displaying images.
2. `base.html`: Base template that other HTML files extend. Contains meta tags, links to stylesheets, and the main content container.
3. `index.html`: Template for the main page where users can upload a dog image.
4. `result.html`: Template for displaying the predicted breed and additional information about the breed.
5. `app.py`: Flask application that handles file uploads, breed prediction, and rendering templates.
6. `model_train.py`: Script for loading the dataset, training the deep learning model, and saving the trained model.

## API Key Setup
To fetch additional breed information, this application uses an external API. You'll need to obtain your own API key for it to work:

1. Visit [API Ninjas](https://api-ninjas.com/) and sign up for an account.
2. Once registered, navigate to the API section and look for the 'Dogs' API.
3. Generate an API key for the Dogs API.
4. Create a `.env` file in the root directory of the project.
5. Add the following line to the `.env` file:

X-API-KEY=your_api_key_here

Replace `YOUR_API_KEY` with the key you obtained from API Ninjas.

## Optional: Setting up a Virtual Environment
If you prefer to run the application in a virtual environment, follow these steps:

1. Create a Virtual Environment
```
python -m venv venv
```

2. Activate the Virtual Environment:
```
source venv/bin/activate #On Windows use venv\Scripts\activate
```

## How to Run
1. Install the required libraries:
```
pip install -r requirements.txt
```

2. Run the Flask application:
```
python app.py
```

3. Open a web browser and navigate to `http://127.0.0.1:5000/` to access the application.

## Future Improvements
- Improve the accuracy of the model by training on a larger dataset.
- Add a feature to provide feedback on the prediction.
- Implement user accounts to save past predictions.

## Acknowledgements
- Dog breed dataset from [Dog Breed Identification Challenge](https://www.kaggle.com/c/dog-breed-identification).
- Bootstrap CSS framework from [Bootstrap](https://getbootstrap.com/).
- Icons and images from [FontAwesome](https://fontawesome.com/) and [Unsplash](https://unsplash.com/).

## License
This project is licensed under the MIT License. See the `LICENSE` file for details."