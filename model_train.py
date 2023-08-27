import pandas as pd
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# Constants
TRAIN_DIR = 'dataset/dog-breed-identification/train/'
LABELS_FILE = 'dataset/dog-breed-identification/labels.csv'
MODEL_SAVE_PATH = 'fine_tuned_dog_breed_model.h5'

def load_data():
    """
    Load the dataset and split it into training and validation sets.
    
    Returns:
    - train_df (DataFrame): Training data.
    - val_df (DataFrame): Validation data.
    """
    labels = pd.read_csv(LABELS_FILE)
    labels['id'] = labels['id'] + '.jpg'
    return train_test_split(labels, test_size=0.2, random_state=42)

def get_generators(train_df, val_df):
    """
    Create ImageDataGenerators for training and validation.
    
    Args:
    - train_df (DataFrame): Training data.
    - val_df (DataFrame): Validation data.
    
    Returns:
    - train_generator: Training data generator.
    - val_generator: Validation data generator.
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Only rescale for validation
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Training generator
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=TRAIN_DIR,
        x_col="id",
        y_col="breed",
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        shuffle=True
    )

    # Validation generator
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=TRAIN_DIR,
        x_col="id",
        y_col="breed",
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        shuffle=False
    )
    
    return train_generator, val_generator

def build_model(num_classes):
    """
    Build and compile the model.
    
    Args:
    - num_classes (int): Number of output classes.
    
    Returns:
    - model: Compiled model.
    """
    # Load the VGG16 model without the top layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Make the base model non-trainable
    for layer in base_model.layers:
        layer.trainable = False

    # Unfreeze the last 4 layers
    for layer in base_model.layers[-4:]:
        layer.trainable = True

    # Create a new model on top of the base model
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def main():
    train_df, val_df = load_data()
    train_generator, val_generator = get_generators(train_df, val_df)
    num_classes = len(train_df['breed'].unique())
    
    model = build_model(num_classes)

    # Define a callback for early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    model.fit(
        train_generator,
        epochs=20,
        validation_data=val_generator,
        callbacks=[early_stop]
    )

    # Save the trained model
    model.save(MODEL_SAVE_PATH)

if __name__ == '__main__':
    main()
