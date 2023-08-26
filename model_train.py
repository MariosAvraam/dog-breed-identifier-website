import pandas as pd
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

# Load the dataset
labels = pd.read_csv('dog-breed-identification/labels.csv')
labels['id'] = labels['id'] + '.jpg'

# Split the data into training and validation sets
train_df, val_df = train_test_split(labels, test_size=0.2, random_state=42)

# Number of classes (dog breeds)
num_classes = len(labels['breed'].unique())

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,  # Reduced rotation
    width_shift_range=0.1,  # Reduced shift
    height_shift_range=0.1,  # Reduced shift
    shear_range=0.1,  # Reduced shear
    zoom_range=0.1,  # Reduced zoom
    horizontal_flip=True,
    fill_mode='nearest'
)


# Only rescale for validation
val_datagen = ImageDataGenerator(rescale=1./255)

# Training generator
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory='dog-breed-identification/train/',  # Adjusted path
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
    directory='dog-breed-identification/train/',  # Adjusted path
    x_col="id",
    y_col="breed",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

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


# Define a callback for early stopping to prevent overfitting
from keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=[early_stop]
)

# Save the trained model
model.save('fine_tuned_dog_breed_model.h5')
