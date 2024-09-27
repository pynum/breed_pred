from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Define the class names
CLASS_NAME = ['labrador_retriever', 'golden_retriever', 'german_shepherd', 'bulldog', 'beagle', 'boxer', 'poodle', 'dachshund', 'pug', 'entlebucher']

# Recreate the model architecture
def create_model():
    input_tensor = Input(shape=(224, 224, 3))
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    output = Dense(len(CLASS_NAME), activation='softmax')(x)
    
    model = Model(inputs=input_tensor, outputs=output)
    return model

# Create the model
model = create_model()

# Load only the weights
model.load_weights('my_model.keras')