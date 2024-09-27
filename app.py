import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Define the class names
CLASS_NAME = ['labrador_retriever', 'golden_retriever', 'german_shepherd', 'bulldog', 'beagle', 'boxer', 'poodle', 'dachshund', 'pug', 'entlebucher']

def load_and_prepare_model(model_path):
    try:
        # Try to load the entire model
        model = load_model(model_path, compile=False)
        print("Loaded entire model successfully.")
    except:
        print("Failed to load entire model. Attempting to load only weights.")
        # If loading entire model fails, try to recreate the architecture and load weights
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(256, activation='relu')(x)
        output = Dense(len(CLASS_NAME), activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=output)
        
        try:
            model.load_weights(model_path)
            print("Loaded weights successfully.")
        except:
            print("Failed to load weights. Using model with ImageNet weights.")
    
    # Ensure the model is built
    model.build((None, 224, 224, 3))
    return model

# Load the model
model = load_and_prepare_model('my_model.keras')

@st.cache_resource
def get_model():
    return model

def predict_image(img):
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    model = get_model()
    prediction = model.predict(img)
    predicted_class = CLASS_NAME[np.argmax(prediction)]
    return predicted_class

def main():
    st.title("Dog Breed Classifier")

    st.write("Upload an image of a dog to classify its breed.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load and display the image
        image = load_img(uploaded_file, target_size=(224, 224))
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Predict the breed
        with st.spinner("Classifying..."):
            breed = predict_image(image)
            st.write(f"Predicted Breed: {breed}")

if __name__ == "__main__":
    main()