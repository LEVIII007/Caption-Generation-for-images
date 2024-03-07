import streamlit as st
from PIL import Image
import numpy as np
import keras
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
vgg_model = VGG16()
vgg_model = keras.models.Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

from keras.models import load_model

model = load_model('img_caption_model.h5')


def idx_to_word(integer, tokenizer):

    # Iterate through the tokenizer's vocabulary
    for word, index in tokenizer.word_index.items():
        # If the integer ID matches the index of a word, return the word
        if index == integer:
            return word

    # If no matching word is found, return None
    return None


def preprocess_image(image):
    img = Image.open(image).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        
       # Tokenize the current caption into a sequence of integers
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
       
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
       
        # Get the index of the word with the highest probability
        yhat = np.argmax(yhat)
        
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
      
    return in_text

# Function to predict caption given an image
def predict_caption_streamlit(image):
    img_array = preprocess_image(image)
    feature = vgg_model.predict(img_array, verbose=0)
    caption = predict_caption(model, feature, tokenizer, 35)
    return caption

# Streamlit app
def main():
    st.title("Image Captioning with Streamlit")

    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Predict caption on button click
        if st.button("Generate Caption"):
            caption_result = predict_caption_streamlit(uploaded_file)
            st.subheader("Generated Caption:")
            st.write(caption_result)

if __name__ == "__main__":
    main()
