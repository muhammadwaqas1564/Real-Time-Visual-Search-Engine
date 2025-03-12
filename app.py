import streamlit as st
import numpy as np
import pickle
import faiss
from PIL import Image
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('inceptionv3_model.h5')

# Load the feature vectors, FAISS index, and image paths
feature_vectors = np.load('feature_vectors.npy')
index = faiss.read_index('faiss_index.index')

with open('paths.pkl', 'rb') as f:
    image_paths = pickle.load(f)

# Normalize the feature vectors for FAISS
faiss.normalize_L2(feature_vectors)

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.convert('RGB')  # Convert to RGB if grayscale
    image = image.resize((224, 224))  # Resize the image
    image = np.array(image) / 255.0  # Normalize the image
    return image

# Function to get feature embedding from the model
def get_feature_embedding(image):
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return model.predict(image)

# Function to search for similar images using FAISS
def search_similar_images_faiss(query_vector, top_n=5):
    faiss.normalize_L2(query_vector)  # Normalize the query vector
    _, indices = index.search(query_vector, top_n)  # Search for top N similar images
    return indices[0]

# Streamlit app
st.title('Image Similarity Search')

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the uploaded image
    img = Image.open(uploaded_file)
    processed_img = preprocess_image(img)

    # Display the uploaded image
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Generate the feature embedding for the uploaded image
    query_vector = get_feature_embedding(processed_img)

    # Search for similar images
    top_5_indices = search_similar_images_faiss(query_vector)

    # Retrieve paths of the top 5 similar images
    top_5_image_paths = [image_paths[i] for i in top_5_indices]

    # Display the similar images
    st.write("Similar Images:")
    cols = st.columns(5)
    for i, img_path in enumerate(top_5_image_paths):
        similar_img = Image.open(img_path)
        with cols[i]:
            st.image(similar_img, caption=f"Similar Image {i+1}", use_column_width=True)
