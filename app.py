import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from ultralytics import YOLO


st.image('https://github.com/sani002/mkpapp/blob/main/Header.png?raw=true')

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after {
                            content:'This app is in its early stage. Thank you.'; 
                            visibility: visible;
                            display: block;
                            position: relative;
                            #background-color: red;
                            padding: 5px;
                            top: 2px;
                        }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

#Add CSS styling for center alignment
st.markdown(
    """
    <style>
    .center {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Create YOLO instance with your model
yolo = YOLO(model='yolov5su.pt')

# Load the pre-trained model
model = tf.keras.models.load_model('MobileNET.h5')

# Define the labels for the categories
labels = {
    0 : 'Aedes_aegypti',
    1 : 'Aedes_albopictus',
    2 : 'Aedes_japonicus',
    3 : 'Aedes_koreicus',
    4 : 'Armigeres_Unknown',
    5 : 'Culex _pipiens',
    6 : 'Culex_quinquefasciatus',
    7 : 'Other _species'
 }

# Function to preprocess the image using YOLO detection and cropping
def preprocess_image(image):
    # Run YOLO detection and get the list of results
    results = yolo.predict(source=image, conf=0.25, save_crop=False)

    # Access the first result in the list
    first_result = results[0]

    # Load the original image using Pillow
    original_image_pil = image
    original_image_np = np.array(original_image_pil)

    # Access the bounding box coordinates
    x_min, y_min, x_max, y_max = first_result.boxes[0].xyxy[0].tolist()

    # Crop the image using NumPy slicing
    cropped_image_np = original_image_np[int(y_min):int(y_max), int(x_min):int(x_max)]

    # Resize the cropped image to 224x224
    resized_image = Image.fromarray(cropped_image_np).resize((224, 224))

    # Convert the resized image to a numpy array
    resized_image_array = np.array(resized_image)

    # Add an extra dimension to match the model's input shape
    resized_image_array = np.expand_dims(resized_image_array, axis=0)

    return resized_image_array

# Function to make predictions
def predict_yolo(image_path):
    # Preprocess the image using YOLO detection and cropping
    processed_image = preprocess_image(image_path)

    # Make the prediction
    prediction = model.predict(processed_image)

    # Get the predicted label index
    label_index = np.argmax(prediction)

    # Get the predicted label
    predicted_label = labels[label_index]

    # Get the confidence level
    confidence = prediction[0][label_index] * 100

    return predicted_label, confidence

# Streamlit app
def main():
    # Center align the heading
    st.markdown("<h1 class='center'>Mosquito Classifier</h1>", unsafe_allow_html=True)

    # Display a file uploader widget
    number = st.radio('Pick one', ['Upload from gallery', 'Capture by camera'])
    if number == 'Capture by camera':
        uploaded_file = st.camera_input("Take a picture")
    else:
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg", "bmp"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Process and classify the image
        predicted_label, confidence = predict_yolo(image)

        # Display the predicted label and confidence
        st.markdown("<h3 class='center'>This might be:</h2>", unsafe_allow_html=True)
        st.markdown(f"<h1 class='center'>{predicted_label}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p class='center'>Confidence: {confidence:.2f}%</p>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
