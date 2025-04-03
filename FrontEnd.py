import os
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import base64

# Disable TensorFlow's OneDNN optimizations warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set the page configuration as the first Streamlit command
st.set_page_config(layout="wide")

# Add custom CSS
st.markdown("""
    <style>
    h1 {
        color: black;
        text-align: center;
        font-size: 48px;
        margin-top: 20px;
    }
    h2 {
        color: Maroon;
        text-align: center;
        font-size: 24px;
        margin-bottom: 40px;
        text-shadow: 2px 2px 2px rgba(255, 255, 255, 0.7);
    }
    .bold-text {
        font-size: 18px;
        font-weight: bold;
    }
    .prediction-text {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 15px;
    }
    .metric-text {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
st.markdown("""
            <style>
            .stTable(accuracy_data) {
                font-size: 100px;
            }
            </style>
            """, unsafe_allow_html=True)

st.markdown('<h1>Traditional Costume Image Classification for Indian States</h1>', unsafe_allow_html=True)
st.markdown('<h2>This application Identifies costume images using four different models</h2>', unsafe_allow_html=True)

# Define a function to read and encode the image file
@st.cache
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Set the background image
def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: 224*224;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

# Radio buttons for selecting between image classification and model comparison
option = st.radio(
    "Select an option",
    ('Image Classification', 'Model Comparison')
)

# File uploader for image classification
upload = st.file_uploader('*Insert image for classification*', type=['png', 'jpg'])

# Load models with custom objects (if any)
def load_custom_model(model_path):
    try:
        from tensorflow.keras.layers import DepthwiseConv2D
        class CustomDepthwiseConv2D(DepthwiseConv2D):
            def __init__(self, **kwargs):
                if 'groups' in kwargs:
                    kwargs.pop('groups')
                super().__init__(**kwargs)

        return load_model(model_path, custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
        return None

# Paths for the models
model1_path = os.path.join(r'D:\Sahana RK\MajorPro\MLfrontend\VGG16.h5')
model2_path = os.path.join(r'D:\Sahana RK\MajorPro\MLfrontend\ResNet50V2.h5')
model3_path = os.path.join(r'D:\Sahana RK\MajorPro\MLfrontend\mobilenetv2_final.h5')
model4_path = os.path.join(r'D:\Sahana RK\MajorPro\MLfrontend\DenseNet121.h5')

# Load the models if the paths are valid
model1 = load_custom_model(model1_path) if os.path.exists(model1_path) else None
model2 = load_custom_model(model2_path) if os.path.exists(model2_path) else None
model3 = load_custom_model(model3_path) if os.path.exists(model3_path) else None
model4 = load_custom_model(model4_path) if os.path.exists(model4_path) else None

# Define class labels
class_labels = ['Chhattisgarh', 'Gujarat', 'Haryana', 'Himachal Pradesh', 
                'Karnataka', 'Kerala', 'Maharashtra', 'Manipur', 'Mizoram', 'West Bengal']

# Descriptions for each class label
class_descriptions = {
    'Chhattisgarh': 'The way Chhattisgarh women wear the sari is known as “kachhora” style; Kachhora sarees are now widely made with batik style of dying. In the local language, the sarees are called Lugda, worn with polkha or blouse. These sarees are handwoven, and the motifs are either geometric or derived from flora and fauna.',
    'Gujarat': 'The traditional Gujarati dresses for men include kediyu or kurta on the top and dhoti or chorno at the bottom. Women in Gujarat wear sarees or chaniya choli.',
    'Haryana': 'Women of Haryana show a special affinity towards colours. Their basic trousseau includes Daaman, Kurti & Chunder.',
    'Himachal Pradesh': 'The traditional dress of Himachal Pradesh is called Kulluvi or Pahadi dress. It is made of woolen fabric, which is warm and suitable for the cold climate of the region.',
    'Karnataka': 'The women of Karnataka traditionally wear sarees, mainly made with silk, while teenage girls adorn silk co-ord sets consisting of a skirt and blouse. ',
    'Kerala': 'the traditional dress is a two-piece set known as the mundum neriyathum.',
    'Maharashtra': 'The traditional dress for women in Maharashtra consists of a saree, choli, and ghagra. The saree, a long piece of cloth draped around the body, is believed to have originated in Maharashtra and dates back to ancient India. ',
    'Manipur': 'Phanek is worn like a sarong. The Manipuri dress is woven with the hand in horizontal line designs. The people also weave special Phanek, those called Mayek Naibi. ',
    'Mizoram': 'Mizoram traditional costume is a stunning combination of brightly coloured fabrics, intricate embroidery, and handwoven fabrics. Women typically wear a Puan or a wraparound skirt with a matching blouse',
    'West Bengal': 'Saree is the signature traditional attire for the women in West Bengal. The saree captures the very essence of the culturally infused state West Bengal is. '
}

# Accuracies for demonstration purposes
model_accuracies = {
    'DenseNet': 100,
    'ResNet': 100,
    'VGG16': 92.0,
    'MobileNet': 90.0
}

def predict_image(image, model):
    # Convert image to RGB if it has an alpha channel
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = image.resize((224, 224))  # Resize the image to the required size
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(image)
    return prediction

def evaluate_model(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    return f1, precision, recall, accuracy

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    st.pyplot(plt)

# Handle the selected option
if option == 'Image Classification':
    if upload is not None:
        im = Image.open(upload)
        st.image(im, caption='Uploaded Image', use_column_width=100)

        st.markdown('<div class="bold-text">Classifying...</div>', unsafe_allow_html=True)

        # Initialize an empty list to store predictions
        predictions = []

        # Get predictions from each model
        if model1 is not None:
            prediction1 = predict_image(im, model1)
            predicted_class1 = np.argmax(prediction1, axis=1)[0]
            st.markdown(f"<div class='prediction-text'>VGG16 Prediction: {class_labels[predicted_class1]}</div>", unsafe_allow_html=True)
            predictions.append(('VGG16', predicted_class1))

        if model2 is not None:
            prediction2 = predict_image(im, model2)
            predicted_class2 = np.argmax(prediction2, axis=1)[0]
            st.markdown(f"<div class='prediction-text'>ResNet Prediction: {class_labels[predicted_class2]}</div>", unsafe_allow_html=True)
            predictions.append(('ResNet', predicted_class2))

        if model3 is not None:
            prediction3 = predict_image(im, model3)
            predicted_class3 = np.argmax(prediction3, axis=1)[0]
            st.markdown(f"<div class='prediction-text'>MobileNet Prediction: {class_labels[predicted_class3]}</div>", unsafe_allow_html=True)
            predictions.append(('MobileNet', predicted_class3))

        if model4 is not None:
            prediction4 = predict_image(im, model4)
            predicted_class4 = np.argmax(prediction4, axis=1)[0]
            st.markdown(f"<div class='prediction-text'>DenseNet Prediction: {class_labels[predicted_class4]}</div>", unsafe_allow_html=True)
            predictions.append(('DenseNet', predicted_class4))

        # Determine the final prediction
        final_prediction_model = max(predictions, key=lambda x: model_accuracies[x[0]])[0]
        final_prediction_class = max(predictions, key=lambda x: model_accuracies[x[0]])[1]

        st.markdown(f"<br><div class='prediction-text'>Final Prediction from {final_prediction_model}: {class_labels[final_prediction_class]}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='prediction-text'>The {final_prediction_model} model is well trained compared to other models, therefore the {final_prediction_model} prediction is considered.</div><br>", unsafe_allow_html=True)
        st.markdown(f"<div class='prediction-text'>Description: {class_descriptions[class_labels[final_prediction_class]]}</div>", unsafe_allow_html=True)

elif option == 'Model Comparison':
    # Display model accuracies in a table
    st.markdown("<h1>Model Accuracies</h1>", unsafe_allow_html=True)
    accuracy_data = {
        'Model': ['VGG16', 'ResNet', 'MobileNet', 'DenseNet'],
        'Accuracy (%)': [model_accuracies['VGG16'], model_accuracies['ResNet'], model_accuracies['MobileNet'], model_accuracies['DenseNet']]
    }
    st.table(accuracy_data)

    #st.markdown("<h2>Additional Information</h2>", unsafe_allow_html=True)
    st.markdown("<p>Below are the model comparison graphs.</p>", unsafe_allow_html=True)

    # Example image for demonstration (you can replace this with your image handling logic)
    example_image_path1 = r"D:\Sahana RK\MajorPro\MLfrontend\model_accuracy_comparison.png"
    example_image_path2 = r"D:\Sahana RK\MajorPro\MLfrontend\Roc_Graph.png"

    if os.path.exists(example_image_path1):
        example_image = Image.open(example_image_path1)
        st.image(example_image, use_column_width=100)
    else:
        st.warning("Example image not found.")

    if os.path.exists(example_image_path2):
        example_image = Image.open(example_image_path2)
        st.image(example_image, use_column_width=100)
    else:
        st.warning("Example image not found.")