import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import streamlit as st
import cv2

# Constants
IMG_SIZE = (224, 224)
MODEL_PATH = 'model/medical_model.h5'

# Define your class names here (match your folder names)
class_names = ['disease', 'normal']  # Replace with your actual classes

# Grad-CAM function
def generate_grad_cam(model, img_array, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (IMG_SIZE[1], IMG_SIZE[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return heatmap

# Load or build model
@st.cache_resource(show_spinner=False)
def get_model():
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
    else:
        inputs = Input(shape=IMG_SIZE + (3,))
        x = Conv2D(32, (3, 3), activation='relu', name='conv2d_0')(inputs)
        x = MaxPooling2D(2, 2)(x)
        x = Conv2D(64, (3, 3), activation='relu', name='conv2d_1')(x)
        x = MaxPooling2D(2, 2)(x)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(len(class_names), activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = get_model()

st.title("🩺 Medical Image Diagnosis with Grad-CAM Explainability")

uploaded_file = st.file_uploader("Upload an image for prediction", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    img_array = np.expand_dims(img_resized, axis=0) / 255.0

    st.image(img_rgb, caption='Uploaded Image', use_container_width=True)

    preds = model.predict(img_array)
    class_idx = np.argmax(preds[0])
    confidence = preds[0][class_idx]
    st.write(f"Prediction: {class_names[class_idx]} with confidence {confidence:.2f}")

    heatmap = generate_grad_cam(model, img_array, 'conv2d_1', pred_index=class_idx)

    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed_img = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)

    st.image(superimposed_img, caption='Grad-CAM Heatmap', use_container_width=True)

else:
    st.info("Please upload an image file to get started.")
