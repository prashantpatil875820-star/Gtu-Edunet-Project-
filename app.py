import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

import config
from live_predict import live_prediction

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Forest Fire Risk Prediction",
    page_icon="🌲",
    layout="wide"
)

# --------------------------------------------------
# Load CNN Model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(config.MODEL_PATH)

model = load_model()

# --------------------------------------------------
# Local Image Prediction
# --------------------------------------------------
def predict_image(img):

    img = img.resize(config.IMAGE_SIZE)

    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    pred = float(model.predict(x, verbose=0)[0][0])

    # Dataset Order:
    # ['fire_risk', 'no_fire_risk']

    if pred < 0.5:
        label = "🔥 Fire Risk"
        confidence = (1 - pred) * 100
    else:
        label = "🌳 No Fire Risk"
        confidence = pred * 100

    return label, confidence


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("🌲 Forest Fire")

page = st.sidebar.selectbox(
    "Navigation",
    [
        "🏠 Home",
        "📷 Upload Image",
        "🛰 Live Satellite",
        "ℹ About"
    ]
)

# ==================================================
# HOME
# ==================================================

if page == "🏠 Home":

    st.title("🌲 Forest Fire Risk Prediction")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", "94.21%")
    col2.metric("Model", "CNN")
    col3.metric("Satellite", "NASA GIBS")

    st.write("")

    st.info("""
This project predicts forest fire risk using Deep Learning.

### Features

✅ Upload Forest Image

✅ Live Satellite Prediction

✅ NASA GIBS Satellite Images

✅ NASA FIRMS Active Fire Hotspots

""")

# ==================================================
# IMAGE PREDICTION
# ==================================================

elif page == "📷 Upload Image":

    st.title("📷 Image Prediction")

    uploaded = st.file_uploader(
        "Upload Forest Image",
        type=["jpg","jpeg","png"]
    )

    if uploaded:

        image = Image.open(uploaded).convert("RGB")

        st.image(image,
                 caption="Uploaded Image",
                 use_container_width=True)

        if st.button("Predict"):

            label, confidence = predict_image(image)

            st.success(label)

            st.progress(int(confidence))

            st.write(f"### Confidence : {confidence:.2f}%")

# ==================================================
# LIVE SATELLITE
# ==================================================

elif page == "🛰 Live Satellite":

    st.title("🛰 Live Satellite Prediction")

    lat = st.number_input(
        "Latitude",
        value=21.1702,
        format="%.6f"
    )

    lon = st.number_input(
        "Longitude",
        value=72.8311,
        format="%.6f"
    )

    if st.button("Download & Predict"):

        with st.spinner("Downloading Satellite Image..."):

            result = live_prediction(lat, lon)

        st.success(result["predicted_label"])

        st.progress(int(result["confidence"]*100))

        st.write(f"### Confidence : {result['confidence']*100:.2f}%")

        st.write(
            f"### NASA FIRMS Hotspots : "
            f"{result['firms_active_fire_hotspots_nearby']}"
        )

        st.image(
            result["image_path"],
            caption="Live Satellite Image",
            use_container_width=True
        )

        if result["firms_active_fire_hotspots_nearby"] > 0:

            st.error("🔥 Active Fire Hotspots Detected")

        else:

            st.success("✅ No Active Fire Hotspots")

# ==================================================
# ABOUT
# ==================================================

else:

    st.title("ℹ About Project")

    st.markdown("""

### Forest Fire Risk Prediction

This project uses a Convolutional Neural Network (CNN)
to detect forest fire risk.

### Technologies

- TensorFlow
- CNN
- Streamlit
- NASA GIBS
- NASA FIRMS

### Accuracy

94.21%

### Dataset

Forest Fire Images

### Author

Devendra Patil

""")