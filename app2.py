import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from datetime import datetime, timezone, timedelta

import config
from live_predict import live_prediction

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Forest Fire Prediction System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# OPEN WEATHER API
# ============================================================

OPENWEATHER_API_KEY = "bc6666ea6cf36a2bc50815e582ed4372"

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* Main Background */
.stApp{
    background:linear-gradient(135deg,#eef5ff,#f9fcff);
}

/* Container */
.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0F2027,#203A43,#2C5364);
}

section[data-testid="stSidebar"] *{
    color:white !important;
}

/* Title */
.title{
    font-size:48px;
    font-weight:800;
    text-align:center;
    color:#ff5722;
    text-shadow:2px 2px 8px rgba(0,0,0,.2);
}

/* Subtitle */
.subtitle{
    text-align:center;
    color:#555;
    font-size:20px;
    margin-bottom:25px;
}

/* Metric Card */
.metric-card{
    background:rgba(255,255,255,.95);
    border-radius:20px;
    padding:20px;
    box-shadow:0 10px 30px rgba(0,0,0,.15);
    transition:.3s;
    border-left:6px solid #ff5722;
}

.metric-card:hover{
    transform:translateY(-5px);
    box-shadow:0 15px 35px rgba(0,0,0,.25);
}

/* Weather Card */
.weather-card{
    background:linear-gradient(135deg,#36D1DC,#5B86E5);
    border-radius:20px;
    padding:20px;
    color:white;
    box-shadow:0 10px 25px rgba(0,0,0,.2);
}

/* Fire Card */
.fire-card{
    background:linear-gradient(135deg,#ff512f,#dd2476);
    border-radius:20px;
    padding:20px;
    color:white;
    box-shadow:0 10px 25px rgba(0,0,0,.2);
}

/* Buttons */
.stButton>button{
    width:100%;
    background:linear-gradient(90deg,#ff5722,#ff9800);
    color:white;
    font-size:18px;
    font-weight:bold;
    border:none;
    border-radius:12px;
    padding:12px;
    transition:.3s;
}

.stButton>button:hover{
    transform:scale(1.03);
    background:linear-gradient(90deg,#ff9800,#ff5722);
}

/* Download Button */
.stDownloadButton>button{
    width:100%;
    background:linear-gradient(90deg,#4CAF50,#2E7D32);
    color:white;
    font-weight:bold;
    border-radius:12px;
}

/* File Uploader */
[data-testid="stFileUploader"]{
    border:2px dashed #ff9800;
    border-radius:15px;
    padding:20px;
    background:white;
}

/* Number Input */
[data-baseweb="input"]{
    border-radius:12px;
}

/* Metric Box */
[data-testid="metric-container"]{
    background:white;
    border-radius:15px;
    padding:15px;
    box-shadow:0 5px 15px rgba(0,0,0,.15);
}

/* Progress */
.stProgress > div > div > div{
    background:linear-gradient(90deg,#00C853,#64DD17);
}

/* Success */
.stSuccess{
    border-radius:12px;
}

/* Warning */
.stWarning{
    border-radius:12px;
}

/* Error */
.stError{
    border-radius:12px;
}

/* Info */
.stInfo{
    border-radius:12px;
}

/* JSON */
.stJson{
    border-radius:15px;
}

/* Footer */
.footer{
    text-align:center;
    margin-top:35px;
    color:#555;
    font-size:15px;
}

/* Horizontal Line */
hr{
    border:none;
    height:2px;
    background:linear-gradient(to right,#ff9800,#ff5722);
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(config.MODEL_PATH)

model = load_model()

# ============================================================
# IMAGE PREDICTION
# ============================================================

def predict_image(img):

    img = img.resize(config.IMAGE_SIZE)

    x = tf.keras.utils.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    pred = float(model.predict(x, verbose=0)[0][0])

    if pred < 0.5:

        label = "🔥 Fire Risk"

        confidence = (1-pred)*100

    else:

        label = "🌳 No Fire Risk"

        confidence = pred*100

    return label, confidence

# ============================================================
# WEATHER FUNCTION
# ============================================================

@st.cache_data(ttl=300)
def get_weather(lat, lon):

    url = (
        f"https://api.openweathermap.org/data/2.5/weather?"
        f"lat={lat}&lon={lon}"
        f"&appid={OPENWEATHER_API_KEY}"
        f"&units=metric"
    )

    try:

        response = requests.get(url, timeout=10)

        data = response.json()
        timezone_offset = data["timezone"]
        local_tz = timezone(timedelta(seconds=timezone_offset))

        if response.status_code != 200:
            return None

        weather = {

            "city":data["name"],

            "country":data["sys"]["country"],

            "temperature":data["main"]["temp"],

            "feels_like":data["main"]["feels_like"],

            "humidity":data["main"]["humidity"],

            "pressure":data["main"]["pressure"],

            "wind":data["wind"]["speed"],

            "visibility":data["visibility"]/1000,

            "description":data["weather"][0]["description"].title(),

            "icon":data["weather"][0]["icon"],

            "clouds":data["clouds"]["all"],

           "sunrise": datetime.fromtimestamp(
                data["sys"]["sunrise"],
                tz=local_tz
            ).strftime("%I:%M %p"),
            
            "sunset": datetime.fromtimestamp(
                data["sys"]["sunset"],
                tz=local_tz
            ).strftime("%I:%M %p"),

        }

        return weather

    except:

        return None

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.image(
    "https://img.icons8.com/color/96/fire-element.png",
    width=90
)

st.sidebar.title("Forest Fire AI")

page = st.sidebar.radio(

    "Navigation",

    [

        "🏠 Home",

        "📷 Upload Image",

        "🛰 Live Satellite",

        "ℹ About"

    ]

)

# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    # ================= HERO SECTION =================

    st.markdown("""
    <div style="
        background:linear-gradient(135deg,#0F2027,#203A43,#2C5364);
        padding:40px;
        border-radius:20px;
        color:white;
        text-align:center;
        box-shadow:0px 8px 25px rgba(0,0,0,0.2);
        margin-bottom:25px;
    ">

    <h1 style="font-size:42px;margin-bottom:10px;">
    Forest Fire Prediction System
    </h1>

    <h3 style="font-weight:400;color:#D6EAF8;">
    AI-Based Forest Fire Risk Detection using Deep Learning
    </h3>

    <p style="font-size:17px;margin-top:20px;">
    CNN Model | NASA GIBS | NASA FIRMS | OpenWeather API
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # ================= DASHBOARD =================

    st.markdown("""
<h2 style="
text-align:center;
background:linear-gradient(90deg,#1B4F72,#2E86C1);
color:white;
padding:15px;
border-radius:12px;
margin-bottom:25px;
box-shadow:0 6px 15px rgba(0,0,0,.15);
">
System Overview
</h2>
""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("AI Model", "CNN")
    c2.metric("Satellite Source", "NASA GIBS")
    c3.metric("Weather API", "OpenWeather")
    c4.metric("Model Accuracy", "94.21%")

    st.markdown("---")

    left, right = st.columns([2,1])

    with left:

        st.subheader("Project Overview")

        st.write("""
The Forest Fire Prediction System is an AI-powered application developed
to identify potential forest fire risk using deep learning techniques
and live satellite imagery.

The system combines satellite images, weather conditions and a trained
Convolutional Neural Network (CNN) model to provide reliable fire risk
prediction in real time.

### Key Features

• Forest Image Classification

• Live Satellite Image Analysis

• Real-Time Weather Information

• NASA FIRMS Hotspot Detection

• Confidence Score Generation

• Fire Risk Prediction

• Prediction Report Download
        """)

    with right:

        st.markdown("""
        <div style="
         background:linear-gradient(135deg,#1B4F72,#2E86C1);
        color:white;
        padding:25px;
        border-radius:18px;
        box-shadow:0px 8px 20px rgba(0,0,0,.15);
        ">

        <h2 style="text-align:center;color:#2C3E50;">
        AI Fire Detection
        </h2>

        <hr>

        <p style="text-align:justify;color:#555;line-height:1.8;">

        The application integrates Deep Learning,
        satellite imagery and weather information
        to estimate forest fire risk and assist
        in environmental monitoring.

        </p>

        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("System Workflow")

    st.markdown("""
    <div style="
    background:white;
    padding:25px;
    border-radius:15px;
    box-shadow:0px 5px 15px rgba(0,0,0,.1);
    line-height:2;
    font-size:17px;
    ">

    <b>Step 1</b> &nbsp;&nbsp; Enter Latitude and Longitude <br><br>

    <b>Step 2</b> &nbsp;&nbsp; Download Live Satellite Image <br><br>

    <b>Step 3</b> &nbsp;&nbsp; Predict Forest Fire Risk using CNN <br><br>

    <b>Step 4</b> &nbsp;&nbsp; Retrieve Current Weather Information <br><br>

    <b>Step 5</b> &nbsp;&nbsp; Display Prediction Results and Weather Dashboard

    </div>
    """, unsafe_allow_html=True)
# ============================================================
# IMAGE PREDICTION
# ============================================================

elif page == "📷 Upload Image":

    st.title("📷 Forest Image Prediction")

    st.write("Upload a forest image for prediction.")

    uploaded = st.file_uploader(

        "Choose Image",

        type=["jpg","jpeg","png"]

    )

    if uploaded:

        image = Image.open(uploaded).convert("RGB")

        col1,col2 = st.columns(2)

        with col1:

            st.image(

                image,

                caption="Uploaded Image",

                use_container_width=True

            )

        with col2:

            st.info("""

### Image Prediction

Press the button below to analyze the image.

The CNN model will detect whether the image
contains fire risk or not.

            """)

            if st.button("🔥 Predict Fire Risk",use_container_width=True):

                with st.spinner("Predicting..."):

                    label, confidence = predict_image(image)

                st.markdown("---")

                if "Fire Risk" in label:

                    st.error(label)

                else:

                    st.success(label)

                st.progress(int(confidence))

                st.metric(

                    "Confidence",

                    f"{confidence:.2f}%"

                )

                if confidence >= 90:

                    st.success("Prediction Confidence : Excellent")

                elif confidence >= 75:

                    st.info("Prediction Confidence : Good")

                else:

                    st.warning("Prediction Confidence : Moderate")

                st.markdown("---")

                st.subheader("Prediction Summary")

                result = {

                    "Prediction":label,

                    "Confidence":f"{confidence:.2f}%",

                    "Model":"CNN",

                    "Image Size":f"{image.size[0]} x {image.size[1]}"

                }

                st.json(result)

                st.download_button(

                    label="⬇ Download Prediction",

                    data=str(result),

                    file_name="prediction_result.txt",

                    mime="text/plain",

                    use_container_width=True

                )

# ============================================================
# LIVE SATELLITE
# ============================================================

elif page == "🛰 Live Satellite":

    st.title("🛰 Live Satellite Fire Prediction")

    st.write("Enter latitude and longitude to download live satellite imagery.")

    col1, col2 = st.columns(2)

    with col1:
        lat = st.number_input(
            "Latitude",
            value=21.1702,
            format="%.6f"
        )

    with col2:
        lon = st.number_input(
            "Longitude",
            value=72.8311,
            format="%.6f"
        )

    st.markdown("")

    if st.button("🚀 Download & Predict", use_container_width=True):

        with st.spinner("Downloading satellite image and weather data..."):

            result = live_prediction(lat, lon)
            weather = get_weather(lat, lon)

        st.success("Prediction Completed Successfully")

        st.markdown("---")

        img_col, info_col = st.columns([1.5,1])

        with img_col:

            st.image(
                result["image_path"],
                caption="Live NASA GIBS Satellite Image",
                use_container_width=True
            )

        with info_col:

            confidence = result["confidence"] * 100
            
            label = result["predicted_label"].lower().strip()
            
            if label == "fire_risk":
                st.error("🔥 FIRE RISK DETECTED")
            
            elif label == "no_fire_risk":
                st.success("🌳 NO FIRE RISK")
            
            else:
                st.warning(f"Unknown Prediction: {label}")

            st.progress(int(confidence))

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

            st.metric(
                "NASA FIRMS Hotspots",
                result["firms_active_fire_hotspots_nearby"]
            )

            if result["firms_active_fire_hotspots_nearby"] > 0:

                st.error("🔥 Active Fire Hotspots Nearby")

            else:

                st.success("✅ No Active Fire Hotspots")

        st.markdown("---")

        # =====================================================
        # WEATHER
        # =====================================================

        st.subheader("🌦 Current Weather")

        if weather:

            st.markdown(
                f"""
                <div class="weather-card">

                <h2>{weather['city']}, {weather['country']}</h2>

                <h3>{weather['description']}</h3>

                </div>
                """,
                unsafe_allow_html=True
            )

            icon_url = (
                f"https://openweathermap.org/img/wn/"
                f"{weather['icon']}@2x.png"
            )

            w1, w2, w3, w4 = st.columns(4)

            with w1:

                st.image(icon_url, width=90)

                st.metric(
                    "🌡 Temperature",
                    f"{weather['temperature']} °C"
                )

                st.metric(
                    "🤒 Feels Like",
                    f"{weather['feels_like']} °C"
                )

            with w2:

                st.metric(
                    "💧 Humidity",
                    f"{weather['humidity']} %"
                )

                st.metric(
                    "📈 Pressure",
                    f"{weather['pressure']} hPa"
                )

            with w3:

                st.metric(
                    "🌬 Wind",
                    f"{weather['wind']} m/s"
                )

                st.metric(
                    "☁ Clouds",
                    f"{weather['clouds']} %"
                )

            with w4:

                st.metric(
                    "🌫 Visibility",
                    f"{weather['visibility']} km"
                )

                st.metric(
                    "🌅 Sunrise",
                    weather["sunrise"]
                )

                st.metric(
                    "🌇 Sunset",
                    weather["sunset"]
                )

        else:

            st.warning("Unable to fetch weather information.")

        st.markdown("---")

        # =====================================================
        # SUMMARY
        # =====================================================

        st.subheader("📋 Prediction Summary")

        summary = {

            "Latitude": lat,

            "Longitude": lon,

            "Prediction": result["predicted_label"],

            "Confidence": f"{confidence:.2f}%",

            "Fire Hotspots":
                result["firms_active_fire_hotspots_nearby"],

            "Temperature":
                weather["temperature"] if weather else "N/A",

            "Humidity":
                weather["humidity"] if weather else "N/A",

            "Wind":
                weather["wind"] if weather else "N/A",

            "Weather":
                weather["description"] if weather else "N/A"

        }

        st.json(summary)

        st.download_button(

            "⬇ Download Report",

            data=str(summary),

            file_name="forest_fire_prediction.txt",

            mime="text/plain",

            use_container_width=True

        )

# ============================================================
# ABOUT
# ============================================================

else:

    st.title("ℹ About Project")

    st.markdown("""

## 🔥 Forest Fire Prediction System

This project predicts forest fire risk using Deep Learning
and live satellite imagery.

### Technologies

- TensorFlow
- Streamlit
- CNN
- NASA GIBS
- NASA FIRMS
- OpenWeather API

### Features

- Upload Forest Image
- Live Satellite Prediction
- Weather Forecast
- Fire Hotspot Detection
- Confidence Score
- Modern Dashboard

### Dataset

Satellite Forest Fire Dataset

""")

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    c1.metric("Framework", "TensorFlow")
    c2.metric("Frontend", "Streamlit")
    c3.metric("Backend", "NASA + OpenWeather")

    st.success("Thank you for using the Forest Fire Prediction System.")

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <hr>

    <div class="footer">

    Developed with ❤️ by <b>Prashant Patil</b>

    <br>

    Forest Fire Prediction using CNN + NASA GIBS +
    NASA FIRMS + OpenWeather API

    </div>
    """,
    unsafe_allow_html=True
)

