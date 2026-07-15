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

.main{
    background:#f4f7fb;
}

.block-container{
    padding-top:2rem;
}

.title{
    font-size:45px;
    font-weight:bold;
    color:#ff5722;
    text-align:center;
}

.subtitle{
    font-size:20px;
    color:gray;
    text-align:center;
}

.metric-card{
    background:white;
    border-radius:15px;
    padding:18px;
    text-align:center;
    box-shadow:0px 4px 10px rgba(0,0,0,.15);
    margin-bottom:15px;
}

.weather-card{
    background:linear-gradient(135deg,#2193b0,#6dd5ed);
    padding:20px;
    border-radius:20px;
    color:white;
}

.fire-card{
    background:linear-gradient(135deg,#ff512f,#dd2476);
    padding:20px;
    border-radius:20px;
    color:white;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:30px;
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

    st.markdown(
        """
        <div class="title">
            🔥 Forest Fire Risk Prediction System
        </div>

        <div class="subtitle">
            AI Powered Forest Fire Detection using CNN, NASA GIBS &
            OpenWeather API
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("🤖 Model", "CNN")
    c2.metric("📡 Satellite", "NASA GIBS")
    c3.metric("🌍 Weather", "OpenWeather")
    c4.metric("🎯 Accuracy", "94.21%")

    st.markdown("---")

    left, right = st.columns([2,1])

    with left:

        st.subheader("🌲 Project Overview")

        st.write("""

This application predicts forest fire risk using a Deep Learning CNN model.

### Main Features

✅ Forest Image Prediction

✅ Live Satellite Prediction

✅ Weather Information

✅ NASA FIRMS Fire Hotspots

✅ OpenWeather API

✅ Confidence Score

✅ Fire Risk Indicator

✅ Modern Dashboard

        """)

    with right:

        st.markdown(
            """
            <div class="metric-card">

            <h2>🔥 AI Fire Detection</h2>

            <h4>Satellite + Weather</h4>

            <p>
            Predict fire risk anywhere in the world using
            live satellite imagery.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.subheader("📌 System Workflow")

    st.write("""

1️⃣ Enter Latitude & Longitude

⬇

2️⃣ Download Live Satellite Image

⬇

3️⃣ CNN Predicts Fire Risk

⬇

4️⃣ Get Current Weather

⬇

5️⃣ Display Fire Risk + Weather Dashboard

""")

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

### Author

**Devendra Patil**

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

    Developed with ❤️ by <b>Devendra Patil</b>

    <br>

    Forest Fire Prediction using CNN + NASA GIBS +
    NASA FIRMS + OpenWeather API

    </div>
    """,
    unsafe_allow_html=True
)

