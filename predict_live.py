"""
End-to-end inference: pull a LIVE satellite image for a lat/lon, run the
trained CNN on it, and cross-check against NASA FIRMS active-fire hotspots.

Usage:
    python predict_live.py --lat 34.2 --lon -118.1
    python predict_live.py --lat 34.2 --lon -118.1 --date 2026-07-10
"""

import argparse
import numpy as np
import tensorflow as tf

import config
from fetch_live_satellite import get_live_sample


def preprocess_image(pil_image):
    img = pil_image.resize(config.IMAGE_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    return np.expand_dims(arr, axis=0)  # model has its own Rescaling layer


def predict(lat, lon, date=None, model_path=None):
    model_path = model_path or config.MODEL_PATH
    model = tf.keras.models.load_model(model_path)

    sample = get_live_sample(lat, lon, date=date)
    x = preprocess_image(sample["image"])
    pred = model.predict(x, verbose=0)

    if config.USE_MULTICLASS:
        class_idx = int(np.argmax(pred[0]))
        confidence = float(pred[0][class_idx])
        label = config.CLASS_NAMES_MULTI[class_idx]
    else:
        prob = float(pred[0][0])

# Keras class_names = ['fire_risk', 'no_fire_risk']
        if prob > 0.5:
           label = "no_fire_risk"
           confidence = prob
        else:
            label = "fire_risk"
            confidence = 1 - prob

    result = {
        "location": (lat, lon),
        "date": date or "latest available",
        "predicted_label": label,
        "confidence": round(confidence, 4),
        "image_path": sample["image_path"],
        "firms_active_fire_hotspots_nearby": sample["hotspot_count"],
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Live forest-fire risk prediction")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--date", type=str, default=None,
                         help="YYYY-MM-DD, defaults to yesterday's GIBS imagery")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    result = predict(args.lat, args.lon, date=args.date, model_path=args.model)

    print("\n=== Forest Fire Risk Prediction (Live) ===")
    for k, v in result.items():
        print(f"{k:35s}: {v}")

    if result["firms_active_fire_hotspots_nearby"] > 0 and \
       result["predicted_label"] in (config.CLASS_NAMES_BINARY[0],):
        print("\n[note] FIRMS detected active fire nearby but the model "
              "predicted low risk — worth reviewing the image/model.")


if __name__ == "__main__":
    main()
