from predict_live import predict


def live_prediction(lat, lon):
    """
    Wrapper function for Streamlit.
    Returns prediction result dictionary.
    """
    return predict(lat, lon)


if __name__ == "__main__":
    result = live_prediction(21.1702, 72.8311)

    print(result)