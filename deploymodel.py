import numpy as np
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import lightgbm
from model import X

app = Flask(__name__)

# Load your model
model_filename = 'flight_delay_prediction_model_lightgbm_bayesian.pkl'
loaded_model = joblib.load(model_filename)

# Load the scaler
scaler_filename = 'scaleroptimize.pkl'
scaler = joblib.load(scaler_filename)

# Preprocessing and adjustment functions
def preprocess_sample_data(data, scaler=None):
    # Define distance bins and labels
    distance_bins = [0, 500, 1000, 1500, 2000, 2500, np.inf]
    distance_labels = ['0-500 km', '501-1000 km', '1001-1500 km', '1501-2000 km', '2001-2500 km', '2500+ km']

    # Simplify weather categories
    simplified_weather = {
        'Partly cloudy': 'Cloudy',
        'Cloudy': 'Cloudy',
        'Sunny': 'Sunny',
        'Patchy rain possible': 'Rain',
        'Clear': 'Clear',
        'Thundery outbreaks possible': 'Storm',
        'Light rain shower': 'Rain',
        'Moderate or heavy rain shower': 'Rain',
        'Overcast': 'Cloudy',
        'Showers': 'Rain',
        'Heavy rain': 'Rain',
        'Thunderstorm': 'Storm',
        'Drizzle': 'Rain',
        'Fog': 'Cloudy',
        'Mist': 'Cloudy',
        'Snow': 'Storm',
        'Sleet': 'Storm',
        'Hail': 'Storm',
        'Light freezing rain': 'Storm',
        'Freezing rain': 'Storm',
        'Ice pellets': 'Storm',
        'Light snow': 'Storm',
        'Moderate snow': 'Storm',
        'Heavy snow': 'Storm',
        'Blizzard': 'Storm',
        'Dust': 'Clear',  # Typically not significant in terms of weather impact
        'Sand': 'Clear',  # Similarly, often considered clear with reduced visibility
        'Smoke': 'Cloudy',  # Can obscure visibility and create hazy conditions
        'Volcanic ash': 'Storm',  # Can be severe and impact travel significantly
        'Squalls': 'Storm',  # Intense bursts of wind and precipitation
        'Gusts': 'Storm',  # Strong wind that can accompany storms
        'High winds': 'Storm',  # Can cause dangerous conditions
    }

    # Convert sample data to DataFrame
    df = pd.DataFrame([data])

    # Create distance categories
    df['Distance_Category'] = pd.cut(df['Distance'], bins=distance_bins, labels=distance_labels, right=False)
    df = df.drop(['Distance'], axis=1)

    # Simplify weather categories
    df['Simplified_Weather'] = df['Simplified_Weather'].map(simplified_weather).fillna('Unknown')

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=['Airline', 'Simplified_Weather', 'Distance_Category', 'From', 'To'])

    # Ensure all columns are present in the training set
    for col in X.columns:
        if col not in df.columns:
            df[col] = 0
    df = df[X.columns]

    # Feature scaling
    if scaler:
        df_scaled = scaler.transform(df)
    else:
        df_scaled = df  # If no scaler, return raw data

    return df_scaled

def adjust_delay_based_on_weather(predicted_delay, weather_condition):
    weather_adjustments = {
        'Sunny': 0,
        'Clear': 0,
        'Cloudy': 5,
        'Rain': 15,
        'Storm': 30,
        'Unknown': 5
    }
    adjustment = weather_adjustments.get(weather_condition, 5)
    return predicted_delay + adjustment

def adjust_delay_based_on_airport_rating(adjusted_delay, airport_rating):
    if airport_rating < 0.5:
        return adjusted_delay + 10
    elif airport_rating < 0.7:
        return adjusted_delay + 5
    return adjusted_delay

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    preprocessed_data = preprocess_sample_data(data, scaler=scaler)
    predicted_delay = loaded_model.predict(preprocessed_data)[0]
    adjusted_delay = adjust_delay_based_on_weather(predicted_delay, data['Simplified_Weather'])
    adjusted_delay = adjust_delay_based_on_airport_rating(adjusted_delay, data['Airport Rating'])
    return jsonify({'predicted_delay': adjusted_delay})

if __name__ == '__main__':
    # Run the server on all interfaces with port 5000
    app.run(host='0.0.0.0', port=5002, debug=True)
