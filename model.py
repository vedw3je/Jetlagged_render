import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt  # Importing matplotlib for plotting

# Load and preprocess data
data = pd.read_csv("Cleaned_Dataset.csv")
data.fillna(method='ffill', inplace=True)

data['Departure Delay'] = data['Departure Delay'].replace(-1, np.nan)
data['Arrival Delay'] = data['Arrival Delay'].replace(-1, np.nan)

# Drop rows where 'Arrival Delay' is NaN
data = data.dropna(subset=['Arrival Delay'])
data = data.dropna(subset=['Departure Delay'])

# Define distance bins and labels
distance_bins = [0, 500, 1000, 1500, 2000, 2500, np.inf]
distance_labels = ['0-500 km', '501-1000 km', '1001-1500 km', '1501-2000 km', '2001-2500 km', '2500+ km']

# Create distance categories
data['Distance_Category'] = pd.cut(data['Distance'], bins=distance_bins, labels=distance_labels, right=False)

# Drop the original Distance column
data = data.drop(['Distance'], axis=1)

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
    'Dust': 'Clear',
    'Sand': 'Clear',
    'Smoke': 'Cloudy',
    'Volcanic ash': 'Storm',
    'Squalls': 'Storm',
    'Gusts': 'Storm',
    'High winds': 'Storm',
}

data['Simplified_Weather'] = data['weather__hourly__weatherDesc__value'].map(simplified_weather)

# Drop unnecessary columns, including the specified weather columns
data.drop(columns='weather__hourly__weatherDesc__value', inplace=True)

# One-hot encode categorical columns including simplified weather and airline categories
data = pd.get_dummies(data, columns=['Airline', 'Simplified_Weather', 'Distance_Category', 'From', 'To'])

# Ensure that all columns in X are numeric
X = data.drop(['Arrival Delay'], axis=1)  # Features
X = X.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric

y = data['Arrival Delay']  # Target variable

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
scaler_filename = 'scaler.pkl'
joblib.dump(scaler, scaler_filename)
print(f"Scaler saved as {scaler_filename}")

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=20)

# Initialize and train the LightGBM model
model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=20)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error with LightGBM: {mse}")
print(f"R^2 Score with LightGBM: {r2}")

# Scatter Plot of Predictions vs Actual Values


# Residual Plot

# Export the trained model
model_filename = 'flight_delay_prediction_model_lightgbm.pkl'
joblib.dump(model, model_filename)
print(f"Model saved as {model_filename}")
