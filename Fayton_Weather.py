"""
Fayton Weather Model - Improved for Weather Detection & Forecasting
Developed by konikhzed with user-specific numbers and Golden Ratio
GitHub: https://github.com/konikhzed/Fayton-Weather
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import requests
import joblib
import warnings
warnings.filterwarnings('ignore')

class FaytonWeatherModel:
    def __init__(self, uncertainty_level=0.1, random_state=42):
        self.φ = (1 + np.sqrt(5)) / 2
        self.φ_inv = 1 / self.φ
        self.φ_angle = 2 * np.pi * self.φ_inv
        
        # User provided numbers
        self.user_numbers = np.array([16.3248, 8.1624, 5.4425, 4.0819, 3.2655, 2.7212, 2.3325, 2.0409, 1.8142])
        self.user_scales = self.user_numbers / self.user_numbers[0]
        self.angles = np.deg2rad(self.user_numbers[:6])
        
        self.uncertainty_level = uncertainty_level
        self.random_state = random_state
        np.random.seed(random_state)
        
    def add_weather_uncertainty(self, X):
        """Enhanced uncertainty with user numbers, angles and spiral modulation"""
        noise = np.random.normal(0, self.uncertainty_level * self.φ_inv, X.shape)
        
        n_samples, n_features = X.shape
        angles = np.arange(n_samples) * self.φ_angle
        spiral = np.sin(angles[:, None] * self.φ_inv + np.arange(n_features))
        
        user_mod = np.sin(self.angles[:n_features]) * np.mean(self.user_scales)
        spiral = spiral * user_mod.mean()
        
        if n_features > 1:
            pressure_effect = X[:, 0] * 0.1 * self.φ_inv * self.user_scales[0]
            spiral += pressure_effect[:, None]
        
        return X + noise + spiral
    
    def fit(self, X, y_class=None, y_temp=None):
        X_uncertain = self.add_weather_uncertainty(X)
        if y_class is not None:
            self.classifier_ = RandomForestClassifier(n_estimators=250, max_depth=15, random_state=self.random_state)
            self.classifier_.fit(X_uncertain, y_class)
        if y_temp is not None:
            self.regressor_ = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=self.random_state)
            self.regressor_.fit(X_uncertain, y_temp)
        return self
    
    def predict_class(self, X):
        return self.classifier_.predict(self.add_weather_uncertainty(X))
    
    def predict_temp(self, X):
        return self.regressor_.predict(self.add_weather_uncertainty(X))
    
    def evaluate(self, X_test, y_class_test=None, y_temp_test=None):
        results = {}
        if y_class_test is not None:
            y_pred = self.predict_class(X_test)
            results['accuracy'] = accuracy_score(y_class_test, y_pred)
            results['f1'] = f1_score(y_class_test, y_pred, average='weighted')
        if y_temp_test is not None:
            y_pred_temp = self.predict_temp(X_test)
            results['rmse'] = np.sqrt(mean_squared_error(y_temp_test, y_pred_temp))
        return results

def fetch_real_weather_data(days=7, city="Tehran"):
    try:
        coords = {
            "Tehran": (35.6892, 51.3890),
            "Mashhad": (36.26, 59.6167),
            "Shiraz": (29.61, 52.53),
            "Isfahan": (32.65, 51.67)
        }
        lat, lon = coords.get(city, (35.6892, 51.3890))
        
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,precipitation,cloud_cover",
            "forecast_days": min(days, 16), "timezone": "Asia/Tehran"
        }
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        hourly = data['hourly']
        df = pd.DataFrame({
            'Temperature': hourly['temperature_2m'],
            'Humidity': hourly['relative_humidity_2m'],
            'Pressure': hourly['pressure_msl'],
            'WindSpeed': hourly['wind_speed_10m'],
            'Precipitation': hourly['precipitation'],
            'CloudCover': hourly['cloud_cover']
        })
        print(f"✅ Fetched {len(df)} real hourly records for {city} ({days} days).")
        return df, city
    except Exception as e:
        print(f"⚠️ API error: {e}. Using synthetic fallback.")
        np.random.seed(42)
        n = days * 24
        df = pd.DataFrame({
            'Temperature': np.random.normal(25, 8, n),
            'Humidity': np.random.normal(50, 20, n),
            'Pressure': np.random.normal(1013, 10, n),
            'WindSpeed': np.random.normal(10, 5, n),
            'Precipitation': np.random.exponential(2, n),
            'CloudCover': np.random.uniform(0, 100, n)
        })
        return df, city

def create_labels(df):
    conditions = []
    for _, row in df.iterrows():
        if row['Precipitation'] > 2 or row['CloudCover'] > 80:
            conditions.append(2)  # Storm/Rainy
        elif row['CloudCover'] > 50:
            conditions.append(1)  # Cloudy
        elif row['WindSpeed'] > 20:
            conditions.append(3)  # Windy
        else:
            conditions.append(0)  # Sunny
    return np.array(conditions)

def run_full_test():
    print("🚀 Fayton Weather Model - Full Enhanced Test\n")
    results = {}
    for days in [3, 7, 14]:
        print(f"\n--- تست {days} روز ---")
        df, city = fetch_real_weather_data(days)
        X = df.values
        y_class = create_labels(df)
        y_temp = df['Temperature'].values
        
        X_train, X_test, y_class_train, y_class_test = train_test_split(X, y_class, test_size=0.25, random_state=42, stratify=y_class)
        _, _, y_temp_train, y_temp_test = train_test_split(X, y_temp, test_size=0.25, random_state=42)
        
        model = FaytonWeatherModel(uncertainty_level=0.12)
        model.fit(X_train, y_class_train, y_temp_train)
        
        metrics = model.evaluate(X_test, y_class_test, y_temp_test)
        print(f"Accuracy: {metrics.get('accuracy', 0):.4f} | RMSE Temp: {metrics.get('rmse', 0):.2f}°C")
        results[days] = metrics
    
    # Save model
    joblib.dump(model, "fayton_weather_model_full.pkl")
    print("\n✅ Model saved successfully!")
    return results

if __name__ == "__main__":
    run_full_test()
    print("\n🎉 تست کامل انجام شد. مدل بهبود یافته با Classification + Regression + Golden Ratio Uncertainty.")