import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import pandas as pd
import joblib
import os

class FaytonWeatherModel:
    def __init__(self, n_estimators=200, max_depth=15, golden_ratio=1.618):
        self.classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        self.regressor = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        self.scaler = StandardScaler()
        self.golden_ratio = golden_ratio
        self.is_fitted = False

    def add_uncertainty(self, data, uncertainty_level=0.1):
        uncertainty = np.random.normal(0, uncertainty_level * self.golden_ratio, size=data.shape)
        return data + uncertainty

    def generate_synthetic_weather_data(self, n_samples=10000):
        np.random.seed(42)
        temperature = np.random.normal(20, 10, n_samples)
        humidity = np.random.uniform(30, 100, n_samples)
        pressure = np.random.normal(1013, 10, n_samples)
        wind_speed = np.random.uniform(0, 30, n_samples)
        visibility = np.random.uniform(5, 20, n_samples)
        X = np.column_stack((temperature, humidity, pressure, wind_speed, visibility))
        
        conditions = []
        for t, h, p, w in zip(temperature, humidity, pressure, wind_speed):
            if t > 25 and h < 60:
                conditions.append('Sunny')
            elif h > 80 or w > 20:
                conditions.append('Storm')
            elif h > 70:
                conditions.append('Rainy')
            else:
                conditions.append('Cloudy')
        y_class = np.array(conditions)
        y_temp = temperature + np.random.normal(0, 2, n_samples)
        return X, y_class, y_temp

    def prepare_data(self, X, y_class, y_temp=None):
        X = self.add_uncertainty(X)
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y_class, y_temp

    def fit(self, X, y_class, y_temp=None):
        X_scaled, y_class, y_temp = self.prepare_data(X, y_class, y_temp)
        self.classifier.fit(X_scaled, y_class)
        if y_temp is not None:
            self.regressor.fit(X_scaled, y_temp)
        self.is_fitted = True
        print('FaytonWeatherModel trained with Golden Ratio uncertainty!')

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError('Model not fitted yet!')
        X = self.add_uncertainty(X)
        X_scaled = self.scaler.transform(X)
        class_pred = self.classifier.predict(X_scaled)
        temp_pred = self.regressor.predict(X_scaled)
        return class_pred, temp_pred

    def evaluate(self, X_test, y_class_test, y_temp_test=None):
        class_pred, temp_pred = self.predict(X_test)
        acc = accuracy_score(y_class_test, class_pred)
        print(f'Weather Classification Accuracy: {acc:.4f}')
        if y_temp_test is not None:
            mse = mean_squared_error(y_temp_test, temp_pred)
            print(f'Temperature Forecast MSE: {mse:.4f}')
        return acc

    def save_model(self, path='fayton_weather_model.joblib'):
        if not self.is_fitted:
            print('Model not trained yet!')
            return
        joblib.dump(self, path)
        print(f'Model saved to {path}')

    def load_model(self, path='fayton_weather_model.joblib'):
        if os.path.exists(path):
            loaded = joblib.load(path)
            self.__dict__.update(loaded.__dict__)
            print(f'Model loaded from {path}')
        else:
            print('No saved model found.')

# Test
if __name__ == "__main__":
    model = FaytonWeatherModel()
    X, y_class, y_temp = model.generate_synthetic_weather_data(5000)
    X_train, X_test, y_class_train, y_class_test, y_temp_train, y_temp_test = train_test_split(X, y_class, y_temp, test_size=0.2, random_state=42)
    model.fit(X_train, y_class_train, y_temp_train)
    model.evaluate(X_test, y_class_test, y_temp_test)
    model.save_model()