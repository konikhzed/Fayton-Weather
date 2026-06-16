import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import requests
import json
from datetime import datetime

# Golden Ratio for uncertainty
PHI = (1 + np.sqrt(5)) / 2

def get_real_weather_data(latitude=35.6892, longitude=51.3890, days=7):
    '''Fetch real weather data from Open-Meteo API for Tehran'''
    url = 'https://api.open-meteo.com/v1/forecast'
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': 'temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,precipitation,cloud_cover',
        'forecast_days': days,
        'timezone': 'Asia/Tehran'
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print('API Error:', response.text)
        return None
    data = response.json()
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'time': pd.date_range(start=data['current']['time'], periods=len(data['hourly']['time']), freq='H'),
        'temperature': data['hourly']['temperature_2m'],
        'humidity': data['hourly']['relative_humidity_2m'],
        'pressure': data['hourly']['pressure_msl'],
        'wind_speed': data['hourly']['wind_speed_10m'],
        'precipitation': data['hourly']['precipitation'],
        'cloud_cover': data['hourly']['cloud_cover']
    })
    return df

def create_weather_labels(df):
    '''Create labels based on weather conditions'''
    df['weather_class'] = 'Sunny'
    df.loc[df['precipitation'] > 0.5, 'weather_class'] = 'Rainy'
    df.loc[(df['cloud_cover'] > 70) & (df['precipitation'] == 0), 'weather_class'] = 'Cloudy'
    df.loc[df['wind_speed'] > 15, 'weather_class'] = 'Stormy'
    return df

def add_golden_uncertainty(X, uncertainty_level=0.1):
    '''Add uncertainty based on Golden Ratio'''
    noise = np.random.normal(0, uncertainty_level * PHI, X.shape)
    return X + noise

def train_and_evaluate(df):
    features = ['temperature', 'humidity', 'pressure', 'wind_speed', 'precipitation', 'cloud_cover']
    X = df[features].values
    y = df['weather_class']
    
    # Add uncertainty
    X_noisy = add_golden_uncertainty(X, uncertainty_level=0.12)
    
    X_train, X_test, y_train, y_test = train_test_split(X_noisy, y, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f'\n=== Fayton Weather Model - Real Data Test ===')
    print(f'Accuracy: {accuracy:.4f}')
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))
    
    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(df['time'][:100], df['temperature'][:100], label='Temperature')
    plt.title('Real Weather Data - Tehran (Sample)')
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.savefig('tehran_real_weather.png')
    print('Plot saved: tehran_real_weather.png')
    return model, accuracy

# Main execution
if __name__ == "__main__":
    print('Fetching real weather data for Tehran...')
    df = get_real_weather_data()
    if df is not None:
        df = create_weather_labels(df)
        model, acc = train_and_evaluate(df)
        print(f'\nModel trained successfully with real data! Accuracy: {acc:.4f}')
    else:
        print('Failed to fetch data.')