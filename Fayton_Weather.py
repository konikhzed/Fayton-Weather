"""
Fayton Weather Model - Improved for Weather Detection & Forecasting
Developed by konikhzed
GitHub: https://github.com/konikhzed/Fayton-Weather
License: MIT
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class FaytonWeatherModel(BaseEstimator, ClassifierMixin):
    """
    Improved Fayton Model specialized for Weather Detection and Forecasting
    Uses Golden Ratio for uncertainty modeling in meteorological data.
    """
    
    def __init__(self, uncertainty_level=0.1, adaptive_noise=True, random_state=42, task='classification'):
        self.φ = (1 + np.sqrt(5)) / 2
        self.φ_inv = 1 / self.φ
        self.φ_angle = 2 * np.pi * self.φ_inv
        
        self.uncertainty_level = uncertainty_level
        self.adaptive_noise = adaptive_noise
        self.random_state = random_state
        self.task = task  # 'classification' or 'regression'
        np.random.seed(random_state)
        
        self.creator = "konikhzed"
        self.model_signature = f"Fayton-Weather-v1.0-{self.creator}"
    
    def add_weather_uncertainty(self, X):
        """Add golden ratio-based uncertainty tailored for weather features"""
        noise = np.random.normal(0, self.uncertainty_level * self.φ_inv, X.shape)
        
        # Weather-specific: simulate temporal/spatial patterns
        n_samples, n_features = X.shape
        angles = np.arange(n_samples) * self.φ_angle
        spiral = np.sin(angles[:, None] * self.φ_inv)
        
        if self.adaptive_noise:
            feature_scales = np.std(X, axis=0) + 1e-8
            spiral = spiral * feature_scales * self.uncertainty_level * 0.5
            
        # Add pressure/temperature correlation simulation
        if n_features > 1:
            pressure_effect = X[:, 0] * 0.1 * self.φ_inv  # Assume first feature is pressure-like
            spiral += pressure_effect[:, None]
        
        return X + noise + spiral
    
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        X_uncertain = self.add_weather_uncertainty(X)
        
        if self.task == 'classification':
            self.model_ = RandomForestClassifier(n_estimators=150, random_state=self.random_state, max_depth=10)
        else:
            self.model_ = RandomForestRegressor(n_estimators=150, random_state=self.random_state, max_depth=10)
            
        self.model_.fit(X_uncertain, y)
        self.n_features_in_ = X.shape[1]
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return self.model_.predict(X)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        if self.task == 'classification':
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            return {'accuracy': acc, 'f1': f1}
        else:
            mse = mean_squared_error(y_test, y_pred)
            return {'mse': mse}
    
    def verify_attribution(self):
        print(f"### Fayton Weather Model by {self.creator} ###")
        print(f"### GitHub: https://github.com/konikhzed/Fayton-Weather ###")
        print(f"### Signature: {self.model_signature} ###\n")

def generate_synthetic_weather_data(n_samples=10000, n_features=8, random_state=42):
    """Generate synthetic weather data for testing"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=6,
        n_redundant=1,
        n_classes=4,  # e.g., Sunny, Rainy, Stormy, Cloudy
        random_state=random_state
    )
    
    # Rename features to weather context
    feature_names = ['Temperature', 'Humidity', 'Pressure', 'WindSpeed', 
                     'Visibility', 'CloudCover', 'Precipitation', 'UV_Index']
    
    df = pd.DataFrame(X, columns=feature_names)
    target_names = ['Sunny', 'Rainy', 'Storm', 'Cloudy']
    
    print(f"Synthetic Weather Dataset: {n_samples} samples, {n_features} features")
    print("Target classes:", target_names)
    return df, y, feature_names, target_names

def run_weather_pipeline(uncertainty_levels=[0.05, 0.1, 0.2]):
    print("🚀 Starting Fayton Weather Model Pipeline\n")
    
    X_df, y, feature_names, target_names = generate_synthetic_weather_data()
    X = X_df.values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    results = []
    for level in uncertainty_levels:
        print(f"\n🔧 Testing uncertainty level: {level}")
        model = FaytonWeatherModel(
            uncertainty_level=level,
            adaptive_noise=True,
            random_state=42,
            task='classification'
        )
        
        model.verify_attribution()
        model.fit(X_train, y_train)
        
        metrics = model.evaluate(X_test, y_test)
        results.append({
            'Uncertainty_Level': level,
            **metrics
        })
        
        print(f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
    
    results_df = pd.DataFrame(results)
    print("\n📊 Final Results:")
    print(results_df)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x='Uncertainty_Level', y='accuracy', marker='o')
    plt.title('Fayton Weather Model Performance')
    plt.xlabel('Uncertainty Level (Golden Ratio based)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig('fayton_weather_performance.png')
    plt.show()
    
    return results_df

if __name__ == "__main__":
    results = run_weather_pipeline()
    print("\n✅ Fayton Weather Model testing completed!")
    print("Model ready for real weather API integration (e.g., OpenWeatherMap).")
