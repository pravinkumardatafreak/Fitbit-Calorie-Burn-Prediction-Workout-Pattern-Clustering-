"""
Fitbit Model Training Script
Trains RandomForestRegressor for Calorie Prediction and PCA + KMeans for Workout Clustering.
Saves model objects for web server consumption.
"""

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def train_and_save_models():
    # 1. Load the dataset
    data_path = "Fitbit_dataset.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    print("Loading dataset...")
    df = pd.read_csv(data_path)

    # Clean Unnamed index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # 2. Preprocess Categorical Variables
    print("Preprocessing data...")
    # Binary encoding for Gender (Male: 0, Female: 1)
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

    # One-hot encode Workout_Type
    workout_cols = ['Workout_Type_Cardio', 'Workout_Type_HIIT', 'Workout_Type_Mixed', 'Workout_Type_Strength', 'Workout_Type_Yoga']
    df = pd.get_dummies(df, columns=['Workout_Type'])

    # Ensure all expected workout columns are present (fill missing with False/0)
    for col in workout_cols:
        if col not in df.columns:
            df[col] = 0

    # Convert boolean columns from get_dummies to integers (0 or 1) for ML model stability
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    # 3. Define Features (X) and Target (y)
    # Matching the notebook cell 13: drop target, Age, and Max_BPM
    X = df.drop(['Calories_Burned (kcal)', 'Age', 'Max_BPM'], axis=1)
    y = df['Calories_Burned (kcal)']

    # 4. Train-Test Split (matching notebook cell 14)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Fit StandardScaler
    print("Fitting Scaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Fit RandomForestRegressor (Calorie prediction)
    print("Training RandomForestRegressor model...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)

    # Calculate test metrics
    y_pred_rf = rf_model.predict(X_test_scaled)
    from sklearn.metrics import r2_score, mean_absolute_error
    r2 = r2_score(y_test, y_pred_rf)
    mae = mean_absolute_error(y_test, y_pred_rf)
    print(f"Random Forest - R2 Score: {r2:.4f}, MAE: {mae:.2f} kcal")

    # 7. PCA + K-Means (Workout pattern clustering)
    print("Fitting PCA and KMeans...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_train_scaled)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(X_pca)

    # Calculate silhouette score for verification
    from sklearn.metrics import silhouette_score
    sil_score = silhouette_score(X_pca, clusters)
    print(f"K-Means (4 Clusters) - Silhouette Score: {sil_score:.4f}")

    # 8. Save artifacts using pickle
    print("Saving artifacts to disk...")
    artifacts = {
        'scaler': scaler,
        'rf_model': rf_model,
        'pca': pca,
        'kmeans': kmeans,
        'feature_names': list(X.columns)
    }

    for name, obj in artifacts.items():
        filename = f"{name}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(obj, f)
        print(f"Saved: {filename}")

    print("Model training complete and all artifacts persisted successfully!")

if __name__ == "__main__":
    train_and_save_models()
