"""
FastAPI Backend for Fitbit Calorie Burn & Workout Clustering Web App
Serves ML predictions and PCA cluster calculations.
"""

import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 1. Initialize FastAPI
app = FastAPI(
    title="Fitbit Metabolic Analytics Engine",
    description="A production-ready API serving calorie burn regression and workout pattern clustering models.",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Define Request Schema
class WorkoutSession(BaseModel):
    gender: str = Field(..., example="Male", description="Gender: 'Male' or 'Female'")
    weight: float = Field(..., example=75.0, description="Weight in kilograms")
    height: float = Field(..., example=1.75, description="Height in meters")
    avg_bpm: float = Field(..., example=140.0, description="Average Heart Rate during session (BPM)")
    resting_bpm: float = Field(70.0, example=65.0, description="Resting Heart Rate (BPM)")
    session_duration: float = Field(..., example=1.0, description="Session Duration in hours (e.g. 0.75 for 45 mins)")
    fat_percentage: float = Field(20.0, example=18.5, description="Body fat percentage")
    water_intake: float = Field(3.0, example=2.5, description="Daily water intake in liters")
    workout_frequency: int = Field(3, example=4, description="Workout sessions per week")
    experience_level: int = Field(1, example=2, description="Experience Level: 0 (Beginner), 1 (Intermediate), 2 (Advanced)")
    workout_type: str = Field(..., example="Cardio", description="Workout Type: 'Cardio', 'HIIT', 'Mixed', 'Strength', 'Yoga'")
    hr_intensity: float = Field(0.7, example=0.75, description="Heart Rate Intensity ratio (0.0 to 1.0)")

# 3. Load ML Artifacts
MODELS_PATH = "."
scaler = None
rf_model = None
pca = None
kmeans = None
feature_names = None

@app.on_event("startup")
def load_models():
    global scaler, rf_model, pca, kmeans, feature_names
    try:
        with open(os.path.join(MODELS_PATH, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        with open(os.path.join(MODELS_PATH, "rf_model.pkl"), "rb") as f:
            rf_model = pickle.load(f)
        with open(os.path.join(MODELS_PATH, "pca.pkl"), "rb") as f:
            pca = pickle.load(f)
        with open(os.path.join(MODELS_PATH, "kmeans.pkl"), "rb") as f:
            kmeans = pickle.load(f)
        with open(os.path.join(MODELS_PATH, "feature_names.pkl"), "rb") as f:
            feature_names = pickle.load(f)
        print("All machine learning models and scalers loaded successfully!")
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        # Note: In production, we'd fail startup, but we'll raise an alert here.

# 4. Helper function to preprocess incoming requests into model features
def preprocess_input(session: WorkoutSession):
    # Calculate BMI
    bmi = session.weight / (session.height ** 2)

    # Base MET and Effective MET estimates based on intensity and workout type
    # These are defaults derived from standard exercise physiology
    base_met = 7.0 if session.workout_type in ["Cardio", "HIIT"] else 4.0
    effective_met = base_met * session.hr_intensity

    # Construct input dict matching columns of X
    user_data = {
        'Gender': 1 if session.gender.lower() == 'female' else 0,
        'Weight (kg)': session.weight,
        'Height (m)': session.height,
        'Avg_BPM': session.avg_bpm,
        'Resting_BPM': session.resting_bpm,
        'Session_Duration (hours)': session.session_duration,
        'Fat_Percentage': session.fat_percentage,
        'Water_Intake (liters)': session.water_intake,
        'Workout_Frequency (days/week)': session.workout_frequency,
        'Experience_Level': session.experience_level,
        'BMI': bmi,
        'Base_MET': base_met,
        'HR_Intensity': session.hr_intensity,
        'Effective_MET': effective_met,
        'Workout_Type_Cardio': 1 if session.workout_type == 'Cardio' else 0,
        'Workout_Type_HIIT': 1 if session.workout_type == 'HIIT' else 0,
        'Workout_Type_Mixed': 1 if session.workout_type == 'Mixed' else 0,
        'Workout_Type_Strength': 1 if session.workout_type == 'Strength' else 0,
        'Workout_Type_Yoga': 1 if session.workout_type == 'Yoga' else 0
    }

    # Align with training columns
    df_user = pd.DataFrame([user_data])
    df_user = df_user[feature_names]
    
    # Scale data
    user_scaled = scaler.transform(df_user)
    return user_scaled

# 5. Define Endpoints
@app.post("/api/predict", summary="Predict Calories Burned")
def predict_calories(session: WorkoutSession):
    if rf_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Calorie Burn prediction model is not loaded.")
    
    try:
        processed_data = preprocess_input(session)
        predicted_cal = rf_model.predict(processed_data)[0]
        return {
            "status": "success",
            "predicted_calories": float(np.round(predicted_cal, 2)),
            "unit": "kcal",
            "explanation": f"Based on your heart rate intensity ({session.hr_intensity * 100}%) and duration ({session.session_duration * 60} mins), your session burned approximately {predicted_cal:.1f} kcal."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/api/cluster", summary="Classify Workout Pattern and Metabolic Zone")
def cluster_workout(session: WorkoutSession):
    if kmeans is None or pca is None or scaler is None:
        raise HTTPException(status_code=503, detail="Clustering models are not loaded.")

    try:
        # Scale input
        processed_data = preprocess_input(session)
        
        # Project using PCA
        pca_coords = pca.transform(processed_data)[0]
        
        # Predict K-Means cluster
        cluster_id = int(kmeans.predict([pca_coords])[0])
        
        # Map clusters to physical zones (Based on our project findings)
        zones = {
            0: {
                "name": "Resting & Active Recovery",
                "description": "Very low intensity session. Focuses on restoring baseline energy systems and biological recovery.",
                "color": "#4caf50"
            },
            1: {
                "name": "Light Aerobic Zone",
                "description": "Fat burning aerobic threshold. Sustainable effort that burns fat as a primary fuel source.",
                "color": "#00bcd4"
            },
            2: {
                "name": "Steady State Cardio",
                "description": "Cardiovascular conditioning. High-capacity aerobic zone building lung and heart endurance.",
                "color": "#ff9800"
            },
            3: {
                "name": "Anaerobic Threshold / HIIT",
                "description": "Maximum metabolic conditioning (high lactic acid). Builds cellular speed and short-burst power.",
                "color": "#e91e63"
            }
        }
        
        return {
            "status": "success",
            "cluster_id": cluster_id,
            "pca_x": float(pca_coords[0]),
            "pca_y": float(pca_coords[1]),
            "zone": zones.get(cluster_id, {"name": "Unknown Zone", "description": "Undefined Metabolic State", "color": "#9e9e9e"})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering error: {str(e)}")

# Mount static frontend directory (created in next steps)
if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
else:
    # If static is not created yet, register a simple landing message
    @app.get("/")
    def index():
        return {"message": "Fitbit Metabolic Analytics Engine API is running. Go to /docs for Swagger documentation."}
