# Fitbit Calorie Burn & Workout Pattern Analysis 🏃‍♂️📊

A comprehensive machine learning pipeline that predicts caloric expenditure and discovers hidden physiological patterns using Fitbit sensor data.

## 🚀 Performance Highlights
| Metric | Value | Status |
| :--- | :--- | :--- |
| **Random Forest R²** | **0.9979** | ✅ Verified |
| **Mean Absolute Error** | **3.78 kcal** | ✅ Verified |
| **Silhouette Score** | **0.4546** | ✅ Verified (4 Clusters) |

## 🧠 Project Overview
This project solves two primary challenges in fitness data science:
1. **Regression:** Predicts calories burned per session. While Linear Regression provided a baseline (MAE: 34.97), the **Random Forest** model captured the non-linear relationship between heart rate and intensity, reducing error by nearly 90%.
2. **Clustering:** Using **PCA** and **K-Means**, I identified 4 distinct workout "zones" (Resting, Light Aerobic, Steady State, and Anaerobic Threshold) with a high Silhouette Score of 0.4546.

## 🛠️ Technical Stack
- **Languages:** Python (Pandas, NumPy)
- **Visualization:** Seaborn, Matplotlib
- **ML Framework:** Scikit-Learn (RandomForestRegressor, KMeans, PCA, StandardScaler)
- **Research:** Exploratory EML-inspired symbolic regression for "God Function" discovery link:
https://arxiv.org/html/2603.21852v2#:~:text=Here%20we%20show%20that%20the,to%20exact%20closed-form%20expressions./
.

## 🧪 Key Insights
- **The Intensity Acceleration:** Calorie burn does not increase linearly with BPM; it accelerates. The Random Forest successfully mapped this curve.
- **Feature Priority:** Sensitivity analysis confirmed that **Avg_BPM** and **Weight** are the strongest predictors, while **Height** has a negligible impact (<0.5 kcal) once BMI is considered.
- **Cluster Optimization:** 4 clusters provided the most mathematically distinct separation of metabolic states compared to 3 or 5.

## 📂 Dataset
The analysis was performed on the [Fitbit Dataset](Fitbit_dataset.csv), containing 14,102 workout records with features like HR_Intensity, BMI, and Workout_Type.
