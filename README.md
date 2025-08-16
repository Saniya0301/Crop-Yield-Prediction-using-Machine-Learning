# 🌾 Crop Yield Prediction using Machine Learning

## 📌 Project Overview
This project aims to predict **crop yield (hg/ha)** based on environmental and agricultural features such as:
- Rainfall
- Pesticide usage
- Average temperature
- Crop type
- Country/Region
- Year  

The dataset contains crop yield data for multiple countries and crops over several years.  
Machine Learning models were trained and compared to build an efficient prediction system.

---
demo 
<img width="821" height="467" alt="{79788C2E-EFEB-40FB-B14C-28E39402EDC8}" src="https://github.com/user-attachments/assets/6ecbe742-5aad-45e6-8950-ba02cd4e7071" />

---

## 📂 Dataset
The dataset (`yield_df.csv`) contains the following columns:

| Column                          | Description |
|---------------------------------|-------------|
| **Area**                        | Country/Region |
| **Item**                        | Crop type (Maize, Wheat, Rice, etc.) |
| **Year**                        | Year of observation |
| **hg/ha_yield**                 | Crop yield (hectograms per hectare) |
| **average_rain_fall_mm_per_year** | Average rainfall per year (mm) |
| **pesticides_tonnes**           | Pesticide usage (tonnes) |
| **avg_temp**                    | Average temperature (°C) |

- **Rows:** ~25,000  
- **Countries:** 100+  
- **Crops:** 10 major crops  

---

## 🔑 Features
- **Data Preprocessing**
  - Removed duplicates
  - OneHotEncoding for categorical variables (`Area`, `Item`)
  - Standard Scaling for numerical features
- **Exploratory Data Analysis**
  - Correlation analysis
  - Crop-wise and Country-wise yield distribution
  - Visualization using Seaborn & Matplotlib
- **Machine Learning Models**
  - Linear Regression
  - Lasso Regression
  - Ridge Regression
  - Decision Tree Regressor 🌳
  - K-Nearest Neighbors (KNN)

---

## 📊 Model Performance
| Model               | MAE ↓ | R² Score ↑ |
|----------------------|---------|-------------|
| Linear Regression    | ~29897 | 0.747 |
| Lasso Regression     | ~29884 | 0.747 |
| Ridge Regression     | ~29853 | 0.747 |
| Decision Tree        | **~5697** | **0.9646** |
| KNN                  | **~4680** | **0.9846** |

✅ **KNN and Decision Tree performed the best** with very high accuracy.  

---

## 🚀 Prediction System
A predictive function was created to estimate crop yield:

```python
def prediction(Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item):
    features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
    transform_features = preprocesser.transform(features)
    predicted_yield = dtr.predict(transform_features).reshape(-1,1)
    return predicted_yield[0][0]

# Example
result = prediction(1990, 1485.0, 121.0, 16.37, 'Albania', 'Maize')
print(result)   # Output: 36613.0


---
🛠️ Tech Stack

Python

Pandas, NumPy

Seaborn, Matplotlib

Scikit-learn

Pickle

---

📈 Visualizations

Count plots of crops and countries

Bar plots of total yield by crop & country

Correlation heatmaps


---
📌 Future Work

Deploy model using Streamlit / Flask

Improve accuracy with Ensemble models (Random Forest, XGBoost)

Add real-time data fetching for prediction

---

👩‍💻 Author

Developed with ❤️ SANIYA CHHABRA
