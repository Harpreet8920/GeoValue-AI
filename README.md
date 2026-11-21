# 🏡 GeoValue AI
### Intelligent House Price Prediction and Real Estate Analytics System
LIVE - https://geovalue-ai-2026.streamlit.app/
GeoValue AI is a machine learning-powered web application designed to estimate residential property listing prices and provide real estate analytics for Bengaluru, India.

The system uses a trained CatBoost regression model to predict property prices based on property characteristics and locality information. It also provides interactive dashboards, similar property comparisons, and explainable AI visualizations using SHAP.

---

## 1. Project Overview

**GeoValue AI** aims to make residential property price estimation more accessible through machine learning and data analytics.

Users can enter property details, receive an estimated listing price, explore property market patterns, compare similar listings, and understand the features influencing model predictions.

### Key Objectives

- Develop a machine learning model for residential property price prediction.
- Analyze property listing patterns across Bengaluru localities.
- Identify relationships between property characteristics and listing prices.
- Provide interpretable model predictions using SHAP.
- Develop an interactive web application using Streamlit.

---

## 2. Key Features

| Feature | Description |
|---|---|
| House Price Prediction | Estimates property listing prices using CatBoost |
| Real Estate Dashboard | Displays dataset statistics and price distributions |
| Market Analytics | Explores locality-level listing patterns |
| Similar Properties | Finds comparable properties based on locality, BHK and area |
| Explainable AI | Visualizes feature contributions using SHAP |
| Prediction Uncertainty | Displays a calibrated prediction interval when available |

---

## 3. Technology Stack

**Programming Language**
- Python

**Machine Learning**
- CatBoost
- Scikit-learn

**Data Processing**
- Pandas
- NumPy

**Data Visualization**
- Plotly
- Matplotlib

**Explainable AI**
- SHAP

**Web Application**
- Streamlit

**Development Environment**
- Google Colab
- Visual Studio Code
- Git and GitHub

---

## 4. Dataset Information

**Dataset:** Bengaluru House Price Dataset

**Source:** [Kaggle – Bengaluru House Price Data](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data)

The original dataset contains 13,320 records and 9 columns.

### Dataset Attributes

| Attribute | Description |
|---|---|
| area_type | Type of property area |
| availability | Property availability information |
| location | Property locality |
| size | Property configuration, such as BHK or bedrooms |
| society | Housing society information |
| total_sqft | Total property area |
| bath | Number of bathrooms |
| balcony | Number of balconies |
| price | Property listing price in lakhs |

The dataset undergoes data cleaning, validation, preprocessing and feature engineering before model development.

**Important:** The dataset represents property listings, not independently verified property transactions.

---

## 5. Machine Learning Model

### Selected Model: CatBoost Regressor

CatBoost is used as the primary regression model for estimating residential property listing prices.

### Model Input Features

1. Location
2. Area Type
3. Availability Status
4. Total Square Feet
5. BHK
6. Bathrooms
7. Balconies
8. Area per BHK
9. Bathrooms per BHK
10. Balconies per BHK

### Target Variable

`price`

The model predicts property prices in Indian lakh rupees (₹ Lakhs).

---

## 6. Explainable AI — SHAP

GeoValue AI integrates SHAP (SHapley Additive exPlanations) to explain the behavior of the trained CatBoost model.

The Explainable AI module provides:

- Global feature importance.
- SHAP feature impact distribution.
- Feature contribution analysis.

SHAP helps users understand how model input features contribute to predictions. These explanations do not establish causal relationships.

---

## 7. Project Architecture

```text
                 DATASET
                    |
                    v
          Data Cleaning & Validation
                    |
                    v
            Feature Engineering
                    |
                    v
             Dataset Splitting
                    |
                    v
           CatBoost Model Training
                    |
                    v
          Model Evaluation & Tuning
                    |
                    v
             SHAP Explainability
                    |
                    v
         Prediction Uncertainty Analysis
                    |
                    v
             Model Export
                    |
                    v
           Streamlit Application
                    |
          +---------+----------+
          |         |          |
          v         v          v
       Dashboard  Prediction  Analytics
          |         |          |
          +---------+----------+
                    |
                    v
             Explainable AI
```

---

## 8. Project Structure

```text
GeoValue-AI/
│
├── app.py
├── requirements.txt
├── README.md
│
└── artifacts/
    │
    ├── model_config.json
    ├── localities.json
    ├── training_reference.csv
    ├── shap_reference.csv
    │
    ├── models/
    │   ├── geovalue_model.cbm
    │   ├── catboost_house_price.cbm
    │   └── lightgbm_house_price.joblib
    │
    ├── evaluation/
    │   ├── eda_summary.json
    │   ├── validation_model_comparison.csv
    │   ├── final_test_predictions.csv
    │   ├── final_test_metrics.csv
    │   ├── test_price_tier_evaluation.csv
    │   └── final_evaluation_report.json
    │
    ├── explainability/
    │   ├── individual_shap_explanation.csv
    │   └── global_shap_importance.csv
    │
    └── uncertainty/
        ├── calibration_residuals.csv
        ├── locality_calibration_scores.csv
        └── calibration_metadata.json
```

---

## 9. Installation and Setup

### Prerequisites

- Python 3.10 or later
- pip
- Visual Studio Code (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/GeoValue-AI.git
```

### Step 2: Navigate to the Project

```bash
cd GeoValue-AI
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the Application

```bash
streamlit run app.py
```

Open the application in your browser:

```text
http://localhost:8501
```

---

## 10. Application Modules

### Dashboard
Displays:
- Total training reference properties.
- Number of represented localities.
- Average and median listing prices.
- Property listing price distribution.
- Top localities by listing count.
- Median listing price by locality.

### House Price Prediction
Accepts property details and generates:
- Estimated property listing price.
- Predicted price per square foot.
- Property configuration summary.
- Calibrated prediction interval when available.

### Market Analytics
Provides:
- Locality filtering.
- Property area versus listing price analysis.
- Price distribution by BHK.
- Property listing records.

### Similar Properties
Finds comparable listings using:
- Locality.
- BHK configuration.
- Property area.

### Explainable AI
Provides SHAP-based global feature importance and feature impact visualizations.

### About GeoValue AI
Describes the project, technologies and dataset scope.

---

## 11. Limitations

- The current model is developed using Bengaluru residential property listing data.
- Predictions may be less reliable for localities that were not represented in the training data.
- Listing prices may differ from actual negotiated or completed transaction prices.
- The model's estimates depend on the quality and coverage of the original dataset.
- Prediction intervals should be interpreted in the context of the calibration methodology and its assumptions.
- The application does not replace professional property valuation.

---

## 12. Future Scope

Potential future enhancements include:

- Integration of verified, regularly updated real estate datasets.
- Geographic visualization of localities.
- Interactive property comparison dashboards.
- Additional machine learning model comparisons.
- Improved uncertainty estimation.
- Property recommendation functionality.
- Cloud deployment and scalable application infrastructure.

---

## 13. Author

**Project:** GeoValue AI  
**Domain:** Machine Learning | Real Estate Analytics | Explainable AI  
**Application Framework:** Streamlit  
**Primary ML Model:** CatBoost Regressor

---

## 14. License

This project is developed for academic and educational purposes. Dataset usage remains subject to the original dataset's license and terms.
