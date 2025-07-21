
import gradio as gr
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin

class LogToPriceWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, np.log1p(y))
        return self

    def predict(self, X):
        return np.expm1(self.model.predict(X))
        
# === Feature engineering function ===
def create_other_features(df):
    X = df.copy()
    X['log_total_area'] = np.log1p(X['total_area'])
    X['area_per_room'] = X['total_area'] / X['rooms']
    X.loc[X['rooms'] == 0, 'area_per_room'] = 0
    for col in ['condition']:
        X[f'is_{col}_unknown'] = (X[col] == 'unknown').astype(int)
    return X


# === Load models and preprocessors ===
catboost_model = joblib.load("final_catboost_model.pkl")
dt_model = joblib.load("decision_tree_pipeline.pkl")
lr_model = joblib.load("linear_pipeline..pkl")
scaler = joblib.load("scaler_latlon.pkl")
clusterer = joblib.load("hdbscan_model..pkl")
feature_engineer = joblib.load("feature_engineer.pkl")


# === Prediction function ===
def predict_price(lat, lon, heating, condition, series, building_type, doc_quality, rooms, total_area, floor, ceiling_height, build_year):
    try:
        input_dict = {
            "lat": [lat],
            "lon": [lon],
            "heating": [heating],
            "condition": [condition],
            "series": [series],
            "building_type": [building_type],
            "doc_quality": [doc_quality],
            "rooms": [rooms],
            "total_area": [total_area],
            "floor": [floor],
            "ceiling_height": [ceiling_height],
            "build_year": [build_year]
        }

        df = pd.DataFrame(input_dict)

        # Predict geo_cluster
        coords_scaled = scaler.transform(df[['lat', 'lon']])
        df["geo_cluster"] = clusterer.predict(coords_scaled)

        # Create other features
        df = create_other_features(df)

        # Apply saved feature transformation
        df = feature_engineer.transform(df)

        # Predict
        cat_pred = np.expm1(catboost_model.predict(df))[0]
        dt_pred = dt_model.predict(df)[0]
        lr_pred = lr_model.predict(df)[0]

        return {
            "CatBoost ($)": f"${cat_pred:,.0f}",
            "Decision Tree ($)": f"${dt_pred:,.0f}",
            "Linear Regression ($)": f"${lr_pred:,.0f}"
        }
    except Exception as e:
        import traceback
        return f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"

# === Interface ===
heating_options = ['central', 'gas', 'unknown', 'autonomous', 'electric', 'mixed', 'solid_fuel', 'no_heating']
condition_options = ['shell_condition', 'euro', 'unknown', 'good', 'average', 'not_finished']
series_options = ['Elite', 'Soviet', 'Individual', 'Economy', 'Stalinka', 'Penthouse']
building_type_options = ['brick', 'panel', 'monolith']
doc_quality_options = ['full', 'no', 'share', 'sales_contract']

interface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Slider(minimum=40.0, maximum=44.0, step=0.0001, label="Latitude"),
        gr.Slider(minimum=72.0, maximum=78.0, step=0.0001, label="Longitude"),
        gr.Dropdown(heating_options, label="Heating"),
        gr.Dropdown(condition_options, label="Condition"),
        gr.Dropdown(series_options, label="Series"),
        gr.Dropdown(building_type_options, label="Building Type"),
        gr.Dropdown(doc_quality_options, label="Document Quality"),
        gr.Slider(minimum=1, maximum=10, step=1, label="Number of Rooms"),
        gr.Slider(minimum=10, maximum=500, step=1, label="Total Area (sq.m)"),
        gr.Slider(minimum=1, maximum=50, step=1, label="Floor"),
        gr.Slider(minimum=2.0, maximum=5.0, step=0.1, label="Ceiling Height (m)"),
        gr.Slider(minimum=1950, maximum=2025, step=1, label="Build Year")
    ],
    outputs=[
        gr.Textbox(label="CatBoost Prediction"),
        gr.Textbox(label="Decision Tree Prediction"),
        gr.Textbox(label="Linear Regression Prediction")
    ],
    title="🏘️ Apartment Price Predictor",
    description="Enter apartment details to get price predictions from CatBoost, Decision Tree, and Linear Regression."
)

if __name__ == "__main__":
    interface.launch()
