import gradio as gr
import pandas as pd
import numpy as np
import joblib
import traceback
from hdbscan.prediction import approximate_predict
from catboost import CatBoostRegressor, Pool
import matplotlib.pyplot as plt

# ======== Feature Engineering ========
def create_other_features(df):
    X = df.copy()
    X['log_total_area'] = np.log1p(X['total_area'])
    X['area_per_room'] = X['total_area'] / X['rooms']
    X.loc[X['rooms'] == 0, 'area_per_room'] = 0
    X['is_condition_unknown'] = (X['condition'] == 'unknown').astype(int)
    return X

# ======== Plotting Function ========
def make_plot(cat, dt, lr):
    fig, ax = plt.subplots()
    ax.bar(["CatBoost", "Decision Tree", "Linear"], [cat, dt, lr], color=["#AEC6CF", "#C5E384", "#FFB347"])
    ax.set_ylabel("Predicted Price (USD)")
    ax.set_title("Model Comparison")
    ax.grid(True, linestyle="--", alpha=0.5)
    return fig

# ======== Load Models ========
catboost_model = joblib.load("final_catboost_model.pkl")
dt_model = joblib.load("decision_tree_pipeline.pkl")
lr_model = joblib.load("linear_pipeline.pkl")
scaler = joblib.load("scaler_latlon.pkl")
clusterer = joblib.load("hdbscan_model..pkl")

# ======== Prediction Function ========
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

        coords_scaled = scaler.transform(df[['lat', 'lon']])
        df["geo_cluster"] = approximate_predict(clusterer, coords_scaled)[0]

        df = create_other_features(df)

        # CatBoost Prediction (log scale → USD)
        pool = Pool(df, cat_features=[
            "heating", "condition", "series", "building_type", "doc_quality", "geo_cluster", "is_condition_unknown"
        ])
        cat_pred = np.expm1(catboost_model.predict(pool)[0])

        # Decision Tree Prediction (already in USD)
        dt_pred = dt_model.predict(df)[0]

        # Linear Regression Prediction (log scale → USD)
        lr_log = lr_model.predict(df)[0]
        lr_pred = np.expm1(lr_log)
        if not np.isfinite(lr_pred):
            lr_pred = -1

        fig = make_plot(cat_pred, dt_pred, lr_pred)
        return f"${cat_pred:,.0f}", f"${dt_pred:,.0f}", f"${lr_pred:,.0f}", fig

    except Exception:
        return "❌", "❌", "❌", None

# ======== Gradio Interface ========
heating_options = ['central', 'gas', 'unknown', 'autonomous', 'electric', 'mixed', 'solid_fuel', 'no_heating']
condition_options = ['shell_condition', 'euro', 'unknown', 'good', 'average', 'not_finished']
series_options = ['Elite', 'Soviet', 'Individual', 'Economy', 'Stalinka', 'Penthouse']
building_type_options = ['brick', 'panel', 'monolith']
doc_quality_options = ['full', 'no', 'share', 'sales_contract']

interface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Slider(minimum=42.8, maximum=42.95, step=0.0001, label="Latitude"),
        gr.Slider(minimum=74.48, maximum=74.7, step=0.0001, label="Longitude"),
        gr.Dropdown(heating_options, label="Heating"),
        gr.Dropdown(condition_options, label="Condition"),
        gr.Dropdown(series_options, label="Series"),
        gr.Dropdown(building_type_options, label="Building Type"),
        gr.Dropdown(doc_quality_options, label="Document Quality"),
        gr.Slider(minimum=1, maximum=10, step=1, label="Number of Rooms"),
        gr.Slider(minimum=10, maximum=1000, step=1, label="Total Area (sq.m)"),
        gr.Slider(minimum=1, maximum=25, step=1, label="Floor"),
        gr.Slider(minimum=2.0, maximum=5.0, step=0.1, label="Ceiling Height (m)"),
        gr.Slider(minimum=1950, maximum=2025, step=1, label="Build Year")
    ],
    outputs=[
        gr.Textbox(label="CatBoost Prediction"),
        gr.Textbox(label="Decision Tree Prediction"),
        gr.Textbox(label="Linear Regression Prediction"),
        gr.Plot(label="Prediction Comparison")
    ],
    title="🏠 Real Estate Price Estimator (Bishkek)",
    description="""
Predict apartment prices in Bishkek using advanced machine learning models (CatBoost, Decision Tree, and Linear Regression).
Adjust the parameters below to see how each feature influences the price.
""",
    article="""
<p style='text-align: center'>
  <a href="https://huggingface.co/spaces/AijanB/Price_prediction" target="_blank">🌐 View on Hugging Face</a>
</p>
"""
)

if __name__ == "__main__":
    interface.launch()
