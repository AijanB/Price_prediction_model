import gradio as gr
import pandas as pd
import numpy as np
import joblib
from hdbscan.prediction import approximate_predict
from catboost import Pool
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from scipy.spatial import distance

# === Load Models ===
catboost_model = joblib.load("final_catboost_model.pkl")
dt_model = joblib.load("decision_tree_pipeline.pkl")
lr_model = joblib.load("linear_pipeline.pkl")
scaler = joblib.load("scaler_latlon.pkl")
clusterer = joblib.load("hdbscan_model..pkl")

X_train = joblib.load("X_full_for_confidence.pkl")
y_train_log = joblib.load("y_full_for_confidence.pkl")

# === Feature Setup ===
categorical = ['heating', 'condition', 'series', 'building_type', 'doc_quality', 'geo_cluster', 'is_condition_unknown']
numeric = [col for col in X_train.columns if col not in categorical]

preprocessor = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical),
    ("num", StandardScaler(), numeric)
])
preprocessor.fit(X_train)

def create_other_features(df):
    df = df.copy()
    df['log_total_area'] = np.log1p(df['total_area'])
    df['area_per_room'] = df['total_area'] / df['rooms']
    df.loc[df['rooms'] == 0, 'area_per_room'] = 0
    df['is_condition_unknown'] = (df['condition'] == 'unknown').astype(int)
    return df

def get_confidence_interval_log_model_catboost(x_raw_df, X_train, y_train_log, model, cat_features, k=20):
    x_encoded = preprocessor.transform(x_raw_df)
    X_encoded = preprocessor.transform(X_train)
    dists = distance.cdist(X_encoded, x_encoded).flatten()
    nearest_idxs = np.argsort(dists)[:k]
    y_neighbors_log = y_train_log.iloc[nearest_idxs].values
    pool = Pool(x_raw_df, cat_features=cat_features)
    y_pred_log = model.predict(pool)[0]
    mean_log = y_neighbors_log.mean()
    std_log = y_neighbors_log.std()
    interval_pred_log = (y_pred_log - 2 * std_log, y_pred_log + 2 * std_log)
    interval_conf_log = (y_pred_log - 2 * std_log / np.sqrt(k), y_pred_log + 2 * std_log / np.sqrt(k))
    return {
        "prediction": np.expm1(y_pred_log),
        "interval_2sigma": tuple(np.expm1(interval_pred_log)),
        "interval_confidence": tuple(np.expm1(interval_conf_log))
    }

def get_confidence_interval_log_model_sklearn(x_raw, X_train, y_train_log, model, preprocessor, k=20):
    x_vec = preprocessor.transform(x_raw)
    X_vec = preprocessor.transform(X_train)
    dists = distance.cdist(X_vec, x_vec).flatten()
    nearest_idxs = np.argsort(dists)[:k]
    y_neighbors_log = y_train_log.iloc[nearest_idxs].values

    y_pred_log = model.predict(x_raw)[0]

    std_log = y_neighbors_log.std()

    interval_pred_log = (y_pred_log - 2 * std_log, y_pred_log + 2 * std_log)
    interval_conf_log = (y_pred_log - 2 * std_log / np.sqrt(k), y_pred_log + 2 * std_log / np.sqrt(k))

    return {
        "prediction": np.expm1(y_pred_log),
        "interval_2sigma": tuple(np.expm1(interval_pred_log)),
        "interval_confidence": tuple(np.expm1(interval_conf_log))
    }


def make_plot(cat, dt, lr):
    fig, ax = plt.subplots()
    ax.bar(["CatBoost", "Decision Tree", "Linear"], [cat, dt, lr], color=["#AEC6CF", "#C5E384", "#FFB347"])
    ax.set_ylabel("Predicted Price (USD)")
    ax.set_title("Model Comparison")
    ax.grid(True, linestyle="--", alpha=0.5)
    return fig

def predict_price(lat, lon, heating, condition, series, building_type, doc_quality, rooms, total_area, floor, ceiling_height, build_year):
    try:
        df = pd.DataFrame({
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
        })

        coords_scaled = scaler.transform(df[['lat', 'lon']])
        df["geo_cluster"] = approximate_predict(clusterer, coords_scaled)[0]
        df = create_other_features(df)

        cat_result = get_confidence_interval_log_model_catboost(df, X_train, y_train_log, catboost_model, categorical)
        dt_result = get_confidence_interval_log_model_sklearn(df, X_train, y_train_log, dt_model, preprocessor)
        lr_result = get_confidence_interval_log_model_sklearn(df, X_train, y_train_log, lr_model, preprocessor)

        cat_text = f"${round(cat_result['prediction']):,}\n±2σ: ${round(cat_result['interval_2sigma'][0]):,} — ${round(cat_result['interval_2sigma'][1]):,}\n95% CI: ${round(cat_result['interval_confidence'][0]):,} — ${round(cat_result['interval_confidence'][1]):,}"
        dt_text = f"${round(dt_result['prediction']):,}\n±2σ: ${round(dt_result['interval_2sigma'][0]):,} — ${round(dt_result['interval_2sigma'][1]):,}\n95% CI: ${round(dt_result['interval_confidence'][0]):,} — ${round(dt_result['interval_confidence'][1]):,}"
        lr_text = f"${round(lr_result['prediction']):,}\n±2σ: ${round(lr_result['interval_2sigma'][0]):,} — ${round(lr_result['interval_2sigma'][1]):,}\n95% CI: ${round(lr_result['interval_confidence'][0]):,} — ${round(lr_result['interval_confidence'][1]):,}"

        fig = make_plot(cat_result['prediction'], dt_result['prediction'], lr_result['prediction'])
        return cat_text, dt_text, lr_text, fig
    except Exception as e:
        import traceback
        return traceback.format_exc(), "❌", "❌", None

# Options
heating_options = ['central', 'gas', 'unknown', 'autonomous', 'electric', 'mixed', 'solid_fuel', 'no_heating']
condition_options = ['shell_condition', 'euro', 'unknown', 'good', 'average', 'not_finished']
series_options = ['Elite', 'Soviet', 'Individual', 'Economy', 'Stalinka', 'Penthouse']
building_type_options = ['brick', 'panel', 'monolith']
doc_quality_options = ['full', 'no', 'share', 'sales_contract']

# Interface
with gr.Blocks(css="""
.gradio-container {
    background: linear-gradient(to bottom right, #f9f9fc, #eaf2f8);
    font-family: 'Segoe UI', sans-serif;
    color: #2c2c2c;
}
.gr-button {
    background-color: #d0dce8 !important;
    color: #1f2a36 !important;
    border: 1px solid #a6b3c5 !important;
    box-shadow: 1px 1px 4px rgba(0,0,0,0.1);
    font-weight: bold;
}
.gr-input, .gr-textbox, .gr-slider, .gr-dropdown {
    border: 1px solid #c5cdd6;
    border-radius: 8px;
    padding: 8px;
    background-color: #ffffff;
    color: #1f2a36;
}
.gr-output {
    font-weight: bold;
    color: #1f2a36;
}
""") as demo:
    gr.Markdown("# 🏠 Real Estate Price Estimator (Bishkek)")
    gr.Markdown("💡 Predict apartment prices in Bishkek using <b>CatBoost</b>, <b>Decision Tree</b>, and <b>Linear Regression</b>.<br>📊 Each prediction includes a ±2σ and 95% confidence interval.")

    with gr.Row():
        with gr.Column():
            lat = gr.Slider(42.8, 42.95, step=0.0001, label="Latitude")
            lon = gr.Slider(74.48, 74.7, step=0.0001, label="Longitude")
            heating = gr.Dropdown(heating_options, label="Heating")
            condition = gr.Dropdown(condition_options, label="Condition")
            series = gr.Dropdown(series_options, label="Series")
            building_type = gr.Dropdown(building_type_options, label="Building Type")
            doc_quality = gr.Dropdown(doc_quality_options, label="Document Quality")
            rooms = gr.Slider(1, 10, step=1, label="Number of Rooms")
            total_area = gr.Slider(10, 1000, step=1, label="Total Area (sq.m)")
            floor = gr.Slider(1, 25, step=1, label="Floor")
            ceiling_height = gr.Slider(2.0, 5.0, step=0.1, label="Ceiling Height (m)")
            build_year = gr.Slider(1950, 2025, step=1, label="Build Year")
            button = gr.Button("Predict")

        with gr.Column():
            cat_text = gr.Textbox(label="CatBoost Prediction + CI")
            dt_text = gr.Textbox(label="Decision Tree Prediction + CI")
            lr_text = gr.Textbox(label="Linear Regression Prediction + CI")
            plot = gr.Plot(label="Prediction Comparison")

    button.click(
        fn=predict_price,
        inputs=[lat, lon, heating, condition, series, building_type, doc_quality, rooms, total_area, floor, ceiling_height, build_year],
        outputs=[cat_text, dt_text, lr_text, plot]
    )

demo.launch()
