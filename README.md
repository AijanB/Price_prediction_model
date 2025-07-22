# 🏠 Real Estate Price Estimator (Bishkek, Kyrgyzstan 🇰🇬)

This app predicts apartment prices in Bishkek using three different machine learning models: **CatBoost**, **Decision Tree**, and **Linear Regression**.

### 🔮 Features
- Predict apartment price based on features like:
  - Location (latitude, longitude)
  - Total area, floor, ceiling height, build year
  - Room count, condition, heating, building type
  - Document quality and series type
- Provides **±2σ** and **95% confidence intervals** for each prediction
- Interactive UI built with **Gradio**
- Visual bar chart comparison of all model predictions

---

### 🚀 How to Run

```bash
pip install -r requirements.txt
python app.py
```

Runs at: `http://localhost:7860`

---

### 📐 Project Architecture

```mermaid
flowchart TD
    A[User Input via Gradio UI] --> B[Feature Engineering]
    B --> C[Geo Clustering with HDBSCAN]
    C --> D[Preprocessing (Encoding, Scaling)]
    D --> E[Log-Transformed Target]
    E --> F[Trained Models]
    F --> G[Prediction + Confidence Intervals]
    G --> H[Gradio Output: Text & Plot]
```

#### 🔄 Steps:
1. **User Input via Gradio**  
   Users input property features via a friendly web interface.

2. **Feature Engineering**  
   Derived features like `log_total_area`, `area_per_room`, and `is_condition_unknown` are computed.

3. **Clustering**  
   Geographic coordinates are scaled and passed to a trained **HDBSCAN** model to assign each input to a location-based cluster.

4. **Preprocessing**  
   - Categorical: OrdinalEncoder  
   - Numerical: StandardScaler  
   Combined using a `ColumnTransformer`.

5. **Model Prediction (log scale)**  
   - Target variable is log-transformed during training.
   - Predictions are inverse-transformed with `np.expm1`.

6. **Prediction Output**  
   Each model returns:
   - Point estimate
   - ±2 standard deviations interval
   - 95% confidence interval using the **k-nearest neighbors** approach

---

### 📏 Confidence Interval Estimation

To evaluate prediction **uncertainty**, we use a **K-Nearest Neighbors (KNN) in latent space** method:

1. For a new input sample, compute its distance to all training points (after preprocessing).
2. Select the `k=20` nearest neighbors based on Euclidean distance.
3. Use their log prices to compute:
   - **±2σ interval**:  
     \[ [\mu - 2\sigma, \mu + 2\sigma] \]
   - **95% Confidence Interval**:  
     \[ [\mu - 2\sigma/\sqrt{k}, \mu + 2\sigma/\sqrt{k}] \]
4. Intervals are then inverse-transformed from log scale using `np.expm1`.

This approach provides **uncertainty-aware predictions** without requiring probabilistic models.

---

### 📊 Output Example

- **CatBoost**:  
  `$84,600  
  ±2σ: $65,100 — $104,900  
  95% CI: $74,200 — $94,300`

- **Comparison Plot**:  
  Bar chart comparing price predictions across models.

---

### 🧠 Models
- `CatBoostRegressor` (native)
- `DecisionTreeRegressor` (sklearn pipeline)
- `LinearRegression` (sklearn pipeline)

---

### 🧰 Tech Stack
- Python, Gradio, CatBoost, scikit-learn, HDBSCAN, Matplotlib
