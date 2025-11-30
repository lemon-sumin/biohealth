import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError


   # 🔥 딥러닝 모델 로드용

# -----------------------------------------------------
# Load Models and Scalers
# -----------------------------------------------------
rf_model = joblib.load("/Users/lemon/Desktop/immune_system/rf2_model.pkl")
nn_model = load_model(
    "/Users/lemon/Desktop/immune_system/nn2_model.h5",
    custom_objects={"mse": MeanSquaredError()}
)
   # 🔥 변경
mlp_model =joblib.load("/Users/lemon/Desktop/immune_system/mlp2_model.pkl")   # 🔥 변경

scaler_X = joblib.load("/Users/lemon/Desktop/immune_system/scaler_X2.pkl")
scaler_y = joblib.load("/Users/lemon/Desktop/immune_system/scaler_y2.pkl")

# -----------------------------------------------------
# Feature Columns (14 features)
# -----------------------------------------------------
feature_cols = [
    "Age (years)", 
    "Gender (1=Male, 2=Female)",
    "Income-to-Poverty Ratio", 
    "Body Mass Index",
    "Height (cm)", 
    "Weight (kg)",
    "Average Sleep Hours",
    "Total Energy Intake (kcal)", 
    "Protein Intake (g)",
    "Cholesterol (mg)", 
    "Phosphorus (mg)",
    "Magnesium (mg)", 
    "Vitamin B6 (mg)",
    "Vitamin B12 (µg)"
]

mean_values = {
    "Total Energy Intake (kcal)": 2100,
    "Protein Intake (g)": 70,
    "Magnesium (mg)": 260,
    "Phosphorus (mg)": 1200,
    "Vitamin B6 (mg)": 1.5,
    "Vitamin B12 (µg)": 4.5,
    "Cholesterol (mg)": 180
}

# -----------------------------------------------------
# Lifestyle Penalty
# -----------------------------------------------------
def compute_lifestyle_penalty(bmi, sleep, energy_kcal, protein_g, activity_level="moderate"):
    factor = 1.0

    if sleep < 5: factor -= 0.20
    elif sleep < 7: factor -= 0.10

    if bmi >= 35: factor -= 0.25
    elif bmi >= 30: factor -= 0.15
    elif bmi >= 27: factor -= 0.10

    if energy_kcal < 1500 or energy_kcal > 3200:
        factor -= 0.10

    if protein_g < 50: factor -= 0.10
    elif protein_g > 200: factor -= 0.05

    if activity_level == "low": factor -= 0.15
    elif activity_level == "high": factor += 0.05

    return round(min(1.0, max(0.4, factor)), 3)

# -----------------------------------------------------
# Immunity Score
# -----------------------------------------------------
def score_range(value, low, high):
    if value < low:
        return max(0, (value / low) * 70)
    elif value > high:
        return 100
    else:
        return 70 + ((value - low) / (high - low)) * 30

def compute_immunity_score(wbc, hgb, mcv, gender, age, lifestyle_factor):

    hgb_low, hgb_high = (13, 17) if gender == 1 else (12, 15)

    s_wbc = score_range(wbc, 4, 11)
    s_hgb = score_range(hgb, hgb_low, hgb_high)
    s_mcv = score_range(mcv, 80, 100)

    base = 0.40*s_wbc + 0.35*s_hgb + 0.25*s_mcv

    if age < 50:
        age_factor = 1.0
        decades = (age - 50) // 10
        age_factor = max(0.70, 1 - decades * 0.045)

    return base * age_factor * lifestyle_factor

# -----------------------------------------------------
# Plot Blood Markers
# -----------------------------------------------------
def plot_blood_position(wbc, hgb, mcv, gender):
    hgb_low, hgb_high = (13, 17) if gender == 1 else (12, 15)

    fig, ax = plt.subplots(figsize=(6,4))

    markers = ["WBC", "HGB", "MCV"]
    values = [wbc, hgb, mcv]
    lows = [4, hgb_low, 80]
    highs = [11, hgb_high, 100]

    colors = ["green" if lows[i] <= values[i] <= highs[i] else "red" for i in range(3)]

    ax.bar(markers, values, color=colors)
    ax.plot(markers, lows, "k--")
    ax.plot(markers, highs, "k--")

    st.pyplot(fig)

# -----------------------------------------------------
# Streamlit UI
# -----------------------------------------------------
st.set_page_config(page_title="AI Immunity Score (3-Model)", page_icon="🧬")
st.title("🧬 AI Immunity Score – Multi-Model Comparison")

st.subheader("👤 Basic Inputs")

age = st.number_input("Age", 1, 100, 25)
gender = st.selectbox("Gender", [1,2])
height = st.number_input("Height (cm)", 100, 220, 170)
weight = st.number_input("Weight (kg)", 30, 200, 70)
bmi = st.number_input("BMI", 10.0, 50.0, 22.0)
sleep = st.number_input("Sleep Hours", 0.0, 15.0, 7.0)

income_choice = st.selectbox("Income Level",
                             ["Low Income", "Middle Income", "Comfortable", "High Income"])

income_map = {"Low Income":1.0, "Middle Income":2.5, "Comfortable":4.0, "High Income":5.0}
income_ratio = income_map[income_choice]

with st.expander("📦 Nutrient Inputs"):
    nutrients = {col: st.number_input(col, value=float(val)) for col, val in mean_values.items()}

# -----------------------------------------------------
# Prepare Input Vector
# -----------------------------------------------------
X_input = np.array([
    age, gender, income_ratio, bmi, height, weight, sleep,
    nutrients["Total Energy Intake (kcal)"],
    nutrients["Protein Intake (g)"],
    nutrients["Cholesterol (mg)"],
    nutrients["Phosphorus (mg)"],
    nutrients["Magnesium (mg)"],
    nutrients["Vitamin B6 (mg)"],
    nutrients["Vitamin B12 (µg)"]
]).reshape(1, -1)

# -----------------------------------------------------
# Predict
# -----------------------------------------------------
if st.button("🔮 Predict with All Models"):

    X_scaled = scaler_X.transform(X_input)

    # Predict 3 models
    rf_pred = scaler_y.inverse_transform(rf_model.predict(X_scaled))[0]
    nn_pred = scaler_y.inverse_transform(nn_model.predict(X_scaled))[0]
    mlp_pred = scaler_y.inverse_transform(mlp_model.predict(X_scaled))[0]

    # Lifestyle factor
    life_factor = compute_lifestyle_penalty(
        bmi=bmi, sleep=sleep,
        energy_kcal=nutrients["Total Energy Intake (kcal)"],
        protein_g=nutrients["Protein Intake (g)"],
    )

    rf_score = compute_immunity_score(*rf_pred, gender, age, life_factor)
    nn_score = compute_immunity_score(*nn_pred, gender, age, life_factor)
    mlp_score = compute_immunity_score(*mlp_pred, gender, age, life_factor)

    # Comparison Table
    compare_df = pd.DataFrame({
        "Model": ["Random Forest", "Neural Network", "MLP"],
        "WBC": [rf_pred[0], nn_pred[0], mlp_pred[0]],
        "HGB": [rf_pred[1], nn_pred[1], mlp_pred[1]],
        "MCV": [rf_pred[2], nn_pred[2], mlp_pred[2]],
        "Immunity Score": [rf_score, nn_score, mlp_score]
    })

    st.table(compare_df)

    st.write("### 🩸 Blood Marker Range (Random Forest)")
    plot_blood_position(*rf_pred, gender)

# #     st.success("3-Model 비교 완료! 🎉")
# import streamlit as st
# import numpy as np
# import pandas as pd
# import joblib
# import matplotlib.pyplot as plt

# # -----------------------------------------------------
# # Load Models and Scalers (NO TENSORFLOW)
# # -----------------------------------------------------
# rf_model = joblib.load("/Users/lemon/Desktop/immune_system/rf2_model.pkl")
# mlp_model = joblib.load("/Users/lemon/Desktop/immune_system/mlp2_model.pkl")

# scaler_X = joblib.load("/Users/lemon/Desktop/immune_system/scaler_X2.pkl")
# scaler_y = joblib.load("/Users/lemon/Desktop/immune_system/scaler_y2.pkl")

# # -----------------------------------------------------
# # Feature Columns
# # -----------------------------------------------------
# feature_cols = [
#     "Age (years)", "Gender (1=Male, 2=Female)", "Income-to-Poverty Ratio",
#     "Body Mass Index", "Height (cm)", "Weight (kg)", "Average Sleep Hours",
#     "Total Energy Intake (kcal)", "Protein Intake (g)", "Cholesterol (mg)",
#     "Phosphorus (mg)", "Magnesium (mg)", "Vitamin B6 (mg)", "Vitamin B12 (µg)"
# ]

# mean_values = {
#     "Total Energy Intake (kcal)": 2100,
#     "Protein Intake (g)": 70,
#     "Magnesium (mg)": 260,
#     "Phosphorus (mg)": 1200,
#     "Vitamin B6 (mg)": 1.5,
#     "Vitamin B12 (µg)": 4.5,
#     "Cholesterol (mg)": 180
# }

# # -----------------------------------------------------
# # Lifestyle Penalty
# # -----------------------------------------------------
# def compute_lifestyle_penalty(bmi, sleep, energy_kcal, protein_g, activity_level="moderate"):
#     factor = 1.0

#     if sleep < 5: factor -= 0.20
#     elif sleep < 7: factor -= 0.10

#     if bmi >= 35: factor -= 0.25
#     elif bmi >= 30: factor -= 0.15
#     elif bmi >= 27: factor -= 0.10

#     if energy_kcal < 1500 or energy_kcal > 3200:
#         factor -= 0.10

#     if protein_g < 50: factor -= 0.10
#     elif protein_g > 200: factor -= 0.05

#     if activity_level == "low": factor -= 0.15
#     elif activity_level == "high": factor += 0.05

#     return round(min(1.0, max(0.4, factor)), 3)

# # -----------------------------------------------------
# # Immunity Score
# # -----------------------------------------------------
# def score_range(value, low, high):
#     if value < low:
#         return max(0, (value / low) * 70)
#     elif value > high:
#         return 100
#     else:
#         return 70 + ((value - low) / (high - low)) * 30

# def compute_immunity_score(wbc, hgb, mcv, gender, age, lifestyle_factor):
#     hgb_low, hgb_high = (13, 17) if gender == 1 else (12, 15)

#     s_wbc = score_range(wbc, 4, 11)
#     s_hgb = score_range(hgb, hgb_low, hgb_high)
#     s_mcv = score_range(mcv, 80, 100)

#     base = (0.40 * s_wbc) + (0.35 * s_hgb) + (0.25 * s_mcv)

#     if age < 50:
#         age_factor = 1.0
#     else:
#         decades = (age - 50) // 10
#         age_factor = max(0.70, 1 - decades * 0.045)

#     return base * age_factor * lifestyle_factor

# # -----------------------------------------------------
# # Plot Blood Markers
# # -----------------------------------------------------
# def plot_blood_position(wbc, hgb, mcv, gender):
#     hgb_low, hgb_high = (13, 17) if gender == 1 else (12, 15)

#     fig, ax = plt.subplots(figsize=(6,4))
#     markers = ["WBC", "HGB", "MCV"]
#     values = [wbc, hgb, mcv]
#     lows = [4, hgb_low, 80]
#     highs = [11, hgb_high, 100]

#     colors = ["green" if lows[i] <= values[i] <= highs[i] else "red" for i in range(3)]

#     ax.bar(markers, values, color=colors)
#     ax.plot(markers, lows, "k--")
#     ax.plot(markers, highs, "k--")

#     st.pyplot(fig)

# # -----------------------------------------------------
# # Streamlit UI
# # -----------------------------------------------------
# st.set_page_config(page_title="AI Immunity Score (2-Model)", page_icon="🧬")
# st.title("🧬 AI Immunity Score – RF & MLP Comparison")

# st.subheader("👤 Basic Inputs")

# age = st.number_input("Age", 1, 100, 25)
# gender = st.selectbox("Gender", [1, 2])
# height = st.number_input("Height (cm)", 100, 220, 170)
# weight = st.number_input("Weight (kg)", 30, 200, 70)
# bmi = st.number_input("BMI", 10.0, 50.0, 22.0)
# sleep = st.number_input("Sleep Hours", 0.0, 15.0, 7.0)

# income_choice = st.selectbox("Income Level",
#                              ["Low Income", "Middle Income", "Comfortable", "High Income"])
# income_map = {"Low Income": 1.0, "Middle Income": 2.5, "Comfortable": 4.0, "High Income": 5.0}
# income_ratio = income_map[income_choice]

# with st.expander("📦 Nutrient Inputs"):
#     nutrients = {col: st.number_input(col, value=float(val)) for col, val in mean_values.items()}

# # -----------------------------------------------------
# # Prepare Input Vector
# # -----------------------------------------------------
# X_input = np.array([
#     age, gender, income_ratio, bmi, height, weight, sleep,
#     nutrients["Total Energy Intake (kcal)"],
#     nutrients["Protein Intake (g)"],
#     nutrients["Cholesterol (mg)"],
#     nutrients["Phosphorus (mg)"],
#     nutrients["Magnesium (mg)"],
#     nutrients["Vitamin B6 (mg)"],
#     nutrients["Vitamin B12 (µg)"]
# ]).reshape(1, -1)

# # -----------------------------------------------------
# # Predict
# # -----------------------------------------------------
# if st.button("🔮 Predict with RF and MLP"):

#     X_scaled = scaler_X.transform(X_input)

#     # Predict
#     rf_pred = scaler_y.inverse_transform(rf_model.predict(X_scaled))[0]
#     mlp_pred = scaler_y.inverse_transform(mlp_model.predict(X_scaled))[0]

#     # Lifestyle factor
#     lf = compute_lifestyle_penalty(
#         bmi=bmi,
#         sleep=sleep,
#         energy_kcal=nutrients["Total Energy Intake (kcal)"],
#         protein_g=nutrients["Protein Intake (g)"]
#     )

#     rf_score = compute_immunity_score(*rf_pred, gender, age, lf)
#     mlp_score = compute_immunity_score(*mlp_pred, gender, age, lf)

#     # Comparison Table
#     compare_df = pd.DataFrame({
#         "Model": ["Random Forest", "MLP"],
#         "WBC": [rf_pred[0], mlp_pred[0]],
#         "HGB": [rf_pred[1], mlp_pred[1]],
#         "MCV": [rf_pred[2], mlp_pred[2]],
#         "Immunity Score": [rf_score, mlp_score]
#     })

#     st.table(compare_df)

#     st.write("### 🩸 Blood Marker Range (Random Forest)")
#     plot_blood_position(*rf_pred, gender)

#     st.success("Model comparison complete! 🎉")
