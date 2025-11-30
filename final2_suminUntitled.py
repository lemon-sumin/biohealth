import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_METAL_ENABLE"] = "0"

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------------------------------------
# Load Models + Scalers
# -----------------------------------------------------
rf_model = joblib.load("rf2_model.pkl")
scaler_X = joblib.load("scaler_X2.pkl")
scaler_y = joblib.load("scaler_y2.pkl")

# -----------------------------------------------------
# Feature Columns 
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
def compute_lifestyle_penalty(bmi, sleep, energy_kcal, protein_g, activity_level):
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
# Immunity Score Calculation
# -----------------------------------------------------
def score_range(value, low, high):
    if value < low: return max(0, (value / low) * 70)
    elif value > high: return 100
    else: return 70 + ((value - low) / (high - low)) * 30

def compute_immunity_score(wbc, hgb, mcv, gender, age, lifestyle_factor):
    hgb_low, hgb_high = (13, 17) if gender == 1 else (12, 15)

    s_wbc = score_range(wbc, 4, 11)
    s_hgb = score_range(hgb, hgb_low, hgb_high)
    s_mcv = score_range(mcv, 80, 100)

    base = 0.50*s_wbc + 0.35*s_hgb + 0.15*s_mcv
    
    if age < 50:
        age_factor = 1.0
    else:
        decades = (age - 50) // 10
        # 나이 증가에 따라 면역 상태는 악화되므로, 점수는 감소해야 함
        age_factor = max(0.7, 1 - decades * 0.148)

    return base * age_factor * lifestyle_factor
    
    

   # if age < 50: age_factor = 1.0
   # else:
    #    decades = (age - 50) // 10
    #    age_factor = max(0.70, 1 - decades * 0.045)

    #return base * age_factor * lifestyle_factor

# -----------------------------------------------------
# Personalized Lifestyle Feedback
# -----------------------------------------------------
def lifestyle_feedback(bmi, sleep, energy, protein, activity):
    fb = []

    # Sleep
    if sleep < 5:
        fb.append("😴 **수면이 매우 부족합니다. 하루 6–8시간으로 늘리는 것을 추천합니다.**")
    elif sleep < 7:
        fb.append("😴 **수면이 약간 부족합니다. 조금만 더 자면 면역 개선에 도움이 됩니다.**")

    # BMI
    if bmi >= 35:
        fb.append("⚠️ **BMI가 매우 높습니다. 체중 관리가 시급합니다.**")
    elif bmi >= 30:
        fb.append("⚠️ **BMI가 높아 면역 기능에 악영향을 줄 수 있습니다.**")
    elif bmi >= 27:
        fb.append("📉 **BMI가 조금 높아 개선하면 더 건강해질 수 있습니다.**")

    # Activity
    if activity == "low":
        fb.append("🏃 **활동량이 너무 적습니다. 하루 30분만 움직여도 면역이 크게 좋아집니다.**")

    # Energy
    if energy < 1500:
        fb.append("🍽 **섭취 칼로리가 너무 부족합니다. 최소 1500 kcal 이상 권장합니다.**")
    elif energy > 3200:
        fb.append("🍽 **칼로리가 너무 많아요. 과다 섭취는 면역 기능에 악영향을 줍니다.**")

    # Protein
    if protein < 50:
        fb.append("🍗 **단백질이 부족합니다. 면역세포 생성에 필수예요.**")
    elif protein > 200:
        fb.append("🍗 **단백질 과다 섭취는 신장에 부담을 줄 수 있습니다.**")

    return fb if fb else ["👍 **생활습관이 전반적으로 매우 양호합니다!**"]

# -----------------------------------------------------
# Blood Marker Visualization
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
st.set_page_config(page_title="AI Immunity Score", page_icon="🧬")
st.title("🧬 AI Immunity Score – Random Forest Model")

st.subheader("👤 Basic Inputs")

age = st.number_input("Age", 1, 100, 25)

gender_label = st.selectbox("Gender", ["Male", "Female"])
gender = 1 if gender_label == "Male" else 2

height = st.number_input("Height (cm)", 100, 220, 170)
weight = st.number_input("Weight (kg)", 30, 200, 70)
bmi = st.number_input("BMI", 10.0, 50.0, 22.0)
sleep = st.number_input("Sleep Hours", 0.0, 15.0, 7.0)

income_choice = st.selectbox("Income Level", ["Low Income","Middle Income","Comfortable","High Income"])
income_map = {"Low Income":1.0,"Middle Income":2.5,"Comfortable":4.0,"High Income":5.0}
income_ratio = income_map[income_choice]

# Activity level (복원)
activity = st.selectbox("Activity Level", ["low", "moderate", "high"])

with st.expander("📦 Nutrient Inputs"):
    nutrients = {
        key: st.number_input(key, value=float(val))
        for key, val in mean_values.items()
    }

# -----------------------------------------------------
# Input Vector (순서 절대 변경 X)
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
# Prediction Section
# -----------------------------------------------------
if st.button("🔮 Predict"):

    X_scaled = scaler_X.transform(X_input)
    rf_scaled = rf_model.predict(X_scaled)
    rf_pred = scaler_y.inverse_transform(rf_scaled)[0]
    wbc, hgb, mcv = rf_pred

    life_factor = compute_lifestyle_penalty(
        bmi=bmi, sleep=sleep,
        energy_kcal=nutrients["Total Energy Intake (kcal)"],
        protein_g=nutrients["Protein Intake (g)"],
        activity_level=activity
    )

    immunity_score = compute_immunity_score(wbc, hgb, mcv, gender, age, life_factor)

    # -----------------------------------------------------
    # Final Score (크게 강조)
    # -----------------------------------------------------
    st.markdown(
        f"""
        <h2 style='text-align: center;'>🧬 <b>FINAL IMMUNITY SCORE</b> 🧬</h2>
        <h1 style='text-align: center; color:#4CAF50;'>
            <b>{immunity_score:.1f} / 100</b>
        </h1>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # Biomarker Table
    # -----------------------------------------------------
    st.subheader("🩸 Predicted Biomarkers")
    df = pd.DataFrame({
        "Biomarker": ["WBC", "Hemoglobin (HGB)", "MCV"],
        "Value": [wbc, hgb, mcv],
        "Normal Range": ["4–11", "13–17 (M) / 12–15 (F)", "80–100"]
    })
    st.table(df)

    st.write("### 🩸 Blood Marker Visualization")
    plot_blood_position(wbc, hgb, mcv, gender)

    # -----------------------------------------------------
    # Personalized Lifestyle Feedback
    # -----------------------------------------------------
    st.subheader("🏃 Personalized Lifestyle Suggestions")
    for fb in lifestyle_feedback(bmi, sleep, nutrients["Total Energy Intake (kcal)"], 
                                 nutrients["Protein Intake (g)"], activity):
        st.write("- " + fb)
