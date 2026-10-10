import json, joblib, pandas as pd, streamlit as st

st.set_page_config(page_title="MessMind", page_icon="🍽️", layout="wide")
model = joblib.load("models/best_model.joblib")
info = json.load(open("models/metrics.json"))
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]; MEALS = ["Breakfast", "Lunch", "Dinner"]

st.title("🍽️ MessMind – AI Food-Waste Predictor for College Mess")
st.caption("Predict how much food will be wasted at the next meal, so the kitchen can cook the right amount.")
tab1, tab2 = st.tabs(["Predict", "Model insights"])

with tab1:
    c1, c2 = st.columns(2)
    day = c1.selectbox("Day of week", DAYS); meal = c1.selectbox("Meal", MEALS)
    pop = c1.slider("Menu popularity (1 = disliked, 5 = favourite)", 1, 5, 3)
    reg = c1.number_input("Students registered", 300, 700, 480)
    temp = c2.slider("Temperature (°C)", 14, 44, 30)
    exam = c2.checkbox("Exam week"); event = c2.checkbox("Event / holiday")
    cost = c2.number_input("Food cost (₹ per kg)", 10, 200, 40)
    x = pd.DataFrame([[DAYS.index(day), MEALS.index(meal), pop, int(exam), int(event), temp, reg]], columns=info["features"])
    waste = max(float(model.predict(x)[0]), 0)
    st.metric("Predicted food waste", f"{waste:.1f} kg")
    a, b, c = st.columns(3)
    a.metric("Cook less by (approx.)", f"{waste * 0.8:.1f} kg")
    b.metric("Saving per meal", f"₹{waste * 0.8 * cost:,.0f}")
    c.metric("Saving per year (3 meals/day)", f"₹{waste * 0.8 * cost * 3 * 365:,.0f}")
    st.info("Tip: cooking ~80% of the predicted waste less keeps a safety buffer so no student goes hungry.")

with tab2:
    st.subheader(f"Best model: {info['best']}")
    st.dataframe(pd.DataFrame(info["results"]).T)
    cols = st.columns(3)
    for col, f in zip(cols, ["model_comparison", "actual_vs_predicted", "feature_importance"]):
        col.image(f"charts/{f}.png")
