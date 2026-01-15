import streamlit as st
import pandas as pd
import joblib

model = joblib.load('titanic_model.pkl')

st.title("🚢 Titanic Survival Predictor")
st.write("Enter the passanger details below to check their survival chance")

pclass = st.selectbox("Passenger Class",[1,2,3], format_func=lambda x: f"Class {x}")
sex = st.selectbox("Gender",["Male","Female"])
fare =  st.slider("Ticket Fare($)", 0, 500, 30)
sibsp = st.slider("Siblings/Spouses Aboard", 0, 8, 0)
parch = st.slider("Parents/Children Aboard", 0, 8, 0)

input_data = pd.DataFrame({
    'Pclass':[pclass],
    'Sex':[sex],
    'SibSp':[sibsp],
    'Parch':[parch],
    'Fare':[fare]
})

if st.button("Predict Survival"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    survival_chance = probability[0][1] * 100

    if prediction[0] == 1:
        st.success(f"🎉 This passenger likely SURVIVED! (Chance: {survival_chance:.1f}%)")
    else:
        st.error(f"💀 This passenger likely DIED. (Chance: {survival_chance:.1f}%)")