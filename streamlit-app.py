# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
import pandas as pd
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from catboost import CatBoostClassifier, Pool
import joblib

# Path of the trained model and data
MODEL_PATH = "model/cat_heart_model.cbm"
DATA_PATH = "data/heart_statlog_cleveland_hungary_final.csv"

st.set_page_config(page_title="Heart Disease Predictor")


st.title("Heart Disease Predictor")
st.subheader("Shap Values for the following dataset")
st.write("""This app was trained on the Heart Disease Dataset Available here: https://www.kaggle.com/datasets/mexwell/heart-disease-dataset.
         The model takes the named factors as input, predicts the risk of a heart disease with a certain probability, then
          calculates the shap values (how much influence each one has on the result)
         """, )
st.write("Warning! This model is supposed to be an experiment and should be used only for educational purposes")

@st.cache_resource
def load_data():
    data = pd.read_csv(DATA_PATH)
    return data


def load_x_y(file_path):
    data = joblib.load(file_path)
    data.reset_index(drop=True, inplace=True)
    return data


def load_model():
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    return model


def calculate_shap(model, X_train, X_test):
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values_cat_train = explainer.shap_values(X_train)
    shap_values_cat_test = explainer.shap_values(X_test)
    return explainer, shap_values_cat_train, shap_values_cat_test


def plot_shap_values(model, explainer, shap_values_cat_train, shap_values_cat_test, X_test, X_train,index):
    # Visualize SHAP values for a specific customer
    fig, ax_2 = plt.subplots(figsize=(5, 5), dpi=200)
    shap.decision_plot(explainer.expected_value, shap_values_cat_test[index],
                       X_test.iloc[index], link="logit")
    st.pyplot(fig)
    plt.close()


def display_shap_summary(shap_values_cat_train, X_train):
    # Create the plot summarizing the SHAP values
    shap.summary_plot(shap_values_cat_train, X_train, plot_type="bar", plot_size=(12, 12))
    summary_fig, _ = plt.gcf(), plt.gca()
    st.pyplot(summary_fig)
    plt.close()


def display_shap_waterfall_plot(explainer, expected_value, shap_values, feature_names, max_display=20):
    # Create SHAP waterfall drawing
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    shap.plots._waterfall.waterfall_legacy(expected_value, shap_values, feature_names=feature_names,
                                           max_display=max_display, show=False)
    st.pyplot(fig)
    plt.close()


def display_summary(model, data, X_train, X_test):
    # Calculate SHAP values
    explainer, shap_values_cat_train, shap_values_cat_test = calculate_shap(model, X_train, X_test)

    # Summarize and visualize SHAP values
    display_shap_summary(shap_values_cat_train, X_train)


def plot_shap(model, data, X_train, X_test,index):
    # Calculate SHAP values
    explainer, shap_values_cat_train, shap_values_cat_test = calculate_shap(model, X_train, X_test)

    # Visualize SHAP values
    plot_shap_values(model, explainer, shap_values_cat_train, shap_values_cat_test, X_test, X_train,index)

    # Waterfall
    display_shap_waterfall_plot(explainer, explainer.expected_value, shap_values_cat_test[index],
                                feature_names=X_test.columns, max_display=20)

model = load_model()
data = load_data()

X_train = load_x_y("data/X_train.pkl")
X_test = load_x_y("data/X_test.pkl")
y_train = load_x_y("data/y_train.pkl")
y_test = load_x_y("data/y_test.pkl")

test_full = X_test
test_full["target"] = y_test



choice = st.radio("Make Your Choice:", ("Feature Importance in Prediction", "User-based SHAP", "Calculate the probability of a heart-disease"))

if choice == "Feature Importance in Prediction":
    display_summary(model, data, X_train=X_train, X_test=X_test)

elif choice == "User-based SHAP":
    st.subheader("Available pacients from Test Dataset:")
    st.write(test_full)
    index = st.number_input("Select a patient and press enter to apply",min_value=0,max_value=len(X_test))
    st.write(f"Selected Patient: {index} has data: \n",)
    st.write(test_full.iloc[index])
    plot_shap(model, data, X_train=X_train, X_test=X_test,index=index)

elif choice == "Calculate the probability of a heart-disease":
    genders = ("Female", "Male")      
    pain_types = ("Typical Angina", "Atypical Angina","Non-Anginal Pain","Asymptomatic")
    binary_choice = ("No","Yes")
    cardiogram = ("Normal","ST-T Wave Abnormality","Probable or definite ventricular hypertrophy by Estes\' Criteria")
    slope = ("Upsloping","Flat","Downsloping")

    age = st.number_input("Age", min_value=10, max_value=100, step=1,value=44)
    gender = st.selectbox("Gender:", genders) 
    chest_pain_type = st.selectbox("Chest Pain Type:", pain_types)
    resting_blood_pressure = st.number_input("Resting Blood Pressure:", min_value=0, max_value=300,step=1,value=150)
    serum_cholesterol = st.number_input("Serum Cholesterol:",min_value=0,max_value=700,step=1,value=288)
    fasting_blood_sugar = st.selectbox("Fasting Sugar > 120mg/dl:", binary_choice)
    cardiogram_results = st.selectbox("Resting Electrocardiogram Results:",cardiogram)
    maximum_hear_rate = st.number_input("Maximum Heart Rate Achieved:",min_value=0,max_value=250,step=1,value=150)
    exercise_angina = st.selectbox("Exercise Induced Angina:", binary_choice)
    oldpeak = st.number_input("Oldpeak:",min_value=-4.0,max_value=8.0,step=0.1,value=3.0)
    slope_of_peak_exercise = st.selectbox("Slope of peak exercise ST segment:",slope)

    confirm_button = st.button("Confirm")

    if confirm_button == True:

        new_pacient_data = pd.DataFrame({
                        "sex": [int(genders.index(gender))],
                        "age": [int(age)],
                        "chest pain type":[int(pain_types.index(chest_pain_type))],
                        "resting bp s":[int(resting_blood_pressure)],
                        "cholesterol":[int(serum_cholesterol)],
                        "fasting blood sugar":[int(binary_choice.index(fasting_blood_sugar))],
                        "resting ecg":[int(cardiogram.index(cardiogram_results))],
                        "max heart rate":[int(maximum_hear_rate)],
                        "exercise angina":[int(binary_choice.index(exercise_angina))],
                        "oldpeak":[int(oldpeak)],
                        "ST slope":[float(slope.index(slope_of_peak_exercise))]
                    })
        st.write(new_pacient_data)
        probability_for_disease = model.predict_proba(new_pacient_data)[:,1]
        st.subheader("Disease Probability: {:.2%}".format(probability_for_disease[0]))
    