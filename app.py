import streamlit as st
import tensorflow
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.image import resize
import joblib
import pandas as pd
import sys

from Text_Helper import Text
txt = Text()

# Load your model
cnn_model = tensorflow.keras.models.load_model("My_CNN_Model")
xgb_model = joblib.load("My_XGB_Model")

st.sidebar.image("Image/sidebar_image.png")
st.sidebar.markdown("# Navigation")
tabs = st.sidebar.selectbox("", ("Home🏠",
                             "Image Prediction🖼️",
                             "Early Stroke Prediction💹",
                             "Papers📃",
                             "About❓")
                            )


#Home Page
if tabs == "Home🏠":
    st.title("Brain Stroke Prediction")
    st.image("Image/home.jpg")
    st.write(txt.home_text)


#Brain CT Image Page
elif tabs == "Image Prediction🖼️":
    st.header("Image Prediction")
    # Define the prediction function
    def get_image_prediction(image, model):
        image_size = 224
        image = load_img(image, color_mode='grayscale',
                        target_size=(image_size, image_size),
                        interpolation='bilinear')
        image = img_to_array(image) / 255.0 
        image = resize(image, [image_size, image_size]) 
        image = tensorflow.expand_dims(image, axis=0)

        prediction = model.predict(image)
        if prediction[0, 0] > 0.5:
            return "Stroke"
        else:
            return "Normal"

    # Upload image
    image_file = st.file_uploader(label="")

    if st.button("Click here to Predict") and image_file is not None:
        result = get_image_prediction(image_file, cnn_model)
        if result is not None:
            if result == "Stroke":
                st.image('Image/stroke_yes.png', caption="")
                st.write("Prediction: Stroke")
            else:
                st.image('Image/stroke_no.png', caption="")
                st.write("Prediction: Normal")

#Early Stroke Prediction Page
elif tabs == "Early Stroke Prediction💹":
    st.header("Input for Early Stroke Prediction")

    # User inputs
    age = st.slider("Select Your Age", 0, 120)
    
    gender = st.selectbox("Select Gender", options=["Male", "Female", "Others"])
    if gender == "Female":
        gender = 0
    elif gender == "Male":
        gender = 1
    elif gender == "Others":
        gender = 2

    hypertension = 1 if st.selectbox(label="Hypertension Value", options=["Yes", "No"]) == "Yes" else 0

    heart_disease = st.number_input("Heart Disease Value") 
    ever_married = 0 if st.selectbox(label="Marrige Stutas", options=["No", "Yes"]) == "No" else 1
    work_type = st.selectbox(label="Work Type", options=["Private", "Self Employed", "Children", "Govt Job", "Never Worked"])
    if work_type == "Govt Job":
        work_type = 0
    elif work_type == "Never Worked":
        work_type = 1
    elif work_type == "Private":
        work_type = 2
    elif work_type == "Self Employed":
        work_type = 3
    elif work_type == "Children":
        work_type = 4

    residence_type = 0 if st.selectbox(label="Resident Type", options=["Rural", "Urban"]) == "Rural" else 1
    avg_glucose_level = st.number_input("Average Glucose Level")
    bmi = st.number_input("BMI")
    smoking_status = st.selectbox(label="Smoking Stutas", options=["Never Smoked", "Formerly Smoked", "Reguler Smokes", "Smokes Sometimes"])
    if smoking_status == "Never Smoked":
        smoking_status = 3
    elif smoking_status == "Smokes Sometimes":
        smoking_status = 0
    elif smoking_status == "Reguler Smokes":
        smoking_status = 2
    elif smoking_status == "Formerly Smoked":
        smoking_status = 1


    def get_data(gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status):
        data = pd.DataFrame([[
        gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status
        ]])
        return data
    
    
    if st.button("Predict Stroke"):
        result = get_data(gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status)
        if age == 0:
            st.markdown("## Please Give Input")
            sys.exit(0)
        
        prediction = xgb_model.predict(result)

        if prediction == 0:
            st.header("There is No Change of Brain Stroke")
            st.image("Image/Health tips for Yes.png")
        elif prediction == 1:
            st.header("There is a Chance of Brain Stroke..Please Be Careful")
            st.image("Image/Health tips for Yes.png")

#Papers Page
elif tabs == "Papers📃":
    st.header("Papers")
    st.write(txt.paper_text)
    


#About Page
elif tabs == "About❓":
    st.header("About")
    st.write(txt.about_text)


