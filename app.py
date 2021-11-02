import streamlit as st 
import pandas as pd
import numpy as np
import os
import pickle
import warnings


st.set_page_config(page_title="Heart disease prediction", page_icon="❤️", layout='centered', initial_sidebar_state="collapsed")

def load_model(modelfile):
	loaded_model = pickle.load(open(modelfile, 'rb'))
	return loaded_model

def main():
    # title
    html_temp = """
    <div>
    <h1 style="color:DarkRed;text-align:left;"> Heart disease prediction  ❤️ </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    col1,col2  = st.columns([2,2])
    
    with col1: 
        with st.beta_expander(" ℹ️ Information", expanded=True):
            st.write("""
            Detecting the risk of heart disease is essential for everyone's health. 31% of global mortality is due to cardiovascular disease. Being able to detect them before the disease occurs would allow better management of patients and increase the chances of survival.
            """)
        '''
        ## How does it work ❓ 
        Complete all the parameters and the machine learning model will predict the most suitable crops to grow in a particular farm based on various parameters
        '''


    with col2:
        df = pd.DataFrame()
        st.subheader(" Find out the most suitable crop to grow in your farm 👨‍🌾")
        df["Age"] = [st.number_input("Age", 1,10000)]
        df["RestingBP"] = [st.number_input("RestingBP", 1,10000)]
        df["Cholesterol"] = [st.number_input("Cholesterol",0.0,100000.0)]
        df["MaxHR"] = [st.number_input("MaxHR", 0.0,100000.0)]
        df["Oldpeak"] = [st.number_input("Oldpeak", 0.0,100000.0)]
        ChestPainType = st.selectbox('Chest pain type', ('Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'))
        RestingECG = st.selectbox('Resting electrocardiogram results', ('Normal', 'Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)', "Showing probable or definite left ventricular hypertrophy by Estes' criteria"))
        ExerciseAngina = st.radio('Exercise-induced angina', ('Yes', 'No'))
        ST_Slope = st.radio('The slope of the peak exercise ST segment', ('Upsloping', 'Flat', 'Downsloping'))
        Sex = st.radio('Sex', ('Male', 'Female'))

        df["FastingBS_0"] = 1

        df["ChestPainType_ASY"] = 1 if ChestPainType == 'Asymptomatic' else 0
        df["ChestPainType_ATA"] = 1 if ChestPainType == 'Atypical Angina' else 0
        df["ChestPainType_NAP"] = 1 if ChestPainType == 'Non-Anginal Pain' else 0
        df["ChestPainType_TA"] = 1 if ChestPainType == 'Typical Angina' else 0

        df["RestingECG_LVH"] = 1 if RestingECG == "Showing probable or definite left ventricular hypertrophy by Estes' criteria" else 0
        df["RestingECG_Normal"] = 1 if RestingECG == "Normal" else 0
        df["RestingECG_ST"] = 1 if RestingECG == "Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)" else 0

        df["ExerciseAngina_Y"] = 1 if ExerciseAngina == 'Yes' else 0
        df["ExerciseAngina_N"] = 1 if ExerciseAngina == 'No' else 0

        df["ST_Slope_Down"] = 1 if ST_Slope == "Downsloping" else 0
        df["ST_Slope_Flat"] = 1 if ST_Slope == "Flat" else 0
        df["ST_Slope_Up"] = 1 if ST_Slope == "Upsloping" else 0
        
        df["Sex_M"] = 1 if Sex == 'Male' else 0
        df["Sex_F"] = 1 if Sex == 'Female' else 0
        
        if st.button('Predict'):

            loaded_model = load_model('model.pkl')
            prediction = loaded_model.predict(df)
            col1.write('''
		    ## Results 🔍 
		    ''')
            if prediction.item() == 1:
                col1.warning('AI detects risk of heart disease, be careful.')
            else:
                col1.success('AI fails to detect risk of heart disease')

    st.warning("Note: This A.I application is for educational/demo purposes only and cannot be relied upon.")
    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
	main()