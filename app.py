# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 13:43:14 2021

@author: Acer
"""

#%%
#main file core pkgs


import streamlit as st
import streamlit.components.v1 as stc
import urllib.request


# Page title and other details
st.set_page_config(page_title='ML model for heart problem prediction',
                   page_icon=":star2:",
                   layout='wide')

# importing our mini Apps

from eda_app import run_eda_app
from ml_app import run_ml_app
from PIL import Image
#import pickle

# defining the area to write and give heading of the web app which available for all the web pages
html_temp = """
                <div style = "background-color:red;padding:10px;border-radius:10px">
		<h2 style="color:white;text-align:center;">App for detecting the wheather person will have heart problem or not. </h2>
		<h3 style="color:blue;text-align:center;">Heart Problem Or Not </h3>
		</div>
        """



def main():
    # st.title("Main App")
    stc.html(html_temp)
    
    menu = ["Home","EDA","ML","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    # st.sidebar.image(r"D:\crome download\human org4.jpg", use_column_width=True)
    
    if choice == "Home":
        st.subheader("Home :sunny:")
        st.write(""":pray: This is ML app which  is used to detect or predict whether a person will
             have the heart problem or not in future. It uses Random Forest Classifier Algorithm to predict
             the output. :clap:""")
        # st.image(r"D:\crome download\human org4.jpg")
        image1 = urllib.request.urlretrieve(r"https://github.com/RohitKumar23-11/streamlit_ml_app/blob/main/1080.jpg")
        st.image(image1,use_column_width=True, caption="Refreah your mind with this Beautiful mountain image.")
        
    elif choice == "EDA":
        run_eda_app()
    elif choice == "ML":
        run_ml_app()
    else:
        st.subheader(":snowflake: About")
        st.success("""This ML web app is created by Rohit Kumar :snowman:. Who is presently a trainee data scientist at 
                 Brainlyst Pvt. Ltd.:sunny:. To create this project I first create a rough ML model in jupyter notebook 
                 and try various ML algorithms like logistic regression, Random Forest Classifier, etc. after performing 
                 these algorithms I select the best model which is RANDOM FOREST CLASSIFIER in this case to predict the 
                 output. The streamlit file need pure python file to run so I create a copy of this jupyter file with 
                 '.py' extension (which is pure python file). With the help of this python file I create streamlit web 
                 app. I download the data from kaggle.com the link of the data is :point_right: -
                 https://www.kaggle.com/ronitf/heart-disease-uci
                 :snowflake:. My Github Link is this :point_right:- https://github.com/RohitKumar23-11?tab=repositories :snowflake:, from here you can download all the files and data directly.""")
        image = urllib.request.urlretrieve(r"https://github.com/RohitKumar23-11/streamlit_ml_app/blob/main/thank-you.jpg")
        st.image(image,use_column_width=True, caption="Thank You so much for using this app.")
    
if __name__ == '__main__':
    main()
