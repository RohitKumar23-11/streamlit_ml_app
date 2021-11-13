# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 19:06:51 2021

@author: Acer
"""

#%%
import streamlit as st

# Load EDA Pkgs

import pandas as pd

# Load Data Visualization Pkgs

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import plotly.express as px
from PIL import Image

#Load Data 
@st.cache
def load_data(data):
    df = pd.read_csv(data)
    return df


def run_eda_app():
    st.subheader("From Exploratory Data Analysis")
    #df = pd.read_csv(r"D:\projects dataset\Employee.csv")
    df = pd.read_csv(r"https://raw.githubusercontent.com/RohitKumar23-11/streamlit_ml_app/main/heart.csv")
    df1 = pd.read_csv(r"https://raw.githubusercontent.com/RohitKumar23-11/streamlit_ml_app/main/clean_heart.csv")
    # st.dataframe(df)
    
    #image = Image.open(r"https://github.com/RohitKumar23-11/streamlit_ml_app/main/eda.jpg")
    #st.image(image,use_column_width=True, width=750,caption="EDA representation image")
    
    submenu = st.sidebar.selectbox("Submenu",['Descriptive','Plots'])
    if submenu == "Descriptive":
        st.subheader("Data details :clipboard:")
        st.warning("To display the duplicate values we use unclear/unprocessed data which have duplicate values.")
        st.dataframe(df)
        
        with st.expander("Data Types"):
            st.write("'O' means Obeject datatype"  )
            st.markdown(df.dtypes.tolist())
        
        with st.expander("Data Describe"):
             st.dataframe(df.describe().T)
             
        # with st.expander("Data information"):
        #     st.dataframe(df.info())
        
        with st.expander("Null Data"):
            st.dataframe(df.isnull().sum())
            
        with st.expander("Columns"):
            st.dataframe(df.columns)
            
        with st.expander("Class Distribution"):
            st.dataframe(df['target'].value_counts())

        with st.expander("Gender Distribution"):
            st.dataframe(df['sex'].value_counts())

        with st.expander("excercise induced angina Distribution"):
            st.dataframe(df['exang'].value_counts())

        with st.expander("ST depression induced by excerise relative to rest Distribution"):
            st.dataframe(df['oldpeak'].value_counts())

        with st.expander("the slope of the peak excercise ST segment Distribution"):
            st.dataframe(df['slope'].value_counts())
            
        with st.expander("checking duplicate values"):
            duplicate = df[df.duplicated()]
            st.dataframe(duplicate)
            
        # with st.expander("total duplicate values"):
        #     st.dataframe(df.duplicated())
            
    elif submenu == "Plots":
        st.subheader("Plots :bar_chart:")
        st.success("To plot the real data in which we train the ML model we use clean/processed data.")
        
        #Layouts 
        col1, col2 = st.columns(2)
        
        with col1:
            # Gender Distribution
            with st.expander(" Dist Plot of Gender (0 for female and 1 for male)"):
                #using seaborn
                fig = plt.figure()
                sns.countplot(df1['sex'])
                st.pyplot(fig)
            
                gen_df = df1['sex'].value_counts().to_frame()
                # st.dataframe(gen_df)
                gen_df = gen_df.reset_index()
                gen_df.columns = ['Gender Type',"Counts"]
                # st.dataframe(gen_df)
                
                p1 = px.pie(gen_df,names='Gender Type',values='Counts')
                st.plotly_chart(p1,use_container_width=True)
                
                # For Leave or not distribution
            with st.expander("Dist of the target "):
                fig = plt.figure()
                sns.countplot(df1['target'])
                st.pyplot(fig)
                
                lev_df = df1['target'].value_counts().to_frame()
                # st.dataframe(gen_df)
                lev_df = lev_df.reset_index()
                lev_df.columns = ['heart disease happend or not',"Counts"]
                # st.dataframe(lev_df)
                p1 = px.pie(lev_df,names='heart disease happend or not',values='Counts')
                st.plotly_chart(p1,use_container_width=True)                
                
            with st.expander("Dist of the exercise induced angina."):
                fig = plt.figure()
                sns.countplot(df1['exang'])
                st.pyplot(fig)
                
                edu_df = df1['exang'].value_counts().to_frame()
                # st.dataframe(gen_df)
                edu_df = edu_df.reset_index()
                edu_df.columns = ['exang',"Counts"]
                # st.dataframe(lev_df)
                p1 = px.pie(edu_df,names='exang',values='Counts')
                st.plotly_chart(p1,use_container_width=True)                
                
            with st.expander(" Dist Plot of thal"):
                #using seaborn
                fig = plt.figure()
                sns.countplot(df1['thal'])
                st.pyplot(fig)
            
                ben_df = df1['thal'].value_counts().to_frame()
                # st.dataframe(gen_df)
                ben_df = ben_df.reset_index()
                ben_df.columns = ['thal',"Counts"]
                # st.dataframe(gen_df)
                
                p1 = px.pie(ben_df,names='thal',values='Counts')
                st.plotly_chart(p1,use_container_width=True)
                
                
            with st.expander(" Dist Plot of resting electrocardiographic results"):
                #using seaborn
                fig = plt.figure()
                sns.countplot(df1['restecg'])
                st.pyplot(fig)
            
                city_df = df1['restecg'].value_counts().to_frame()
                city_df = city_df.reset_index()
                city_df.columns = ['restecg',"Counts"]
    
                
                p1 = px.pie(city_df,names='restecg',values='Counts')
                st.plotly_chart(p1,use_container_width=True)
            
            with st.expander("Count plot of the age distribution."):
                fig = plt.figure()
                sns.countplot(df1['age'])
                st.pyplot(fig)
                
                age_df = df1['age'].value_counts().to_frame()
                age_df = age_df.reset_index()
                age_df.columns = ['Age Counts',"Counts"]
    
                
                p1 = px.pie(age_df,names='Age Counts',values='Counts')
                st.plotly_chart(p1,use_container_width=True)
                
            with st.expander("Count plot of the chest pain tpye distribution."):
                fig = plt.figure()
                sns.countplot(df1['cp'])
                st.pyplot(fig)
                
                join_df = df1['cp'].value_counts().to_frame()
                join_df = join_df.reset_index()
                join_df.columns = ['cp',"Counts"]
    
                
                p1 = px.pie(join_df,names='cp',values='Counts')
                st.plotly_chart(p1,use_container_width=True)
            
            with st.expander("Countplot of the ca distribution."):
                fig = plt.figure()
                sns.countplot(df1['ca'])
                st.pyplot(fig)
                
                pay_df = df1['ca'].value_counts().to_frame()
                pay_df = pay_df.reset_index()
                pay_df.columns = ['ca',"Counts"]
    
                
                p1 = px.pie(pay_df,names='ca',values='Counts')
                st.plotly_chart(p1,use_container_width=True)
                
            with st.expander("Countplot of the slope of the peak exercise ST segment distribution."):
                fig = plt.figure()
                sns.countplot(df1['slope'])
                st.pyplot(fig)
                
                pay_df = df1['slope'].value_counts().to_frame()
                pay_df = pay_df.reset_index()
                pay_df.columns = ['slope',"Counts"]
    
                
                p1 = px.pie(pay_df,names='slope',values='Counts')
                st.plotly_chart(p1,use_container_width=True)
                
            with st.expander("Comparison of Gender and target."):
                fig = plt.figure()
                sns.barplot(x='sex',y='target',data=df1)
                st.pyplot(fig)
                
                
                
        with col2:
            with st.expander("Gender Distribution"):
                st.dataframe(gen_df)
            with st.expander("target Distribution"):
                st.dataframe(lev_df)
            with st.expander("oldpeak details of the people"):
                st.dataframe(edu_df)
            with st.expander("thal Distribution"):
                st.dataframe(ben_df)
            with st.expander("restecg Distribution"):
                st.dataframe(city_df)
            with st.expander("Age Distribution"):
                st.dataframe(age_df)
            with st.expander("chest pain type Distribution"):
                st.dataframe(join_df)
            with st.expander("ca info."):
                st.dataframe(pay_df)
                
        # st.subheader("Now sns disrtibution and comapring starts")
                
        with col1:
            with st.expander("comparing the age with target."):
                fig = plt.figure()
                sns.barplot(x='age',y='target',data=df1)
                st.pyplot(fig)
            with st.expander("comparing the resting blood pressure with target."):
                fig = plt.figure()
                sns.barplot(x='trestbps',y='target',data=df1)
                st.pyplot(fig)
            with st.expander("comparing the fasting blood sugar with target."):
                fig = plt.figure()
                sns.barplot(x='fbs',y='target',data=df1)
                st.pyplot(fig)
                
        with col2:
            with st.expander("comparing the slope with target."):
                fig = plt.figure()
                sns.barplot(x='slope',y='target',data=df1)
                st.pyplot(fig)
            with st.expander("comparing the thal with  target."):
                fig = plt.figure()
                sns.barplot(x='thal',y='target',data=df1)
                st.pyplot(fig)
            with st.expander("comparing the chest pain type with target."):
                fig = plt.figure()
                sns.barplot(x='cp',y='target',data=df1)
                st.pyplot(fig)
                
        # st.subheader("checking")
        # with st.expander("checking"):
        #     p2 = px.bar(df,x='Age',y='LeaveOrNot')
        #     st.plotly_chart(p2,use_container_width=True)
            
        st.subheader("Outlier Detection Using Box Plot. :art:")
        with st.expander("Outlier Detection Plot for Age using Gender"):
            fig = plt.figure()
            sns.boxplot(df1['age'])
            st.pyplot(fig)
            
            p3 = px.box(df1,x='age',color='sex')
            st.plotly_chart(p3,use_container_width=True)
            
        with st.expander("Outlier Detection Plot for resting blood pressure using Gender"):
            fig = plt.figure()
            sns.boxplot(df1['trestbps'])
            st.pyplot(fig)
            
            p3 = px.box(df1,x='trestbps',color='sex')
            st.plotly_chart(p3,use_container_width=True)
            
        with st.expander('Outlier Detection plot for maximum heart rate achieved using Gender'):
            fig = plt.figure()
            sns.boxplot(df1['thalach'])
            st.pyplot(fig)
            
            p3 = px.box(df1,x='thalach',color='sex')
            st.plotly_chart(p3,use_container_width=True)
        
        with st.expander("Outlier detection plot for serum cholestoral in mg/dl using Gender"):
            # fig = plt.figure()
            # sns.boxplot(df['chol'])
            # st.pyplot(fig)
            
            p3 = px.box(df1,x='chol',color='sex')
            st.plotly_chart(p3,use_container_width=True)
            
        st.subheader("Correlation Plot. :art:")
        with st.expander("correlation Plot"):
           corr_matrix = df1.corr()
           fig = plt.figure(figsize=(20,10))
           sns.heatmap(corr_matrix,annot=True)
           st.pyplot(fig)
           
           p4 = px.imshow(corr_matrix)
           st.write("Correlation matrix using plotly below. :memo:")
           st.plotly_chart(p4,use_container_width=True)
        
            
        
            


                
