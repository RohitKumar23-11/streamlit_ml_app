import streamlit as st 
import joblib
import os
import numpy as np
from PIL import Image

attrib_info = """
#### Attribute Information:
    - Age 29-77
    - Sex  Male, Female
    - cp 0, 1, 2, 3
    - trestbps 94 to 200.
    - chol 126 to 564
    - fbs 0 and 1
    - restecg 0, 1, 2
    - thalach 71 to 202
    - exang 0 and 1
    - oldpeak 0 to 6.2
    - slope 0, 1, 2
    - ca 0, 1, 2, 3, 4
    - thal 0, 1, 2, 3
    - target 0 and 1

"""

label_dict = {"0":0,"1":1}
sex_dict = {"Female":0,"Male":1}
target_dict = {"0":1,"1":1}

['Age','Sex','cp','trestbps','chol','fbs','restecg','thalach',
 'exang','lodpeak','slope','cs','thal','target']

def get_fvalue(val):
	feature_dict = {"0":0,"1":1}
	for key,value in feature_dict.items():
		if val == key:
			return value 
        
def get_value(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return value 
        
@st.cache(allow_output_mutation=True)
def load_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model


def run_ml_app():
	st.subheader("Machine Learning Section. :computer:")
	loaded_model = load_model(r"https://github.com/RohitKumar23-11/streamlit_ml_app/blob/main/randomforestclassifier_1.pkl")

	with st.expander("Attributes Info"):
		st.markdown(attrib_info,unsafe_allow_html=True)
        
    # Layout
	col1,col2 = st.columns(2)

	with col1:
		age = st.number_input("Age",29,77)
		sex = st.radio("Gender",("Female","Male"))
		cp = st.slider("cp",0,3)
		trestbps = st.slider("trestbps",94,200,1) 
		chol = st.slider("cholestrol",126,564,5)
		fbs = st.radio("fbs",["0","1"]) 

		
	
	with col2:
		restecg = st.slider("restecg",0,2)
		thalach = st.slider("thalach",71,202,2) 
		exang = st.radio("exang",["0","1"]) 
		oldpeak = st.number_input("oldpeak",0,7,1) 
		slope = st.slider("slope",0,2)
		ca = st.slider("ca",0,4) 
		thal = st.slider("thal",0,3) 
		
        
	with st.expander("Your Selected Options"):
		result = {'Age':age,
		'Gender':sex,
		'cp':cp,
		'trestbps':trestbps,
		'chol':chol,
		'fbs':fbs,
		'restecg':restecg,
		'thalach':thalach,
		'exang':exang,
		'oldpeak':oldpeak,
		'slope':slope,
		'ca':ca,
		'thal':thal}
		st.write(result)
		encoded_result = []
		for i in result.values():
			if type(i) == int:
				encoded_result.append(i)
			elif i in ["Female","Male"]:
				res = get_value(i,sex_dict)
				encoded_result.append(res)
			else:
				encoded_result.append(get_fvalue(i))


		st.write(encoded_result)
	with st.expander("Prediction Results"):
		single_sample = np.array(encoded_result).reshape(1,-1)

		
		prediction = loaded_model.predict(single_sample)
		pred_prob = loaded_model.predict_proba(single_sample)
		st.write(prediction)
		if prediction == 1:
			st.warning("Positive Risk-{}".format(prediction[0]))
			pred_probability_score = {"Negative chance":pred_prob[0][0]*100,"Positive chance":pred_prob[0][1]*100}
			st.subheader("Prediction Probability Score")
			st.json(pred_probability_score)
		else:
			st.success("Negative Risk-{}".format(prediction[0]))
			pred_probability_score = {"Negative chance":pred_prob[0][0]*100,"Positive chance":pred_prob[0][1]*100}
			st.subheader("Prediction Probability Score")
			st.json(pred_probability_score)


        

