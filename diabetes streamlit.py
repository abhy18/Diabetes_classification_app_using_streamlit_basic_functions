import streamlit as st
import numpy as np 
import pickle

st.markdown(
  """
  <style>
  .main {
      background-color: orange
  }
  .sidebar .sidebar-content {
      background: url("/content/gok.png")
  }
  </style>
  """,
  unsafe_allow_html=True
)
st.sidebar.title('Super options')

from toolz.functoolz import return_none
st.title('Diabetes classification')
st.header('Using streamlit')
model = pickle.load(open('nodel.pkl','rb'))

def diabetes_pred(input_data):
  input_arr = np.asarray(input_data)
  input_arr_re = input_arr.reshape(1,-1)

  predicts = model.predict(input_arr_re)
  print(predicts)

  if (predicts[0] ==0):
    return 'Not diabetic'
  else:
    return 'Diabetic'

def main():
  p =st.text_input('Enter no of pregnancies')
  g =st.text_input('Enter Gulcose level')
  bp =st.text_input('Enter blood pressure level')
  th =st.text_input('enter the thickness value')
  In = st.text_input('Enter the Insulin level')
  bmi = st.text_input('Enter the BMI')
  dpf = st.text_input('DiabetesPedigreeFunction')
  age = st.number_input('Enter the age',0,120)

  diagnosis = ''

  if st.button('Diabetes Test result'):
    diagnosis = diabetes_pred([p,g,bp,th,In,bmi,dpf,age])
  
  st.success(diagnosis)

if __name__ == '__main__':
  main()




gtype= ["Line","Scatter","pie"]
import pandas as pd 
df = pd.read_csv('/content/diabetes.csv')
gtype1 =  st.selectbox('Select the graph type',gtype)
if gtype1 == "Line":
  st.line_chart(df['Pregnancies'])
elif gtype1 =="Scatter":
  st.scatter_chart(df['Pregnancies'])
else:
  st.bar_chart(df['Pregnancies'])
