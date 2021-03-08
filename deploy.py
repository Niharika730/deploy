import streamlit as st
import numpy as np
import pandas as pd
from pickle import dump,load
st.header("Support Vector Machine (rbf)")
clf1=load(open('pickle/svmr.pkl','rb'))
col1 = st.number_input("Enter values for col1:")
col2 = st.number_input("Enter values for col2:")
click=st.button("SUBMIT")
st.write(np.array(clf1.predict([[col1,col2]])))
