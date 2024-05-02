# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv(r"C:\Users\jilla\Downloads\diabetes.csv")

# HEADINGS
st.title('Diabetes Checkup')

st.subheader('Training Data Stats')
st.write(df.describe())
st.subheader('Visualization')
st.bar_chart(df)

# X AND Y DATA
X = df.drop(['Outcome'], axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# FUNCTION
def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0, 67, 20)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 21, 88, 33)

    user_report = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    report_data = pd.DataFrame(user_report, index=[0])
    return report_data

# PATIENT DATA
user_data = user_report()

# MODEL
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test, rf.predict(X_test)) * 100) + '%')
user_result = rf.predict(user_data)
st.subheader('Your report: ')
output = 'You are healthy' if user_result[0] == 0 else 'You are not healthy'
st.write(output)
