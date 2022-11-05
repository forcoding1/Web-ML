from enum import unique
from tkinter import Checkbutton
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import seaborn as sns


import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


#functions

def add_params(clf_name):
    params = dict()
    if clf_name == 'SVM':
        c = st.sidebar.slider('C', 0.01, 15.0)
        params['C'] = c
    elif clf_name == 'KNN':
        k = st.sidebar.slider('K', 1, 15)
        params['K'] = k
    elif clf_name == 'Logistic Regression':
        s = st.sidebar.selectbox('Solver', ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'))
        params['S'] = s
    elif clf_name == 'Descision Tree':
        n = st.sidebar.selectbox('n_features', 1, 100)
        params['N'] = n
    else:
        st.write("Choose an algorithm to continue")
    return params

def classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C = params['C'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors = params['K'])
    elif clf_name == 'Logistic Regression':
        clf = LogisticRegression(solver=params['S'])
    elif clf_name == 'Naieve Bayes':
        clf = GaussianNB()
    elif clf_name == 'Decision Tree':
        clf = DecisionTreeClassifier(n_features = params['N'])
    return clf

st.title('Machine Learning App')
image = Image.open('pain.jpg')
st.image(image, use_column_width = True)
st.write('By Pain')
st.sidebar.write('Classification')


def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    options = ['EDA', 'Visualisation', 'Model selection', 'About me']
    chose = st.sidebar.selectbox('Operations', options)

    if chose == 'EDA':
        st.subheader('Exploratory Data Analysis')
        data = st.file_uploader('Choose any file', type=['csv', 'json', 'txt', 'xlsx'])
        if data:
            st.success("File uploading successful")
            df = pd.read_csv(data)
            st.dataframe(df.head(50))
            if st.checkbox("Shape"):
                st.write(df.shape)
            if st.checkbox("Total Columns"):
                st.write(df.columns)
            if st.checkbox("Multiple columns"):
                selected_columns = st.multiselect("Select columns",df.columns)
                st.write(df[selected_columns])
            if st.checkbox("Summary"):
                st.write(df.describe().T)
            if st.checkbox("Null Values"):
                st.write(df.isnull().sum())
            if st.checkbox("Data Types"):
                st.write(df.dtypes.astype(str))
            if st.checkbox("Correlation"):
                st.write(df.corr())

    if chose == 'Visualisation':
        st.subheader('Visualisation')
        data = st.file_uploader('Choose any file', type=['csv', 'json', 'txt', 'xlsx'])
        if data:
            st.success("File uploading successful")
            df = pd.read_csv(data)
            if st.checkbox("all columns"):
                st.dataframe(df)
            elif st.checkbox("Custom columns"):
                chose = st.multiselect("Select columns", df.columns)
                df = df[chose]
                st.dataframe(df)
            if st.checkbox("Pairplot"):
                st.write(sns.pairplot(df, diag_kind='kde'))
                st.pyplot()
            if st.checkbox("Heatmap"):
                st.write(sns.heatmap(df.corr(), annot = True))
                st.pyplot()
            if st.checkbox("Pie chart"):
                pie_col = st.selectbox("Select column to plot", df.columns)
                st.write(df[pie_col].value_counts().plot.pie(autopct = "%1.1f%%"))
                st.pyplot()
    if chose == 'Model selection':
        st.subheader('Model selection')
        data = st.file_uploader('Choose any file', type=['csv', 'json', 'txt', 'xlsx', 'data'])
        if data:
            st.success("File uploading successful")
            df = pd.read_csv(data)
            if st.checkbox("all columns"):
                st.dataframe(df)
                a = 1
            elif st.checkbox("Custom columns"):
                sel_col = st.multiselect("Select the preferred columns", df.columns)
                df = df[sel_col]
                st.dataframe(df)
                a = 1
            if a == 1 :
                x = df.iloc[:,0:-1]
                st.write(x.shape)
                y = df.iloc[:,-1]
                st.write(y.shape)
                seed = st.sidebar.slider("Seed", 1,200)
                classifier_name = st.sidebar.selectbox('Classifier',('SVM', 'KNN', 'Logistic Regression','Descision Tree', 'Naieve Bayes'))

                params = add_params(classifier_name)

                clf = classifier(classifier_name, params)

                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)

                clf.fit(x_train, y_train)

                y_pred = clf.predict(x_test)

                accuracy = accuracy_score(y_test, y_pred)


                st.write('Classifier: ', classifier_name)
                st.write('Accuracy: ', accuracy)
            

if __name__ == '__main__':
    main()
