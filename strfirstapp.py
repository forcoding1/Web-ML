from enum import unique
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
from sklearn.metrics import accuracy_score

st.title('Machine Learning App')
image = Image.open('pain.jpg')
st.image(image, use_column_width = True)
st.write('By Pain')
st.write("""# A simple data app using streamlit""")
st.sidebar.write('Classification')

dataset_name = st.sidebar.selectbox('Dataset',('Iris', 'Breast Cancer', 'Wine'))

classifier_name = st.sidebar.selectbox('Classifier',('SVM', 'KNN', 'Logistic Regression'))

def get_dataset(name):
    data = None
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Breast Cancer':
        data = datasets.load_breast_cancer()
    elif name == 'Wine':
        data = datasets.load_wine()
    x = data.data
    y = data.target

    return x,y

x,y = get_dataset(dataset_name)

st.dataframe(x)
st.write('Shape of your data: ', x.shape)
st.write('Unique target variables: ' , len(np.unique(y)))

st.set_option('deprecation.showPyplotGlobalUse', False) #error handling for plotting graphs

fig = plt.figure()
sns.boxplot(data = x, orient='h')
st.pyplot()

plt.hist(x)
st.pyplot()

def add_params(clf_name):
    params = dict()
    if clf_name == 'SVM':
        c = st.sidebar.slider('C', 0.01, 15.0)
        params['C'] = c
    elif clf_name == 'KNN':
        k = st.sidebar.slider('K', 1, 15)
        params['K'] = k
    else:
        s = st.sidebar.selectbox('Solver', ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'))
        params['S'] = s
    return params

params = add_params(classifier_name)

def classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C = params['C'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors = params['K'])
    else:
        clf = LogisticRegression(solver=params['S'])
    return clf

clf = classifier(classifier_name, params)

xtrain, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

clf.fit(xtrain, y_train)

y_pred = clf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)


st.write('Classifier: ', classifier_name)
st.write('Accuracy: ', accuracy)