import numpy as np

import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# find possbile states
def convertToStringThenToFloat(x):
    return genderValue(str(x).replace("'", "").replace(" ", "").replace("b", "").replace(".", ""))


def genderValue(xString):
    #print(xString + "1")
    return {
        #workclass column values
        "Private": 1.0,
        "Self-emp-not-inc": 2.0,
        "Self-emp-inc": 3.0,
        "Federal-gov": 4.0,
        "Local-gov": 5.0,
        "State-gov": 6.0,
        "Without-pay": 7.0,
        "Never-worked": 8.0,
        #education column values
        "Preschool": 1.0,
        "1st-4th": 2.0,
        "5th-6th": 3.0,
        "7th-8th": 4.0,
        "9th": 5.0,
        "10th": 6.0,
        "11th": 7.0,
        "12th": 8.0,
        "HS-grad": 9.0,
        "Some-college": 10.0,
        "Assoc-acdm": 11.0,
        "Assoc-voc": 12.0,
        "Prof-school": 13.0,
        "Bachelors": 14.0,
        "Masters": 15.0,
        "Doctorate": 16.0,
        #marital status column values
        "Never-married": 1.0,
        "Divorced": 2.0,
        "Separated": 3.0,
        "Widowed": 4.0,
        "Married-spouse-absent": 5.0,
        "Married-AF-spouse": 6.0,
        "Married-civ-spouse": 7.0,
        #occupation column values
        "Tech-support": 1.0,
        "Craft-repair": 2.0,
        "Other-service": 3.0,
        "Sales": 4.0,
        "Exec-managerial": 5.0,
        "Prof-specialty": 6.0,
        "Handlers-cleaners": 7.0,
        "Machine-op-inspct": 8.0,
        "Adm-clerical": 9.0,
        "Farming-fishing": 10.0,
        "Transport-moving": 11.0,
        "Priv-house-serv": 12.0,
        "Protective-serv": 13.0,
        "Armed-Forces": 14.0,
        #relationship column values
        "Wife": 1.0,
        "Own-child": 2.0,
        "Husband": 3.0,
        "Not-in-family": 4.0,
        "Other-relative": 5.0,
        "Unmarried": 6.0,
        #race column values
        "White": 1.0,
        "Asian-Pac-Islander": 2.0,
        "Amer-Indian-Eskimo": 3.0,
        "Other": 4.0,
        "Black": 5.0,
        #
        "Male": 1.0,
        "Female": 2.0,
        #native country
        "United-States": 1.0,
        "Cambodia": 2.0,
        "England": 3.0,
        "Puerto-Rico": 4.0,
        "Canada": 5.0,
        "Germany": 6.0,
        "Outlying-US(Guam-USVI-etc)": 7.0,
        "India": 8.0,
        "Japan": 9.0,
        "Greece": 10.0,
        "South": 11.0,
        "China": 12.0,
        "Cuba": 13.0,
        "Iran": 14.0,
        "Honduras": 15.0,
        "Philippines": 16.0,
        "Italy": 17.0,
        "Poland": 18.0,
        "Jamaica": 19.0,
        "Vietnam": 20.0,
        "Mexico": 21.0,
        "Portugal": 22.0,
        "Ireland": 23.0,
        "France": 24.0,
        "Dominican-Republic": 25.0,
        "Laos": 26.0,
        "Ecuador": 27.0,
        "Taiwan": 28.0,
        "Haiti": 29.0,
        "Columbia": 30.0,
        "Hungary": 31.0,
        "Guatemala": 32.0,
        "Nicaragua": 33.0,
        "Scotland": 34.0,
        "Thailand": 35.0,
        "Yugoslavia": 36.0,
        "El-Salvador": 37.0,
        "Trinadad&Tobago": 38.0,
        "Peru": 39.0,
        "Hong": 40.0,
        "Holand-Netherlands": 41.0,
        #salary
        "<=50K": 1.0,
        ">50K": 2.0
    }.get(xString, 0.0)

# main function
if __name__ == '__main__':
    #cancer_data = np.genfromtxt(fname='adult.csv',dtype='float', delimiter=',',missing_values="nan", filling_values=1, converters={1: convertToStringThenToFloat, 3: convertToStringThenToFloat, 5: convertToStringThenToFloat, 6: convertToStringThenToFloat, 7: convertToStringThenToFloat, 8: convertToStringThenToFloat, 9: convertToStringThenToFloat, 13: convertToStringThenToFloat, 14: convertToStringThenToFloat})
    cancer_data = np.genfromtxt(fname='adult.csv', dtype='float', delimiter=',', converters={1: convertToStringThenToFloat, 3: convertToStringThenToFloat, 5: convertToStringThenToFloat, 6: convertToStringThenToFloat, 7: convertToStringThenToFloat, 8: convertToStringThenToFloat, 9: convertToStringThenToFloat, 13: convertToStringThenToFloat, 14: convertToStringThenToFloat})
    cancer_data1 = np.genfromtxt(fname='adult.test.csv', dtype='float', delimiter=',', converters={1: convertToStringThenToFloat, 3: convertToStringThenToFloat, 5: convertToStringThenToFloat, 6: convertToStringThenToFloat, 7: convertToStringThenToFloat, 8: convertToStringThenToFloat, 9: convertToStringThenToFloat, 13: convertToStringThenToFloat, 14: convertToStringThenToFloat})
    print(len(cancer_data1))
    print(cancer_data1)
    print(cancer_data1.shape)
    #train
    X = cancer_data[:, range(0, 13)]
    X1 = cancer_data[:, 14]
    #test
    Y = cancer_data1[:, range(0, 13)]
    Y1 = cancer_data1[:, 14]

    #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)
    # training set turned to 1-d array
    #y_train = y_train.ravel()

    #classifier testing set turned to 1-d array
    #y_test = y_test.ravel()

    for K in range(25):
        K_value = K+1
        neigh = KNeighborsClassifier(n_neighbors = K_value, weights='uniform', algorithm='auto')
        #fitting training set with all values
        #neigh.fit(X_train, y_train)
        neigh.fit(X,X1)
        #predicts training income based on nearest neighbor classifier
        #y_pred = neigh.predict(X_test)
        y_pred = neigh.predict(Y)
        #print result
        print("Accuracy is ", accuracy_score(Y1,y_pred)*100,"% for K-Value:",K_value)
