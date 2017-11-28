from sklearn import datasets
from sklearn import svm
from scipy.spatial import distance
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
import numpy as np
import matplotlib.pyplot as plot


def workClassValue(x):
    return workClassValueHelper(str(x).replace("'", "").replace(" ", "").replace("b", "").replace(".", ""))


def workClassValueHelper(xString):
    return {
        #workclass column values
        "Private": 1.0,
        "Self-emp-not-inc": 2.0,
        "Self-emp-inc": 3.0,
        "Federal-gov": 4.0,
        "Local-gov": 5.0,
        "State-gov": 6.0,
        "Without-pay": 7.0,
        "Never-worked": 8.0
    }.get(xString, 0.0)


def educationValue(x):
    return educationValueHelper(str(x).replace("'", "").replace(" ", "").replace("b", "").replace(".", ""))


def educationValueHelper(xString):
    return {
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
    }.get(xString, 0.0)


def maritalStatusValue(x):
    return maritalStatusValueHelper(str(x).replace("'", "").replace(" ", "").replace("b", "").replace(".", ""))


def maritalStatusValueHelper(xString):
    return {
        #marital status column values
        "Never-married": 1.0,
        "Divorced": 2.0,
        "Separated": 3.0,
        "Widowed": 4.0,
        "Married-spouse-absent": 5.0,
        "Married-AF-spouse": 6.0,
        "Married-civ-spouse": 7.0
    }.get(xString, 0.0)


def occupationValue(x):
    return occupationValueHelper(str(x).replace("'", "").replace(" ", "").replace("b", "").replace(".", ""))


def occupationValueHelper(xString):
    return {
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
        "Armed-Forces": 14.0
    }.get(xString, 0.0)


def relationshipValue(x):
    return relationshipValueHelper(str(x).replace("'", "").replace(" ", "").replace("b", "").replace(".", ""))


def relationshipValueHelper(xString):
    return {
        #relationship column values
        "Wife": 1.0,
        "Own-child": 2.0,
        "Husband": 3.0,
        "Not-in-family": 4.0,
        "Other-relative": 5.0,
        "Unmarried": 6.0
    }.get(xString, 0.0)


def raceValue(x):
    return raceValueHelper(str(x).replace("'", "").replace(" ", "").replace("b", "").replace(".", ""))


def raceValueHelper(xString):
    return {
        #race column values
        "White": 1.0,
        "Asian-Pac-Islander": 2.0,
        "Amer-Indian-Eskimo": 3.0,
        "Other": 4.0,
        "Black": 5.0,
    }.get(xString, 0.0)


def genderValue(x):
    return genderValueHelper(str(x).replace("'", "").replace(" ", "").replace("b", "").replace(".", ""))


def genderValueHelper(xString):
    return {
        #gender
        "Male": 1.0,
        "Female": 2.0
    }.get(xString, 0.0)


def countryValue(x):
    return countryValueHelper(str(x).replace("'", "").replace(" ", "").replace("b", "").replace(".", ""))


def countryValueHelper(xString):
    #print(xString + "1")
    return {
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
        "Holand-Netherlands": 41.0
    }.get(xString, 0.0)


def salaryValue(x):
    return salaryValueHelper(str(x).replace("'", "").replace(" ", "").replace("b", "").replace(".", ""))


def salaryValueHelper(xString):
    return {
        #salary
        "<=50K": 1.0,
        ">50K": -1.0
    }.get(xString, 0.0)

def euclidian(x):
    return distance.euclidean((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), x, w=None)


if __name__ == '__main__':
    #get all data
    X_data = np.genfromtxt(fname='adult.csv', dtype='float', delimiter=',', converters={1: workClassValue, 3: educationValue, 5: maritalStatusValue, 6: occupationValue, 7: relationshipValue, 8: raceValue, 9: genderValue, 13: countryValue, 14: salaryValue})
    Y_data = np.genfromtxt(fname='adult.test.csv', dtype='float', delimiter=',', converters={1: workClassValue, 3: educationValue, 5: maritalStatusValue, 6: occupationValue, 7: relationshipValue, 8: raceValue, 9: genderValue, 13: countryValue, 14: salaryValue})

    # create blank data structure
    x_training_euclidian = np.empty(shape=[1], dtype='float')
    x_training_euclidian = np.delete(x_training_euclidian, 0, axis=0)

    # compute euclidian distance for first 13 attributes
    #normalize
    for x in X_data[:, :13]:
        x_training_euclidian = np.append(x_training_euclidian, [euclidian(tuple(x))], axis=0)

    #get 14th attribute
    x_training_country = X_data[:, 13]

    #print ndarray size
    print(x_training_euclidian.shape)
    print(x_training_country.shape)

    #add one more column so i can merge later
    x_training_country = np.expand_dims(x_training_country, axis=1)
    x_training_euclidian = np.expand_dims(x_training_euclidian, axis=1)

    #print ndarray size
    print(x_training_euclidian.shape)
    print(x_training_country.shape)

    #merge the 2 ndarrays
    x_training_data = np.concatenate((x_training_euclidian, x_training_country), axis=1)

    #get classifier
    x_training_result = X_data[:, 14]

    # create blank data structure
    y_testing_euclidian = np.empty(shape=[1], dtype='float')
    y_testing_euclidian = np.delete(y_testing_euclidian, 0, axis=0)

    # compute euclidian distance for first 13 attributes
    for y in Y_data[:, :13]:
        y_testing_euclidian = np.append(y_testing_euclidian, [euclidian(tuple(y))], axis=0)

    #get 14th attribute
    y_testing_country = Y_data[:, 13]

    #print ndarray size
    print(y_testing_euclidian.shape)
    print(y_testing_country.shape)

    #add one more column so i can merge later
    y_testing_country = np.expand_dims(y_testing_country, axis=1)
    y_testing_euclidian = np.expand_dims(y_testing_euclidian, axis=1)

    #print ndarray size
    print(y_testing_euclidian.shape)
    print(y_testing_country.shape)

    #merge the 2 ndarrays
    y_testing_data = np.concatenate((y_testing_euclidian, y_testing_country), axis=1)
    y_testing_result = Y_data[:, 14]

    #svm function
    #svc = svm.LinearSVC(C=1.0).fit(x_training_data,x_training_result)
    svc = svm.SVC(kernel='rbf', gamma=.07, C=1.0).fit(x_training_data, x_training_result)

    #predict using testing
    Z = svc.predict(y_testing_data)

    print(Z.shape)
    print(accuracy_score(y_testing_result, Z))
    precision, recall, fscore, support = score(y_testing_result, Z)

    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))

    # Put the result into a color plot
    #Z = Z.reshape(xx.shape)
    #plot.contourf(xx, yy, Z, cmap=plot.cm.coolwarm, alpha=0.8)

    #plot here
    #plot.scatter(x_training_data[:,0], x_training_data[:, 1], c=x_training_result,cmap=plot.cm.coolwarm)
    #plot.xlabel('13 combined')
    #plot.ylabel('country')
    #plot.title('hello')
    #plot.show()

