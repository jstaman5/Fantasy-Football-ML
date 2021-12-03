#SVM

import pandas as pd
import numpy as np
import pylab as pl
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os

def main():
    data_dir = os.path.abspath(os.path.dirname(__file__)) + "/../data"
    data = pd.read_csv(data_dir+"/Qbs.csv", sep=',')
    data = data.drop(['Player', 'Year'], axis = 1)

    data = data.apply(LabelEncoder().fit_transform)

    def normalizeData(data):
        return  (data-np.min(data)) / (np.max(data) - np.min(data))

    fantasy_points = data['FantasyPoints'].values
    age = normalizeData(data['Age'].values)
    comp = normalizeData(data['Cmp%'].values)
    yard_att = normalizeData(data['Yards/Attempt'].values)
    sacks = normalizeData(data['Sacks'].values)
    nfl_rating = normalizeData(data['Rating'].values)

    X = np.array([ np.ones(shape = age.shape),age, comp, yard_att, sacks,nfl_rating ], dtype=np.float32).T
    y = np.array(fantasy_points)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    svc = svm.SVC(kernel='poly').fit(X_train, y_train)

    predicted_linear = svc.predict(X_test)
    print(predicted_linear)
    print(y_test)
    predicted_linear.sort()
    y_test.sort()
    plt.figure()
    plt.scatter(np.arange(1,np.size(y_test) + 1),predicted_linear,label ="Predicted Fantasy Points")
    plt.scatter(np.arange(1,np.size(y_test) + 1),y_test,label = "Actual Fantasy Points")
    #plt.legend()
    plt.ylabel("Fantasy Points")
    plt.xlabel("QB")
    plt.title("Scatter Plot of Estimated and Actual Fantasy Points for Each Quarterback")
    plt.show()
    #print("SVM + Linear \t\t-> " + str(accuracy_score(y_test, predicted_linear)))
    #print(classification_report(y_test, predicted_linear))

    return

    
if __name__ == "__main__":
    main()