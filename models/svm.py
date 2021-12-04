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

def normalizeData(data):
        return  (data-np.min(data)) / (np.max(data) - np.min(data))

def main():
    
    position_csv = ["/QBs.csv", "/TEs.csv", "/WRs.csv"]

    data_dir = os.path.abspath(os.path.dirname(__file__)) + "/../data"
    
    for pos in position_csv:
        data = pd.read_csv(data_dir + pos, sep=',')
        
        data = data.drop(['Player', 'Year'], axis = 1)
        data = data.apply(LabelEncoder().fit_transform)

        fantasy_points = data['FantasyPoints'].values

        if pos == "/QBs.csv":
            age = normalizeData(data['Age'].values)
            comp = normalizeData(data['Cmp%'].values)
            yard_att = normalizeData(data['Yards/Attempt'].values)
            sacks = normalizeData(data['Sacks'].values)
            nfl_rating = normalizeData(data['Rating'].values)

            X = np.array([ np.ones(shape = age.shape),age, comp, yard_att, sacks,nfl_rating ], dtype=np.float32).T
            plt_pos = "QB"

        elif pos == "/TEs.csv":
            snaps = normalizeData(data['Snap %'].values)
            targets = normalizeData(data['Targets'].values)
            target_share = normalizeData(data['Target Share'].values)
            catch = normalizeData(data['Catch %'].values)
            forty = normalizeData(data['40 yd time'].values)

            X = np.array([ np.ones(shape = snaps.shape),snaps, targets, target_share, catch, forty], dtype=np.float32).T
            plt_pos = "TE"
        
        elif pos == "/WRs.csv":
            targets = normalizeData(data['Targets'].values)
            fd = normalizeData(data['First Downs'].values)
            ybc = normalizeData(data['YBC'].values)
            yac = normalizeData(data['YAC'].values)
            adot = normalizeData(data['ADOT'].values)
            bt = normalizeData(data['Broken Tackles'].values)
            dp = normalizeData(data['Drop %'].values)
            picks = normalizeData(data['Interceptions'].values)
            qbr = normalizeData(data['QB Rating'].values)

            X = np.array([ np.ones(shape = targets.shape), targets, fd, ybc, yac, adot, bt, dp, picks, qbr], dtype=np.float32).T
            plt_pos = "WR"

        y = np.array(fantasy_points)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

        svc = svm.SVC(kernel='poly').fit(X_train, y_train)

        
        predicted_linear = svc.predict(X_test)
        result = list(zip(y_test, predicted_linear))
        
        result.sort()

        y_test = [x for (x,y) in result]
        predicted_linear = [y for (x,y) in result]
        
        plt.figure()
        plt.scatter(np.arange(1,np.size(y_test) + 1),predicted_linear,label ="Predicted Fantasy Points")
        plt.scatter(np.arange(1,np.size(y_test) + 1),y_test,label = "Actual Fantasy Points")
        #plt.legend()
        plt.ylabel("Fantasy Points")
        plt.xlabel(plt_pos)
        plt.title("Scatter Plot of Estimated and Actual Fantasy Points for Each {}".format(plt_pos))
        plt.show()
        #print("SVM + Linear \t\t-> " + str(accuracy_score(y_test, predicted_linear)))
        #print(classification_report(y_test, predicted_linear))

    return

    
if __name__ == "__main__":
    main()