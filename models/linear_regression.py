#linear regression

import math

import numpy as np
from numpy.lib.function_base import gradient 
import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sbn
import os

# model evaluation - MSE criterion function
def J(y, y_pred, w, beta):
    jtran = np.transpose(y_pred - y)
    Jdata = (1/(2*np.size(y))) * (np.dot(jtran,y_pred -y)) 
    Jmodel = (np.dot(w,np.transpose(w)))
    return Jdata, beta*Jmodel, Jdata + beta*Jmodel

# model evaluation - R2 statistical score
def r2(y, y_pred):
    mean_y = np.mean(y)
    ss_res = np.sum(np.square(y-y_pred))
    ss_tot = np.sum(np.square(y-mean_y))
    return 1 - (ss_res / ss_tot)

# model training - MSE gradient descent
def gd(X, y, eta, beta, epsi=0.01, kMax=100000):
    print_stuff = False
    max_w_change = []
    rows,cols = np.shape(X)
    w = np.zeros(6)
    for k in range(kMax):
        #make gradient
        trans = np.transpose(X)
        inside = np.dot(X,w) - y
        oldW = w
        grad = (1/rows) * (np.dot(trans,inside)) + 2*beta*w

        #change w and store max change
        w = oldW - eta*grad
        max_w_change.append(max(abs(w-oldW)))
        
        if((max_w_change[k]) < epsi): #checks if max change is smaller than epsilon
            break

        if print_stuff:
            Jdata, bj, jbj = J(y, np.dot(X,w), w, beta)
            print(Jdata,bj, jbj, max_w_change)
    
    return w, max_w_change,k


def main():
    sbn.set()

    # read data from file, produce related info and plots
    data_dir = os.path.abspath(os.path.dirname(__file__)) + "/../data"
    data = pd.read_csv(data_dir+"/Qbs.csv", sep=',')
    data = data.drop(['Player', 'Year'], axis = 1)

    def normalizeData(data):
        return  (data-np.min(data)) / (np.max(data) - np.min(data))

    fantasy_points = data['FantasyPoints'].values
    age = normalizeData(data['Age'].values)
    comp = normalizeData(data['Cmp%'].values)
    yard_att = normalizeData(data['Yards/Attempt'].values)
    sacks = normalizeData(data['Sacks'].values)
    nfl_rating = normalizeData(data['Rating'].values)

    info = data.describe()
    print(info)
    
    sbn.heatmap(data.corr()) #-- use show data.corr() output

    plt.show()

    sbn.pairplot(data) #-- consider hue='Country_code'
    plt.show()

    # create data arrays used below, split into train/test, standardize and prepend by 1s

    X = np.array([ np.ones(shape = age.shape),age, comp, yard_att, sacks,nfl_rating ], dtype=np.float32).T
    y = np.array(fantasy_points)
    print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # evaluate hyperparameters on training data
    eta_values =  [ 0.001, 0.01, 0.1 ]
    beta_values = [ 0.0, 0.01, 0.05, 0.1, 0.5, 1.0 ]
    eps_values = [0.001, 0.01]

    bestEta = 0
    bestBeta = 0
    bestR2 = 0
    besteps = 0

    #loops through all epsilon, eta, and beta values
    for eps in eps_values:
        for eta in eta_values:
            iters = []
            dataFidelity = []
            modelTerm = []
            sum = []
            for beta in beta_values:
                #runs gradient descent function
                w,c,k = gd(X_train,y_train,eta,beta,epsi = eps)
                y_pred = np.dot(X_train,w)
                calcR2 = r2(y_train,y_pred)
                iters.append(k)
                jdata, jbetaw, jtotal = J(y_train,y_pred,w,beta)
                dataFidelity.append(jdata)
                modelTerm.append(jbetaw)
                sum.append(jtotal)

                if (calcR2 > bestR2) and (eps == .01) :
                    bestR2 = calcR2
                    bestEta = eta
                    bestBeta = beta
                    besteps = eps
            
            #plots iteration vs beta values for each eta
            '''fig, ax = plt.subplots(1,2)
            width = .25
            labels = ['0.0','0.0','0.01','0.05','0.1','0.5','1.0']
            x = np.arange(6)
            fig.suptitle("Eta = {}   epsilon = {}".format(eta,eps))
            ax[0].scatter(beta_values,iters)
            ax[0].set_title("Iterations vs Beta Values")
            ax[0].set_xlabel("Beta Values")
            ax[0].set_ylabel("Iterations")
            
            #plots MSE vs beta for each eta
            ax[1].set_title("MSE Minimization Results vs Beta Values")
            rects1 = ax[1].bar(x-width,dataFidelity,width,label = "Fidelity")
            rects2 = ax[1].bar(x,modelTerm, width, label = "Model")
            rects3 = ax[1].bar(x +width, sum, width, label = "Sum")
            ax[1].set_xticklabels(labels)
            ax[1].set_xlabel("Beta Values")
            ax[1].set_ylabel("MSE Units")
            ax[1].legend()
            plt.show()'''

    print(bestEta,bestBeta,besteps,bestR2)

    #plot for max weight vector difference vs iterations
    '''w,wchange, k = gd(X_train, y_train,.1 ,1.0)


    plt.figure()
    plt.scatter(np.arange(1,k+ 2),wchange)
    plt.suptitle("Eta = {}   Beta = {}".format(.1,1.0))
    plt.title("Max Weight Difference vs Iterations")
    plt.ylabel("Max Weight Vectir Difference")
    plt.xlabel("Iterations")
    plt.ylim([-.05,2])
    
    plt.show()'''

    # select eta and beta, retrain and run test data
    w,wchange, k = gd(X_train, y_train, bestEta,bestBeta)
    print(w)
    pred = np.dot(X_test, w)

    plt.figure()
    
    
    result = list(zip(y_test, pred))
    
    result.sort()

    y_test = [x for (x,y) in result]
    pred = [y for (x,y) in result]
    plt.scatter(np.arange(1,np.size(y_test) + 1),pred,label ="Predicted Fantasy Points")
    plt.scatter(np.arange(1,np.size(y_test) + 1),y_test,label = "Actual Fantasy Points")
    plt.legend()
    plt.ylabel("Fantasy Points")
    plt.xlabel("QB")
    plt.title("Scatter Plot of Estimated and Actual Fantasy Points for Each Quarterback")

    plt.show()

main()
