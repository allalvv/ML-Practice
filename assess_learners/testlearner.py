"""
Test a learner. 
"""

import numpy as np
import math
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rf
import BagLearner as bl
import sys
import util
import bagging_dt as bd
from timeit import default_timer as timer
import sklearn
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



if __name__=="__main__":
    datafile='Istanbul.csv'

    testX, testY, trainX, trainY = None, None, None, None
    permutation = None
    author = None
    with util.get_learner_data_file(datafile) as f:
        alldata = np.genfromtxt(f, delimiter=',')
        # Skip the date column and header row if we're working on Istanbul data
        if datafile == 'Istanbul.csv':
            alldata = alldata[1:, 1:]
        datasize = alldata.shape[0]
        cutoff = int(datasize * 0.6)
        permutation = np.random.permutation(alldata.shape[0])
        col_permutation = np.random.permutation(alldata.shape[1] - 1)
        train_data = alldata[permutation[:cutoff], :]
        # trainX = train_data[:,:-1]
        trainX = train_data[:, col_permutation]
        trainY = train_data[:, -1]
        test_data = alldata[permutation[cutoff:], :]
        # testX = test_data[:,:-1]
        testX = test_data[:, col_permutation]
        testY = test_data[:, -1]
    i=100
    while(i):
    # create a learner and train it
    #learner = lrl.LinRegLearner(verbose = True) # create a LinRegLearner
    #learner.addEvidence(trainX, trainY) # train it
    #print learner.author()

    #learner = bd.bagging_dt(leaf_size=50, verbose=False)  # constructor
    #learner.addEvidence(trainX, trainY)  # training step
    #Y = learner.query(trainX)  # query
        start = timer()
        learner = dt.DTLearner(leaf_size = 50, verbose = False) # constructor
        learner.addEvidence(trainX, trainY)  # training step
        predY = learner.query(trainX)  # query

    # evaluate in sample
    #predY = learner.query(trainX) # get the predictions
        end = timer()
        print(end - start)
        rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
        print
        print "In sample results"
        print "RMSE: ", rmse
        c = np.corrcoef(predY, y=trainY)
        print "corr: ", c[0,1]

        start = timer()
        predY = learner.query(testX)  # get the predictions

        end = timer()
        print(end - start)
        rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
        print

        rmse_BL_Test = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
        c_BL_Test = np.corrcoef(predY, y=testY)
        print "Out of sample results"
        print "RMSE: ", rmse
        c = np.corrcoef(predY, y=testY)
        print "corr: ", c[0,1]
        rmse_BL_Test = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
        c_BL_Test = np.corrcoef(predY, y=testY)
    i=i-10


