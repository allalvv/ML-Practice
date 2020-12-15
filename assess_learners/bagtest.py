import LinRegLearner as lr
import numpy as np

class InsaneLearner(object):


    def __init__(self,learner=lr.LinRegLearner , kwargs={}, bags = 20, boost = False, verbose = False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        pass

    def author(self):
        return 'alvov3'  # replace tb34 with your Georgia Tech username

    def addEvidence(self, dataX, dataY):
        self.learners = []  # create an emoty array where we will store the learners
        for i in range(0, self.bags):
            self.learners.append(self.learner(**self.kwargs))  # create the learners (as many as the # of bags)
        if self.boost == False:
            for l in self.learners:
                a = np.random.randint(0, high=X.shape[0], size=X.shape[0])
                dataX = X[a]
                dataY = Y[a]
                l.addEvidence(dataX, dataY)
        return self.learners
    def query(self, points):  # query the Y values by scnanning the TREE and then average the Y's

        Result = []
        for l in self.learners:
            Result.append(l.query(points))
        if self.verbose:
            print np.mean(Result, axis=0)
        return np.mean(Result, axis=0)  # average the Y's of each bag.

