import LinRegLearner as lr
import BagLearner as bl
import numpy as np


class InsaneLearner(object):
    def __init__(self, learner=lr.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=False):
        self.learner,self.kwargs,self.bags,self.boost,self.verbose=learner,kwargs,bags,boost,verbose
        pass

    def author(self):
        return 'alvov3'  # replace tb34 with your Georgia Tech username

    def addEvidence(self, dataX, dataY):
        self.bagResults = []
        for i in range(0, 20):
            learner = bl.BagLearner(self.learner, self.kwargs, self.bags, self.boost, self.verbose)
            learner.addEvidence(dataX, dataY)
            Y = learner.query(dataX)
            self.bagResults.append(Y)
        return self.bagResults

    def query(self, points):
        return np.mean(self.bagResults, axis=0)   # returns mean of all the results from the bag   learners (count 20).


if __name__ == "__main__":
    print "alvov3"