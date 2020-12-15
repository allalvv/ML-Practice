import numpy as np
import DTLearner as dt
import RTLearner as rt
import LinRegLearner as lrl
import InsaneLearner as it


class BagLearner(object):


    def __init__(self, learner, kwargs, bags, boost, verbose):
        self.learner = learner  # type of lerner that bag gets (Linear regression, DT,RT....
        self.kwargs = kwargs
        self.bags = bags
        self.boost = False  # Boosting is an optional topic and not required.
        self.verbose = verbose
        pass


    def author(self):
        return 'alvov3'  # replace tb34 with your Georgia Tech username


    def addEvidence(self, dataX, dataY):
        self.learners = []
        # self.kwargs = {"k": 10}
        for i in range(0, self.bags):
            self.learners.append(self.learner(**self.kwargs))

        if self.boost == False:  # Boosting is an optional topic and not required.
            for bag in self.learners:
                # create bags with random data for every model (learner)
                trainSize = int(0.6 * len(dataY))
                bagRows = np.random.randint(len(dataX), size=trainSize)
                bagDataX = dataX[bagRows, :]
                bagDataY = dataY[bagRows]
                bag.addEvidence(bagDataX, bagDataY)
        else:
            print "-------------------------------------------------------------","\n"
            print "Boosting is an optional topic and not required."
            print "-------------------------------------------------------------","\n"

        # print logs if verbose=True
        self.printLogs("BagLearner", "addEvidence", self.learners)
        return self.learners

    def query(self, points):
        bagResults = []  # saves results from all the bags
        for i in self.learners:
            bagResults.append(i.query(points))

        # print logs if verbose=True
        self.printLogs("BagLearner", "query", np.mean(bagResults, axis=0))

        return np.mean(bagResults, axis=0)  # returns mean of all the results from the bags.

    def printLogs(self, className, functionName, value):
        if self.verbose:
            print "\n"
            print ("Log - ", className, "||", functionName, "result:")
            print "\n"
            print value
            print "\n"
            print "-------------------------------------------------------------"
            print "\n"


if __name__ == "__main__":

   # print learner.author()
    print "the secret clue is 'zzyzx'"