import numpy as np


class RTLearner(object):


    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        pass


    def author(self):
        return 'alvov3'  # replace tb34 with your Georgia Tech username


    def printLogs(self, className, functionName, value):
        if self.verbose:
            print "\n"
            print ("Log - ", className, "||", functionName, "result:")
            print "\n"
            print value
            print "\n"
            print "-------------------------------------------------------------"
            print "\n"


    def buildTree(self, dataX, dataY):
        # Stopping criteria:
        # stop if number of rows <= the number of leaves allowed.
        if dataX.shape[0] <= self.leaf_size:
            return np.array([[-1, np.mean(dataY), np.nan, np.nan]])
        # if	all	data.y same:	return	[leaf,	data.y,	NA,	NA]
        # stop if  all y are the same
        elif len(set(dataY)) == 1:
            return np.array([[-1, np.mean(dataY), np.nan, np.nan]])
        # Building a tree
        else:
            samplesNum = dataX.shape[0]  # get row number
            featuresNum = dataX.shape[1]  # get col number

            # find a good split
            # true-not to split, search for a new value
            notToSplit = True

            while notToSplit:
                splitFeat = np.random.randint(featuresNum)  # get random feature to split
                splitVal = (dataX[np.random.randint(0, samplesNum), splitFeat] + dataX[
                    np.random.randint(0, samplesNum), splitFeat]) / 2

                # check if the split value are not max or min
                sortedBySplitFeature = np.sort(dataX[:, splitFeat])
                if splitVal != sortedBySplitFeature[0] and splitVal != sortedBySplitFeature[-1]:
                    notToSplit = False
                elif len(set(dataX[:, splitFeat])) == 1:  # create a leaf if feature has all the same values.
                    return np.array([[-1, np.mean(dataY), np.nan, np.nan]])
                if (len(set(dataX[:, splitFeat])) == 2):
                    return np.array([[-1, np.mean(dataY), np.nan, np.nan]])
            # build left tree
            leftTree = self.buildTree(dataX[dataX[:, splitFeat] <= splitVal], dataY[dataX[:, splitFeat] <= splitVal])

            # print logs if verbose=True
            self.printLogs("RTLearner", "buildTree -leftTree", leftTree)

            # build right tree
            rightTree = self.buildTree(dataX[dataX[:, splitFeat] > splitVal], dataY[dataX[:, splitFeat] > splitVal])

            # print logs
            self.printLogs("RTLearner", "buildTree-rightTree", rightTree)

            # set tree root
            root = np.array([[splitFeat, splitVal, 1, leftTree.shape[0] + 1]])
            # create a tree
            self.tree = np.append(root, np.append(leftTree, rightTree, axis=0), axis=0)

            # print logs if verbose=True
            self.printLogs("RTLearner", "buildTree - final tree", self.tree)

            return self.tree


    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        self.tree = self.buildTree(dataX, dataY)


    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        # get size of points
        pointsLen = len(points)
        predictedY = np.empty(pointsLen)

        for i in range(0, pointsLen):
            jumpToIndex = 0  # start from root
            # Check for a leaf
            while ~np.isnan(self.tree[jumpToIndex, 2]):
                factor = int(self.tree[jumpToIndex, 0])
                splitValue = self.tree[jumpToIndex, 1]
                if points[i, factor] <= splitValue:
                    jumpToIndex = jumpToIndex + 1
                else:
                    jumpToIndex = jumpToIndex + int(self.tree[jumpToIndex, 3])

            predictedY[i] = self.tree[jumpToIndex, 1]

        # print logs if verbose=True
        self.printLogs("RTLearner", "query - predicted data for Y", predictedY)

        return predictedY


if __name__ == "__main__":
    #print learner.author()
    print "the secret clue is 'zzyzx'"
