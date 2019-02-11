import copy
import logging
import pandas as pd
import numpy as np

from collections import Counter

from sklearn import preprocessing, utils
import sklearn.model_selection as ms
from scipy.sparse import isspmatrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import os
import seaborn as sns

from abc import ABC, abstractmethod

# TODO: Move this to a common lib?
OUTPUT_DIRECTORY = '../output'

if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)
if not os.path.exists('{}/images'.format(OUTPUT_DIRECTORY)):
    os.makedirs('{}/images'.format(OUTPUT_DIRECTORY))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def plot_pairplot(title, df, class_column_name=None):
    plt = sns.pairplot(df, hue=class_column_name)
    return plt


# Adapted from https://stats.stackexchange.com/questions/239973/a-general-measure-of-data-set-imbalance
def is_balanced(seq):
    n = len(seq)
    classes = [(clas, float(count)) for clas, count in Counter(seq).items()]
    k = len(classes)

    H = -sum([(count/n) * np.log((count/n)) for clas, count in classes])
    return H/np.log(k) > 0.75


class DataLoader(ABC):
    def __init__(self, path, verbose, seed):
        self._path = path
        self._verbose = verbose
        self._seed = seed

        self.features = None
        self.classes = None
        self.testing_x = None
        self.testing_y = None
        self.training_x = None
        self.training_y = None
        self.binary = False
        self.balanced = False
        self._data = pd.DataFrame()

    def load_and_process(self, data=None, preprocess=True):
        """
        Load data from the given path and perform any initial processing required. This will populate the
        features and classes and should be called before any processing is done.

        :return: Nothing
        """
        if data is not None:
            self._data = data
            self.features = None
            self.classes = None
            self.testing_x = None
            self.testing_y = None
            self.training_x = None
            self.training_y = None
        else:
            self._load_data()
        self.log("Processing {} Path: {}, Dimensions: {}", self.data_name(), self._path, self._data.shape)
        if self._verbose:
            old_max_rows = pd.options.display.max_rows
            pd.options.display.max_rows = 10
            self.log("Data Sample:\n{}", self._data)
            pd.options.display.max_rows = old_max_rows

        if preprocess:
            self.log("Will pre-process data")
            self._preprocess_data()

        self.get_features()
        self.get_classes()
        self.log("Feature dimensions: {}", self.features.shape)
        self.log("Classes dimensions: {}", self.classes.shape)
        self.log("Class values: {}", np.unique(self.classes))
        class_dist = np.histogram(self.classes)[0]
        class_dist = class_dist[np.nonzero(class_dist)]
        self.log("Class distribution: {}", class_dist)
        self.log("Class distribution (%): {}", (class_dist / self.classes.shape[0]) * 100)
        self.log("Sparse? {}", isspmatrix(self.features))

        if len(class_dist) == 2:
            self.binary = True
        self.balanced = is_balanced(self.classes)

        self.log("Binary? {}", self.binary)
        self.log("Balanced? {}", self.balanced)

    def scale_standard(self):
        self.features = StandardScaler().fit_transform(self.features)
        if self.training_x is not None:
            self.training_x = StandardScaler().fit_transform(self.training_x)

        if self.testing_x is not None:
            self.testing_x = StandardScaler().fit_transform(self.testing_x)

    def build_train_test_split(self, test_size=0.3):
        if not self.training_x and not self.training_y and not self.testing_x and not self.testing_y:
            self.training_x, self.testing_x, self.training_y, self.testing_y = ms.train_test_split(
                self.features, self.classes, test_size=test_size, random_state=self._seed, stratify=self.classes
            )

    def get_features(self, force=False):
        if self.features is None or force:
            self.log("Pulling features")
            self.features = np.array(self._data.iloc[:, 0:-1])

        return self.features

    def get_classes(self, force=False):
        if self.classes is None or force:
            self.log("Pulling classes")
            self.classes = np.array(self._data.iloc[:, -1])

        return self.classes

    def dump_test_train_val(self, test_size=0.2, random_state=123):
        ds_train_x, ds_test_x, ds_train_y, ds_test_y = ms.train_test_split(self.features, self.classes,
                                                                           test_size=test_size,
                                                                           random_state=random_state,
                                                                           stratify=self.classes)
        pipe = Pipeline([('Scale', preprocessing.StandardScaler())])
        train_x = pipe.fit_transform(ds_train_x, ds_train_y)
        train_y = np.atleast_2d(ds_train_y).T
        test_x = pipe.transform(ds_test_x)
        test_y = np.atleast_2d(ds_test_y).T

        train_x, validate_x, train_y, validate_y = ms.train_test_split(train_x, train_y,
                                                                       test_size=test_size, random_state=random_state,
                                                                       stratify=train_y)
        test_y = pd.DataFrame(np.where(test_y == 0, -1, 1))
        train_y = pd.DataFrame(np.where(train_y == 0, -1, 1))
        validate_y = pd.DataFrame(np.where(validate_y == 0, -1, 1))

        tst = pd.concat([pd.DataFrame(test_x), test_y], axis=1)
        trg = pd.concat([pd.DataFrame(train_x), train_y], axis=1)
        val = pd.concat([pd.DataFrame(validate_x), validate_y], axis=1)

        tst.to_csv('data/{}_test.csv'.format(self.data_name()), index=False, header=False)
        trg.to_csv('data/{}_train.csv'.format(self.data_name()), index=False, header=False)
        val.to_csv('data/{}_validate.csv'.format(self.data_name()), index=False, header=False)

    @abstractmethod
    def _load_data(self):
        pass

    @abstractmethod
    def data_name(self):
        pass

    @abstractmethod
    def _preprocess_data(self):
        pass

    @abstractmethod
    def class_column_name(self):
        pass

    @abstractmethod
    def pre_training_adjustment(self, train_features, train_classes):
        """
        Perform any adjustments to training data before training begins.
        :param train_features: The training features to adjust
        :param train_classes: The training classes to adjust
        :return: The processed data
        """
        return train_features, train_classes

    def reload_from_hdf(self, hdf_path, hdf_ds_name, preprocess=True):
        self.log("Reloading from HDF {}".format(hdf_path))
        loader = copy.deepcopy(self)

        df = pd.read_hdf(hdf_path, hdf_ds_name)
        loader.load_and_process(data=df, preprocess=preprocess)
        loader.build_train_test_split()

        return loader

    def log(self, msg, *args):
        """
        If the learner has verbose set to true, log the message with the given parameters using string.format
        :param msg: The log message
        :param args: The arguments
        :return: None
        """
        if self._verbose:
            logger.info(msg.format(*args))


class BankMarketingData(DataLoader):

    def __init__(self, path='data/BankMarketingData.csv', verbose=False, seed=1):
    #def __init__(self, path='BankMarketingData.csv', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
       # self._data = pd.read_excel(self._path, header=1, index_col=0)

        self._data =pd.read_csv(self._path)
    def data_name(self):
        return 'BankMarketingData'

    def class_column_name(self):
        return 'Bank Marketing Data'

    def _preprocess_data(self):
        #self._data = pd.read_csv('data/BankMarketingData.csv')
        print("Data has", len(self._data), "rows and", len(self._data.columns), "columns.")
        if self._data.isnull().values.any():
            print("Warning: Missing Data")

        to_encode = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week',
                    'poutcome']
        df = self._data[to_encode]
        df = pd.get_dummies(df).astype('category')
        df_others = self._data.drop(to_encode, axis=1)
        self._data = pd.concat([df_others, df], axis=1)
        column_order = list(self._data)
        column_order.insert(0, column_order.pop(column_order.index('y')))

        self._data = self._data.loc[:, column_order]
        self._data['y'].replace(['no', 'yes'], [0, 1], inplace=True)
        self._data['y'] = self._data['y'].astype('category')

        numericcols = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx',
                       'cons.conf.idx',
                       'euribor3m', 'nr.employed']
        df_num = self._data[numericcols]
        df_stand = (df_num - df_num.min()) / (df_num.max() - df_num.min())
        df_bank_categorical = self._data.drop(numericcols, axis=1)
        self._data = pd.concat([df_bank_categorical, df_stand], axis=1)
        self._data.columns = range(self._data.shape[1])
        column_order = list(self._data)
        column_order.insert(63, column_order.pop(column_order.index(0)))

        self._data = self._data.loc[:, column_order]
        self._data.describe(include='all')

    def pre_training_adjustment(self, train_features, train_classes):
        """
        Perform any adjustments to training data before training begins.
        :param train_features: The training features to adjust
        :param train_classes: The training classes to adjust
        :return: The processed data
        """
        # #standardize
        # names = train_features.columns
        # # Create the Scaler object
        # scaler = preprocessing.StandardScaler()
        # # Fit data on the scaler object
        # scaled_df = scaler.fit_transform(train_features)
        # train_features = pd.DataFrame(scaled_df, columns=names)

        # standardize

        # # Create the Scaler object
        # scaler = preprocessing.StandardScaler()
        # # Fit data on the scaler object
        # train_features = scaler.fit_transform(train_features)

        return train_features, train_classes

class CreditApprovalData(DataLoader):

    def __init__(self, path='data/crx.data', verbose=False, seed=1):
    #def __init__(self, path='crx.data', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        self._data = pd.read_csv(self._path, header=None)

    def data_name(self):
        return 'CreditApprovalData'

    def class_column_name(self):
        return 'Credit Approval Data'

    def _preprocess_data(self):
        #dropp null value
        self._data.dropna(how='all')
        self._data = self._data[(self._data != '?').all(axis=1)]

        # replase nan values with mean
        # self._data=self._data.fillna(self._data.mean())

        to_encode = [0, 3, 4, 5, 6, 8, 9, 11, 12]
        df = self._data[to_encode]
        df = pd.get_dummies(df).astype('category')
        df_others = self._data.drop(to_encode, axis=1)
        df_credit = pd.concat([df_others, df], axis=1)
        column_order = list(df_credit)
        column_order.insert(0, column_order.pop(column_order.index(15)))
        df_credit = df_credit.loc[:, column_order]
        df_credit[15].replace(['+', '-'], [1, 0], inplace=True)
        df_credit[15] = df_credit[15].astype('category')


        numericcols = [1, 2, 7,10,13,14]
        df_num = df_credit[numericcols]
        if df_num.isnull().values.any():
            print("Warning: Missing Data")
        df_num = df_num.astype(float).fillna(0.0)
        df_stand = (df_num - df_num.min()) / (df_num.max() - df_num.min())
        df_credit_categorical = df_credit.drop(numericcols, axis=1)

        self._data = pd.concat([df_credit_categorical, df_stand], axis=1)
        self._data.columns = range(self._data.shape[1])
        column_order = list(self._data)
        column_order.insert(46, column_order.pop(column_order.index(0)))
        self._data = self._data.loc[:, column_order]

        self._data.describe(include='all')

    def pre_training_adjustment(self, train_features, train_classes):
        #standardize

        # # Create the Scaler object
        # scaler = preprocessing.StandardScaler()
        # # Fit data on the scaler object
        # train_features = scaler.fit_transform(train_features)

        return train_features, train_classes


if __name__ == '__main__':
    # cd_data = BankMarketingData(verbose=True)
    # cd_data.load_and_process()

    ca_data = CreditApprovalData(verbose=True)
    ca_data.load_and_process()
