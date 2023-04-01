"""
Machine Learning on frequency distribution
"""

import prefpy_io
import sys
import time
import logging
import json
import glob
import random
from random import sample
from itertools import permutations
from itertools import combinations
from itertools import chain
import numpy as np
from profile import *
from mechanism import *
from preference import *
from features import *
from scipy.special import comb
# import glovar

import config

from collections import OrderedDict

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor as MLPR
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV, RidgeClassifierCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
# from sklearn.externals import joblib
import joblib
from numpy import *
from ax import optimize

#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation
#from keras.layers import Conv1D, GlobalMaxPooling1D
#from keras.wrappers.scikit_learn import KerasRegressor

# import torch
# from torch.autograd import Variable
# import torchvision.transforms as transforms
# import torchvision.datasets as dsets
# import torch.nn
# import torch.optim as optim
import argparse
import scipy.stats as ss
np.random.seed(123)

from matplotlib import colors
from matplotlib.ticker import PercentFormatter
# Create a random number generator with a fixed seed for reproducibility
rng = np.random.default_rng(19680801)

# read json data from input file and creates training and test sets
# input: file name, file must be a pickled python dictionary
#   expects a python dictionary with two arrays with key names 'X' and 'y' where each row corresponds to a profile
# output: Xtrain, ytrain, Xtest, ytest
#   a split of the dataset into training and test sets
def read_input(inputfile, kfold):
    with open(inputfile) as f:
        dataset = json.loads(f.read())
    X = np.array(dataset['X'])
    # ***********************************
    kf = KFold(n_splits=kfold)
    train_index = []
    test_index = []
    for train, test in kf.split(X):
        train_index.append(train)
        test_index.append(test)

    return train_index, test_index


# read json data from input file and creates training and test sets
# for k-fold cross validation
def read_multi_input(inputfiles, kfold):
    # dataset = dict()
    # print(inputfiles)
    # X, y, filenames, orders = list(), list(), list(), list()
    X, y, filenames = list(), list(), list()
    for inputfile in inputfiles:
        with open(inputfile) as f:
            dataset = json.loads(f.read())
            # print(dataset)
            X.extend(dataset['X'])
            y.extend(dataset['y'])
            filenames.extend(dataset['filenames'])
            # orders.extend(dataset['orders'])
            # print(np.array(tempset['X']).shape)
            # dataset = {**dataset, **tempset}
    X = np.array(X)
    # print(X.shape)
    # ***********************************
    kf = KFold(n_splits=kfold)
    train_index = []
    test_index = []
    for train, test in kf.split(X):
        train_index.append(train)
        test_index.append(test)

    y = np.array(y)
    filenames = np.array(filenames)
    # orders = np.array(orders)
    return train_index, test_index, X, y, filenames#, orders


def read_input2(inputfile, ind):
    with open(inputfile) as f:
        dataset = json.loads(f.read())
    X = np.array(dataset['X'])
    y = np.array(dataset['y'])
    filenames = np.array(dataset['filenames'])  # 20191209
    orders = np.array(dataset['orders'])  # 20191209
    # print(filenames)
    # print("ok1=",np.shape(y))
    # m = np.shape(y)[1]
    logging.debug('read dataset: ' + str(np.shape(X)) + ',' + str(np.shape(y)))
    # ***********************************
    # Xtrain, Xtest, ytrain, ytest, filenames_train, filenames_test = train_test_split(X, y, filenames, test_size=0.0)
    # ***********************************
    # Xtest = Xtrain
    # ytest = ytrain
    # filenames_test = filenames_train
    # ***********************************
    Xtrain, Xtest, ytrain, ytest, filenames_train, filenames_test, orders_train, orders_test = \
        X[ind[0]:ind[1]], X[ind[2]:ind[3]], y[ind[0]:ind[1]], y[ind[2]:ind[3]], \
        filenames[ind[0]:ind[1]], filenames[ind[2]:ind[3]], orders[ind[0]:ind[1]], orders[ind[2]:ind[3]]
    # filenames_train = list(filenames_train)
    # filenames_test = list(filenames_test)
    # orders_train = list(orders_train)
    # orders_test = list(orders_test)
    # print(filenames_test)
    logging.debug('training set shape: ' + str(np.shape(Xtrain)) + ',' + str(np.shape(ytrain)))
    logging.debug('test set shape: ' + str(np.shape(Xtest)) + ',' + str(np.shape(ytest)))
    return Xtrain, ytrain, filenames_train, orders_train, Xtest, ytest, filenames_test, orders_test#, m


def read_simple_input(inputfile):
    # print(inputfile)
    with open(inputfile) as f:
        dataset = json.loads(f.read())
    X = np.array(dataset['X'])
    y = np.array(dataset['y'])
    filenames = np.array(dataset['filenames'])  # 20191209
    orders = np.array(dataset['orders'])  # 20191209
    return X, y, filenames, orders


# measure performance of learning algorithm and report
# input: ytest = true labels (winner/non-winner), yhattest = predicted labels (winner/non-winner)
# output: None
def measures(ytest, yhattest, clf_name, name='test set'):
    print(1.0 * np.sum(ytest.flatten()) / (np.shape(ytest)[0]))
    measurements = OrderedDict()
    # min_max_scaler = MinMaxScaler()
    # yhattest = min_max_scaler.fit_transform(yhattest.flatten().reshape(np.shape(yhattest))).astype(int)
    yhattest_flat = yhattest.flatten()
    ytest_flat = ytest.flatten()
    measurements['clf_name'] = clf_name
    measurements['name'] = name
    measurements['mean_squared_error'] = metrics.mean_squared_error(ytest_flat, yhattest_flat)
    logging.info('------------------------------------')
    for measure in measurements:
        logging.info(measure + ': ' + str(measurements[measure]))
    with open(config.mloutputfile, 'a') as fo:
        fo.write(json.dumps(measurements, indent=4, separators=(',', ': ')))


# dumb baseline
# input: training and test set.
# output: None
def baseline(Xtrain, ytrain, Xtest, ytest):
    (n, m) = np.shape(ytest)
    ytesthat = np.zeros((n, m))
    # print(n,m)
    for i in range(n):
        tmp = ytest[i, :]
        # print(tmp)
        # tmp = np.where(tmp == 1)[0][0]
        tmp = np.where(tmp >= 1)[0][0]
        ytesthat[i, tmp] = 1
    measures(ytest, ytesthat, 'baseline')


def output(ytesthat, filenames, method_name):
    with open(config.mloutputfile + '.' + method_name, 'w') as fo:
        # print(len(filenames))
        for i in range(len(filenames)):
            filenamesplit = filenames[i].rsplit('\\')
            # print(filenamesplit)
            name = filenamesplit.pop()

            # ordersplit = orders[i].rsplit('\\')
            # order = str(orders[i])
            # filename = dataset+'/'+name
            prediction = [str(ytesthat[i])]  # 1113
            # fo.write(name + '\t' + order + '\t' + ' '.join(prediction) + '\n')
            fo.write(name + '\t' + ' '.join(prediction) + '\n')


# learn and test a neural network
# input: training and test set.
# output: None
def nn(X, y, kfold, filenames_test, orders_test):
    regressor = MLPR(hidden_layer_sizes=(10 ** 2,), activation='logistic')
    y_pred = cross_val_predict(regressor, X, y, cv=kfold)
    measures(y, y_pred, 'nn')
    output(y_pred, filenames_test, orders_test, 'nn')


def nn2(Xtrain, ytrain, Xtest, ytest, filenames_test, orders_test, model_name, tup):
    regressor = MLPR(hidden_layer_sizes=tup, activation='logistic', early_stopping=True, validation_fraction=0.1, verbose=True, max_iter=6400)
    regressor.fit(Xtrain, ytrain)
    # save model
    joblib.dump(regressor, model_name)
    measures(ytrain, regressor.predict(Xtrain), 'nn', name='training set')
    yhattest = regressor.predict(Xtest)
    measures(ytest, yhattest, 'nn')
    output(yhattest, filenames_test, orders_test, 'nn')


def random_forest(Xtrain, ytrain, Xtest, ytest, filenames_test, orders_test, model_name, param):
    estimator = RandomForestRegressor(max_features=param[0], n_estimators=param[1], max_depth=param[2],
                                      min_samples_leaf=param[3], oob_score=param[4], random_state=param[5])  # before 20191122
    # estimator.fit(Xtrain, ytrain)
    estimator.fit(Xtrain, ytrain)
    # save model
    joblib.dump(estimator, model_name)
    measures(ytrain, estimator.predict(Xtrain), 'random_forest', name='training set')
    yhattest = estimator.predict(Xtest)
    measures(ytest, yhattest, 'random_forest')
    output(yhattest, filenames_test, orders_test, 'random_forest')


def linearreg(Xtrain, ytrain, Xtest, ytest, filenames_test, orders_test, model_name):
    regressor = LinearRegression()
    regressor.fit(Xtrain, ytrain)
    # save model
    joblib.dump(regressor, model_name)
    measures(ytrain, regressor.predict(Xtrain), 'LinearRegression', name='training set')
    yhattest = regressor.predict(Xtest)
    measures(ytest, yhattest, 'LinearRegression')
    output(yhattest, filenames_test, orders_test, 'LinearRegression')


def logisticreg(Xtrain, ytrain, Xtest, ytest, filenames_test, orders_test, model_name):
    regressor = LogisticRegression(random_state=123, solver='saga', n_jobs=-1, max_iter=100)
    regressor.fit(Xtrain, ytrain)
    # save model
    joblib.dump(regressor, model_name)
    measures(ytrain, regressor.predict(Xtrain), 'LogisticRegression', name='training set')
    yhattest = regressor.predict(Xtest)
    measures(ytest, yhattest, 'LogisticRegression')
    output(yhattest, filenames_test, orders_test, 'LogisticRegression')


# acutally do multilabel learning
# input: training and test set, a classification algorithm instantiated
# output: None
def multilabel_actual(baseclf, baseclf_name, Xtrain, ytrain, Xtest, ytest, filenames_test, orders_test, model_name):
    clf = OneVsRestClassifier(baseclf)
    clf.fit(Xtrain, ytrain)
    # save model
    joblib.dump(clf, model_name)
    yhattest = clf.predict(Xtest)
    measures(ytest, yhattest, baseclf_name)
    try:
        yhattest_proba = clf.predict_proba(Xtest)
        output(yhattest_proba, filenames_test, orders_test, baseclf_name)
    except:
        output(yhattest, filenames_test, orders_test, baseclf_name)


# try this !!!
# higher level function to learn and test using various multi label prediction algorithms
# input: training and test set.
# output: None
def multilabel(Xtrain, ytrain, Xtest, ytest, filenames_test, orders_test, model_name):
    # baseclf = LogisticRegressionCV()
    # multilabel_actual(baseclf, 'logistic', Xtrain, ytrain, Xtest, ytest, filenames_test, orders_test, model_name)
    # baseclf = SVC(kernel='linear', class_weight='balanced')
    # multilabel_actual(baseclf, 'svc_linear', Xtrain, ytrain, Xtest, ytest, filenames_test, orders_test, model_name)
    # baseclf = SVC(kernel='rbf', class_weight='balanced')
    # multilabel_actual(baseclf, 'svc_rbf', Xtrain, ytrain, Xtest, ytest, filenames_test, orders_test, model_name)
    baseclf = SVR(kernel='rbf')
    multilabel_actual(baseclf, 'svc_rbf', Xtrain, ytrain, Xtest, ytest, filenames_test, orders_test, model_name)
    # baseclf = KernelRidge(kernel='rbf')
    # multioutput_actual(baseclf, 'kernel_ridge_rbf', Xtrain, ytrain, Xtest, ytest, filenames_test, orders_test, model_name)


# acutally do multioutput learning
# input: training and test set, a classification algorithm instantiated
# output: None
def multioutput_actual(baseclf, baseclf_name, Xtrain, ytrain, Xtest, ytest, filenames_test, model_name):
    clf = MultiOutputClassifier(baseclf)
    clf.fit(Xtrain, ytrain)
    print("---after fitting---")
    # save model
    joblib.dump(clf, model_name)
    yhattest = clf.predict(Xtest)
    measures(ytest, yhattest, baseclf_name)
    try:
        yhattest_proba = clf.predict_proba(Xtest)
        output(yhattest_proba, filenames_test, baseclf_name)
    except:
        output(yhattest, filenames_test, baseclf_name)


def pointwise_rankerror(y_pred, y):
    N = len(y)
    rank_pred = ss.rankdata(y_pred.detach())
    # print(rank_pred)
    rank_y = ss.rankdata(y.detach())
    # print(rank_y)
    # print("-=", sum((rank_pred - rank_y)**2),N)
    return sum((rank_pred - rank_y)**2)/N


# higher level function to learn and test using various classifiers using multioutput option
# learns a separate classifier for each candidate. each label is treated as a separate problem
# needed to extend algorithms that do not support multilabel output
# input: training and test set
# output: None
def multioutput(Xtrain, ytrain, Xtest, ytest, filenames_test, model_name):  # 20220224
    # baseclf = RidgeClassifierCV(class_weight='balanced')
    # multioutput_actual(baseclf, 'ridge', Xtrain, ytrain, Xtest, ytest, filenames_test)
    # baseclf = KernelRidge(kernel='rbf')
    # multioutput_actual(baseclf, 'kernel_ridge_rbf', Xtrain, ytrain, Xtest, ytest, filenames_test, model_name)   # 20220224
    # baseclf = KernelRidge(kernel='linear')
    # multioutput_actual(baseclf, 'kernel_ridge_linear', Xtrain, ytrain, Xtest, ytest, filenames_test)
    print("---Now processing multioutput_actual---")
    baseclf = LogisticRegression(random_state=123, solver='saga', n_jobs=-1, max_iter=1000)
    multioutput_actual(baseclf, 'LogisticRegression', Xtrain, ytrain, Xtest, ytest, filenames_test, model_name)


# initialize stuff
def initialize(args):
    if len(args) == 2:
        config.inputfile = args[0]
        config.mloutputfile = args[1]
    else:
        logging.error('2 command line arguments expected: input_file, output_file')
    with open(config.mloutputfile, 'w') as fo:
        pass


# Build and configure the deep neural network model using Keras library
# input: m is the # candidates, mfts is the # features for each profile. this sets the output units and the input
# size respectively
# output: a Keras model
# structure: 1D convolution layer -> Pooling -> Dense -> Output
# all activations are sigmoid
# def build_deep(m, mfts):
#     filters = m + 1
#     kernel_size = 3
#     hidden_dims = m
#     # create a sequential model
#     model = Sequential()
#     # add a 1D convolutional layer
#     model.add(Conv1D(filters, kernel_size, activation='sigmoid', input_shape=(1, mfts), strides=1, dilation_rate=1,
#                      padding='same'))
#     # group filters
#     model.add(GlobalMaxPooling1D())
#     # usual dense hidden layer with dropout
#     model.add(Dense(hidden_dims))
#     model.add(Dropout(0.2))
#     model.add(Activation('relu'))
#     # output layer with m output units and sigmoid activation
#     model.add(Dense(m))
#     model.add(Activation('sigmoid'))
#     model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
#     return model


# Run deep neural networks.
# Currently using regression model
# TODO: implement class balance
# def deep(Xtrain, ytrain, Xtest, ytest, filenames_test):
#     (n, m) = np.shape(ytrain)
#     logging.debug('ytrain shape' + str((n, m)))
#     (nsamples, mfts) = np.shape(Xtrain)
#     logging.debug('Xtrain shape' + str((nsamples, mfts)))
#     Xtrain_tensor = Xtrain.reshape(Xtrain.shape[0], 1, Xtrain.shape[1])
#     Xtest_tensor = Xtest.reshape(Xtest.shape[0], 1, Xtest.shape[1])
#     logging.debug('Xtrain_tensor shape' + str(np.shape(Xtrain_tensor)))
#     regressor = KerasRegressor(build_fn=build_deep, m=m, mfts=mfts, epochs=200, batch_size=5, verbose=1)
#     regressor.fit(Xtrain_tensor, ytrain)
#     measures(ytrain, regressor.predict(Xtrain_tensor), 'deep', name='training set')
#     yhattest = regressor.predict(Xtest_tensor)
#     logging.debug('deep prediction shape ' + str(np.shape(yhattest)))
#     measures(ytest, yhattest, 'deep')
#     output(yhattest, filenames_test, 'deep')


# training function
# read dataset with features and winners
def train(param):


    # -----------Nov 29 2019-----------

    # tenfold cross validation
    # input dataset
    kfold = 10

    ml_features = glob.glob(config.mlinputfile)
    ml_features = sorted(ml_features)
    print("Start reading features.")
    time0 = time.perf_counter()
    train_index, test_index, X, y, filenames = read_multi_input(ml_features, kfold)
    time1 = time.perf_counter()
    print("Complete reading features, taking {:.4f} s.".format(time1 - time0))
    # X = np.delete(X, -2, axis=1)  # delete feature:lp
    # X = np.delete(X, list(range(glovar.m * glovar.m)), axis=1)  # 20211207 delete positional matrix
    print("X size=", X.shape)

    # y = y.reshape(-1, 1) # 20211210

    print("y size=", y.shape)
    # print(np.isnan(X).any())  # False
    # print(np.isnan(y).any())  # False
    print("X is finite? ", np.isfinite(X).any())
    print("y is finite? ", np.isfinite(y).any())

    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    print("X size=", X.shape)
    # print(np.isnan(X).any())  # False
    # print(np.isnan(y).any())  # False
    print("X is finite? ", np.isfinite(X).any())
    print("y is finite? ", np.isfinite(y).any())


    # m = np.shape(y)[1]
    # logging.debug('read dataset: ' + str(np.shape(X)) + ',' + str(np.shape(y)))

    # For Neural Network
    # # hidden_layer_size = [(100, ), (64, ), (128, ), (256, ), (256, 128), (128, 128), (128, 64), (256, 64), (256, 256), (512, 256)]
    # # hidden_layer_size = [(200,), (128,), (256,), (512,), (512, 256), (256, 256), (256, 128), (512, 128), (512, 512), (1024, 512)]  # 20171231-1831


    #
    # # 20191204 for len1
    # param = [(60, 120, 50, 1, True, 123),
    #          (60, 130, 50, 1, True, 123),  # 272408.378628825 best
    #          (70, 120, 50, 1, True, 123),  # 489402.6703553372
    #          (70, 130, 50, 1, True, 123),
    #          (20, 30, 20, 1, False, 123),  # 722858.9371350572 before 20191122
    #          (20, 30, 20, 1, True, 123),  # 722858.9371350572 before 20191122
    #          (70, 140, 50, 1, True, 123),  # 647561.7051221252
    #          (75, 130, 50, 1, True, 123),  # 597827.8243595813
    #          (75, 135, 50, 1, True, 123),  # 618898.9596279871
    #          (80, 150, 50, 1, True, 123)]  # 489402.47381186247
    #
    for i in range(0, kfold):
        train = train_index[i]
        test = test_index[i]
        print("Test index range: {}~{}".format(test[0], test[-1]))
        time2 = time.perf_counter()
        X_train, X_test, y_train, y_test, filenames_train, filenames_test = \
            X[train], X[test], y[train], y[test], filenames[train], filenames[test]
        time3 = time.perf_counter()
        print("Complete splitting datasets, taking {:.4f} s.".format(time3 - time2))
        model_name = config.model_name_prefix + '-test_' + str(test[0]) +'-'+ str(test[-1]) + '.pkl'
        print("Start creating Model:{}.".format(model_name))
        print("average y test=", np.mean(y[test]))

        try:
            # nn2(X_train, y_train, X_test, y_test, filenames_test, orders_test, model_name, param[i])
            # random_forest(X_train, y_train, X_test, y_test, filenames_test, orders_test, model_name, param[i])
            multioutput(X_train, y_train, X_test, y_test, filenames_test, model_name)

            time4 = time.perf_counter()
            print('Model {} with Parameters {} Completed! Use {:.4f} s.'.format(i + 1, param[i], time4 - time3))
        except MemoryError as error:
            # Output expected MemoryErrors.
            log_exception(error)

        break
    end = time.perf_counter()
    print("Complete k-fold cross validation. Total time is {:.4f} s.".format(end - time0))


def log_exception(exception: BaseException, expected: bool = True):
    """Prints the passed BaseException to the console, including traceback.

    :param exception: The BaseException to output.
    :param expected: Determines if BaseException was expected.
    """
    output = "[{}] {}: {}".format('EXPECTED' if expected else 'UNEXPECTED', type(exception).__name__, exception)
    print(output)


def test_model():

    # ind = [None, None, None, None]
    # ml_features = glob.glob(config.mlinputfile)
    # ml_features = sorted(ml_features)
    # X, y, filenames, orders = list(), list(), list(), list()
    # for ml_file in ml_features:
    #     X_, y_, filenames_, orders_, _, _, _, _ = read_input2(ml_file, ind)
    #     X += X_.tolist()
    #     y += y_.tolist()
    #     filenames += filenames_
    #     orders += orders_
    # print("-----------start computing------------")
    # # print(X_train)
    # print("X size", len(X))
    # print("-1", filenames[-1])
    # print("2569599", filenames[2469599])
    # print("2569600", filenames[2469600])
    # print("2569601", filenames[2469601])
    # print("2569602", filenames[2469602])
    #
    # break_ind = 2469600
    # X_train, y_train, filenames_train, orders_train = np.array(X[0:break_ind]), np.array(y[0:break_ind]), filenames[0:break_ind], orders[0:break_ind]
    # X_test, y_test, filenames_test, orders_test = np.array(X[break_ind:]), np.array(y[break_ind:]), filenames[break_ind:], orders[break_ind:]
    #
    # print("X train size", len(X_train))
    # print("X size size", len(X_test))

    # load model
    # time0 = time.perf_counter()
    # mdl = joblib.load(config.model_name)
    # time1 = time.perf_counter()
    # print("okkkk")
    # yhattest = mdl.predict(X_test)
    # measures(y_test, yhattest, 'random_forest')
    # output(yhattest, filenames_test, orders_test, 'random_forest')

    # print(mdl.predict(X_test[0:1, :]))
    # time2 = time.perf_counter()
    # print(X_test[0:1, :])
    # print("load time = ", time1 - time0)
    # print("computing time = ", time2 - time1)

    # kfold = 10
    #
    # ml_features = glob.glob(config.mlinputfile)
    # ml_features = sorted(ml_features)
    # print("Start reading features.")
    # time0 = time.perf_counter()
    # train_index, test_index, X, y, filenames, orders = read_multi_input(ml_features, kfold)
    # time1 = time.perf_counter()
    # print("Complete reading features, taking {:.4f} s.".format(time1 - time0))
    #
    # print("X size=", X.shape)

    os.chdir(config.profile_folder)
    filenames = sorted(glob.glob(config.profile_filenames))
    print(filenames[0], filenames[9999], filenames[10000], filenames[-1])
    print(filenames[0], filenames[999], filenames[1000], filenames[-1])
    # filenames = filenames[2000:7223]


def predict():

    config.mltest = config.result_path + 'M10N10k-soi2-100-dist_3.json'
    config.model_name = config.models_path + 'report-nn-M10N10ksoi2-all1M-pr3-test_n-a.pkl'
    config.mloutputfile = config.result_path + 'report-nn-M10N10ksoi2-all1M-pr3-test-d3-100.json'
    X_test, y_test, filenames_test, orders_test = read_simple_input(config.mltest)
    print("X_test size = {}".format(X_test.shape))
    print("-------Features completed!----------")
    mdl = joblib.load(config.model_name)
    yhattest = mdl.predict(X_test)
    measures(y_test, yhattest, 'nn')
    output(yhattest, filenames_test, orders_test, 'nn')


def test():
    FORMAT = "%(asctime)s %(levelname)s %(module)s %(lineno)d %(funcName)s:: %(message)s"
    logging.basicConfig(filename='common.log', filemode='a', level=logging.DEBUG, format=FORMAT)

    # size = (900000, 231)
    config.mlinputfile = config.result_path + 'M16N10k-soc3-100k-experiment-20220223.json'
    config.write_model_name = config.models_path + 'model-KRR-M16N10k-soc3-100k-experiment-20220223.pkl'
    config.mloutputfile = config.result_path + 'output-KRR-M16N10k-soc3-100k-experiment-20220223.json'
    config.model_name_prefix = 'model-KRR-M16N10k-soc3-100k-experiment-20220223'
    # param = (128, 256)
    # param = [(400,), (256,), (512,), (1024,), (1024, 512), (512, 512), (512, 256), (1024, 256), (1024, 1024),
    #                      (2048, 1024)]  # 20171231-1835 # 20200512 (512, 256) best
    # param = [(1024,), (2048,), (256, 128), (256, 256), (512, 128), (512, 256), (512, 512), (512, 256, 128), (512, 128, 128),
    #          (1024, 256)]  # 20210816 # (512, 256, 128) best 63879.280989763305
    param = [(2048, 1024), (2048, 512), (512, 256, 256), (1024, 512, 128), (1024, 512, 256), (2048, 512, 128), (2048, 1024, 512),
             (4096, 256), (4096, 1024), (4096, 2048)]  # 20211218
    # param = [(1024, 256)]  # 20210816 # (2048,) best
    # For Random Forest, (max_features, n_estimator, max_depth, min_sample_leaf, oob_score, random_state)
    # param = [(60, 120, 50, 1, True, 123),
    #          (60, 130, 50, 1, True, 123),  # 272408.378628825 best
    #          (30, 120, 50, 1, True, 123),  # 489402.6703553372
    #          (30, 130, 50, 1, True, 123),
    #          (20, 30, 20, 1, False, 123),  # 722858.9371350572 before 20191122
    #          (20, 30, 20, 1, True, 123),  # 722858.9371350572 before 20191122
    #          (40, 140, 50, 1, True, 123),  # 647561.7051221252
    #          (45, 130, 50, 1, True, 123),  # 597827.8243595813  "mean_squared_error": 153962.35597861145
    #          (50, 135, 50, 1, True, 123),  # 618898.9596279871
    #          (55, 150, 50, 1, True, 123)]  # 489402.47381186247
    train(param)

    # config.write_model_name = config.models_path + 'report-nn-M10N10ksoi2-all1.1M_2M-d9-test_1M_1.1M-param-16_16.pkl'
    # config.mloutputfile = config.result_path + 'report-nn-M10N10ksoi2-all1.1M_2M-d9-test_1M_1.1M-param-16_16.json'
    # param = (16, 16)
    # train(param)
    #
    # config.write_model_name = config.models_path + 'report-nn-M10N10ksoi2-all1.1M_2M-d9-test_1M_1.1M-param-16_16_16.pkl'
    # config.mloutputfile = config.result_path + 'report-nn-M10N10ksoi2-all1.1M_2M-d9-test_1M_1.1M-param-16_16_16.json'
    # param = (16, 32, 16)
    # train(param)


def test_features():
    FORMAT = "%(asctime)s %(levelname)s %(module)s %(lineno)d %(funcName)s:: %(message)s"
    logging.basicConfig(filename='common.log', filemode='a', level=logging.DEBUG, format=FORMAT)

    config.mlinputfile = config.result_path + 'resultsM5N10k-soc3-1m-experiment-20211122.json'
    ml_features = glob.glob(config.mlinputfile)
    ml_features = sorted(ml_features)
    print("Start reading features.")
    kfold = 10
    time0 = time.perf_counter()
    train_index, test_index, X, y, filenames, orders = read_multi_input(ml_features, kfold)
    time1 = time.perf_counter()
    print("Complete reading features, taking {:.4f} s.".format(time1 - time0))

    print("X size=", X.shape)
    # print(np.isnan(X).any())  # False
    # print(np.isnan(y).any())  # False
    print("X is finite? ", np.isfinite(X).any())
    print("y is finite? ", np.isfinite(y).any())
    print("X0=", X[1], y[1])

    # fig, ax1 = plt.subplots()
    # colors0 = ['r', 'g', 'c', 'm', 'b', 'y', 'k']
    # filenames = ['l1norm vs min dist', 'l2norm vs min dist', 'fnorm vs min dist', 'linfnorm vs min dist']
    # for i in [1]:
    # # i = 0
    #     ax1.plot(X[:, i-4], y, '.', color=colors0[i], label=filenames[i], markersize=0.5)
    #
    # # plt.xscale('log')
    # plt.setp(ax1.get_xticklabels(), fontsize=12)
    # plt.legend(loc='lower right')
    # plt.xlabel("norm")
    # plt.ylabel("min dist")
    # plt.show()
    #
    # fig, ax2 = plt.subplots()
    # n_bins = 200
    #
    # # N is the count in each bin, bins is the lower-limit of the bin
    # N, bins, patches = ax2.hist(y, bins=n_bins)
    #
    # # We'll color code by height, but you could use any scalar
    # fracs = N / N.max()
    #
    # # we need to normalize the data to 0..1 for the full range of the colormap
    # norm = colors.Normalize(fracs.min(), fracs.max())
    #
    # # Now, we'll loop through our objects and set the color of each accordingly
    # for thisfrac, thispatch in zip(fracs, patches):
    #     color = plt.cm.viridis(norm(thisfrac))
    #     thispatch.set_facecolor(color)
    #
    # plt.show()




if __name__ == '__main__':
    # FORMAT = "%(asctime)s %(levelname)s %(module)s %(lineno)d %(funcName)s:: %(message)s"
    # logging.basicConfig(filename='common.log', filemode='a', level=logging.DEBUG, format=FORMAT)
    # train(sys.argv)

    # test_model()

    # predict()

    # Torchmain()

    test()

    # test_features()