from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np
import data_loader as dtl
import logistic_method as algs


def hamdist(str1, str2):
    """Count the # of differences between equal length strings str1 and str2"""
    diffs = 0
    for ch1, ch2 in zip(str1, str2):
        if ch1 != ch2:
            diffs += 1
    return diffs


def getaccuracy(ytest, predictions):
    correct = 0


    for i in range(len(ytest)):

        if ytest[i] == predictions[i]:
            correct += 1
    return (correct/float(len(ytest))) * 100.0

def geterror(ytest, predictions):
    return (100.0-getaccuracy(ytest, predictions))
def y_digit(ytrain):
	k=np.zeros(ytrain.shape[0])
	for i in range(ytrain.shape[0]):
		if len(np.where(ytrain[i]==1)[0])!=0:
			k[i]=np.where(ytrain[i]==1)[0][0]
	return k

if __name__ == '__main__':
    trainsize = 1000
    testsize = 500
    numruns = 10

    classalgs = {#'Random': algs.Classifier(),
                 #'Naive Bayes': algs.NaiveBayes({'usecolumnones': False}),
                 #'Naive Bayes Ones': algs.NaiveBayes({'usecolumnones': True}),
                 #'Linear Regression': algs.LinearRegressionClass(),
                 'Logistic Regression': algs.LogitReg({'regularizer':'l2'}),
                 #'Neural Network': algs.NeuralNet({'epochs': 500,'stepsize':0.06}),
                #'kernel Logistic Regression': algs.KernelLogitReg({'regularizer':'l2','kernel':'hamming'}),

    }
    numalgs = len(classalgs)

    parameters = (
        {'regwgt': 0.0, 'nh': 16},
        #{'regwgt': 0.01, 'nh': 8},
        #{'regwgt': 0.05, 'nh': 16},
        #{'regwgt': 0.1, 'nh': 32},
                      )
    numparams = len(parameters)

    errors = {}
    for learnername in classalgs:
        errors[learnername] = np.zeros((numparams,numruns))

    for r in range(numruns):
        #trainset, testset = dtl.load_susy(trainsize,testsize)
        #trainset, testset = dtl.load_susy_complete(trainsize,testsize)
        trainset, testset = dtl.load_train_and_test_data(1000,500)

        print(('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0],r))

        for p in range(numparams):
            params = parameters[p]
            for learnername, learner in classalgs.items():
                # Reset learner for new parameters
                learner.reset(params)
                print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                # Train model
                learner.learn(trainset[0], trainset[1])
                # Test model
                predictions = learner.predict(testset[0])
                ytestset = y_digit(testset[1])

                print ytestset
                print predictions
                print hamdist(ytestset,predictions)

                error = geterror(ytestset, predictions)

                print ('Error for ' + learnername + ': ' + str(error))
                errors[learnername][p,r] = error


    for learnername, learner in classalgs.items():
        besterror = np.mean(errors[learnername][0,:])
        bestparams = 0
        for p in range(numparams):
            aveerror = np.mean(errors[learnername][p,:])
            if aveerror < besterror:
                besterror = aveerror
                bestparams = p

        # Extract best parameters
        learner.reset(parameters[bestparams])
        print ('Best parameters for ' + learnername + ': ' + str(learner.getparams()))
        print ('Average error for ' + learnername + ': ' + str(besterror) + ' +- ' + str(np.std(errors[learnername][bestparams,:])/math.sqrt(numruns)))
