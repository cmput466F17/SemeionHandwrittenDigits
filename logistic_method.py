from __future__ import division  # floating point division
import numpy as np
import utilities as utils
import math


class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """

    def __init__(self, parameters={}):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}

    def reset(self, parameters):
        """ Reset learner """
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        try:
            utils.update_dictionary_items(self.params, parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """

    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest


class LogitReg(Classifier):
    def __init__(self, parameters={}):
        # Default: no regularization
        self.params = {'regwgt': 0.0, 'regularizer': 'None'}
        self.reset(parameters)


    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        if self.params['regularizer'] is 'l1':
            self.regularizer = (utils.l1, utils.dl1)
        elif self.params['regularizer'] is 'l2':
            self.regularizer = (utils.l2, utils.dl2)
        else:
            self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))


    def logit_cost(self, theta, X, y):
        """
        Compute cost for logistic regression using theta as the parameters.
        """
        cost=0.0
        ### YOUR CODE HERE
        yhat =utils.sigmoid(np.dot(X, theta.T))
        cost =-np.dot(y,np.log(yhat))-np.dot((1-y),np.log(1-yhat))+self.params['regwgt']*self.regularizer[0](theta)
        ###END YOUR CODE
        return cost

    def logit_cost_grad(self, theta, X, y):
        """
        Compute gradients of the cost with respect to theta.
        """
        grad = np.zeros(len(theta))
        ### YOUR CODE HERE
        grad=np.dot((utils.sigmoid(np.dot(X, theta.T)) - y).T,X)+self.params['regwgt']*self.regularizer[1](theta)
        #ask ta
        return grad

    def learn(self, Xtrain, ytrain):
        """
        Learn the weights using the training data
        """
        self.weights = np.zeros(Xtrain.shape[1], )
        ### YOUR CODE HERE
        stepsize = 0.03
        epoch =1000
        w = np.zeros((ytrain.shape[1],Xtrain.shape[1]))

        for i in range(epoch):
            Xtrain, ytrain = self.unison_shuffled_copies(Xtrain, ytrain)
            for j in range(Xtrain.shape[0]):
                X = np.array(Xtrain[j, :], ndmin=2)
                y = np.array(ytrain[j,:],ndmin = 2)
                g= self.logit_cost_grad(w,X,y)
                w = w - (stepsize * 1.0/(i + 1))*g
        self.weights = w

        ### END YOUR CODE

    def predict(self, Xtest):
        """
        Use the parameters computed in self.learn to give predictions on new
        observations.
        """
        ### YOUR CODE HERE
        value = utils.sigmoid(np.dot(Xtest, self.weights.T))
        ytest = np.zeros(value.shape)
        for i in range(value.shape[0]):
            maxIndex = 0
            maxValue = 0
            for j in range(value.shape[1]):
                if value[i][j]>maxValue:
                    maxIndex = j
                    maxValue = value[i][j]
            ytest[i][maxIndex] = 1
        for i in ytest:
            print i
        ytest = self.y_digit(ytest)



        ### END YOUR CODE
        assert len(ytest) == Xtest.shape[0]
        return ytest
    def unison_shuffled_copies(self, x1, x2):
        randomize = np.arange(len(x1))
        np.random.shuffle(randomize)
        return x1[randomize],x2[randomize]

    def y_digit(self,ytrain):
        k = np.zeros(ytrain.shape[0])
        for i in range(ytrain.shape[0]):
            if len(np.where(ytrain[i] == 1)[0]) != 0:
                k[i] = np.where(ytrain[i] == 1)[0][0]
        return k


