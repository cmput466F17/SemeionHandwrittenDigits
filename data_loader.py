import csv
import numpy as np
def load_train_and_test_data(trainsize,testsize):
    filename = 'datasets/sem.csv'



    dataset = loadcsv(filename)
    trainset, testset = splitdataset(dataset,trainsize, testsize,featureoffset=256)
    return trainset,testset
def loadcsv(filename):
    dataset = np.genfromtxt(filename, delimiter=',')
    return dataset

def splitdataset(dataset, trainsize, testsize, testdataset=None, featureoffset=None, outputfirst=None):
    """
    Splits the dataset into a train and test split
    If there is a separate testfile, it can be specified in testfile
    If a subset of features is desired, this can be specifed with featureinds; defaults to all
    Assumes output variable is the last variable
    """
    # Generate random indices without replacement, to make train and test sets disjoint
    # np.random.seed(123)
    randindices = np.random.choice(dataset.shape[0], trainsize + testsize, replace=False)
    featureend = dataset.shape[1] - 10
    outputlocation = featureend
    if featureoffset is None:
        featureoffset = 0
    if outputfirst is not None:
        featureoffset = featureoffset + 1
        featureend = featureend + 1
        outputlocation = 0

    Xtrain = dataset[randindices[0:trainsize], featureoffset:featureend]
    ytrain = dataset[randindices[0:trainsize], outputlocation:dataset.shape[1]]
    Xtest = dataset[randindices[trainsize:trainsize + testsize], featureoffset:featureend]
    ytest = dataset[randindices[trainsize:trainsize + testsize], outputlocation:dataset.shape[1]]



    return ((Xtrain, ytrain), (Xtest, ytest))

train,test = load_train_and_test_data(1000,500)
print test[1]