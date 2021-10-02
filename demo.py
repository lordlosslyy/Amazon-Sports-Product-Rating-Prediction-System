import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from urllib.request import urlopen
import scipy.optimize
import random
        
def parseData(fname):
    for l in open(fname):
        l = l.replace("true", "True")
        l = l.replace("false", "False")
        yield eval(l)

data = list(parseData("/Users/yunyi/Documents/Sports_and_Outdoors_5.json"))
random.shuffle(data)

train_data = data[:500000]

itemCount = defaultdict(int)
userCount = defaultdict(int)

for d in train_data:
    itemCount[d['asin']] += 1
    userCount[d['reviewerID']] += 1

N = len(train_data)
nUsers = len(userCount)
nItems = len(itemCount)
users = list(userCount.keys())
items = list(itemCount.keys())
userBiases = defaultdict(float)
itemBiases = defaultdict(float)
#The actual prediction function of our model is simple: Just predict using a global offset (alpha), a user offset (beta_u in the slides), and an item offset (beta_i)

def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)

def prediction(user, item):
    return alpha + userBiases[user] + itemBiases[item]
#We'll use another library in this example to perform gradient descent. This library requires that we pass it a "flat" parameter vector (theta) containing all of our parameters. This utility function just converts between a flat feature vector, and our model parameters, i.e., it "unpacks" theta into our offset and bias parameters.

def unpack(theta):
    global alpha
    global userBiases
    global itemBiases
    alpha = theta[0]
    userBiases = dict(zip(users, theta[1:nUsers+1]))
    itemBiases = dict(zip(items, theta[1+nUsers:]))
    
# The "cost" function is the function we are trying to optimize. Again this is a requirement of the gradient descent library we'll use. In this case, we're just computing the (regularized) MSE of a particular solution (theta), and returning the cost.
def cost(theta, labels, lamb):
    unpack(theta)
    predictions = [prediction(d['reviewerID'], d['asin']) for d in train_data]
    cost = MSE(predictions, labels)
    print("MSE = " + str(cost))
    for u in userBiases:
        cost += lamb*userBiases[u]**2
    for i in itemBiases:
        cost += lamb*itemBiases[i]**2
    return cost

#The derivative function is the most difficult to implement, but follows the definitions of the derivatives for this model as given in the lectures. This step could be avoided if using a gradient descent implementation based on (e.g.) Tensorflow.
def derivative(theta, labels, lamb):
    unpack(theta)
    N = len(train_data)
    dalpha = 0
    dUserBiases = defaultdict(float)
    dItemBiases = defaultdict(float)
    for d in train_data:
        u,i = d['reviewerID'], d['asin']
        pred = prediction(u, i)
        diff = pred - (d['overall'])
        dalpha += 2/N*diff
        dUserBiases[u] += 2/N*diff
        dItemBiases[i] += 2/N*diff
    for u in userBiases:
        dUserBiases[u] += 2*lamb*userBiases[u]
    for i in itemBiases:
        dItemBiases[i] += 2*lamb*itemBiases[i]
    dtheta = [dalpha] + [dUserBiases[u] for u in users] + [dItemBiases[i] for i in items]
    return np.array(dtheta)

ratingMean = sum([d['overall'] for d in train_data]) / len(train_data)
alpha = ratingMean

alwaysPredictMean = [ratingMean for d in train_data] 
labels = [d['overall'] for d in train_data]

lambdaList = [1e-3, 1e-4, 1e-5, 1e-6] 
MSEList = []
thetaList = []

for lamb in lambdaList:
    theta, curMSE, _ = scipy.optimize.fmin_l_bfgs_b(cost, [alpha] + [0.0]*(nUsers+nItems), derivative, args = (labels, lamb), maxiter = 30)
    MSEList.append(curMSE)
    thetaList.append(theta)

###
valid_data = data[500000:1000000]

validMSEList = []

for theta in thetaList:
    predictions = []
    unpack(theta) 
    for d in valid_data: 
        p = ratingMean
        if d['reviewerID'] in userBiases:
            p += userBiases[d['reviewerID']]
        if d['asin'] in itemBiases:
            p += itemBiases[d['asin']]
        predictions.append(p)
        
    label_valid = [d['overall'] for d in valid_data]

    valid_mse = MSE(predictions, label_valid)
    validMSEList.append(valid_mse)
    print('MSE of validation set {}'.format(valid_mse))


baselineV = MSE(alwaysPredictMean, labels)
baselineArr = [baselineV] * len(thetaList)

plt.plot(lambdaList, MSEList, label = "trainData MSE")
plt.plot(lambdaList, validMSEList, label = "validationData MSE") 
plt.plot(lambdaList, baselineArr, label = "Baseline") 

plt.xscale("log")
plt.xlabel("Lambda (log scale)")
plt.ylabel("MSE")

plt.title("MSE in Different set and Different Lambda")
plt.legend()
plt.show()
