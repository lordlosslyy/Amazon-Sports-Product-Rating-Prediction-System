#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from urllib.request import urlopen
import scipy.optimize
import random
from sklearn import svm
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def parseDataFromURL(fname):
    for l in urlopen(fname):
        yield eval(l)
        
def parseData(fname):
    #global count
    for l in open(fname):
        l = l.replace("true", "True")
        l = l.replace("false", "False")
        yield eval(l)
    
data = []
data = list(parseData("/Users/yunyi/Documents/Sports_and_Outdoors_5.json"))
print("done")
print(len(data))
print(data[0])

random.shuffle(data)

print(data[0])

train_data = data[:500000]

itemCount = defaultdict(int)
userCount = defaultdict(int)


for d in train_data:
    #GamesEachUser[user].add(game)
    #UsersEachGame[game].add(user)
    
    #if game not in TotalGames:
    #    TotalGames.append(game)
    itemCount[d['asin']] += 1
    userCount[d['reviewerID']] += 1
    #totalPlayed += 1

N = len(train_data)
nUsers = len(userCount)
nItems = len(itemCount)
users = list(userCount.keys())
items = list(itemCount.keys())


# In[9]:


userBiases = defaultdict(float)
itemBiases = defaultdict(float)
#The actual prediction function of our model is simple: Just predict using a global offset (alpha), a user offset (beta_u in the slides), and an item offset (beta_i)

#def time_transform(time):
#    return math.log(time+1, 2)

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


# In[10]:


ratingMean = sum([d['overall'] for d in train_data]) / len(train_data)
print(ratingMean)
alpha = ratingMean


# In[11]:


alwaysPredictMean = [ratingMean for d in train_data] 
labels = [d['overall'] for d in train_data]

print(MSE(alwaysPredictMean, labels))


# In[13]:


#lambdaList = [1e-5] 
lambdaList = [1e-3, 1e-4, 1e-5, 1e-6] 
MSEList = []
thetaList = []

for lamb in lambdaList:
    theta, curMSE, _ = scipy.optimize.fmin_l_bfgs_b(cost, [alpha] + [0.0]*(nUsers+nItems),
                             derivative, args = (labels, lamb), maxiter = 30)
    MSEList.append(curMSE)
    thetaList.append(theta)


# In[14]:


#plt.plot(cValue, testBER, label = "Test BER") 
plt.plot(lambdaList, MSEList, label = "trainData MSE") 
#plt.plot(lambdaList, validationBER, label = "Validation BER") 

plt.xscale("log")
plt.xlabel("Lambda")
plt.ylabel("MSE")

plt.title("MSE in Different Lambda")
plt.legend()
plt.show()


# In[15]:


print(MSEList)


# In[16]:


valid_data = data[500000:1000000]
test_data = data[1000000:1500000]


# In[17]:


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
    


# In[18]:


testMSEList = []
for theta in thetaList:
    predictions = []
    unpack(theta) 
    #    global userBiases
    #global itemBiases
    for d in test_data: 
        p = ratingMean
        if d['reviewerID'] in userBiases:
            p += userBiases[d['reviewerID']]
        if d['asin'] in itemBiases:
            p += itemBiases[d['asin']]
        predictions.append(p)
        #predictions.append(prediction(d['reviewerID'], d['asin']))
    label_valid = [d['overall'] for d in test_data]

    test_mse = MSE(predictions, label_valid)
    testMSEList.append(test_mse)
    print('MSE of test set {}'.format(test_mse))
    


# In[19]:


ratingMeanT = sum([d['overall'] for d in test_data]) / len(test_data)
print(ratingMeanT)

alwaysPredictMeanT = [ratingMeanT for d in test_data] 
labelsT = [d['overall'] for d in test_data]


# In[20]:


baselineV = MSE(alwaysPredictMean, labels)
baselineArr = [baselineV] * len(thetaList)


# In[21]:


lambdaList1 = lambdaList 
MSEList1 = MSEList 
validMSEList1 = validMSEList 
testMSEList1 = testMSEList
baselineArr1 = baselineArr 
thetaList1 = thetaList

plt.plot(lambdaList, MSEList, label = "trainData MSE")
plt.plot(lambdaList, validMSEList, label = "validationData MSE") 
plt.plot(lambdaList, testMSEList, label = "testData MSE") 
plt.plot(lambdaList, baselineArr, label = "Baseline") 

plt.xscale("log")
plt.xlabel("Lambda (log scale)")
plt.ylabel("MSE")

plt.title("MSE in Different set and Different Lambda")
plt.legend()
plt.show()


# In[22]:


plt.plot(lambdaList, MSEList, label = "trainData MSE")
plt.plot(lambdaList, validMSEList, label = "validationData MSE") 
plt.plot(lambdaList, testMSEList, label = "testData MSE") 

plt.xscale("log")
plt.xlabel("Lambda (log scale)")
plt.ylabel("MSE")

plt.title("MSE in Different set and Different Lambda")
plt.legend()
plt.show()

