# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 15:37:14 2019

@author: 
     Magdalena Fijalkowska
"""

import pickle
import numpy
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import style
import sklearn
from collections import Counter
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn import neighbors
import numpy as np
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


class KNN():
    
    def __init__(self, dist, class_Y):
        self.dist = dist
        self.class_Y = class_Y
    
    
digits = load_digits()

def build_in_KNN():
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    print("Skleran machine learning model accuracy: ", accuracy)

def Unique_Classification(Array):
    x = numpy.array(Array)
    unique, counts = numpy.unique(x, return_counts=True)

    return dict(zip(unique,counts))

    


def display_data():
    
    print("DATASET DESCRIPTION")
    print("Number of classes: ", len(digits.target_names))
    print("Number of instances: ", digits.data.shape[0])
    print ("Number of train instances: ", X_train.shape[0])
    print ("Number of test instances: ", X_test.shape[0])
    
    digit_data = digits.data
    digit_data.reshape(-1)
    MIN = min(digit_data[0])
    MAX = max(digit_data[0])
    for i in range(len(digits.data)): 
        min_val = min(digit_data[i])
        max_val = max(digit_data[i])
        
        if (min_val<MIN):
            MIN = min_val
        if (max_val>MAX):
            MAX = max_val
            
    print("Min feature value is: ", min_val)
    print("Max feature value is: ", max_val)

    classes =Unique_Classification(digits.target)

    for x, y in classes.items():
        print("The class ", x, " has " , y, " instances")
    
        

#define X, y
X = digits.data
y = digits.target

# split into training and test data
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0)

#print("min", min(digits.data))
# define scaler
sc = StandardScaler()
sc.fit(X_train)

#scale data
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


#def KNN
def k_nearest_neighbor(train_data, class_data, predict, k): 
    
    #reshape each matrix to a 1d-array
    train_data.reshape(-1)
    predict.reshape(-1) 
    
    #create array to store distances for X_train digits
    distances = []
    
    #create list of objects to store class and distance
    for i in range(len(X_train)):
        distances.append(KNN(99999, class_data[i]))
    
    
    #calculate euclidean_distance between random digit and X_train digits and store them in the array    
    for i in range(len(X_train)):
        
        euclidean_distance = 0.0
        for ii in range(64): 
            euclidean_distance += (train_data[i][ii] - predict[ii])**2
        
        euclidean_distance = sqrt(euclidean_distance)
        distances[i].dist = (euclidean_distance)
    
    distances.sort(key = lambda x: x.dist, reverse= False)
    
    #create array to store first k nearest classes
    arr = []
    for i in range(k):
        arr.append(distances[i].class_Y)
        
    
    occurance_count = Counter(arr)
    
    vote_result = occurance_count.most_common(1)[0][0]
    return vote_result



    
    
def main():
    
    display_data()
    build_in_KNN()
    user_query()
    Puckle_out = open("KNN_Classifier_Scratch.pickle","rb")
    KNN_Scratch = pickle.load(Puckle_out)
    #print(KNN_Scratch(X_train, y_train, X_test[0], k=5))
    arr_error  = []
    print("Accuracy of sklearn KNN: ", KNN_Classifier.score(X_test,y_test))
    correct = 0
    total = 0
    for i in range(len(X_test)):
        vote = k_nearest_neighbor(X_train, y_train, X_test[i], k=5)
        if y_test[i] == vote: 
            correct += 1
        else:
            arr_error.append(i)
        total += 1
    
    accuracy = float(correct)/total
    print("Accuracy of self implemented KNN: ", accuracy)
    print("List of indexes of incorrect predictions for implemented KNN model: ", arr_error)
    
    
def user_query():
    
    
    num = int(input("Index of matrix to classify: "))
    #using already saved and trained model to find the class
    result = KNN_Scratch(X_train, y_train, X_test[num], k=5)
    example = X_test[num]
    example = example.reshape(1,-1)

    result2 = KNN_Classifier.predict(example)

    print("Self implemented KNN classified digit as: ", result)
    print("Sklearn KNN classified digit as: ", result2)

main()

Puckle_out = open("KNN_Classifier_Scratch.pickle","rb")
KNN_Scratch = pickle.load(Puckle_out)

