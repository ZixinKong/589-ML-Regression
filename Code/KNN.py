#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import modules needed
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold

def compute_error(y_hat, y):
	# mean absolute error
	return np.abs(y_hat - y).mean()

# create K nearest neighbors model
def knn_model(num_neighbor, train_x, train_y, test_x):
    
    knn = KNeighborsRegressor(n_neighbors=num_neighbor)
    knn.fit(train_x, train_y)
    return knn.predict(test_x)


# implement 5-fold-cross-validation
def cross_validation(num_neighbors, train_x, train_y):
    MAE = [] # record avg MAE with differnt num_neighbor(k) 
    kf = KFold(5)
    for k in num_neighbors:
        print (k)
        temp = [] # record MAE for each training with different num_neighbor(k)
        for train_index, test_index in kf.split(train_x):
            print ("run")
            x_train, x_test = train_x[train_index], train_x[test_index]
            y_train, y_test = train_y[train_index], train_y[test_index]
    
            y_predicted = knn_model(k, x_train, y_train, x_test)
            temp.append(compute_error(y_predicted, y_test))
            
            
        MAE.append(np.mean(temp))
        
# create a dictionary with key=num_neighbor, value=corresponding Avg of MAE when k=key
    return dict(zip(num_neighbors, MAE))


##########################################################################

