#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge


def compute_error(y_hat, y):
	# mean absolute error
	return np.abs(y_hat - y).mean()

# create decision tree
def ridge_model(a, train_x, train_y, test_x):
    
    ridge = Ridge(alpha=a)
    ridge.fit(train_x, train_y)
    return ridge.predict(test_x)

# implement 5-fold-cross-validation
def cross_validation(alphas, train_x, train_y):
    MAE = [] # record avg MAE with differnt num_neighbor(k) 
    kf = KFold(5)
    for a in alphas:
        temp = [] # record MAE for each training with different num_neighbor(k)
        for train_index, test_index in kf.split(train_x):
            x_train, x_test = train_x[train_index], train_x[test_index]
            y_train, y_test = train_y[train_index], train_y[test_index]
    
            y_predicted = ridge_model(a, x_train, y_train, x_test)
            temp.append(compute_error(y_predicted, y_test))
        MAE.append(np.mean(temp))
        
# create a dictionary with key=num_neighbor, value=corresponding Avg of MAE when k=key
    
    
    return dict(zip(alphas, MAE))
    
##############################################################################