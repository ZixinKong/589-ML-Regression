#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import modules needed
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold

def compute_error(y_hat, y):
	# mean absolute error
	return np.abs(y_hat - y).mean()

# create decision tree
def decision_tree(d, train_x, train_y, test_x):
    
    dt = DecisionTreeRegressor(criterion='mae', max_depth=d, min_samples_split=3, min_samples_leaf=2)
    dt.fit(train_x, train_y) 
    return dt.predict(test_x)

# implement 5-fold-cross-validation
def cross_validation(max_depths, train_x, train_y):
    MAE = [] # record avg MAE with differnt num_neighbor(k) '
    kf = KFold(10)
    for d in max_depths:
        temp = [] # record MAE for each training with different num_neighbor(k)
        for train_index, test_index in kf.split(train_x):
            x_train, x_test = train_x[train_index], train_x[test_index]
            y_train, y_test = train_y[train_index], train_y[test_index]
    
            y_predicted = decision_tree(d, x_train, y_train, x_test)
            temp.append(compute_error(y_predicted, y_test))
        MAE.append(np.mean(temp))
        
# create a dictionary with key=num_neighbor, value=corresponding Avg of MAE when k=key
    return dict(zip(max_depths, MAE))

##########################################################################