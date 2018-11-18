# Import python modules
import numpy as np
import kaggle
import matplotlib.pyplot as plt
import KNN
import trees
import ridge
import lasso
import power


from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

# Read in train and test data
def read_data_power_plant():
	print('Reading power plant dataset ...')
	train_x = np.loadtxt('../../Data/PowerOutput/data_train.txt')
	train_y = np.loadtxt('../../Data/PowerOutput/labels_train.txt')
	test_x = np.loadtxt('../../Data/PowerOutput/data_test.txt')

	return (train_x, train_y, test_x)

def read_data_localization_indoors():
	print('Reading indoor localization dataset ...')
	train_x = np.loadtxt('../../Data/IndoorLocalization/data_train.txt')
	train_y = np.loadtxt('../../Data/IndoorLocalization/labels_train.txt')
	test_x = np.loadtxt('../../Data/IndoorLocalization/data_test.txt')

	return (train_x, train_y, test_x)

# Compute MAE
def compute_error(y_hat, y):
	# mean absolute error
	return np.abs(y_hat - y).mean()

############################################################################

###prepare data 
train_x_1, train_y_1, test_x_1 = read_data_power_plant()
train_x_2, train_y_2, test_x_2 = read_data_localization_indoors()

#print('Train=', train_x_2.shape) #shape: (row, column) Train= (19937, 400)
#print('Test=', test_x_2.shape) # Test= (1111, 400)

"""
# Questions 1: Decision Trees
# (b)
print ("Now is for question 1(b): decision tree model in power plant dataset ...")
max_depths = [3,6,9,12,15]
print ("Find best max depth using 5-fold cross validation...")
dt_output, run_time = trees.cross_validation(max_depths, train_x_1, train_y_1)
print ("Max depth - Average MAE of each max depth: ", dt_output)

best_parameter = min(dt_output, key = dt_output.get)
print ("The best parameter for decision tree model in power plant dataset is: ", best_parameter)

# draw run time plot
print ("Running time plot is: ")
plt.plot(max_depths, run_time, '-o')
plt.xlabel("Value of max depth")
plt.ylabel("Running time (milliseconds)")
plt.title("Average time to perform cross-validation with different max_depth under Decision Trees (Power plant dataset)")
plt.show()

print ("Now training full dataset with max depth =", best_parameter)
predicted_y = trees.decision_tree(best_parameter, train_x_1, train_y_1, test_x_1)

# Output file location
file_name = '../Predictions/PowerOutput/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)
print ("\n")

# (c)
print ("Now is for question 1(c): decision tree model in indoor localization dataset ...")

max_depths = [20,25,30,35,40]
print ("Find best max depth using 5-fold cross validation...")
dt_output, run_time = trees.cross_validation(max_depths, train_x_2, train_y_2)
print ("Max depth - Average MAE of each max depth: ", dt_output)

best_parameter = min(dt_output, key = dt_output.get)
print ("The best parameter for decision tree model in indoor localization dataset is: ", best_parameter)

# draw run time plot
print ("Running time plot is: ")
plt.plot(max_depths, run_time, '-o')
plt.xlabel("Value of max depth")
plt.ylabel("Running time (milliseconds)")
plt.title("Average time to perform cross-validation with different max_depth under Decision Trees (Indoor localization dataset)")
plt.show()

print ("Now training full dataset with max depth =", best_parameter)
predicted_y = trees.decision_tree(best_parameter, train_x_2, train_y_2, test_x_2)

# Output file location
file_name = '../Predictions/IndoorLocalization/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)
print ("\n")

# Question 3: Nearest neighbors ################################
# (a) 
print ("Now is for question 3(a): Nearest neighbors model in power plant dataset ...")

num_neighbors = [3,5,10,20,25] 
print ("Find best k using 5-fold cross validation...")
knn_output = KNN.cross_validation(num_neighbors, train_x_1, train_y_1)
print ("k - Average MAE of each k: ", knn_output)

best_parameter = min(knn_output, key = knn_output.get)
print ("The best parameter for KNN model in power plant dataset is: ", best_parameter)

print ("Now training full dataset with k =", best_parameter)
predicted_y = KNN.knn_model(best_parameter, train_x_1, train_y_1, test_x_1)

# Output file location
file_name = '../Predictions/PowerOutput/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)
print ("\n")

###(b)
print ("Now is for question 3(b): Nearest neighbors model in indoor localization dataset ...")

num_neighbors = [3,5,10,20,25] 
print ("Find best k using 5-fold cross validation...")
knn_output = KNN.cross_validation(num_neighbors, train_x_2, train_y_2)
print ("k - Average MAE of each k: ", knn_output)

best_parameter = min(knn_output, key = knn_output.get)
print ("The best parameter for KNN model in indoor localization dataset is: ", best_parameter)

print ("Now training full dataset with k =", best_parameter)
predicted_y = KNN.knn_model(best_parameter, train_x_2, train_y_2, test_x_2)

# Output file location
file_name = '../Predictions/IndoorLocalization/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)
print ("\n")


# Question 4: Linear Model #######################################
# (b)
print ("Now is for question 4(b): Linear model in power plant dataset ...")

alphas_1 = [10**-6, 10**-4, 10**-2, 1, 10]

print ("Find best alpha of ridge using 5-fold cross validation ...")
ridge_output = ridge.cross_validation(alphas_1, train_x_1, train_y_1)
print ("Alpha - Average MAE of each alpha (ridge): ", ridge_output)

para1 = min(ridge_output, key = ridge_output.get)
print ("The best parameter for ridge model in power plant dataset is: ", para1)


print ("Find best alpha of lasso using 5-fold cross validation ...")
lasso_output = lasso.cross_validation(alphas_1, train_x_1, train_y_1)
print ("Alpha - Average MAE of each alpha (lasso): ", lasso_output)

para2 = min(lasso_output, key = lasso_output.get)
print ("The best parameter for lasso model in power plant dataset is: ", para2)

if ridge_output[para1] < lasso_output[para2]:
    best_parameter = para1
    print ("The best model for power plant dataset is: ridge with aplha=", para1)
    print ("Now training full dataset using ridge model with aplha=", best_parameter)
    predicted_y = ridge.ridge_model(best_parameter, train_x_1, train_y_1, test_x_1)
else:
    best_parameter = para2
    print ("The best model for power plant dataset is: lasso with aplha=", para2)
    print ("Now training full dataset using lasso model with alpha=", best_parameter)
    predicted_y = ridge.ridge_model(best_parameter, train_x_1, train_y_1, test_x_1)

# Output file location
file_name = '../Predictions/PowerOutput/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)         
print ("\n")

# (c)
print ("Now is for question 4(c): Linear model in indoor localization dataset ...")

alphas_2 = [10**-4, 10**-2, 1, 10]

print ("Find best alpha of ridge using 5-fold cross validation ...")
ridge_output = ridge.cross_validation(alphas_2, train_x_2, train_y_2)
print ("Alpha - Average MAE of each alpha (ridge): ", ridge_output)

para1 = min(ridge_output, key = ridge_output.get)

print ("The best parameter for ridge model in indoor localization dataset is: ", para1)


print ("Find best alpha of lasso using 5-fold cross validation ...")
lasso_output = lasso.cross_validation(alphas_2, train_x_2, train_y_2)
print ("Alpha - Average MAE of each alpha (lasso): ", lasso_output)

para2 = min(lasso_output, key = lasso_output.get)
print ("The best parameter for lasso model in indoor localization dataset is: ", para2)

if ridge_output[para1] < lasso_output[para2]:
    best_parameter = para1
    print ("The best model for indoor localization dataset is: ridge with aplha=", para1)
    print ("Now training full dataset using ridge model with aplha=", best_parameter)
    predicted_y = ridge.ridge_model(best_parameter, train_x_2, train_y_2, test_x_2)
else:
    best_parameter = para2
    print ("The best model for indoor localization dataset is: lasso with aplha=", para2)
    print ("Now training full dataset using lasso model with alpha=", best_parameter)
    predicted_y = ridge.ridge_model(best_parameter, train_x_2, train_y_2, test_x_2)

# Output file location
file_name = '../Predictions/IndoorLocalization/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)
print ("\n")

# Question 5: Kaggle competition ######################################3
# (a)
print ("Now is for question 5(a): Improve model in power plant dataset ...")
print ("After testing, I choose 10-fold cross validation for model selection, the details of testing k-fold is in report.")

max_depths = [6, 7, 8, 9, 10, 11, 12]
print ("I choose the decision tree model, the max depth as candidates are: ", max_depths)

#sel = GenericUnivariateSelect()
#x1_new = sel.fit_transform(train_x_1, train_y_1)
#test1_new = sel.transform(test_x_1)

print ("Find best max depth using 10-fold cross validation: ...")
power_output = power.cross_validation(max_depths, train_x_1, train_y_1)
print ("Max depth - Average MAE of each max depth: ", power_output)

best_parameter1 = min(power_output, key = power_output.get)
print ("The best parameter for power plant model is: ", best_parameter1)

#predicted_y = power.decision_tree(best_parameter, train_x_1, train_y_1, test_x_1)
print  ("Now training full dataset with max depth=", best_parameter1)
sel = DecisionTreeRegressor(criterion='mae', max_depth=10, min_samples_split=3, min_samples_leaf=2)
sel.fit(train_x_1, train_y_1) 
predicted_y = sel.predict(test_x_1)

# Output file location
file_name = '../Predictions/PowerOutput/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)
print ("\n")
"""

# (b)
print ("Now is for question 5(b): Improve model in indoor localization dataset ...")
print ("After testing, I choose 5-fold cross validation for model selection, the details of testing k-fold is in report.")
#k = [5,10,15,20]
#max_depths = [30,35,40,45,50,60]
num_neighbors = [1,2,3,4,5]
print ("I choose the KNN model, the num of neighbors as candidates are: ", num_neighbors)
#pca=PCA(n_components=1)  
#y2_new = pca.fit_transform(train_y_2) 
#print (y2_new)


#sel = GenericUnivariateSelect()

#sel = SelectKBest(f_regression, k=100)
#print (sel.score_func)
# print len(train_y_2)
#x2_new = sel.fit_transform(train_x_2, y2_new)


#test2_new = sel.transform(test_x_2)

#print('Train=', x2_new.shape) #shape: (row, column) Train= (19937, 400)
#print('Test=', test2_new.shape) # Test= (1111, 400)


print ("Find best k using 5-fold cross validation: ...")
indoor_output = KNN.cross_validation(num_neighbors, train_x_2, train_y_2)
print ("k - Average MAE of each k: ", indoor_output)

best_parameter2 = min(indoor_output, key = indoor_output.get)
print ("The best parameter for indoor localization model is: ", best_parameter2)

#predicted_y = KNN.knn_model(best_parameter, train_x_2, train_y_2, test_x_1)

#nComp = 2
#yhat = np.dot(pca.transform(test2_new)[:,:nComp], pca.components_[:nComp,:])
#print (yhat)
#print (predicted_y)
#print (yhat)


# best _parameter=4
"""
print  ("Now training full dataset with k=", best_parameter2)
knn = KNeighborsRegressor(n_neighbors=4, leaf_size=40)
knn.fit(train_x_2, train_y_2)
predicted_y = knn.predict(test_x_2)


# Output file location
file_name = '../Predictions/IndoorLocalization/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)

"""




