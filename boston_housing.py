
# coding: utf-8

# # Machine Learning Engineer Nanodegree
# ## Model Evaluation & Validation
# ## Project: Predicting Boston Housing Prices

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from sklearn.cross_validation import ShuffleSplit

# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
get_ipython().magic(u'matplotlib inline')

# Load the Boston housing dataset
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
    
# Success
print "Boston housing dataset has {} data points with {} variables each.".format(*data.shape)


# ### Implementation: Calculate Statistics
# For your very first coding implementation, you will calculate descriptive statistics about the Boston housing prices. Since `numpy` has already been imported for you, use this library to perform the necessary calculations. These statistics will be extremely important later on to analyze various prediction results from the constructed model.
# 
# In the code cell below, you will need to implement the following:
# - Calculate the minimum, maximum, mean, median, and standard deviation of `'MEDV'`, which is stored in `prices`.
#   - Store each calculation in their respective variable.

# TODO: Minimum price of the data
minimum_price = np.min(prices);

# TODO: Maximum price of the data

maximum_price = np.max(prices);

# TODO: Mean price of the data
mean_price =np.mean(prices);

# TODO: Median price of the data
median_price =np.median(prices);

# TODO: Standard deviation of prices of the data
std_price = np.std(prices);

# Show the calculated statistics
print "Statistics for Boston housing dataset:\n"
print "Minimum price: ${:,.2f}".format(float(minimum_price))
print "Maximum price: ${:,.2f}".format(float(maximum_price))
print "Mean price: ${:,.2f}".format(float(mean_price))
print "Median price ${:,.2f}".format(float(median_price))
print "Standard deviation of prices: ${:,.2f}".format(float(std_price))


#plot of features versus price

import matplotlib.pyplot as plt
plt.figure(figsize=(20, 5))
plt.subplot(1,3,1)
plt.plot(data['RM'],prices,'o')
plt.ylabel('prices')
plt.xlabel('Number of rooms')
plt.subplot(1,3,2)
plt.plot(data['LSTAT'],prices,'o')
plt.ylabel('prices')
plt.xlabel('LSTAT')
plt.subplot(1,3,3)
plt.plot(data['PTRATIO'],prices,'o')
plt.ylabel('prices')
plt.xlabel('PTRATIO')



from sklearn.metrics import r2_score

def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    
    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    score =r2_score(y_true,y_predict)
    
    # Return the score
    return score




# Calculate the performance of this model
score = performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
print "Model has a coefficient of determination, R^2, of {:.3f}.".format(score)




# TODO: Import 'train_test_split'
from sklearn import cross_validation
# TODO: Shuffle and split the data into training and testing subsets
X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, prices, test_size=0.2, random_state=0)

# Success
print "Training and testing split was successful."




# Produce learning curves for varying training set sizes and maximum depths
vs.ModelLearning(features, prices)


vs.ModelComplexity(X_train, y_train)

#Implementing Gridsearch using svm, kernel and comparing scores with cross validation

from sklearn import svm
X_train2, X_test2, y_train2, y_test2 = cross_validation.train_test_split(features, prices, test_size=0.2, random_state=0)
clf = svm.SVC(kernel='linear', C=1, probability=True).fit(X_train2, y_train2)
print clf.score(X_test2, y_test2)
scores = cross_validation.cross_val_score(clf, features, prices, cv=5)
print (scores)
print scores.mean()

#fit model function


# TODO: Import 'make_scorer', 'DecisionTreeRegressor', and 'GridSearchCV'
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn import svm
import matplotlib.pyplot as plt
def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)

    # TODO: Create a decision tree regressor object
    regressor =DecisionTreeRegressor()

    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth':(1,2,3,4,5,6,7,8,9,10)}

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    # TODO: Create the grid search object
    grid = GridSearchCV(regressor, params, scoring_fnc,cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_



# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

# Produce the value for 'max_depth'
print "Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth'])


# Produce a matrix for client data
client_data = [[5, 17, 15], # Client 1
               [4, 32, 22], # Client 2
               [8, 3, 12]]  # Client 3

# Show predictions
for i, price in enumerate(reg.predict(client_data)):
    print "Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price)
    
print features.describe()

print prices.describe()

vs.PredictTrials(features, prices, fit_model, client_data)
