# CSC 420 - Machine Learning
# Attempt to predict if a stock will go up or down tomorrow from the closing
# prices from the past year.
# 
# Apply several models to the problem, such as logistic regression, k-nearest
# neighbors, and support vector machines.
# 
# With none of the sections commented out, this script took about six minutes
# to run.
#
# Jordan Turley

import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn import metrics

import numpy as np

RANDOM_STATE = 12345
TARGET_NAMES = ['decrease', 'increase']

def main():
	# Get all of the data
	x, y = readData()

	print(x.shape)
	print(y.shape)

	# Split into training and test data
	xTrain, xTest, yTrain, yTest = train_test_split(x, y, random_state = RANDOM_STATE)

	# Use StandardScaler to normalize the data
	scaler = StandardScaler()
	scaler.fit(xTrain)
	xTrain = scaler.transform(xTrain)
	xTest = scaler.transform(xTest)

	# First, try logistic regression
	rkf = RepeatedKFold(n_splits = 10, n_repeats = 5, random_state = RANDOM_STATE)

	# LR achieved a score of 0.816
	# I commented out this section to save runtime
	# To see the results, uncomment this section
	'''parameters = {
		'penalty': ['l1', 'l2'],
		'C': np.arange(0.1, 2.1, 0.1)
	}

	lr = LogisticRegression()
	clfLR = GridSearchCV(lr, parameters, cv = rkf)
	clfLR.fit(xTrain, yTrain)
	print('LR Best score:', clfLR.best_score_)
	for param_name in sorted(parameters.keys()):
		print("%s: %r" % (param_name, clfLR.best_params_[param_name]))
	'''

	# Now try k-Nearest Neighbors
	# kNN achieved a score of 0.8208
	# Set parameters to use for grid search for kNN
	parameters = {
		'n_neighbors': np.arange(1, 11, 1), # 1, 2, ... 9, 10
		'metric': ['euclidean', 'l1', 'manhattan', 'chebyshev'] # Try a few distance metrics
	}

	knn = KNeighborsClassifier()
	clfKNN = GridSearchCV(knn, parameters, cv = rkf)
	clfKNN.fit(xTrain, yTrain)
	print('kNN Best score:', clfKNN.best_score_)
	for param_name in sorted(parameters.keys()):
		print("%s: %r" % (param_name, clfKNN.best_params_[param_name]))

	# Finally, try support vector machines
	# SVM achieved a score of 0.816
	# I commented out this code to save time running it
	# To see the results just uncomment this section
	'''parameters = {
		'kernel': ['linear', 'sigmoid', 'rbf', 'poly'],
		'C': np.arange(0.1, 2.1, 0.1) # 0.1, 0.2, ... 1.9, 2.0
	}
	svm = SVC()
	clfSVM = GridSearchCV(svm, parameters, cv = rkf)
	clfSVM.fit(xTrain, yTrain)
	print('SVM Best score:', clfSVM.best_score_)
	for param_name in sorted(parameters.keys()):
		print("%s: %r" % (param_name, clfSVM.best_params_[param_name]))
	'''

	# The one that did the best was k-nearest neighbors
	# with k = 6 and using chebyshev distance
	# Now, try kNN on the test set and see how it does
	predictions = clfKNN.predict(xTrain)

	# Success rate
	print('Success rate:', np.mean(predictions == yTrain))

	# Metrics: Precision, Recall, F1-Score, and Support
	print(metrics.classification_report(yTrain, predictions, target_names = TARGET_NAMES))

	# Confusion matrix
	print(metrics.confusion_matrix(yTrain, predictions, labels = TARGET_NAMES))

	predictions = clfKNN.predict(xTest)

	# Success rate
	print('Success rate:', np.mean(predictions == yTest))

	# Metrics: Precision, Recall, F1-Score, and Support
	print(metrics.classification_report(yTest, predictions, target_names = TARGET_NAMES))

	# Confusion matrix
	print(metrics.confusion_matrix(yTest, predictions, labels = TARGET_NAMES))

def readData():
	'''Read in all of the stock prices and if it increased or decreased
	over the last 30 days. Return the data in numpy arrays
	'''
	# 'cd' into data folder and get all files
	os.chdir('data_1_day')
	files = os.listdir()

	classes = []
	attributes = []

	# Loop over every filename
	for filename in files:
		if filename == '.DS_Store':
			continue

		# Get all of the contents of each file
		historyFile = open(filename)
		history = historyFile.readlines()
		historyFile.close()

		# Get the increase or decrease
		c = history[-1].rstrip()
		classes.append(c)
		del history[-1]

		# Get the prices
		a = []
		for priceStr in history:
			priceStr = priceStr.rstrip()
			price = float(priceStr)
			a.append(price)

		attributes.append(a)

	return np.array(attributes), np.array(classes)

def readDataPreSorted(filename):
	'''Reads in the training, validation, or test set. Reads in the tickers in
	the set from the given filename, then reads those stock prices in from
	their data files, returning two lists: the attributes and the classes.
	I used this when I was setting the training, validation, and test sets.
	If I had more data, I would use this.
	'''
	f = open(filename)

	classes = []
	attributes = []

	# Go through each ticker in the filename
	for ticker in f:
		ticker = ticker.rstrip() # Strip off \n at end of line
		if ticker == '.DS_Store':
			continue

		# Open the file containing the price history
		historyFile = open('data/' + ticker + '.txt')
		history = historyFile.readlines()
		historyFile.close()

		# Get the increase or decrease
		c = history[0].rstrip()
		classes.append(c)
		del history[0]

		# Get the prices
		a = []
		for priceStr in history:
			priceStr = priceStr.rstrip()
			price = float(priceStr)
			a.append(price)

		attributes.append(a)

	f.close()

	return attributes, classes

if __name__ == '__main__':
	main()