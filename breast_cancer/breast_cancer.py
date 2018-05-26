# CSC 420 - Machine Learning
# 
# Apply SVM and kNN to sklearn's breast cancer dataset.
# 
# Jordan Turley and Eric Murrell

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

RANDOM_STATE = 12345

def main():
	# Read in the data, print out the shape and description
	data = load_breast_cancer()
	print(data.data.shape)
	print(data.target.shape)
	#print(data.DESCR)

	# The data isn't balanced: 212 are malignant, 357 are benign.
	# 569 instances, 30 features. Target is a 0 or 1
	
	# Split the data into training and test sets
	xTrain, xTest, yTrain, yTest = train_test_split(data.data, data.target, random_state = RANDOM_STATE)

	# Scale the data based on the training data
	scaler = StandardScaler()
	scaler.fit(xTrain)
	xTrain = scaler.transform(xTrain)
	xTest = scaler.transform(xTest)

	# Init all the parameters we want to grid search over
	parameters = {
		'kernel': ['linear', 'sigmoid', 'rbf', 'poly'],
		'C': np.arange(0.1, 2, 0.1) # 0.1, 0.2, ... 1.9, 2.0
	}

	# Repeated k folds for grid search
	rkf = RepeatedKFold(n_splits = 10, n_repeats = 10, random_state = RANDOM_STATE)

	# Grid search and find the best parameters
	svc = SVC()
	clf = GridSearchCV(svc, parameters, cv = rkf)
	clf.fit(xTrain, yTrain)
	print('Best score:', clf.best_score_)
	for param_name in sorted(parameters.keys()):
		print("%s: %r" % (param_name, clf.best_params_[param_name]))

	# Use these parameters to predict for the test set
	predictions = clf.predict(xTest)

	# Print out evaluation metrics
	print('Success Rate:', clf.score(xTest, yTest))
	print(metrics.classification_report(yTest, predictions, target_names = ['benign', 'malignant'], digits = 3)) # , target_names = target_names
	print(metrics.confusion_matrix(yTest, predictions))
	print()

	# Try another model and see how it performs
	parameters = {
		'n_neighbors': np.arange(1, 11, 1) # k = 1, 2, ... 10
	}

	# Now try k nearest neighbors
	knn = KNeighborsClassifier()
	clf = GridSearchCV(knn, parameters, cv = rkf)
	clf.fit(xTrain, yTrain)
	print('Best score:', clf.best_score_)
	for param_name in sorted(parameters.keys()):
		print("%s: %r" % (param_name, clf.best_params_[param_name]))

	# Generate predictions
	predictions = clf.predict(xTest)

	# Print out evaluation metrics
	print('Success Rate:', clf.score(xTest, yTest))
	print(metrics.classification_report(yTest, predictions, target_names = ['benign', 'malignant'], digits = 3)) # , target_names = target_names
	print(metrics.confusion_matrix(yTest, predictions))

if __name__ == '__main__':
	main()