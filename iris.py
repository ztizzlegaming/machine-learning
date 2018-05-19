# Several models applied to the iris dataset.
# Logistic regression, k-Nearest Neighbors, Support Vector Machines, and a
# feedforward neural network.
# 
# Data Preparation:
# Get the dataset from sklearn
# Rescale and normalize the x data using sklearn's StandardScaler
# Convert the y data to a binary matrix rather than categories (0, 1, 2)
# 
# Evaluation:
# I am using k-fold cross-validation with 10 folds to evaluate each model.
# I compute the success rate, precision, recall, and f1-score, as well as the
# confusion matrix.
# 
# Dependencies:
# sklearn
# keras
# tensorflow
# 
# Run:
# python3 ffn_iris.py
# 
# Jordan Turley

import numpy as np

# Import data processing stuff from sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV

# Import LR, KNN, and SVM from sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Import evaluation stuff from sklearn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Import neural network stuff from keras
from tensorflow import set_random_seed
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier

SEED = 42
K_FOLDS = 10

def main():
	np.random.seed(SEED)
	set_random_seed(SEED)

	# Load iris data
	iris = load_iris()
	data = iris['data']
	target = iris['target']

	# Normalize x data
	scaler = StandardScaler()
	xScaled = scaler.fit_transform(data)

	# Run the data through each of the models
	irisLogisticRegression(xScaled, target)
	print()
	iriskNN(xScaled, target)
	print()
	irisSVM(xScaled, target)
	print()
	irisFFN(xScaled, target)

def irisLogisticRegression(x, y):
	print('Logistic Regression:')

	# First use grid search to find the best values for the parameters
	parameters = {
		'penalty': ['l1', 'l2'],
		'C': np.arange(0.1, 2.1, 0.1)
	}

	rkf = RepeatedKFold(n_splits = K_FOLDS, n_repeats = 5, random_state = SEED)

	lr = LogisticRegression()
	clfLR = GridSearchCV(lr, parameters, cv = rkf)
	clfLR.fit(x, y)
	for param_name in sorted(parameters.keys()):
		print("%s: %r" % (param_name, clfLR.best_params_[param_name]))

	# Build the model on the best parameters
	lr = LogisticRegression(C = clfLR.best_params_['C'], penalty = clfLR.best_params_['penalty'])

	# Now use k folds to evaluate the model on the best parameters
	kf = KFold(n_splits = K_FOLDS, shuffle = True, random_state = SEED)

	# Initialize variables to hold evaluation metrics
	cfLR = np.zeros(shape = (3, 3))
	successRateLR = 0
	precisionLR = 0
	recallLR = 0

	# Loop over each fold
	for trainIdx, testIdx in kf.split(x):
		xTrain, xTest = x[trainIdx], x[testIdx]
		yTrain, yTest = y[trainIdx], y[testIdx]

		# Train on this fold
		lr.fit(xTrain, yTrain)
	
		# Make predictions on test set
		predictions = lr.predict(xTest)

		# Calculate evaluation metrics
		cfLR += confusion_matrix(yTest, predictions)
		successRateLR += np.mean(yTest == predictions)
		precisionLR += precision_score(yTest, predictions, average = 'weighted')
		recallLR += recall_score(yTest, predictions, average = 'weighted')

	# Average the metrics for each fold
	successRateLR /= K_FOLDS
	precisionLR /= K_FOLDS
	recallLR /= K_FOLDS

	# Calculate F1 from precision and recall
	f1LR = 2 / (1 / precisionLR + 1 / recallLR)

	# Print out all the results
	print('Success rate:', successRateLR)
	print('Precision:', precisionLR)
	print('Recall:', recallLR)
	print('F1-score:', f1LR)
	print(cfLR)


def iriskNN(x, y):
	print('k-Nearest Neighbors:')

	# First use grid search to find the best parameters
	parameters = {
		'n_neighbors': np.arange(1, 11, 1), # 1, 2, ... 9, 10
		'metric': ['euclidean', 'l1', 'manhattan', 'chebyshev'] # Try a few distance metrics
	}

	rkf = RepeatedKFold(n_splits = K_FOLDS, n_repeats = 5, random_state = SEED)

	knn = KNeighborsClassifier()
	clfKNN = GridSearchCV(knn, parameters, cv = rkf)
	clfKNN.fit(x, y)
	for param_name in sorted(parameters.keys()):
		print("%s: %r" % (param_name, clfKNN.best_params_[param_name]))

	# Build model on the best parameters
	knn = KNeighborsClassifier(n_neighbors = clfKNN.best_params_['n_neighbors'], metric = clfKNN.best_params_['metric'])

	# Use k folds cross validation to evaluate model
	kf = KFold(n_splits = K_FOLDS, shuffle = True, random_state = SEED)

	# Init variables to hold metrics
	cfKNN = np.zeros(shape = (3, 3))
	successRateKNN = 0
	precisionKNN = 0
	recallKNN = 0

	# Loop over each fold
	for trainIdx, testIdx in kf.split(x):
		xTrain, xTest = x[trainIdx], x[testIdx]
		yTrain, yTest = y[trainIdx], y[testIdx]

		# Fit the model to the training data of the fold
		knn.fit(xTrain, yTrain)
		
		# Make predictions for the test data of the fold
		predictions = knn.predict(xTest)

		# Compute the metrics
		cfKNN += confusion_matrix(yTest, predictions)
		successRateKNN += np.mean(yTest == predictions)
		precisionKNN += precision_score(yTest, predictions, average = 'weighted')
		recallKNN += recall_score(yTest, predictions, average = 'weighted')

	# Average the metrics for each fold
	successRateKNN /= K_FOLDS
	precisionKNN /= K_FOLDS
	recallKNN /= K_FOLDS

	# Calculate f1 from precision and recall
	f1KNN = 2 / (1 / precisionKNN + 1 / recallKNN)

	# Print out everything
	print('Success rate:', successRateKNN)
	print('Precision:', precisionKNN)
	print('Recall:', recallKNN)
	print('F1-score:', f1KNN)
	print(cfKNN)

def irisSVM(x, y):
	print('Support Vector Machines:')

	# Use grid search to find the best parameters
	parameters = {
		'kernel': ['linear', 'sigmoid', 'rbf', 'poly'],
		'C': np.arange(0.1, 2.1, 0.1) # 0.1, 0.2, ... 1.9, 2.0
	}

	rkf = RepeatedKFold(n_splits = K_FOLDS, n_repeats = 5, random_state = SEED)

	svm = SVC()
	clfSVM = GridSearchCV(svm, parameters, cv = rkf)
	clfSVM.fit(x, y)
	for param_name in sorted(parameters.keys()):
		print("%s: %r" % (param_name, clfSVM.best_params_[param_name]))

	# Build the model for the best parameters
	svm = SVC(kernel = clfSVM.best_params_['kernel'], C = clfSVM.best_params_['C'])

	# Use k folds cross validation to evaluate the model
	kf = KFold(n_splits = K_FOLDS, shuffle = True, random_state = SEED)

	# Init variables to hold metrics
	cfSVM = np.zeros(shape = (3, 3))
	successRateSVM = 0
	precisionSVM = 0
	recallSVM = 0

	# Loop over each fold
	for trainIdx, testIdx in kf.split(x):
		xTrain, xTest = x[trainIdx], x[testIdx]
		yTrain, yTest = y[trainIdx], y[testIdx]

		# Train the model on training data of fold
		svm.fit(xTrain, yTrain)
	
		# Make predictions on test set of fold
		predictions = svm.predict(xTest)

		# Calculate the evaluation metrics
		cfSVM += confusion_matrix(yTest, predictions)
		successRateSVM += np.mean(yTest == predictions)
		precisionSVM += precision_score(yTest, predictions, average = 'weighted')
		recallSVM += recall_score(yTest, predictions, average = 'weighted')

	# Average the metrics for each fold
	successRateSVM /= K_FOLDS
	precisionSVM /= K_FOLDS
	recallSVM /= K_FOLDS

	# Calculate f1 from precision and recall
	f1SVM = 2 / (1 / precisionSVM + 1 / recallSVM)

	# Print out everything
	print('Success rate:', successRateSVM)
	print('Precision:', precisionSVM)
	print('Recall:', recallSVM)
	print('F1-score:', f1SVM)
	print(cfSVM)

def irisFFN(x, y):
	print('Feedforward Neural Network:')
	
	# Convert y data to binary matrix
	yBinary = to_categorical(y)

	# Build the neural network
	model = Sequential()
	model.add(Dense(4, activation = 'relu', input_shape = (4,))) # Hidden layer
	model.add(Dense(3, activation = 'softmax')) # Output layer

	# Compile model
	model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
	
	# K-folds cross validation
	kf = KFold(n_splits = K_FOLDS, shuffle = True, random_state = SEED)

	# Init metrics to evaluate the model on
	cfFFN = np.zeros(shape = (3, 3))
	successRateFFN = 0
	precisionFFN = 0
	recallFFN = 0

	# Loop over each fold
	for trainIdx, testIdx in kf.split(x):
		xTrain_, xTest_ = x[trainIdx], x[testIdx]
		yTrain_, yTest_ = y[trainIdx], y[testIdx]
		yTrainBin_, yTestBin_ = yBinary[trainIdx], yBinary[testIdx]

		# Reset the model and train on the training set
		model.reset_states()
		model.fit(xTrain_, yTrainBin_, epochs = 20, batch_size = 1, verbose = 0)

		# Success rate
		results = model.evaluate(xTest_, yTestBin_, verbose = 0)
		successRateFFN += results[1]

		# Confusion matrix and evaluation metrics
		yPred = model.predict_classes(xTest_)

		# Calculate confusion matrix, precision, and recall
		cfFFN += confusion_matrix(yTest_, yPred)
		precisionFFN += precision_score(yTest_, yPred, average = 'weighted')
		recallFFN += recall_score(yTest_, yPred, average = 'weighted')
	
	# Average metrics
	successRateFFN /= K_FOLDS
	precisionFFN /= K_FOLDS
	recallFFN /= K_FOLDS

	# Calculate F1-score
	f1FFN = 2 / (1 / precisionFFN + 1 / recallFFN)

	# Print out everything
	print('Success rate:', successRateFFN)
	print('Precision:', precisionFFN)
	print('Recall:', recallFFN)
	print('F1-score:', f1FFN)
	print(cfFFN)

if __name__ == '__main__':
	main()