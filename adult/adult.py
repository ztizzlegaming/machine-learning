# Apply a feedforward neural network and k-nearest neighbors to the adult
# dataset. Attempt to predict income (<=50K or >50K) for a person.
# 
# Data preparation:
# I edited the dataset a bit. The test set was a bit different from the
# training set, so I removed the first line and the periods from the class.
# I used pandas to read in the two sets from each file and used one hot
# encoding to encode each of the categorical attributes. I also encoded
# the class as a 0 or one rather than '<=50K' and '>50K'
# 
# Evaluation:
# I split the training set into training and validation sets. I trained the models
# and evaluated them on the validation set. As we will see, the neural network
# performed slightly better on the validation set, so I evaluated the neural
# network on the test set to get a final accuracy score.
# 
# Dependencies:
# pandas
# numpy
# sklearn
# tensorflow
# keras
# 
# Run:
# python3 adult.py
# 
# Jordan Turley

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier

from keras.models import Sequential
from keras.layers import Dense

def main():
	# Read in training data
	training = pd.read_csv('adult.data.txt', sep = ', ', header = None, engine = 'python')
	trainX = training.ix[:, :13]
	trainY = training.ix[:, 14]

	# Read in test data
	test = pd.read_csv('adult.test.txt', sep = ', ', header = None, engine = 'python')
	testX = test.ix[:, :13]
	testY = test.ix[:, 14]

	# Combine the two sets into one for one hot encoding
	x = trainX.append(testX)

	# Encode all x values that are not numbers
	# workclass, education, marital-status, occupation, relationship, race,
	# sex, and native-country
	labelsToEncode = [1, 3, 5, 6, 7, 8, 9, 13]
	for labelIdx in labelsToEncode:
		x = encodeLabels(x, labelIdx)
	
	# Split back up the training and test sets
	trainX = x.iloc[:32561, :]
	testX = x.iloc[32561:, :]

	# Normalize x data
	scaler = StandardScaler()
	scaler.fit(trainX)
	trainX = scaler.transform(trainX)
	testX = scaler.transform(testX)

	# Encode y data
	encoderY = LabelEncoder()
	encoderY.fit(trainY)
	trainY = encoderY.transform(trainY)
	testY = encoderY.transform(testY)

	# Split the training data into training and validation
	trainX, validateX, trainY, validateY = train_test_split(trainX, trainY) # 75% train, 25% validate

	# Build the model
	model = Sequential()
	model.add(Dense(108, activation = 'relu', input_shape = (108,))) # Hidden layer
	model.add(Dense(64, activation = 'relu')) # Hidden layer
	model.add(Dense(32, activation = 'relu')) # Hidden layer
	model.add(Dense(1, activation = 'sigmoid')) # Output layer

	# Compile model and fit to training data
	model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	model.fit(trainX, trainY, epochs = 10, batch_size = 16, verbose = 1)

	results = model.evaluate(validateX, validateY, verbose = 0)
	print(results)

	# Make predictions for the validation set
	predictionsValidate = model.predict_classes(validateX)

	print('FFN:')
	print('Accuracy:', accuracy_score(validateY, predictionsValidate))
	print(confusion_matrix(validateY, predictionsValidate))
	print()

	# Apply k-nearest neighbors to the data
	'''
	This code takes about an hour to run, so I commented it out.
	The best performing parameters are used below.

	knn = KNeighborsClassifier()
	parameters = {
		'n_neighbors': np.arange(10, 21, 1), # 1, 2, ... 19, 20
		'metric': ['euclidean', 'l1', 'manhattan', 'chebyshev'] # Try a few distance metrics
	}

	# Use grid search to find the best parameters
	knn = KNeighborsClassifier()
	clfKNN = GridSearchCV(knn, parameters, n_jobs = -1) # This takes a while, run on all cores
	clfKNN.fit(trainX, trainY)
	for param_name in sorted(parameters.keys()):
		print("%s: %r" % (param_name, clfKNN.best_params_[param_name]))

	'''

	# k-Nearest Neighbors performed best with k = 19 and using l1 distance
	knn = KNeighborsClassifier(n_neighbors = 19, metric = 'l1')
	knn.fit(trainX, trainY)

	# Evaluate knn model on validation set
	predictionsValidate = knn.predict(validateX)

	print('kNN, k = 19, distance = l1')
	print('Accuracy:', accuracy_score(validateY, predictionsValidate))
	print(confusion_matrix(validateY, predictionsValidate))
	print()

	# FFN: about 85% correct, kNN: about 83% correct
	# Neural net does slightly better, evaluate neural network on test set
	predictions = model.predict_classes(testX)

	# Print out the final results on the test set
	print('FFN test set:')
	print('Accuracy:', accuracy_score(testY, predictions))
	print(classification_report(testY, predictions, target_names = ['<=50K', '>50K'], digits = 3)) # , target_names = target_names
	print(confusion_matrix(testY, predictions))

def encodeLabels(x, idx):
	'''Encodes the attributes at the given column of x using one hot encoding.
	'''
	# Get the encoding of the column
	ohe = x[idx].str.get_dummies()

	# Join the two together
	x = pd.concat([x, ohe], axis = 1)

	# Delete the original column
	del x[idx]

	return x

if __name__ == '__main__':
	main()