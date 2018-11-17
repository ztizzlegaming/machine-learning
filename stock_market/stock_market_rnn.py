# CSC 420 - Machine Learning
# Attempt to predict if a stock will go up or down tomorrow from the closing
# prices from the past year.
# 
# Apply a recurrent neural network to the problem
# 
# This script uses a library called keras, which also uses tensorflow
# 
# Jordan Turley

import os
import random
import numpy
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

# 93 increased, 407 decreased

RANDOM_STATE = 7
BATCH_SIZE = 1
EPOCHS = 100

TARGET_NAMES = ['decrease', 'increase']

def runModel(stock, look_back, blocks, layers):
	'''Reads in the values for a given stock from its file in data/, then
	converts the values to time series. The model uses the past few days as
	the series to try to predict if the stock will increase or decrease
	tomorrow. The number of days to use is given by look_back.
	Blocks is the number of blocks that the neural network uses.
	Layers is the number of hidden layers to add to the neural network.
	'''
	numpy.random.seed(RANDOM_STATE)
	
	# Use pandas to read in the values of the stock
	dataframe = read_csv('data_1_day/' + stock + '.txt', usecols=[0], engine='python', header = None)
	
	# Get the values out
	dataset = dataframe.values

	# Get the last increase or decrease for the next day
	lastIncDec = dataset[-1, 0]

	# Remove the last value
	dataset = dataset[:-1]

	# Split the data into training, validation, and test sets
	# We do this manually to preserve the time series
	train_size = int(len(dataset) * 0.5)
	validation_size = int(len(dataset) * 0.25)
	test_size = len(dataset) - train_size - validation_size

	# Split up the data
	train = dataset[:train_size, :]
	validate = dataset[train_size:train_size + validation_size, :]
	test = dataset[train_size + validation_size:, :]

	# The last value of the training set goes into the validation,
	# and the last of validation goes into test, so we have to manually
	# set if the last one increases or decreases
	lastTrainX = 'increase'
	if validate[0, 0] <= train[-1, 0]:
		lastTrainX = 'decrease'

	lastValidateX = 'increase'
	if test[0, 0] <= validate[-1, 0]:
		lastValidateX = 'decrease'

	# Split up the data and convert it to series
	trainX, trainY = create_dataset(train, lastTrainX, look_back)
	validateX, validateY = create_dataset(validate, lastValidateX, look_back)
	testX, testY = create_dataset(test, lastIncDec, look_back)

	# Rescale the data using StandardScaler
	scaler = StandardScaler()
	scaler.fit(trainX)
	trainX = scaler.transform(trainX)
	validateX = scaler.transform(validateX)
	testX = scaler.transform(testX)

	# Reshape the data to look like a time series
	trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
	validateX = numpy.reshape(validateX, (validateX.shape[0], validateX.shape[1], 1))
	testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

	# Encode y as 0 or 1 for decrease or increase
	encoder = LabelEncoder()
	encoder.fit(trainY)
	encodedTrainY = encoder.transform(trainY)
	encodedValidateY = encoder.transform(validateY)
	encodedTestY = encoder.transform(testY)

	# Build our recurrent neural network model
	model = Sequential()

	# Add hidden layers
	for i1 in range(layers - 1):
		model.add(LSTM(blocks, batch_input_shape=(BATCH_SIZE, look_back, 1), stateful=True, return_sequences=True))

	model.add(LSTM(blocks, batch_input_shape=(BATCH_SIZE, look_back, 1), stateful = True))
	
	# Add the output layer
	model.add(Dense(1))
	
	# Compile the model. Binary cross entropy because this is a binary problem
	model.compile(loss = 'binary_crossentropy', optimizer = 'adam')
	
	# Train model over many epochs
	for i1 in range(EPOCHS):
		print('Iteration', (i1 + 1))
		model.fit(trainX, encodedTrainY, epochs = 1, batch_size = BATCH_SIZE, verbose = 2)
		model.reset_states()

	# Make predictions
	trainPredictions = model.predict_classes(trainX, batch_size = BATCH_SIZE)
	validationPredictions = model.predict_classes(validateX, batch_size = BATCH_SIZE)
	testPrediction = model.predict_classes(testX, batch_size = BATCH_SIZE)

	# Find the success rate
	trainPerf = numpy.mean(trainPredictions == encodedTrainY)
	validatePerf = numpy.mean(validationPredictions == encodedValidateY)
	testPerf = numpy.mean(testPrediction == encodedTestY)

	# Print out the success rates
	print('Performance train:', trainPerf)
	print('Performance validate:', validatePerf)
	print('Performance test:', testPerf)

	# Make predictions on the test set
	testPrediction = encoder.inverse_transform(testPrediction)

	# Metrics: Precision, Recall, F1-Score, and Support
	print(metrics.classification_report(testY, testPrediction, target_names = TARGET_NAMES))

	# Confusion matrix
	print(metrics.confusion_matrix(testY, testPrediction))

	return trainPerf, validatePerf, testPerf

def create_dataset(dataset, nex, look_back):
	'''Creates a dataset of x and y values for a given number of days to look
	back.
	dataset contains the stock closing values
	nex is the final class, as it cannot be seen in the dataset
	look_back is the number of days to look back and get data to try to predict
	the next day
	'''
	dataX, dataY = [], []
	for i1 in range(len(dataset) - look_back):
		x = dataset[i1:i1 + look_back, 0]
		dataX.append(x)
		if (dataset[i1 + 1] > dataset[i1]):
			dataY.append(['increase'])
		else:
			dataY.append(['decrease'])

	# Add in the last class
	# Because we are looking one past the last entry in the dataset, we have
	# to do this manually.
	dataX.append(dataset[-look_back:, 0])
	dataY.append([nex])

	return numpy.array(dataX), numpy.array(dataY)

def main():
	# Get all of the stocks in the data_1_day folder
	os.chdir('data_1_day')
	files = os.listdir()
	os.chdir('../')
	stocks = []
	for f in files:
		if f == '.DS_Store':
			continue
		stocks.append(f[:-4])

	# These values were determined by validateFindHyperparameters()
	lookBack = 1
	blocks = 5
	layers = 3

	# Shuffle, select 50 to evaluate
	random.seed(RANDOM_STATE)
	randStocks = random.sample(stocks, 50)

	scores = []

	'''
	# This code was ran on 50 random stocks to see which performed well and which didn't
	for stock in randStocks:
		trainScore, valScore, testScore = runModel(stock, lookBack, blocks, layers)
		scores.append({'stock': stock, 'test_score': testScore})

	bestStock = ''
	bestScore = 0
	worstStock = ''
	worstScore = 1
	for score in scores:
		stock = score['stock']
		testScore = score['test_score']

		if testScore > bestScore:
			bestStock = stock
			bestScore = testScore
		
		if testScore < worstScore:
			worstStock = stock
			worstScore = testScore

	print('Best:', bestStock, bestScore)
	print('Worst:', worstStock, worstScore)'''

	# The best performing stock was MAT. Run the model on only this stock
	runModel('MAT', lookBack, blocks, layers)

	# The worst performing stock was LB
	runModel('LB', lookBack, blocks, layers)

def validateFindHyperparams():
	'''Performs grid search over five different stocks for several values of
	number of days to look back, values for blocks of the LSTM network, and 
	number of hidden layers in the network. Prints out the best score and the
	hyperparameters that achieved this score.
	This took over a day to run, because so many neural networks had to be
	trained and evaluated
	'''
	stocks = ['AET', 'CHD', 'DUK', 'MSFT', 'SCG'] # AET increased, all others decreased
	lookBackVals = [1, 2, 3, 4, 5]
	blockVals = [1, 2, 3, 4, 5]
	layerVals = [1, 2, 3, 4, 5]

	scores = []

	# Loop over each combination of the values
	for lookBack in lookBackVals:
		for block in blockVals:
			for layer in layerVals:
				avgVScore = 0
				# Evaluate the stock with these parameters
				for s in stocks:
					avgVScore += runModel(s, lookBack, block, layer)

				# Average the scores for the five stocks
				avgVScore /= len(stocks)
				scores.append({
					'look_back': lookBack,
					'block': block,
					'layer': layer,
					'score': avgVScore
				})

	# Find the best score
	bestScore = 0
	bestScoreObj = {}
	for score in scores:
		if score['score'] > bestScore:
			bestScoreObj = score

	print(bestScoreObj)

if __name__ == '__main__':
	main()
