# CSC 420 - Machine Learning
# Apply Multinomial Naive Bayes to Recipe Data
# 
# First, extract cookbook_import_pages_current.zip.
# 
# Then, run recipe_classify.py to hand classify pages as recipes or
# non-recipes, which generates data.txt, or use the data.txt file that is
# included in the directory.
# 
# Run this script to apply naive bayes to the data. Misclassified data will be
# printed out, as well as the precision, recall, F1-score, and support. The
# confusion matrix is also shown.
# 
# The database is downloaded here:
# http://recipes.wikia.com/wiki/Special:Statistics
# 
# Jordan Turley

# Constants
DATA_FILE = 'data.txt'
TEST_PROPORTION = 0.2

import random

# Import everything we need from sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics

# Numpy for error rate
import numpy as np

def main():
	# Read in the data from the file
	data, classes = readData(DATA_FILE)
	target_names = ['y', 'n']

	# Split the data into training and test sets
	training_data, test_data, training_classes, test_classes = train_test_split(data, classes, test_size = TEST_PROPORTION, random_state = 0)

	# Create the pipeline for performing Multinomial Naive Bayes
	text_clf = Pipeline([('vect', CountVectorizer()),
						 ('tfidf', TfidfTransformer()),
						 ('clf', MultinomialNB())
	])

	# Train the model with the training data
	text_clf.fit(training_data, training_classes)

	# Predict y or n for the test set
	test_predict = text_clf.predict(test_data)
	print(test_predict)

	# Go through and print out the ones that were incorrectly classified
	for i1 in range(len(test_classes)):
		real_class = test_classes[i1]
		predicted_class = test_predict[i1]
		if real_class != predicted_class:
			print(i1)
			print(test_data[i1])
			print('Predicted', predicted_class, 'but real is', real_class)
			print()

	# Success rate
	print('Success rate:', np.mean(test_predict == test_classes))

	# Metrics: Precision, Recall, F1-Score, and Support
	print(metrics.classification_report(test_classes, test_predict, target_names = target_names))

	# Confusion matrix
	print(metrics.confusion_matrix(test_classes, test_predict))

def readData(filename):
	'''Reads the data from the given filename and splits up the data and class,
	then returns the data in two arrays that has been randomized.'''
	file = open(filename)

	# Get the array of lines out of the file
	lines = file.readlines()

	file.close()

	# Get the number of instances from the first line
	count = int(lines[0])

	data = []

	# Go through each group, make it into a dictionary,
	# and stick it in the array
	for i1 in range(count):
		classification = lines[i1 * 2 + 1].strip()
		line = lines[i1 * 2 + 2].strip()

		data.append({
			'line': line,
			'class': classification
		})
	
	# Shuffle the data
	random.shuffle(data)

	# Go through and split the data into two arrays
	text = []
	classes = []
	for elem in data:
		text.append(elem['line'])
		classes.append(elem['class'])

	return text, classes

if __name__ == '__main__':
	main()