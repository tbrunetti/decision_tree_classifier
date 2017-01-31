import argparse
from collections import Counter
import numpy as np
from sklearn import tree, metrics
import pydotplus
import matplotlib as plt
from sklearn.externals.six import StringIO
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split
from imblearn.combine import SMOTETomek
import pandas
import graphviz
import os
import sys


def smotetomek(input, outcome):
	#create SMOTE Tomek link resampling object
	smotetomek = SMOTETomek()
	data_resampled, ground_truth_resampled = smotetomek.fit_sample(input, outcome)
	return data_resampled, ground_truth_resampled


# return ratio of imbalance in data set
def check_imbalance(input, outcome):
	# read and store data
	data = pandas.read_csv(input, delimiter=',', header=0)
	freq = data[outcome].value_counts()
	outcome_class = data[outcome].value_counts().index.tolist()
	total_obs = sum(freq)
	
	for i in outcome_class:
		print 'There are ' + str(freq[i]) + " samples in class " + str(i) +' (' + str(((float(freq[i])/float(total_obs))*100.0)) + '%)'
	
	
def manage_imbalance(input, outcome, method):
	# read and store data
	data = pandas.read_csv(input, delimiter=',', header=0)
	ground_truth = data.loc[:, str(outcome)] # store true outcomes
	
	# remove predictor variable from dataset
	data.drop(str(outcome), axis=1, inplace=True)
	column_names = list(data) # extract header information
	
	# convert to numpy arrays
	data = data.as_matrix()
	ground_truth = ground_truth.as_matrix()

	if method == 'SMOTETomek_option':
		data_resampled, ground_truth_resampled = smotetomek(input=data, outcome=ground_truth)
		print 'The data set has the following distribution post-balancing:'
		for key in Counter(ground_truth_resampled):
			print 'There are ' + str(Counter(ground_truth_resampled)[key]) + " samples in the balanced class " + str(key) +' (' + str(((float(Counter(ground_truth_resampled)[key])/float(len(ground_truth_resampled)))*100.0)) + '%)'

		return data_resampled, ground_truth_resampled


if __name__ == '__main__':
	parser=argparse.ArgumentParser("Cleans data and builds and trains decision tree")
	parser.add_argument('-dataInput', required=True, dest='matrixFile', help='Full path to comma-delimited "matrix" file')
	parser.add_argument('-predictColumn', required=True, dest='target_outcome', help='Exact column name in dataset to be used as predicted outcome')
	parser.add_argument('--balance_method', default='SMOTETomek_option', dest='balance_method', help='method for balancing data set')
	args=parser.parse_args()

	check_imbalance(input=args.matrixFile, outcome=args.target_outcome)
	data_resampled, ground_truth_resampled = manage_imbalance(input=args.matrixFile, outcome=args.target_outcome, method=args.balance_method)