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

def smote_enn(input, outcome):
	# create SMOTE ENN resampling object
	smoteenn = SMOTESENN()
	data_resampled, ground_truth_resampled = smoteenn.fit_sample(input, outcome)
	return data_resampled, ground_truth_resampled


# return ratio of imbalance in data set
def check_imbalance(input, outcome, **kwargs):
	output_results = open(str(kwargs['name']) + '_metrics.txt', 'a+')

	# read and store data
	data = pandas.read_csv(input, delimiter=',', header=0)
	freq = data[outcome].value_counts()
	outcome_class = data[outcome].value_counts().index.tolist()
	total_obs = sum(freq)
	
	for i in outcome_class:
		output_results.write('There are ' + str(freq[i]) + " samples in class " + str(i) +' (' + str(((float(freq[i])/float(total_obs))*100.0)) + '%)' + '\n')
	
	
def manage_imbalance(input, outcome, method, **kwargs):
	
	output_results = open(str(kwargs['name']) + '_metrics.txt', 'a+')

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
		output_results.write('The data set has the following distribution post-balancing:'+'\n')
		output_results.write('method of balancing:  SMOTE for underbalancing and Tomek Links for cleaning/undersampling' + '\n')
		for key in Counter(ground_truth_resampled):
			output_results.write('There are ' + str(Counter(ground_truth_resampled)[key]) + " samples in the balanced class " + str(key) +' (' + str(((float(Counter(ground_truth_resampled)[key])/float(len(ground_truth_resampled)))*100.0)) + '%)' +'\n')

	elif method == 'SMOTE_ENN_option':
		data_resampled, ground_truth_resampled = smote_enn(input=data, outcome=ground_truth)
		output_results.write('The data set has the following distrubtion post-balancing:' + '\n')
		output_results.write('method of balancing:  SMOTE for underbalancing and edited nearest neighbors (ENN) for oversampling' + '\n')
		for key in Counter(ground_truth_resampled):
			output_results.write('There are ' + str(Counter(ground_truth_resampled)[key]) + " samples in the balanced class " + str(key) +' (' + str(((float(Counter(ground_truth_resampled)[key])/float(len(ground_truth_resampled)))*100.0)) + '%)' +'\n')


	return data_resampled, ground_truth_resampled, column_names
