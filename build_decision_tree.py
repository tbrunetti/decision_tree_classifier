import argparse
from time import gmtime, strftime
import numpy as np
from sklearn import tree, metrics
import pydotplus
from sklearn.externals.six import StringIO
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split
import pandas
import graphviz
import os
import sys


#def build_decision_tree(data, prediction):
def build_decision_tree(input, outcome):

	# read and store data
	data = pandas.read_csv(input, delimiter=',', header=0)
	ground_truth = data.loc[:, str(outcome)] # store true outcomes
	
	# remove predictor variable from dataset
	data.drop(str(outcome), axis=1, inplace=True)
	column_names = list(data) # extract header information
	
	# convert to numpy arrays
	data = data.as_matrix()
	ground_truth = ground_truth.as_matrix()

	# split data into training and test set
	# setting a random state ensures data is split exact same way everytime alg is run assuming input
	# data is the same
	data_training, data_test, outcome_training, outcome_test = train_test_split(
		data, ground_truth, test_size=args.pct_test, random_state=args.seed)

	# build classifier using training data
	tree_classifier = tree.DecisionTreeClassifier(min_samples_leaf=args.min_samples, max_depth=args.depth, criterion=args.split)
	tree_classifier = tree_classifier.fit(data_training, outcome_training)

	# have classifier predict class of test data and store prediction in predicted_outcome
	predicted_test_outcome = tree_classifier.predict(data_test)
	dot_data = StringIO()
	tree.export_graphviz(tree_classifier, out_file=dot_data, impurity=True, filled=True, rounded=True, feature_names=column_names)
	graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
	graph.write_pdf(str(args.name) + '_decision_tree_path' + '.pdf')

	# use test set to determine how well classifier performs
	test_score = tree_classifier.score(data_test, outcome_test)
	acc = metrics.accuracy_score(outcome_test, predicted_test_outcome)
	output_results.write("The score of the classifer using the test set is " + str(test_score)+'\n')
	output_results.write("The accuracy of the classifier using the test set is " + str(acc)+'\n')
	
	# create list of class names
	if args.class_names != None:
		class_names = args.class_names.split(',')
	else:
		class_names = [str(x) for x in list(set(ground_truth.flat))]
	
	# checks if predicted outcome classes are binary and whether to build confusion matrix
	if len(class_names) > 2:
		print "Predicted classes are not binary, more than 2 classes" + '\n' + str(class_names)
		print 'True negative, false positive, false negative, true postive confusion matrix will not be created'
	else:
		print "Calculating confusion matrix..."
		# get confusion matrix to show how samples were classified
		# only works for binary outcomes
		tn, fp, fn, tp = metrics.confusion_matrix(outcome_test, predicted_test_outcome).ravel()
		output_results.write('Confusion matrix results on test set:' + '\n' + \
							'The number of true negatives is '+ str(tn) +'\n' \
							'The number of false positives is ' + str(fp) +'\n' + \
							'The number of false negatives is ' + str(fn) + '\n' + \
							'The number of true positives is ' + str(tp) + '\n')

	# write out statistics precision, recall, f1-score, support
	output_results.write(str(metrics.classification_report(outcome_test, predicted_test_outcome, target_names=class_names)) + '\n')
		
	return data, ground_truth, tree_classifier, data_training, data_test, outcome_training, outcome_test


def model_cross_validation(input, outcome):
	output_results.write('------------------CROSS-VALIDATION OF MODEL--------------------' + '\n')
	make_tree = tree.DecisionTreeClassifier(min_samples_leaf=args.min_samples, max_depth=args.depth, criterion=args.split)
	cv_scores = cross_val_score(make_tree, input, outcome, cv=args.cross_val, scoring=args.score_method)
	output_results.write(str(args.cross_val) + '-fold cross-validation scores:' + '\n' + str(cv_scores) + '\n')
	output_results.write('The mean cross-validation score is:' + '\t' + str(cv_scores.mean()))

if __name__ == '__main__':
	parser=argparse.ArgumentParser("Cleans data and builds and trains decision tree")
	parser.add_argument('-dataInput', required=True, dest='matrixFile', help='Full path to comma-delimited "matrix" file')
	parser.add_argument('-predictColumn', required=True, dest='target_outcome', help='Exact column name in dataset to be used as predicted outcome')
	parser.add_argument('--run_name', default=strftime("%Y-%m-%d__%H:%M:%S", gmtime()), dest='name', type=str, help='name of run to store results')
	parser.add_argument('--pct_test', default=0.20, dest='pct_test', type=float, help='[FLOAT] 0.0-1.0, percent of data to use for testing classifier, default=0.20')
	parser.add_argument('--class', default=None, dest='class_names', type=str, help='comma separated list of predicted class names corresponding to predicted_outcome values in ascending order')
	parser.add_argument('--minLeaf', default=5, dest='min_samples', type=int, help='[INT] Minimum samples in a node for splitting, default=5')
	parser.add_argument('--maxDepth', default=3, dest='depth', type=int, help='[INT] Maximum depth to build tree, default=3')
	parser.add_argument('--split_criteria', default='gini', dest='split', help='gini or entropy, default=gini')
	parser.add_argument('--seed', default=None, dest='seed', type=int, help='[INT] set seed to ensure data is split the same way every run, assuming no changes in input, default=0')
	parser.add_argument('--cross_val', default=10, dest='cross_val', type=int, help='[INT] for k-fold cross validation of model use, default=10')
	parser.add_argument('--scoring', default='accuracy', dest='score_method', help='sting indicating how cross validation should be scored, default=accuracy (see sklearn.model_selection)' )
	#parser.add_argument('--missing_continuous', dest='impute_cont', action='store_true', help='indicates missing values of continuous type are in data set; This option will handle missing values by imputation')
	#parser.add_argument('--missing_categorical', dest='impute_cat', action='store_true', help='indicates missing values of categorical type are in data set; this option will handle missing values by assigning negative one to NA cells')
	#parser.add_argument('--rowNames', dest='rownames', action='store_true', help='row names are included in dataInput in first column')
	#parser.add_argument('--colNames', dest='colnames', action='store_true', help='column names are included in dataInput in first row')
	args=parser.parse_args()
	save_parameters = open(str(args.name) + '_run_parameters.txt', 'w')
	for key in vars(args):
		save_parameters.write(str(key) + ':' + '\t' + str(vars(args)[key]) + '\n')	

	# files to keep results and run information
	output_results = open(str(args.name) + '_metrics.txt', 'w')
	#formatted_matrix = data_formatting.format_input(data=args.matrixFile, rows=args.rownames, cols=args.colnames)
	
	#build_decision_tree(data=formatted_matrix, prediction=args.predicted_outcome)
	data, ground_truth, tree_classifier, data_training, data_test, outcome_training, outcome_test = build_decision_tree(input=args.matrixFile, outcome=args.target_outcome)
	model_cross_validation(input=data, outcome=ground_truth)