import argparse
import numpy as np
from sklearn import tree, metrics
import pydotplus
import matplotlib as plt
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
		#matrix = metrics.confusion_matrix(outcome_test, predicted_test_outcome)
		#plt.ylabel('Ground Truth validation set')
		#plt.xlabel('Predicted Outcome validation set')
		#plt.matshow(matrix)
		#plt.show()

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
	
	# builds new tree with all data using cross-validation
	make_tree = tree.DecisionTreeClassifier(min_samples_leaf=args.min_samples, max_depth=args.depth, criterion=args.split)
	cv_scores = cross_val_score(make_tree, input, outcome, cv=args.cross_val, scoring=args.score_method)
	
	# write results of cross-validation both individual and mean scores
	output_results.write(str(args.cross_val) + '-fold cross-validation scores:' + '\n' + str(cv_scores) + '\n')
	output_results.write('The mean cross-validation score is:' + '\t' + str(cv_scores.mean()))