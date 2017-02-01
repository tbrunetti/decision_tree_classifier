#import data_formatting
#import missing_data_calculations
import build_decision_tree
import imbalance_set
import argparse
import os
import sys
from time import gmtime, strftime


def run(**kwargs):

	save_parameters = open(str(kwargs['name']) + '_run_parameters.txt', 'w')
	for key in vars(args):
		save_parameters.write(str(key) + ':' + '\t' + str(vars(args)[key]) + '\n')	

	if kwargs['balance'] == None:
		#build_decision_tree(data=formatted_matrix, prediction=args.predicted_outcome)
		data, ground_truth, tree_classifier, data_training, data_test, outcome_training, outcome_test = build_decision_tree.build_decision_tree(input=kwargs['matrixFile'], outcome=kwargs['target_outcome'], method=kwargs['balance'], cols=None, **kwargs)
		build_decision_tree.model_cross_validation(input=data, outcome=ground_truth, **kwargs)
	
	else:
		imbalance_set.check_imbalance(input=kwargs['matrixFile'], outcome=kwargs['target_outcome'], **kwargs)
		data_resampled, ground_truth_resampled, column_names = imbalance_set.manage_imbalance(input=kwargs['matrixFile'], outcome=kwargs['target_outcome'], method=kwargs['balance'],**kwargs)
		data, ground_truth, tree_classifier, data_training, data_test, outcome_training, outcome_test = build_decision_tree.build_decision_tree(input=data_resampled, outcome=ground_truth_resampled, method=kwargs['balance'], cols=column_names, **kwargs)
		build_decision_tree.model_cross_validation(input=data, outcome=ground_truth, **kwargs)

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
	parser.add_argument('--balance_method', default=None, dest='balance', help='method for balancing data set')
	args = parser.parse_args()
	kwargs = vars(args)
	run(**kwargs)
