#import data_formatting
#import missing_data_calculations
import build_decision_tree
import argparse
import os
import sys
from time import gmtime, strftime

def run():
	
	save_parameters = open(str(args.name) + '_run_parameters.txt', 'w')
	for key in vars(args):
		save_parameters.write(str(key) + ':' + '\t' + str(vars(args)[key]) + '\n')	

	# files to keep results and run information
	output_results = open(str(args.name) + '_metrics.txt', 'w')
	#formatted_matrix = data_formatting.format_input(data=args.matrixFile, rows=args.rownames, cols=args.colnames)
	
	#build_decision_tree(data=formatted_matrix, prediction=args.predicted_outcome)
	data, ground_truth, tree_classifier, data_training, data_test, outcome_training, outcome_test = build_decision_tree(input=args.matrixFile, outcome=args.target_outcome)
	model_cross_validation(input=data, outcome=ground_truth)

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
	
	run()