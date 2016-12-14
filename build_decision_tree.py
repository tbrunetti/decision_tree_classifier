import argparse
from collections import deque
import numpy
import pandas
from sklearn import tree
import data_formatting


def missing_categorical():
	pass;


def missing_continuous():
	pass;


def build_decision_tree(data, prediction):
	pass;

if __name__ == '__main__':
	parser=argparse.ArgumentParser("Cleans data and builds and trains decision tree")
	parser.add_argument('-dataInput', required=True, dest='matrixFile', help='Full path to tab-delimited "matrix" file')
	parser.add_argument('-predictColumn', required=True, dest='predicted_outcome', help='[INT] column number in dataset that has variable to predict/correlate i.e. first column = 1, second column =2, etc... ')
	parser.add_argument('-contVars', default=None, dest='------------', help='a comma separated list of all columns that are continuous variables')
	parser.add_argument('--missing_continuous', dest='impute_cont', action='store_true', help='indicates missing values of continuous type are in data set; This option will handle missing values by imputation')
	parser.add_argument('--missing_categorical', dest='impute_cat', action='store_true', help='indicates missing values of categorical type are in data set; this option will handle missing values by assigning negative one to NA cells')
	parser.add_argument('--rowNames', dest='rownames', action='store_true', help='row names are included in dataInput in first column')
	parser.add_argument('--colNames', dest='colnames', action='store_true', help='column names are included in dataInput in first row')
	args=parser.parse_args()	

	

	formatted_matrix = data_formatting.format_input(data=args.matrixFile, rows=args.rownames, cols=args.colnames)
	
	build_decision_tree(data=formatted_matrix, prediction=args.predicted_outcome)