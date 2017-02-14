# decision_tree_classifier
In progress:
* data_formatting.py
* missing_data_calculations.py
* other methods for balancing data

###Overview
-----------
![Alt text](https://github.com/tbrunetti/decision_tree_classifier/blob/master/pipeline_overview.jpg)

###Software Requirements
-------------------------
* Python minimum version requirement 2.7.6
* Python scikit-learn version >=  0.18.0 (http://scikit-learn.org/stable)
* imbalanced-learn version >= 0.2.1 (https://pypi.python.org/pypi/imbalanced-learn)
* SciPy libraries (https://scipy.org)
  * Pandas version >= 0.19.2
  * Numpy version >= 1.12
  * Matplotlib version >= 2.0.0
* Pydotplus version >= 2.0.2 (https://pypi.python.org/pypi/pydotplus)
* Graphviz (http://www.graphviz.org)
* graphviz for Python version >= 0.5.2 (https://pypi.python.org/pypi/graphviz)

###User Input Requirements
---------------------------
* comma separated file including header at top and with predictor column

###Installation and Configuration
----------------------------------
It is critical that all the above listed software requirements are met and are in your Python path.  Most of the requirements can easily be installed using pip:
```
python -m pip install --upgrade pip
pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose
pip install -U scikit-learn
pip install -U imbalanced-learn
pip install pydotplus
pip install graphviz
```
The only software that may be more difficult to install is Graphviz from http://www.graphviz.org.  Download the appropriate package for your system and follow the build instructions pertaining to your OS.  Make sure it is visible or within your Python path.
 
Next, clone the repository into a directory
```
mkdir ~/my_project
cd ~/my_project
git clone https://github.com/tbrunetti/decision_tree_classifier.git
cd decision_tree_classifier
```
###Running the Software
------------------------
To run the software only two arguments are required: *-dataInput* and *-predictColumn*  

*-dataInput* must be a full path to a **comma-separated value (csv) file including a header** of all the data that is to be considered to use in building the decision tree.  It must **include the predictor column**.  

*-predictColumn* must be the **EXACT** name of the column in the -dataInput that is the outcome to be predicted  

For example if the data file was called file.csv and the name of the outcome to be predicted in the file is named "OutcomeToPredict", the format of the minimum command to build a decision tree is as follows:
```
python run_build_and_predict.py -dataInput /path/to/file.csv -predictColumn OutcomeToPredict
```
The other Python files are modules that are made and imported into run_build_and_predict.py.  Therefore, they should never be called themselves as they will not run on their own.  

It is important to note that although the minimum command above will build a decision tree model, there are several options for the user to fine tune and control the rules of the model.  The above overview flowchart illustrates the options and default values.  
* **--run****_****name**   can be used to specify output file name prefix.  If no name is specified the default is a time-date stamp
* **--pct****_****test** is a float that ranges from 0.0 to 1.0.  This set the fraction of data to be used/saved for the test/validation set.  The default is 0.20, i.e. 20% of data is used for validation and the remaining 80% is used as training.
* **--class**  a comma-separated list of class names in ascending order for the predictor column.  For example, if 0 in predictor column corresponds to No and 1 in predictor column corresponds to yes, the user can type: No,Yes following the class flag.  If classes are not specified, chronological integers beginning with 0 will be used as substitute for outcome class names.  
* **--minLeaf** is an integer used to determine the minimum number of samples that need to be in a node/leaf for splitting.  The default is 5.
* **--maxDepth** is an integer used to determine the maximum depth a tree can have before splitting stops.  The default value is 3.
* **--split****_****criteria** is the calculation to determine how much information or homogeneity is at the node.  Used to determine which column in dataset is most useful for splitting the tree.  There are two options: gini or entropy.  The default is gini.
* **--seed**  this can be set to any integer.  This ensures that the data is split in the same exact way each time to reproduce the same results.  If this parameter is not set, the default is None.  This means every time the program is run a different tree or outcome may result.  NOTE! as of right now, if balance_method is selected this seed is overridden since the oversampling introduces new unique data.  Will be fixed soon.
* **--cross****_****val** this is an integer that is used to determine the k-fold cross validation for the model.  The default is 10.  i.e. The data is split into 10 parts, and each iteration 1 part is reserved for validation and the remaining 9 parts are used to train.  This is repeated 10 times until each part has been reserved and tested for validation.
* **--scoring** the method to indicate how the cross-validation should be scored.  Default is accuracy.  For all possible options see sklearn.model_selection
* **--balance****_****method**  If the data being used is not balanced in terms of equal occurences of the predicted outcome, the user can choose to balance the data set.  The default is None, meaning no balancing will occur.  The other option is to type in SMOTETomek_option.  This will attempt to balance the dataset by oversampling the minority group and undersampling the majority group.  Oversampling will try to cluster the imbalanced group and add artificial samples that carry attributes of the minority cluster and add it to the dataset.  Other options to become available in the future.  

For a quick help guide for all user options run the following command:
```
python run_build_and_predict.py -h
```
###Expected Output
-------------------
There are a total of **4 files** that result from running the program:
* decision_tree_path.pdf
* full_dataset_decision_tree_path.pdf
* metrics.txt
* run_parameters.txt  

The two PDF files are illustrations of the decision tree.  The full_dataset_decision_tree is probably a little overfit as it uses the entire dataset.  I would recommend using decision_tree_path.pdf.  

metrics.txt give run statistics such as balancing, accuracy, cross-validation results, model accuracy, etc...  

run_parameters.txt give the user all the parameters and flags that were set
