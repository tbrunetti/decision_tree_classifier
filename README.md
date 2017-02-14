# decision_tree_classifier
In progress
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

It is important to note that although the minimum command above will build a decision tree model, there are several options for the user to fine tune and control the rules of the model.  

###Expected Output
-------------------
There are a total of **4 files** that result from running the program:
* decision_tree_path.pdf
* full_dataset_decision_tree_path.pdf
* metrics.txt
* run_parameters.txt
