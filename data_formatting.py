from collections import deque
import pandas

# imported into build_decision_tree.py
# used to organize and format data into data frame in prep for missing data and decision tree

def format_input(data, rows, cols):
	
	with open(data) as user_input:
		
		data_values = []
		rowNames = []
		
		if rows == True and cols== True:
			header = next(user_input)
			header = line.rstrip('\n').split('\t')
			# assuming empty tab in beginning where row names are listed
			header.pop(0)
			
			for line in user_input:
				line = deque(line.rstrip('\n').split('\t'))
				rowNames.append(line[0]) 		# stores row name
				line.popleft() 					# gets rid of row name
				data_values.append(list(line))

			#store data in dataframe with labels supplied by user in matrix
			formatted_matrix = pandas.DataFrame(data_values, index=rowNames, columns=header)
			
			return formatted_matrix
				
		elif rows == True and cols == False:
			for line in user_input:
				line = deque(line.rstrip('\n').split('\t'))
				rowNames.append(line[0])
				line.popleft()
				data_values.append(list(line))

			# only adds matrix names to rows, columns are just numbered according to pandas
			formatted_matrix = pandas.DataFrame(data_values, index=rowNames)
			
			return formatted_matrix
		
		elif rows == False and cols == True:
			header = next(user_input)
			header = line.rstrip('\n').split('\t')
			# assuming empty tab in beginning where row names are listed
			header.pop(0)
			
			for line in user_input:
				line = line.rstrip('\n').split('\t')
				data_values.append(line)

			#store data in dataframe with column labels only, row names are default of pandas numbering
			formatted_matrix = pandas.DataFrame(data_values, columns=header)
			
			return formatted_matrix
		
		#in this instance it means row and column names were not supplied by the user
		else:
			for line in user_input:
				line = line.rstrip('\n').split('\t')
				data_values.append(line)

			# no customized labeling of rows or columns.  Defaults to pandas labels	
			formatted_matrix = pandas.DataFrame(data_values)
			
			return formatted_matrix
