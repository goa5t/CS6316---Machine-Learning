# Starting code for CS6316/4501 HW3, Fall 2016
# By Weilin Xu

import numpy as np
import pandas as pd

from math import sqrt
from sklearn.svm import SVC
import random
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,MinMaxScaler

# Attention: You're not allowed to use the model_selection module in sklearn.
#            You're expected to implement it with your own code.
# from sklearn.model_selection import GridSearchCV

class SvmIncomeClassifier:
    def __init__(self):
        random.seed(0)

    def load_data(self, csv_fpath, csv_fpath2):
        col_names_x = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status','occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss','hours-per-week', 'native-country']
        col_names_y = ['label']

        numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship','race', 'sex', 'native-country']

        # 1. Data pre-processing.
        # Hint: Feel free to use some existing libraries for easier data pre-processing.
	#Cat features: 1, 3, 5, 6, 7, 8, 9, 13, 14
	#num features: 0, 2, 4, 10, 11, 12, 15

	cat_cols = [1, 3, 5, 6, 7, 8, 9, 13, 14]
	num_cols = [0, 2, 4, 10, 11, 12, 15]

	# read in data first, and apply column names
        train = pd.read_csv(csv_fpath, names = col_names_x + col_names_y, header = None)
	test = pd.read_csv(csv_fpath2, names = col_names_x + col_names_y, header = None)

	#print "column names"
	#print train.columns
	
	#get cat values into a 2D numpy array
	train_cat_val = np.array(train[cat_cols])
	test_cat_val = np.array(test[cat_cols])
	train_cat_val = np.concatenate((train_cat_val, test_cat_val), axis=0)
	#print "cat vals"
	#print train_cat_val
	#print "length:"
	#print train_cat_val.shape

	#label encoder to encode values
	enc_label = LabelEncoder()
	train_data = enc_label.fit_transform(train_cat_val[:,0])
	for i in range (1, train_cat_val.shape[1]):
		enc_label = LabelEncoder()
		train_data = np.column_stack((train_data, enc_label.fit_transform(train_cat_val[:,i])))
	
	#print "train data"
	#print train_data

	

	enc = OneHotEncoder()
	#train_cat_data = enc.fit_transform(train_cat_val)
	enc.fit(train_data)
	#train_cat_data = enc_label.fit(train_data)
	train_cat_data = enc.transform(train_data)
	#print "enc nvals"
	#print enc.n_values_

	#create a list of columns to create a DF from original array
	cols = [str(categorical_cols[i]) + '_' + str(j) for i in range(0, len(cat_cols)-1) for j in range(0,enc.n_values_[i]) ]
	cols.append('label_0')
	cols.append('label_1')
	#print "cols = "
	#print cols
	train_cat_data_df = pd.DataFrame(train_cat_data.toarray(), columns=cols)


	#print train_cat_data_df
	#need to delete old continuous columns
	train = train.drop(categorical_cols, axis=1)
	
	y = train_cat_data_df[['label_1']]
	y = y.iloc[:38842]
	#delete y cols from training set
	train = train.drop(col_names_y, axis=1)
	"""
	print "train before adding encoded"
	print train
	print "print y "
	print y
	"""
	
	##need to scale the numerical cols
	scaler = MinMaxScaler()
	train = pd.DataFrame(scaler.fit_transform(train), columns=numerical_cols)
	
	
	#need to add the encoded values
	del cols[-2:]
	train[cols] = train_cat_data_df[cols]
	
	x = train
	
	

	
	"""
	print "print y "
	print y
	print "print x:"
	print x
	"""
	#label_1 is 1 if >50k, 0 if <=50k
        return x, y   

    def train_and_select_model(self, training_csv, testing_csv):
        x_train, y_train = self.load_data(training_csv, testing_csv)

        # 2. Select the best model with cross validation.
        # Attention: Write your own hyper-parameter candidates.
	
	
	
        param_set = [
                    	{'kernel': 'linear', 'C': 13, 'degree': 1, 'gamma': 0.0594},
			{'kernel': 'poly', 'C': 28, 'degree': 0, 'gamma': 0.0594},
			{'kernel': 'poly', 'C': 29, 'degree': 0, 'gamma': 0.0594},
                     	{'kernel': 'poly', 'C': 28, 'degree': 1, 'gamma': 0.0594},
                     	{'kernel': 'poly', 'C': 29, 'degree': 1, 'gamma': 0.0594},
			
			
        ]
        pass
	num_params = 5
	#cross-validation, 3-fold
	length = 38841
	it = 12947
	best_score = 0
	best_model = 0
	
	
	###Fold 1#######
	print "fold: 1"
	cv_train = x_train._slice(slice(0, 25894), 0)
	cv_test = x_train._slice(slice(25894, 38841), 0)
	cv_train_y = y_train._slice(slice(0, 25894), 0)
	cv_test_y = y_train._slice(slice(25894, 38841), 0)
	
	tmpscore_test = 0
	tmpscore_train = 0
	train_total = [0, 0, 0, 0, 0]
	test_total = [0, 0, 0, 0, 0]

	#calculate models
	for i in range (0, num_params):
		print "setting SVM for param:%d, kernel:%s, C:%d, degree:%d, gamma:%f..." % (i, param_set[i]["kernel"], param_set[i]["C"], param_set[i]["degree"], param_set[i]["gamma"])
		clf0 = SVC(C=param_set[i]["C"], kernel=param_set[i]["kernel"], degree=param_set[i]["degree"], gamma=param_set[i]["gamma"])
		clf0.fit(cv_train, cv_train_y['label_1'].values)
		print "predicting..."
		#train score
		predict0 = clf0.predict(cv_train)
		tmpscore_train = accuracy_score(cv_train_y['label_1'].values, predict0)
		print "accuracy on train_data is:"
		print tmpscore_train
		train_total[i] = tmpscore_train
	
		#test score
		predict0 = clf0.predict(cv_test)
		tmpscore_test = accuracy_score(cv_test_y['label_1'].values, predict0)
		print "accuracy on test_data is:"
		print tmpscore_test
		test_total[i] = tmpscore_test
		
	
	###Fold 2######
	print "fold: 2"
	cv_train = x_train._slice(slice(12947, 38841), 0)
	cv_test = x_train._slice(slice(0, 12947), 0)
	cv_train_y = y_train._slice(slice(12947, 38841), 0)
	cv_test_y = y_train._slice(slice(0, 12947), 0)

	tmpscore = 0
	#calculate models
	for i in range (0, num_params):
		print "setting SVM for param:%d, kernel:%s, C:%d, degree:%d, gamma:%f..." % (i, param_set[i]["kernel"], param_set[i]["C"], param_set[i]["degree"], param_set[i]["gamma"])
		clf0 = SVC(C=param_set[i]["C"], kernel=param_set[i]["kernel"], degree=param_set[i]["degree"], gamma=param_set[i]["gamma"])
		clf0.fit(cv_train, cv_train_y['label_1'].values)
		print "predicting..."
		#train score
		predict0 = clf0.predict(cv_train)
		tmpscore_train = accuracy_score(cv_train_y['label_1'].values, predict0)
		print "accuracy on train_data is:"
		print tmpscore_train
		train_total[i] = train_total[i] + tmpscore_train	
	
		#test score
		predict0 = clf0.predict(cv_test)
		tmpscore_test = accuracy_score(cv_test_y['label_1'].values, predict0)
		print "accuracy on test_data is:"
		print tmpscore_test
		test_total[i] = test_total[i] + tmpscore_test		


	###Fold 3#######
	print "fold: 3"

	cv_train = x_train._slice(slice(0, 12947), 0).append(x_train._slice(slice(25894, 38841), 0) )
	cv_test = x_train._slice(slice(12947, 25894), 0)
	cv_train_y = y_train._slice(slice(0, 12947), 0).append(y_train._slice(slice(25894, 38841), 0) )
	cv_test_y = y_train._slice(slice(12947, 25894), 0)
	
	tmpscore = 0
	#calculate models
	for i in range (0, num_params):
		print "setting SVM for param:%d, kernel:%s, C:%d, degree:%d, gamma:%f..." % (i, param_set[i]["kernel"], param_set[i]["C"], param_set[i]["degree"], param_set[i]["gamma"])
		clf0 = SVC(C=param_set[i]["C"], kernel=param_set[i]["kernel"], degree=param_set[i]["degree"], gamma=param_set[i]["gamma"])
		clf0.fit(cv_train, cv_train_y['label_1'].values)
		print "predicting..."
		#train score
		predict0 = clf0.predict(cv_train)
		tmpscore_train = accuracy_score(cv_train_y['label_1'].values, predict0)
		print "accuracy on train_data is:"
		print tmpscore_train
		train_total[i] = train_total[i] + tmpscore_train

		#test score
		predict0 = clf0.predict(cv_test)
		tmpscore_test = accuracy_score(cv_test_y['label_1'].values, predict0)
		print "accuracy on test_data is:"
		print tmpscore_test
		test_total[i] = test_total[i] + tmpscore_test
	

	### Average train and test values, then compute model using best test average score
	best_param = -1
	train_avg = [0, 0, 0, 0, 0]
	test_avg = [0, 0, 0, 0, 0]
	for i in range (0, num_params):
		train_avg[i] = train_total[i] / 3.0
		test_avg[i] = test_total[i] / 3.0
		print "Train average for param:%d, kernel:%s, C:%d, degree:%d, gamma:%f..." % (i, param_set[i]["kernel"], param_set[i]["C"], param_set[i]["degree"], param_set[i]["gamma"]) 
		print train_avg[i]
		print "Test average for param:%d, kernel:%s, C:%d, degree:%d, gamma:%f..." % (i, param_set[i]["kernel"], param_set[i]["C"], param_set[i]["degree"], param_set[i]["gamma"]) 
		print test_avg[i]
		##find the highest average test score, compute model based on those params
		if test_avg[i] > best_score:
			best_score = test_avg[i]
			best_param = i

	##end of for loop

	print "highest test average = %f" % best_score
	print "Best model = param:%d, kernel:%s, C:%d, degree:%d, gamma:%f..." % (best_param, param_set[best_param]["kernel"], param_set[best_param]["C"], param_set[best_param]["degree"], param_set[best_param]["gamma"])
	
	##set best model based on highest test avg
	best_model = SVC(C=param_set[best_param]["C"], kernel=param_set[best_param]["kernel"], degree=param_set[best_param]["degree"], gamma=param_set[best_param]["gamma"]) 
	best_model.fit(x_train, y_train['label_1'].values)

	###return statement########
        return best_model, best_score

    def predict(self, test_csv, training_csv, trained_model):
        x_test, _ = self.load_data(test_csv, training_csv)
        predictions = trained_model.predict(x_test)
        return predictions

    def output_results(self, predictions):
        # 3. Upload your Python code, the predictions.txt as well as a report to Collab.
        # Hint: Don't archive the files or change the file names for the automated grading.
        with open('predictions.txt', 'w') as f:
            for pred in predictions:
                if pred == 0:
                    f.write('<=50K\n')
                else:
                    f.write('>50K\n')

if __name__ == '__main__':
    training_csv = "salary.labeled.csv"
    testing_csv = "salary.2Predict.csv"
    clf = SvmIncomeClassifier()
    trained_model, cv_score = clf.train_and_select_model(training_csv, testing_csv)
    print "The best model was scored %.2f" % cv_score
    predictions = clf.predict(testing_csv, training_csv, trained_model)
    clf.output_results(predictions)


