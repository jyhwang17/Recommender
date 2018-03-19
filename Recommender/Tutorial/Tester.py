
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
#import tempfile


# IMPORT URLIB
from six.moves import urllib

import numpy as np
import tensorflow as tf

#Sckit learn
import sklearn as skl
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer


import Model
currentPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentPath+"/MatrixUtils")
import MatrixUtils as mu


FLAG = None
tf.logging.set_verbosity(tf.logging.INFO)

if __name__ == "__main__":
	

	#training_set = tf.contrib.learn.datasets.base.load_csv_without_header(filename='data/train2.csv',target_dtype=np.int, features_dtype=np.float64)
	#test_set = tf.contrib.learn.datasets.base.load_csv_without_header(filename='data/test.csv',target_dtype=np.int, features_dtype=np.float64)
	#predict_set= tf.contrib.learn.datasets.base.load_csv_without_header(filename='data/predict.csv',target_dtype=np.int, features_dtype=np.float64)
	#print(training_set.target)
	#print(training_set,test_set, predict_set)
	
	model_params = {"learning_rate": 0.001}

	# Instantiate Estimator
	nn = tf.estimator.Estimator(model_fn=Model.model_fn,params=model_params)
	
	# Train neural Network for regression
	nn.train(input_fn=lambda: Model.train_input_fn('data/',''), steps=5000)
	
	# validation
	#nn.evaluate(input_fn=lambda: Model.eval_input_fn('data/',''),steps=1000)
	#print out predictions
	#It supposed to be written...

	'''
	predictions = nn.predict(input_fn=lambda: Model.serving_input_fn(''))

	for i, p in enumerate(predictions):
		print("Prediction %s: %s " %s (i+1, p["ages"]))
	'''

