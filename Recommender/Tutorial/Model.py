import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.python.estimator.export.export import build_raw_serving_input_receiver_fn
from tensorflow.python.estimator.export.export_output import PredictOutput

INPUT_USER_DATA = "userHistory"
INPUT_MASK = "observedMask"

SIGNATURE_NAME = "serving_default"
LEARNING_RATE = 0.00000001
currentPath = os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
sys.path.append(currentPath+"/MatrixUtils")
import MatrixUtils as mu

def model_fn(features, labels, mode, params):
    
	print(features)
	print(labels)


	#params should contain learning_rate,userCount,itemCount
	# Connect the first hidden layer to input layer
    # (features["x"]) with relu activation
	first_hidden_layer = tf.layers.dense(features[INPUT_USER_DATA], 2000, activation=tf.nn.relu)

    # Connect the second hidden layer to first hidden layer with relu
	second_hidden_layer = tf.layers.dense(first_hidden_layer, 2000, activation=tf.nn.relu)

    # Connect the output layer to second hidden layer (no activation fn)
	output_layer = tf.layers.dense(second_hidden_layer, 13474)

    # Reshape output layer to 1-dim Tensor to return predictions
	predictions = tf.reshape(output_layer, [-1,13474])

	# Provide an estimator spec for `ModeKeys.PREDICT`.
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"scores": predictions},
            export_outputs={SIGNATURE_NAME: PredictOutput({"scores": predictions})})

    # Calculate loss using mean squared error
	#loss = tf.losses.mean_squared_error(labels , predictions)
	
	loss2= tf.reduce_mean(tf.square((labels-predictions))*features[INPUT_USER_DATA] )

	optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
	train_op = optimizer.minimize( loss=loss2, global_step=tf.train.get_global_step())

    # Calculate root mean squared error as additional eval metric
	eval_metric_ops = {
        "rmse": tf.metrics.root_mean_squared_error(
            labels, predictions)
    }#shoulde be modified

    # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
	return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss2,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)

def serving_input_fn(params):
    tensor = tf.placeholder(tf.float32, shape=[1, 13474])# should be modified
    return build_raw_serving_input_receiver_fn({ INPUT_USER_DATA: tensor})()


params = {"learning_rate": LEARNING_RATE}


def train_input_fn(training_dir, params):
    return _input_fn(training_dir, 'movie_train.csv')


def eval_input_fn(training_dir, params):
    return _input_fn(training_dir, 'test.csv')


def _input_fn(training_dir, training_filename):
	training_set = tf.contrib.learn.datasets.base.load_csv_without_header(filename=os.path.join(training_dir, training_filename), target_dtype=np.int, features_dtype=np.float32)
	trainMat = {}
	itemCount = 13474
	print(training_set.data)
	for userId,itemId,rating in training_set.data:
		userId = int(userId)
		itemId = int(itemId)
		mu.set(trainMat,userId,itemId,1.0)

	trainArray=[] #
	for userId in trainMat:
		trainArray.append(mu.input_vector(trainMat[userId],0.0,itemCount))
		
	trainArray = np.array(trainArray)

	return tf.estimator.inputs.numpy_input_fn(
        x={INPUT_USER_DATA: trainArray},
        y= trainArray,
        num_epochs=None,
        shuffle=True)()

