import tensorflow as tf
#abalon predictor

def model_fn(features, label , mode, hyperparameters):
	# Logic to do the following:
	# 1. Configure the model via TensorFlow operations
	# 2. Define the loss function for training/evaluation
	# 3. Define the training operation/optimizer
	# 4. Generate predictions
	# 5. Return predictions/loss/trainop/eval_metric_ops in EstimatorSpec object
	input_layer = tf.feature_column.input_layer(
			features=features, feature_columns=[age,height,weight]
			)

	hidden_layer = tf.layers.dense(
			inputs=input_later,units=10, activation=tf.nn.relu
			)
	second_hidden_layer = tf.layers.dense(
			inputs=hidden_layer, units=20, activation = tf.nn.relu
			)
	output_layer = tf.layers.dense(
			inputs=second_hidden_layer, units=1,activation=None
			)

	# Reshape output layer to 1-dim Tensor to return predictions
	
	predictions = tf.reshape(output_layer, [-1])
	predictions_dict = {"ages": predictions}

