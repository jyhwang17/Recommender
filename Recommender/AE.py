import tensorflow as tf
import numpy as np
import random
import math
import sys
import os
import time

currentPath = os.path.dirname(os.path.realpath(__file__))

sys.path.append(currentPath+"/MatrixUtils")
sys.path.append(currentPath+"/DataUtils")
import MatrixUtils as mu
import DataUtils as du

class Recommender():

	def __init__(self,userCount,itemCount,trainMat,testMat,trainConf,trainMask,trainMask2):

		self.userCount = userCount
		self.itemCount = itemCount
		#log
		self.bestP=0.0
		self.bestR=0.0
		self.bestE=0.0

		#default Parameters
		self.lr = 0.0005
		self.dim1 = 2000
		self.dim2 = 800
		self.dim3 = 400
		self.reg_u = 0.1
		self.reg_l = 0.1
		self.keepProb = 0.2

		#data		
		self.trainMat = trainMat
		self.testMat = testMat
		self.trainConf= trainConf
		self.trainMask = trainMask
		self.trainMask2 = trainMask2
		
		#Xavier initialization
		scale = math.sqrt(6.0/(self.itemCount + self.dim1))
		scale2= math.sqrt(6.0/(self.dim1 + self.dim2))
		scale3= math.sqrt(6.0/(self.dim2 + self.dim3))

		#placeholders for input data
		self.input_data = tf.placeholder(tf.float32, [None, self.itemCount])		
		self.input_conf = tf.placeholder(tf.float32, [None, self.itemCount])
		self.input_mask = tf.placeholder(tf.float32, [None, self.itemCount])
		self.input_mask2= tf.placeholder(tf.float32, [None, self.itemCount])
				
		#drop-out keep prob
		self.keep_prob = tf.placeholder(tf.float32)
		self.keep_prob2= tf.placeholder(tf.float32)

		#Auto-Encoders
		self.weights = { 'encode_1': tf.Variable(tf.random_uniform([self.itemCount,self.dim1],-scale,scale,seed=123456789)),
						 'encode_2': tf.Variable(tf.random_uniform([self.dim1,self.dim2],-scale2,scale2,seed=123456789)),
						 'encode_3': tf.Variable(tf.random_uniform([self.dim2,self.dim3],-scale3,scale3,seed=123456789)),
						 'decode_3': tf.Variable(tf.random_uniform([self.dim3,self.dim2],-scale3,scale3,seed=123456789)),
						 'decode_2': tf.Variable(tf.random_uniform([self.dim2,self.dim1],-scale2,scale2,seed=123456789)),
						 'decode_1': tf.Variable(tf.random_uniform([self.dim1,self.itemCount],-scale,scale,seed=123456789))}
		
		self.biases = { 'encode_1': tf.Variable(tf.random_uniform([self.dim1],-scale,scale,seed=123456789)),
						'encode_2': tf.Variable(tf.random_uniform([self.dim2],-scale2,scale2,seed=123456789)),
						'encode_3': tf.Variable(tf.random_uniform([self.dim3],-scale3,scale3,seed=123456789)),
						'decode_3': tf.Variable(tf.random_uniform([self.dim2],-scale3,scale3,seed=123456789)),
						'decode_2': tf.Variable(tf.random_uniform([self.dim1],-scale2,scale2,seed=123456789)),
						'decode_1': tf.Variable(tf.random_uniform([self.itemCount],-scale,scale,seed=123456789))}
		self.layers=[]
		self.layers.append(self.input_data)#0
		self.layers.append(tf.nn.dropout(tf.nn.elu(tf.matmul(self.layers[0], self.weights['encode_1'])+ self.biases['encode_1']), self.keep_prob) ) #1
		self.layers.append(tf.nn.dropout(tf.nn.elu(tf.matmul(self.layers[1], self.weights['encode_2'])+ self.biases['encode_2']),self.keep_prob)   )#2
		self.layers.append(tf.nn.dropout(tf.nn.elu(tf.matmul(self.layers[2], self.weights['encode_3'])+ self.biases['encode_3']),self.keep_prob2)  )#3	
		self.layers.append(tf.nn.dropout(tf.nn.elu(tf.matmul(self.layers[3], self.weights['decode_3'])+ self.biases['decode_3']),self.keep_prob2)  )#4
		self.layers.append(tf.nn.dropout(tf.nn.elu(tf.matmul(self.layers[4], self.weights['decode_2'])+ self.biases['decode_2']),self.keep_prob)   )#5
		self.layers.append(tf.nn.dropout(tf.nn.elu(tf.matmul(self.layers[5], self.weights['decode_1'])+ self.biases['decode_1']),self.keep_prob)   )#6
		self.prediction = self.layers[6]
		self.prediction2= self.prediction*self.input_mask
	
		#		
		self.loss = tf.reduce_mean(tf.reduce_sum( tf.square(0-self.prediction)*self.reg_l*self.input_mask2 +tf.log(1+tf.exp(self.prediction-0.5))*self.input_conf*self.input_mask2 )+
								   tf.reduce_sum( tf.square(1-self.prediction)*self.reg_u*self.input_mask  +tf.log(1+tf.exp(0.5-self.prediction))*self.input_conf*self.input_mask  ) )
		self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

	def ready(self):

		config = tf.ConfigProto()
		config.gpu_options.allow_growth=True
		self.sess = tf.Session(config=config)
		self.sess.run(tf.global_variables_initializer())
	
	def evalK(self,topk,ep):

		precision=0.0
		recall=0.0
		uCnt=len(testMat)
		for userId in testMat:
			Indata = np.array(mu.input_vector(self.trainMat[userId],0.0,self.itemCount)).reshape(1,self.itemCount)
			Inmask = np.array(mu.input_vector(self.trainMask[userId],0.0,self.itemCount)).reshape(1,self.itemCount)
			result = self.sess.run(self.prediction,feed_dict={self.input_data: Indata, self.input_mask: Inmask, self.keep_prob:1.0,self.keep_prob2:1.0 })
			result = np.argsort(-result).flatten().tolist()
			k,hit=0.0,0.0
			for itemId in result:
				if k==topk:break
				if mu.is_visited(self.trainMat,userId,itemId): continue
				k+=1.0
				if mu.is_visited(self.testMat,userId,itemId):
					hit+=1.0
			precision+=(hit/k)
			recall+=hit/float(len(testMat[userId]))
		
		precision/=uCnt
		recall/=uCnt
		if(precision > self.bestP):
			self.bestP = precision
			self.bestR = recall
			self.bestE = ep
		
		#print("Precision@%d: %lf Recall@%d: %lf"%(topk,precision,topk,recall))		
		

	def train(self,batchSize,epochCount):

		allItems = range(self.itemCount)
		userIdList = list(self.trainMat.keys())
		for epoch in range(1,epochCount+1):
			random.shuffle(userIdList)
			start_time=time.time()
			for batchId in range( int(len(userIdList)/batchSize)):
				start = batchId * batchSize
				end = start + batchSize		
				batchUser = []				
				batchData = []
				batchConf = []
				batchMask = []
				batchMask2 = []
				
				for i in range(start,end):
					userId = userIdList[i]
					batchUser.append(userId)
					batchData.append((mu.input_vector(self.trainMat[userId],0.0,self.itemCount )))
					batchConf.append((mu.input_vector(self.trainConf[userId],0.1,self.itemCount)))
					batchMask.append((mu.input_vector(self.trainMask[userId],0.0,self.itemCount)))
					batchMask2.append((mu.input_vector(self.trainMask2[userId],1.0,self.itemCount)))
				
				batchData = np.array(batchData)
				batchConf = np.array(batchConf)
				batchMask = np.array(batchMask)
				batchMask2= np.array(batchMask2)
	
				self.sess.run(self.optimizer, feed_dict={self.input_data:batchData, self.input_conf:batchConf, self.input_mask:batchMask, self.input_mask2:batchMask2,self.keep_prob:1.0,self.keep_prob2:self.keepProb})
			#print("\nEpoch: [%d/%d]"%(epoch,epochCount))			
			self.evalK(10,epoch)
			#print("Time: %lf"%(time.time()-start_time))



if __name__ == "__main__":
	
	try:
		batchSize = int(sys.argv[1])
		learnRate = float(sys.argv[2])
		dim = int(sys.argv[3])
		weight = float(sys.argv[4])
		reg_u = float(sys.argv[5])
		reg_l = float(sys.argv[6])
		keepProb = float(sys.argv[7])
		DATA_NAME = (sys.argv[8])
	except:
		print("batchSize,learnRate dim conf gam+ gam- keepProb The name of data\n")
		exit()


	#print("batchSize,learnRate,dim,conf,gam+,gam-,keepProb::epoch::bestPrecision");
	userCount,itemCount,trainSet,testSet = du.prep_dataset(DATA_NAME)	
	
	trainMat = {}#base 0.0
	trainConf = {}#base 0.1
	trainMask = {}#base0.0 for observed
	trainMask2 = {}#base 1.0 for unseen

	for	(userId,itemId,rating) in trainSet:
		userId = int(userId)
		itemId = int(itemId)
		rating = float(rating)
		mu.set(trainMat,userId,itemId,1.0)
		mu.set(trainConf,userId,itemId,weight)
		mu.set(trainMask,userId,itemId,1.0)
		mu.set(trainMask2,userId,itemId,0.0)		
		

	testMat={}

	for (userId,itemId,rating) in testSet:
		userId = int(userId)
		itemId = int(itemId)
		rating = float(rating)
		mu.set(testMat,userId,itemId,1.0)

	rec = Recommender(userCount,itemCount,trainMat,testMat,trainConf,trainMask,trainMask2)
	rec.ready()
	#set parameter
	rec.dim1 =dim
	rec.dim2 =(4*dim/10)
	rec.dim3 = (2*dim/10)
	rec.lr = learnRate
	rec.reg_u = reg_u
	rec.reg_l = reg_l
	rec.keepProb = keepProb
	
	rec.train(batchSize,5)
	#batchSize,learnRate,dim,conf,gam+,gam-,keepProb::epoch::bestPrecision
	print("%d,%f,%d,%f,%f,%f,%f::%d::%f"%(batchSize,rec.lr,rec.dim1,weight,rec.reg_u,rec.reg_l,rec.keepProb,rec.bestE,rec.bestP) );
	
	
