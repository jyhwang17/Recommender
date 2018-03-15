import tensorflow as tf
import numpy as np
import random
import math
import sys
import os
import time
import heapq
import bottleneck
currentPath = os.path.dirname(os.path.realpath(__file__))

sys.path.append(currentPath+"/MatrixUtils")
sys.path.append(currentPath+"/DataUtils")
import MatrixUtils as mu
import DataUtils as du

class Recommender():

	def __init__(self,userCount,itemCount,trainMat,testMat,trainMask,trainMask2):
		
		self.userCount = userCount
		self.itemCount = itemCount

		#log
		self.bestP = 0.0
		self.bestR = 0.0
		self.bestE = 0.0

		#default Parameters
		self.dim = 30
		self.reg_u = 0.005
		self.reg_l = 0.005
		self.keepProb = 1.0
		self.lr = 0.01
		#data		
		self.trainMat = trainMat
		self.testMat = testMat
		self.trainMask = trainMask
		self.trainMask2 = trainMask2
		
		#variables
		self.U = tf.Variable(tf.random_normal([userCount,self.dim],0,0.001,seed=123456789))
		self.V = tf.Variable(tf.random_normal([itemCount,self.dim],0,0.001,seed=123456789))
		self.pos_ub = tf.Variable( tf.random_normal([userCount,1],mean=1.00,stddev=0.00,seed=123456789))
		self.pos_lb = tf.Variable( tf.random_normal([userCount,1],mean=0.50,stddev=0.00,seed=123456789))
		self.neg_ub = tf.Variable( tf.random_normal([userCount,1],mean=0.50,stddev=0.00, seed=123456789))
		self.neg_lb = tf.Variable( tf.random_normal([userCount,1],mean=0.00,stddev=0.00, seed=123456789))
		
		#placeholders
		self.R = tf.placeholder(tf.float32,[None,itemCount])
		self.conf = tf.placeholder(tf.float32,[None,itemCount])
		self.mask = tf.placeholder(tf.float32,[None,itemCount])
		self.mask2= tf.placeholder(tf.float32,[None,itemCount])
		
		self.user_Indices = tf.placeholder(tf.int32,[None])
		self.item_Indices = tf.placeholder(tf.int32,[None])
		self.U_selected = tf.gather(self.U,self.user_Indices)
		self.V_selected = tf.gather(self.V,self.item_Indices)
		
		self.pred = tf.matmul(self.U_selected, tf.transpose(self.V_selected))
		self.pred2= self.pred-(self.mask*10000.0)
		self.line1 = tf.gather(self.pos_ub,self.user_Indices)
		self.line2 = tf.gather(self.pos_lb,self.user_Indices)
		self.line3 = tf.gather(self.neg_ub,self.user_Indices)
		self.line4 = tf.gather(self.neg_lb,self.user_Indices)
		
		self.loss = tf.reduce_sum( tf.square(self.line4-self.pred)*self.reg_l*self.mask2 + tf.log(1+tf.exp(self.pred-self.line3))*self.conf*self.mask2)+ tf.reduce_sum( tf.square(self.pred-self.line1)*self.reg_u*self.mask  + tf.log(1+tf.exp(self.line2-self.pred))*self.conf*self.mask)
		
		self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss, var_list=[self.U,self.V])

	def ready(self):
		
		config = tf.ConfigProto()
		config.gpu_options.allow_growth=True
		self.sess = tf.Session(config=config)
		self.sess.run(tf.global_variables_initializer())
	def evalK3(self,topk,ep):

		precision,recall = 0.0,0.0
		uCnt = len(self.testMat)
		allItems = range(self.itemCount)
		
		for userId in self.testMat:
			result = self.sess.run(self.pred2,feed_dict={self.user_Indices:[userId],self.item_Indices: allItems, self.mask: np.array([mu.input_vector(self.trainMask[userId],0.0,self.itemCount)])})
			result = result.flatten()
			bound = topk
			topIndices = bottleneck.argpartition(-result,bound)[:bound]
			k,hit=0.0,0.0
			for itemId in topIndices:
				k+=1.0
				if mu.is_visited(self.testMat,userId,itemId):
					hit+=1.0
			precision+=(hit/k)
			recall+=hit/float(len(self.testMat[userId]))

		precision/=uCnt
		recall/=uCnt

		print("[ver3]Precision@%d: %lf Recall@%d: %lf"%(topk,precision,topk,recall))

	def evalK2(self,topk,ep):

		precision = 0.0
		recall = 0.0
		uCnt = len(self.testMat)
		allItems = range(self.itemCount)

		for userId in self.testMat:
			result = self.sess.run(self.pred, feed_dict={self.user_Indices:[userId],self.item_Indices: allItems })
			topkList = mu.topk_selection(self.trainMat, userId,topk, result.flatten().tolist(), allItems)
			k,hit=0.0,0.0
			for pref,itemId in topkList:
				if mu.is_visited(self.testMat,userId,itemId): hit+=1.0
			precision+=(hit/len(topkList))
			recall+=hit/float(len(self.testMat[userId]))
		
		precision/=uCnt
		recall/=uCnt
		
		print("[ver2]Precision@%d: %lf Recall@%d: %lf"%(topk,precision,topk,recall))

	def evalK(self,topk,ep):

		precision =0.0
		recall =0.0
		uCnt=len(self.testMat)
		allItems = range(self.itemCount)
		for userId in testMat:
			result = self.sess.run(self.pred, feed_dict={self.user_Indices:[userId], self.item_Indices: allItems })
			result = np.argsort(-result).flatten().tolist()
			k,hit=0.0,0.0
			for itemId in result:
				if k== topk:break
				if mu.is_visited(self.trainMat,userId,itemId): continue
				k+=1.0
				if mu.is_visited(self.testMat,userId,itemId):
					hit+=1.0
			precision+=(hit/k)
			recall+=hit/float(len(self.testMat[userId]))
		
		precision/=uCnt
		recall/=uCnt
		
		if precision > self.bestP:
			self.bestP = precision
			self.bestR = recall
			self.bestE = ep
		
		print("Precision@%d: %lf Recall@%d: %lf"%(topk,precision,topk,recall))
	
	def set_confidence(self,trainConf):
		self.trainConf = trainConf

	def load_latentvector(self):
		pass

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
	
				self.sess.run(self.optimizer, feed_dict={self.R:batchData, self.conf:batchConf, self.mask:batchMask, self.mask2:batchMask2, self.user_Indices:batchUser, self.item_Indices: allItems })
			
			print("Epoch: [%d/%d] "%(epoch,epochCount))			
			#self.evalK(10,epoch)
			start_time2 = time.time()
			self.evalK3(10,epoch)
			print("Total_Time:%lf Evaluation_Time:%lf "%(time.time()-start_time, time.time()- start_time2 ));



if __name__ == "__main__":

	try:
		batchSize = int(sys.argv[1])
		learnRate = float(sys.argv[2])
		dim = int(sys.argv[3])
		weight = float(sys.argv[4])
		reg_u = float(sys.argv[5])
		reg_l = float(sys.argv[6])
		keepProb = float(sys.argv[7])
		DATA_NAME= (sys.argv[8])
	except:
		print("batchSize,learnRate dim conf gam+ gam- keepProb The name of data\n");
		exit()

	userCount,itemCount,trainSet,testSet = du.prep_dataset(DATA_NAME)	
	print("UserCount:%d itemCount:%d Records:%d\n"%(userCount,itemCount,len(trainSet)))	
	trainMat = {}#base 0.0
	trainConf = {}#base 0.1
	trainMask = {}#base0.0 for observed
	trainMask2 = {}#base 1.0 for unseen

	for (userId,itemId,rating) in trainSet:
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

	rec = Recommender(userCount,itemCount,trainMat,testMat,trainMask,trainMask2)
	rec.ready()
	
	#set parameter
	rec.set_confidence(trainConf)
	rec.dim = dim;
	rec.lr = learnRate
	rec.reg_u = reg_u
	rec.reg_l = reg_l
	rec.keepProb = keepProb
	rec.train(batchSize,5)
	print("%d,%f,%d,%f,%f,%f,%f::%d::%f"%(batchSize,rec.lr,rec.dim,weight,rec.reg_u,rec.reg_l,rec.keepProb,rec.bestE,rec.bestP))

