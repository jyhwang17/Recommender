#-*-coding: utf-8 -*-
import numpy as np
import os

def prep_dataset(filename):
	currentPath=os.path.dirname(os.path.realpath(__file__))

	userDict={}
	itemDict={}
	userCount=0
	itemCount=0
	trainCoo = []
	testCoo = []
	
	for line in open(currentPath+"/DataSet/"+filename+"_Train"):
		userId,itemId,val = line.strip().split(" ")
		trainCoo.append([int(userId),int(itemId),float(val)])
		if userId not in userDict:
			userDict[userId]=1
			userCount+=1
		if itemId not in itemDict:
			itemDict[itemId]=1
			itemCount+=1

	for line in open(currentPath+"/DataSet/"+filename+"_Test"):
		userId,itemId,val = line.strip().split(" ")
		testCoo.append([int(userId),int(itemId),float(val)])
		if userId not in userDict:
			userDict[userId]=1
			userCount+=1
		if itemId not in itemDict:
			itemDict[itemId]=1
			itemCount+=1


	return userCount,itemCount,np.array(trainCoo),np.array(testCoo)

