import heapq

def set(mat,userId,itemId,val):

	if userId in mat:
		mat[userId][itemId] =val
	else:
		mat[userId] = {itemId:val}

def add(mat,userId,itemId,val):

	if userId in mat and itemId in mat[userId]:
		mat[userId][itemId] += val
	elif userId in mat:
		mat[userId][itemId] = val
	else:
		mat[userId] = {itemId:val}
		
def is_visited(mat,userId,itemId):

	if userId in mat and itemId in mat[userId]:
		return True
	else:
		return False

def input_vector(baseDict,basevalue,dimension):
	ret = [basevalue]*dimension
	for i in baseDict.keys():
		ret[i] = baseDict[i]
	return ret;

def topk_selection(mat,userId,topk,pref,items):
	candidates = zip(pref,items)
	heap = []
	for score,itemId in candidates:
		if is_visited(mat,userId,itemId): continue
										
		if len(heap) < topk or score > heap[0][0]:
			if len(heap)==topk: heapq.heappop(heap)
			heapq.heappush(heap,(score,itemId))
	return heap

