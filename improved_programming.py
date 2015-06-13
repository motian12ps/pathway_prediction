import sys
import numpy as np 
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt
from operator import itemgetter, attrgetter

class Pathway(object):
	"""docstring for pathway"""
	'''this is a dictionary to store all of pathways'''
	'''key is the name '''

	indexList=[]	#ordered sequence of pathway name, pathway index can be obtained by Pathway.indexList.index(pathwayname)
	pathcounter=0
	ID=None
	name=None
	coefficient={}
	def __init__(self, name):
		super(Pathway, self).__init__()
		self.ID = Pathway.pathcounter
		Pathway.pathcounter+=1
		#linesplit = line.split("	")
		self.name = name
		Pathway.indexList.append(self.name)
		Pathway.coefficient[self.ID]=0



class Sample(object):
	"""docstring for ClassName"""
	indexList=[]
	#coefDict={}
	ID=None
	samplecount=0
	sortList={}
	xvalueDict={}
	def __init__(self, line):
		super(Sample, self).__init__()
		linesplit=line.split("_")
		self.ID=linesplit[1]
		Sample.indexList.append(linesplit[1])
		Sample.xvalueDict[Sample.samplecount]=[]
		Sample.sortList[Sample.samplecount]=[]
		Sample.samplecount+=1
	def clear(self):
		Sample.indexList=[]
		Sample.xvalueDict={}
		Sample.sortList=[]
		Sample.samplecount=0
		

class Parameter(object):
	"""docstring for Parameter"""
	K=100 #the number of aimed features
	ordertype=None #the order of pathway values by p-value,q-value, difference
	typeindex=1
	isreverse=False
	orderList=[]
	def __init__(self):
		super(Parameter, self).__init__()

	def clear(self):
		Parameter.K=100
		Parameter.ordertype=None
		Parameter.typeindex=1
		Parameter.isreverse=False
		Parameter.orderList=[]

class TestSample(object):
	"""docstring for ClassName"""
	indexList=[]
	#coefDict={}
	ID=None
	samplecount=0
	sortList={}
	xvalueDict={}
	def __init__(self, line):
		super(TestSample, self).__init__()
		linesplit=line.split("_")
		self.ID=linesplit[1]
		TestSample.indexList.append(linesplit[1])
		TestSample.xvalueDict[TestSample.samplecount]=[]
		TestSample.sortList[TestSample.samplecount]=[]
		TestSample.samplecount+=1
	def clear(self):
		TestSample.indexList=[]
		TestSample.xvalueDict={}
		TestSample.sortList={}
		TestSample.samplecount=0


class TestPathway(object):
	"""docstring for pathway"""
	'''this is a dictionary to store all of pathways'''
	'''key is the name '''
	indexList=[]
	pathcounter=0
	ID=None
	name=None
	coefficient={}
	def __init__(self, name):
		super(TestPathway, self).__init__()
		self.ID = TestPathway.pathcounter
		TestPathway.pathcounter+=1
		#linesplit = line.split("	")
		self.name = name
		TestPathway.indexList.append(self.name)
		TestPathway.coefficient[self.ID]=0

	def clear(self):
		TestPathway.pathcounter=0
		TestPathway.indexList=[]
		TestPathway.coefficient={}



def loadData(argv):
	if len(argv)!=6:
		FileTraining=open('training_dataset.txt','r')
		#FileTesting=open('testing_dataset.txt','r')
		Parameter.ordertype='q-value'
		Parameter.isreverse=True
	else:
		FileTraining=open(argv[1],'r')
		FileTesting=open(argv[2],'r')	
		Parameter.K=argv[3]
		Parameter.ordertype=argv[4]
		Parameter.isreverse=argv[5]

	data_training=FileTraining.readlines()
	
	
	data_training_y=[]
	data_training_x=[]
	


	linecount_training=0
	
	linetotal_training=len(data_training)




	for line in data_training:
		if linecount_training==0:
			#print line
			linesplit=line.split("	")
			for i in range(0,len(linesplit)):
				if "Norm" in linesplit[i]:
					data_training_y.append(1)
					Sample(linesplit[i])
				elif "Tumour" in linesplit[i]:
					data_training_y.append(-1)
					Sample(linesplit[i])
				elif Parameter.ordertype in linesplit[i]:
					Parameter.typeindex=i 
			linecount_training+=1
		else:
			linesplit=line.split("	")
			pathwayname=linesplit[0]
			Pathway(pathwayname)
			indexpathway=Pathway.indexList.index(pathwayname)
			sortlist=[]
			ordervalue=float(linesplit[Parameter.typeindex])
			for i in range(2,2+len(Sample.indexList)):
				sortist=Sample.sortList[i-2]	
				pair=(float(linesplit[i]),ordervalue,indexpathway)
				sortist.append(pair)
						
			#print tmp,len(tmp)
			linecount_training+=1
			#print float(linesplit[Parameter.typeindex])
	#print Sample.sortList[1],len(Sample.sortList[1])

	flag=True
	'''sort by Parameter.ordertype'''
	for sampleIndex,List in Sample.sortList.iteritems():
		#print sampleIndex
		list1=[]
		List=sorted(List,key=itemgetter(1),reverse=Parameter.isreverse)
		readlength=0
		if Parameter.K<len(List):
			readlength=Parameter.K
		else:
			readlength=len(List)
		
		for i in range(0,readlength):
			if flag==True:
				Parameter.orderList.append(List[i][2])				
			list1.append(List[i][0])
		data_training_x.append(list1)

		Sample.xvalueDict[sampleIndex]=list1
		flag=False
	#print List
	#print Sample.xvalueDict[0]
	#print Parameter.orderList

	np_training_x=np.array(data_training_x)
	np_training_y=np.array(data_training_y)


	return np_training_x,np_training_y

def train(np_training_x,np_training_y):
	logreg = linear_model.LogisticRegression()
	logreg.fit(np_training_x,np_training_y)

	
	x_min, x_max = np_training_x[:, 0].min() - .5, np_training_x[:, 0].max() + .5
	y_min, y_max = np_training_x[:, 1].min() - .5, np_training_x[:, 1].max() + .5
	h=0.2
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))


	plt.scatter(np_training_x[:, 0], np_training_x[:, 1], c=np_training_y, edgecolors='k', cmap=plt.cm.Paired)
	plt.xlabel('Sepal length')
	plt.ylabel('Sepal width')

	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	plt.xticks(())
	plt.yticks(())
	#plt.show()

	return logreg


	# print data_training_y,len(data_training_y)
	# print Sample.indexList,len(Sample.indexList)
	# print Sample.coefDict,len(Sample.coefDict)
	# print Parameter.typeindex
	# print Pathway.indexList,len(Pathway.indexList)



def loadTestData(argv):
	if len(argv)!=6:
		FileTesting=open('testing_dataset1.txt','r')
	else:
		FileTesting=open(argv[2],'r')

	data_testing=FileTesting.readlines()
	data_testing_y=[]
	data_testing_x=[]

	linecount_testing=0
	linetotal_testing=len(data_testing)

	for line in data_testing:
		#print line
		if linecount_testing==0:
			#print line
			linesplit=line.split("	")
			for i in range(0,len(linesplit)):
				if "Norm" in linesplit[i]:
					data_testing_y.append(1)
					TestSample(linesplit[i])
				elif "Tumour" in linesplit[i]:
					data_testing_y.append(-1)
					TestSample(linesplit[i])
				# elif Parameter.ordertype in linesplit[i]:
				# 	Parameter.typeindex=i 
			linecount_testing+=1
		else:
			linesplit=line.split("	")
			pathwayname=linesplit[0]
			pathwayindex=Pathway.indexList.index(pathwayname)

			#print pathwayname,pathwayindex
			#TestPathway(linesplit[0])
			#sortlist=[]
			#ordervalue=float(linesplit[Parameter.typeindex])

			for i in range(2,2+len(TestSample.indexList)):
				pathwayvalue=float(linesplit[i])
				sortlist=TestSample.sortList[i-2]
				tupleentry=(pathwayvalue,pathwayindex)
				sortlist.append(tupleentry)
				pass
			# for i in range(2,2+len(TestSample.indexList)):
			# 	coefList=TestSample.sortList[i-2]	
			# 	pair=(float(linesplit[i]),ordervalue)
			# 	coefList.append(pair)
						
			#print tmp,len(tmp)
			linecount_testing+=1
	#print TestSample.sortList[0][0]
	#print len(TestSample.sortList[0])
	#print TestSample.indexList
	'''filter the pathways we want'''
	for i1 in range(0,len(TestSample.indexList)):
		xlist=[]
		for i2 in range(0,len(Parameter.orderList)):
			target_index=Parameter.orderList[i2]
			#print target_index
			for i3 in range(0,len(TestSample.sortList[i1])):
				tupleentry=TestSample.sortList[i1][i3]
				currentvalue=tupleentry[0]
				currentindex=tupleentry[1]
				if currentindex==target_index:
					xlist.append(currentvalue)
				else:
					continue
		#print xlist,len(xlist)
		data_testing_x.append(xlist)
	
	np_testing_x=np.array(data_testing_x)
	np_testing_y=np.array(data_testing_y)

	#print np_testing_x,np_testing_x.shape
	#print np_testing_y,np_testing_y.shape
	return np_testing_x,np_testing_y

'''np_testing_y is optional'''
def predict(np_testing_x,logreg,np_testing_y=None):
	predictlist=logreg.predict(np_testing_x)

	count=0
	if np_testing_y!=None:
		for i in range(0,len(predictlist)):
			if predictlist[i]==np_testing_y[i]:
				count=count
			elif predictlist[i]!=np_testing_y[i]:
				count+=1
		accuracy=1-float(count)/float(len(np_testing_y))
	#print Parameter.K,accuracy
	print predictlist, len(predictlist)
	count1=0
	for i in range(0,len(predictlist)):
		if predictlist[i]==1:
			count1+=1
	print count1/float(len(predictlist))
	# print np_testing_y

def clear():
	Pathway.indexList=[]
	Pathway.pathcounter=0
	Pathway.coefficient={}

	Sample.indexList=[]
	Sample.xvalueDict={}
	Sample.sortList={}
	Sample.samplecount=0

	Parameter.K=100
	Parameter.ordertype=None
	Parameter.typeindex=1
	Parameter.isreverse=False
	Parameter.orderList=[]

	TestSample.indexList=[]
	TestSample.xvalueDict={}
	TestSample.sortList={}
	TestSample.samplecount=0

	TestPathway.pathcounter=0
	TestPathway.indexList=[]
	TestPathway.coefficient={}



def main(argv):
	#path=Pathway("1234")
	for k in range(7,8):
		Parameter.K=k
		np_training_x,np_training_y=loadData(argv)
		logreg=train(np_training_x,np_training_y)
		np_testing_x,np_testing_y=loadTestData(argv)
		#predict(np_testing_x,logreg,np_testing_y)
		#print np_testing_x
		predict(np_testing_x,logreg)
		clear()
	# np_training_x,np_training_y=loadData(argv)
	# logreg=train(np_training_x,np_training_y)
	# np_testing_x,np_testing_y=loadTestData(argv)
	# predict(np_testing_x,logreg,np_testing_y)
	
if __name__ == '__main__':
	main(sys.argv)
		
