import sys
import numpy as np 
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt


def loadData(numoffeatures):
	File_y=open('042815_y.txt','r')
	File_x=open('042815_x.txt','r')

	File_y_predict=open('042715_y.txt','r')
	File_x_predict=open('042715_x.txt','r')

	data_x=File_x.readlines()
	data_y=File_y.readlines()

	data_x_predict=File_x_predict.readlines()
	data_y_predict=File_y_predict.readlines()

	dataset_x=[]
	dataset_y=[]
	dataset_x_predict=[]
	dataset_y_predict=[]

	linecount_x=0
	linecount_y=0
	linecount_x_predict=0
	linecount_y_predict=0

	linetotal_x=len(data_x)
	linetotal_y=len(data_y)
	linetotal_x_predict=len(data_x_predict)
	linetotal_y_predict=len(data_y_predict)

	#print linetotal_x,linetotal_y

	aimfeatures=numoffeatures

	'''load y, y has two different values -1,1 if y=1, the patient has illness, otherwise, the patient is normal'''

	for line_y in data_y:
		#print line_y
		if linecount_y==0:
			linesplit=line_y.replace('\n','').split('	')
			for i in range(1, len(linesplit)):
				label=linesplit[i].split('_')[0]
				if label=='Norm':
					dataset_y.append(1)
				elif label=='Tumour':
					dataset_y.append(-1)
			#print dataset_y,len(dataset_y)
			#print linesplit
			linecount_y+=1
		else:
			break

	for line_y in data_y_predict:
		#print line_y
		if linecount_y_predict==0:
			linesplit=line_y.replace('\n','').split('	')
			for i in range(1, len(linesplit)):
				label=linesplit[i].split('_')[0]
				if label=='Norm':
					dataset_y_predict.append(1)
				elif label=='Tumour':
					dataset_y_predict.append(-1)
			#print dataset_y_predict,len(dataset_y_predict)
			#print linesplit
			linecount_y_predict+=1
		else:
			break

	'''load x'''
	for line_x in data_x:
		if linecount_x==0:
			#print line_x
			linecount_x+=1
		else:
			#print line_x
			linesplit=line_x.replace('\n','').split("	")
			listnum=[]
			if (len(linesplit)>=aimfeatures):
				length=aimfeatures
			else:
				length=len(linesplit)
			for i in range(1,length):
				value=float(linesplit[i])
				listnum.append(value)
				#print listnum
			dataset_x.append(listnum)
			#print listnum
			#print len(listnum)
			linecount_x+=1

	'''load x'''
	for line_x in data_x_predict:
		if linecount_x_predict==0:
			#print line_x
			linecount_x_predict+=1
		else:
			#print line_x
			linesplit=line_x.replace('\n','').split("	")
			listnum=[]
			if (len(linesplit)>=aimfeatures):
				length=aimfeatures
			else:
				length=len(linesplit)
			for i in range(1,length):
				value=float(linesplit[i])
				listnum.append(value)
				#print listnum
			dataset_x_predict.append(listnum)
			#print listnum
			#print len(listnum)
			linecount_x_predict+=1

	np_y=np.array(dataset_y)
	np_x=np.array(dataset_x)
	np_y_predict=np.array(dataset_y_predict)
	np_x_predict=np.array(dataset_x_predict)
	print np_y.shape,np_x.shape,np_y_predict.shape,np_x_predict.shape

	return np_x,np_y,np_x_predict,np_y_predict



def learnandpredict(np_x,np_y,np_x_predict,np_y_predict,numoffeatures):
	logreg = linear_model.LogisticRegression()
	logreg.fit(np_x,np_y)
	predictlist=logreg.predict(np_x_predict)
	#print np_y_predict
	#print predictlist
	count=0
	for i in range(0,len(predictlist)):
		if predictlist[i]==np_y_predict[i]:
			count=count
		elif predictlist[i]!=np_y_predict[i]:
			count+=1
	accuracy=1-float(count)/float(len(np_y_predict))
	print numoffeatures,accuracy
	print np_y_predict
	print predictlist


	x_min, x_max = np_x[:, 0].min() - .5, np_x[:, 0].max() + .5
	y_min, y_max = np_x[:, 1].min() - .5, np_x[:, 1].max() + .5
	h=0.2
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

	coef=logreg.coef_
	#coef=logreg.densify()
	score=logreg.decision_function(np_x)
	#print len(coef)
	#print coef.shape
	#print coef
	#print score
	# plt.scatter(np_x[:, 0], np_x[:, 1], c=np_y, edgecolors='k', cmap=plt.cm.Paired)
	# plt.xlabel('Sepal length')
	# plt.ylabel('Sepal width')

	# plt.xlim(xx.min(), xx.max())
	# plt.ylim(yy.min(), yy.max())
	# plt.xticks(())
	# plt.yticks(())

	# plt.show()

def main(argv):
	for numoffeatures in range(3,272):
		np_x,np_y,np_x_predict,np_y_predict=loadData(numoffeatures)
		
		learnandpredict(np_x,np_y,np_x_predict,np_y_predict,numoffeatures)

	pass

if __name__ == '__main__':
	main(sys.argv)