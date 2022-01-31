import cma
from pprint import pprint
import copy
import numpy as np
import pandas as pd 
from math import exp,fabs
from pandas import read_csv
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.svm import SVC

from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
# used for normalization
from sklearn.preprocessing import  MinMaxScaler
from sklearn.preprocessing import StandardScaler

dfData = read_csv("../data/data_0.csv", header=None, sep=',')
dfData=dfData.values
dfLabels = read_csv("../data/labels.csv", header=None)
dfLabels=dfLabels.values.ravel()
Dimension=int(len(dfData[0]))

def testingModel(xtrain):
	
	cost=0.0
	
	#x = np.zeros(Dimension)

	#for i in range (0,Dimension):
	#	x[i] = (2.0 * xtrain[i] - 1.0)/1.0

	
	# a few hard-coded values
	numberOfFolds = 10
	
	# list of classifiers, selected on the basis of our previous paper "
	classifierList = [
		
			#[GradientBoostingClassifier(n_estimators=300), "GradientBoostingClassifier(n_estimators=300)"],
			#[RandomForestClassifier(n_estimators=300), "RandomForestClassifier(n_estimators=300)"],
			[LogisticRegression(solver='lbfgs',), "LogisticRegression"],
			[PassiveAggressiveClassifier(),"PassiveAggressiveClassifier"],
			#[SGDClassifier(), "SGDClassifier"],
			[SVC(kernel='linear'), "SVC(linear)"],
			#[RidgeClassifier(), "RidgeClassifier"],
			#[BaggingClassifier(n_estimators=300), "BaggingClassifier(n_estimators=300)"],
			

			]
	

	X=dfData
	scaler = MinMaxScaler()
	X = scaler.fit_transform(X)
	
	for indexX in range(0, len(X[0])):
		for indexY in range (0, len(X)):
			tempVal=X[indexY,indexX]
			if tempVal<=xtrain[indexX]:
				tempVal=0
			elif tempVal>xtrain[indexX]:
				tempVal=1
			X[indexY,indexX]=tempVal
	
	y=dfLabels

	
	labels=np.max(y)+1
	# prepare folds
	skf = StratifiedKFold(n_splits=numberOfFolds, shuffle=True)
	indexes = [ (training, test) for training, test in skf.split(X, y) ]
	
	# this will be used for the top features
	topFeatures = dict()
	
	# iterate over all classifiers
	classifierIndex = 0
	
	classifierMean=[]
	
	for originalClassifier, classifierName in classifierList :
		
		#print("\nClassifier " + classifierName)
		classifierPerformance = []

		
		# iterate over all folds
		
		indexFold = 0

		

		for train_index, test_index in indexes :
			
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]
			
			# let's normalize, anyway
			# MinMaxScaler StandardScaler Normalizer
			#scaler = MinMaxScaler()
			#X_train = scaler.fit_transform(X_train)
			#X_test = scaler.fit_transform(X_test)
			
			#choose Majority
			maxLabel=-1
			Majority=-1
			for nLabel in range(0,labels):
				sizeClass=len(y_train[y_train==nLabel])
				if sizeClass>maxLabel:
					maxLabel=sizeClass
					Majority=nLabel
			#print("Majority Class"+str(Majority))
			
			# Separate majority and minority classes
			df_majority = X_train[y_train==Majority]
			#print("df_majority "+str(len(df_majority)))
			
			
			df_upsampled=df_majority
			for nLabel in range(0,labels):
				if nLabel !=Majority :		
					df_minorityTemp = X_train[y_train==nLabel]
					#print("df_minority "+str(len(df_minorityTemp)))
						# Upsample minority class
					df_minority_upsampledTemp = resample(df_minorityTemp, 
													 replace=True,     # sample with replacement
													 n_samples=len(df_majority),    # to match majority class
													 random_state=123) # reproducible results
					#print("df_minority_upsampled "+str(len(df_minority_upsampledTemp))+" label "+str(nLabel))
					# Combine majority class with upsampled minority class
					df_upsampled = np.concatenate([df_upsampled, df_minority_upsampledTemp])
					
					#print('df_upsampled '+str(len(df_upsampled)))
					#print('df_upsampled2D  '+str(len(df_upsampled[0])))
			
			#df_upsampled=np.concatenate([df_majority, df_upsampled])
			X_train=df_upsampled
			
			arrMajority=[]
			arrMajority = [Majority for i in range(len(df_majority))] 
			#print('arrMajority '+str(len(arrMajority)))
			
			arrMinority=arrMajority
			for nLabel in range(0,labels):
				if nLabel !=Majority :
					arr2=[]
					arr2 = [nLabel for i in range(len(df_majority))] 
					#print('arrMinority '+str(len(arr2))+" label "+str(nLabel))
					arrMinority=np.concatenate([arrMinority, arr2])
			
			y_train=arrMinority
			#print('y_train '+str(len(y_train)))
			
			
			classifier = copy.deepcopy(originalClassifier)
			classifier.fit(X_train, y_train)
			scoreTraining = classifier.score(X_train, y_train)
			scoreTest = classifier.score(X_test, y_test)
			
			
			
			#print("\ttraining: %.4f, test: %.4f" % (scoreTraining, scoreTest))
			classifierPerformance.append( scoreTest )


		classifierIndex+=1
		classifierMean.append(np.mean(classifierPerformance))
		#line ="%s \t %.4f \t %.4f \n" % (classifierName, np.mean(classifierPerformance), np.std(classifierPerformance))
	cost=1-(np.mean(classifierMean))
		
	return cost



if __name__ == "__main__":
	es = cma.CMAEvolutionStrategy(Dimension * [0.5], 0.01, {'bounds': [0, 1]})
	print(Dimension)
	
	
	while not es.stop():
		X = es.ask()
		es.tell(X, [testingModel(x) for x in X])
		es.logger.add()
		es.disp() 
	es.result_pretty()
	xtrain=es.result[0]
	print(xtrain)
	
	X=dfData
	scaler = MinMaxScaler()
	X = scaler.fit_transform(X)
	
	for indexX in range(0, len(X[0])):
		for indexY in range (0, len(X)):
			tempVal=X[indexY,indexX]
			if tempVal<=xtrain[indexX]:
				tempVal=0
			elif tempVal>xtrain[indexX]:
				tempVal=1
			X[indexY,indexX]=tempVal

	pd.DataFrame(X).to_csv("../data/Xreduced.csv", header=None, index =None)
	pd.DataFrame(xtrain).to_csv("../data/xtrain.csv", header=None, index =None)
