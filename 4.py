from __future__ import print_function
from abc import ABCMeta, abstractmethod
from pyspark import SparkContext, RDD, since
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.util import JavaLoader, JavaSaveable
from pyspark.mllib.tree import DecisionTreeModel, DecisionTree, RandomForestModel, RandomForest
from pyspark.mllib.classification import LogisticRegressionModel,LogisticRegressionWithSGD
# $example on$
from pyspark.mllib.util import MLUtils
from operator import add
from pyspark.mllib.linalg import SparseVector
import random
import copy
class BaggingClassifier():
	def __init__(self,
				 n_estimators=3, 
				 sample_probability=0.9, 
				 features_num=100, 
				 oob_score=False, 
				 warm_start=False):

		self.n_estimators = n_estimators
		self.sample_probability = sample_probability
		self.features_num = features_num
		#self.oob_score = oob_score
		#self.warm_start = warm_start
		self.sampledFeatureIndex=[[] for i in range(n_estimators)]
		self.precision=None
		self.recall=None
		self.F1score=None

	def __randomSelectFeature(self,line,iteration_num):#No sampling on sample, iteration_num corresponds to the right sampledFeatureIndex 
		line_label=line.label
		features=line.features
		#is it neccessary?
		if not self.sampledFeatureIndex[iteration_num]:
			self.sampledFeatureIndex[iteration_num]=copy.deepcopy(self.__reserviorSampling(self.features_num,len(line.features)))
		new_features={}
		count=0
		for i in self.sampledFeatureIndex[iteration_num]:
			if features[i]!=0:
				new_features[count]=features[i]
			count+=1
		if bool(new_features)==False:
			new_features[len(self.sampledFeatureIndex[iteration_num])-1]=0
		return LabeledPoint(line_label,SparseVector(len(self.sampledFeatureIndex[iteration_num]),new_features))

	def __reserviorSampling(self,k,n):# select k from n, if k is larger than n, all is selected
		sample=[]
		for i in range(n):
			if i<k:
				sample.append(i)
			elif i>=k and random.random()<k/float(i+1):
				replace_index=random.randint(0,len(sample)-1)
				sample[replace_index]=i
		return sample

	def __randomSelect(self,myRDD,n):
		result=None
		while True:
			result=myRDD.map(lambda x:self.__randomSelectFeature(x,n)).sample(False,self.sample_probability)
			if result.isEmpty():
				continue
			else:
				break
		return result

	def __ramdomSelect_predict(self,myRDD,n):
		result=None
		while True:
			result=myRDD.map(lambda x:self.__randomSelectFeature(x,n)).map(lambda x:x.features)
			if result.isEmpty():
				continue
			else:
				break
		return result	

	def fit(self, data, classifier, argument):
		model = []
		sampeled_list=data.take(1)
		total_features=len(sampeled_list[0].features)
		#It will evaluate the sample to be picked up
		for i in range(self.n_estimators):
			self.sampledFeatureIndex[i]=self.__reserviorSampling(self.features_num,total_features)
		for i in range(self.n_estimators):
			cdata=self.__randomSelect(data,i)
			argument['data']=cdata# The argument should have the same type
			model.append(classifier.train(**argument))# **argument is the named variables
		return model

	#unpack nested tuple to a list
	def __unpack(self,x):
		result_list=[]
		self.__unpack_helper(x,result_list)
		return	result_list
	
	def __unpack_helper(self,tup,result_list):
		if (type(tup[0]) is tuple)==False:
			result_list.insert(0,tup[1])
			result_list.insert(0,tup[0])
			return
		else:
			result_list.insert(0,tup[1])
			self.__unpack_helper(tup[0],result_list)
		return

	def __mostFrequent(self,x):
		mydict={}
		frequent_key=None
		value=None
		for i in x:
			if i in mydict:
				mydict[i]+=1
			else:
				mydict[i]=1
		for key in mydict:
			if frequent_key==None:
				frequent_key=key
				value=mydict[key]
			else:
				if mydict[key]>value:
					frequent_key=key
					value=mydict[key]
		return frequent_key

	def predict(self,data,models):
		rdd_list=[]
		count=0# count corresponds to the right sampledIndex
		for model in models:
			cdata_feature=self.__ramdomSelect_predict(data,count)#cdara is the RDD selected by the feature
			prediction_result=model.predict(cdata_feature).zipWithUniqueId().map(lambda x:[x[1],x[0]])
			rdd_list.append(prediction_result)
			count+=1

		for i in range(len(rdd_list)-1):
			if i==0:
				joined_result=rdd_list[0].join(rdd_list[1]).join(rdd_list[2])
			else:
				joined_result=joined_result.join(rdd_list[i+1])
		joined_result=joined_result.map(lambda x:self.__unpack(x[1])).map(lambda x:self.__mostFrequent(x))
		predictionAndLabels=joined_result.zipWithUniqueId().\
							map(lambda x: (x[1],float(x[0])) ).\
							join( data.map(lambda x:float(x.label)).\
							zipWithUniqueId().map( lambda x: (x[1],x[0])) ).\
							map(lambda x: x[1])
		#predictionAndLabels.foreach(print)
		metrics = MulticlassMetrics(predictionAndLabels)
		self.precision=metrics.precision()
		self.recall=metrics.recall()
		self.F1score=metrics.fMeasure()
		return joined_result

if __name__ == "__main__":
	sc = SparkContext(appName = 'testML')
	data = MLUtils.loadLibSVMFile(sc, 'test_original.txt')
	baggingClass = BaggingClassifier(n_estimators=6,sample_probability=0.9)
	myclassifier=LogisticRegressionWithSGD()
	modelC = baggingClass.fit(data,myclassifier,{'iterations':int(1)})
	#print(baggingClass.sampledFeatureIndex)
	baggingClass.predict(data,modelC)
	mylist=[]