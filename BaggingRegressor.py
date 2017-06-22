from __future__ import print_function
from abc import ABCMeta, abstractmethod
from pyspark import SparkContext, RDD, since
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.util import JavaLoader, JavaSaveable
# $example on$
from pyspark.mllib.util import MLUtils
from operator import add
from pyspark.mllib.linalg import SparseVector
import random
import copy
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel

class  BaggingRegressor(object):
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
		self.MSE=None

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
			elif i>=k and random.random()<k/float(i+1):# The porbability of being chosen is k/i (i is from 1, so add 1 in coding)
				replace_index=random.randint(0,len(sample)-1)#the probability is always 1/k
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
			model.append(classifier.trainRegressor(**argument))# **argument is the named variables
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

	def __average(self,line):
		mysum=float(0)
		count=0
		for x in line:
			mysum+=x
			count+=1
		return mysum/count

	def predict(self,data,models):
		rdd_list=[]
		count=0# count corresponds to the right sampledIndex
		for model in models:
			cdata_feature=self.__ramdomSelect_predict(data,count)#cdara is the RDD selected by the feature
			prediction_result=model.predict(cdata_feature)
			#prediction_result.foreach(print)
			rdd_list.append(prediction_result)
			count+=1
		for i in range(len(rdd_list)-1):
			if i==0:
				joined_result=rdd_list[0].zip(rdd_list[1])
			else:
				joined_result=joined_result.zip(rdd_list[i+1])
		joined_result=joined_result.map(lambda x:self.__unpack(x)).map(lambda x:self.__average(x))
		labelsAndPredictions = data.map(lambda lp: lp.label).zip(joined_result)
		self.MSE=labelsAndPredictions.map(lambda lp: (lp[0] - lp[1])*(lp[0]-lp[1])).sum()/float(data.count())
		print('Mean Squared Error = ' + str(self.MSE))
		return joined_result

if __name__=="__main__":
	sc = SparkContext(appName = 'testML')
	data = MLUtils.loadLibSVMFile(sc, 'regression_linear.txt')
	myBagging=BaggingRegressor(n_estimators=10,sample_probability=0.8)
	singleRegressor=GradientBoostedTrees()
	models=myBagging.fit(data,singleRegressor,{"categoricalFeaturesInfo":{}, "numIterations":3})
	myBagging.predict(data,models)


