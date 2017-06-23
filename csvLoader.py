from __future__ import print_function
from pyspark import SparkContext, RDD
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import SparseVector
class csvLoader(object):
	def __init__(self):
		pass

	def __processLine(self,line,label_index,stringMaps,skip_indexes):
		count=0
		features=[]
		label=None
		feature_count=0
		for item in line:
			if count in skip_indexes:
				count+=1
				continue			
			if count in stringMaps:
				if count==label_index:
					label=stringMaps[count][line[count]]
					count+=1
				else:
					features.append( (feature_count, stringMaps[count][line[count]]) )
					count+=1
					feature_count+=1
			else:
				if count==label_index:
					label=line[count]
					count+=1
				else:
					features.append( (feature_count, line[count]) )
					count+=1
					feature_count+=1
		return LabeledPoint(label,SparseVector(feature_count,features))

	def loadCSVwithHeader(self,sc,path,label='',featureToSkip=[]):
		myrdd=sc.textFile(path)
		header=myrdd.first()#type is unicode
		label_index=0
		skip_indexes=[]
		index=0
		for x in header.split(','):
			if x==label:
				label_index=index
			elif x in featureToSkip:
				skip_indexes.append(index)
			index+=1

		myrdd=myrdd.filter(lambda x:x!=header).map(lambda x:x.split(',')) #remove the header 
		first_line=myrdd.take(1)
		string_indexes=[]#This list contains all the string
		count=0
		for x in first_line:
			try:
				x=float(x)
			except Exception:
				string_indexes.append(count)
			count+=1

		#find the map
		stringMaps={}# The map of maps, key is the index should be replaced
		for index in string_indexes:
			stringMaps[index]=(myrdd.map(lambda x:x[index]).distinct().zipWithIndex().collectAsMap())

		myrdd=myrdd.map(lambda x:self.__processLine(x,label_index,stringMaps,skip_indexes))
		return myrdd

if __name__=="__main__":
	sc=SparkContext(appName="test")
	myloader=csvLoader()
	rdd1=myloader.loadCSVwithHeader(sc,'data/csv_loader_test.csv','status',['PPE','MDVP:Fo(Hz)'])
	rdd1.foreach(print)
	print(rdd1.take(1)[0].features[20])
