from __future__ import print_function
from pyspark import SparkContext, RDD, since
from csvLoader import csvLoader
if __name__=="__main__":
	sc=SparkContext(appName="test")
	myloader=csvLoader()

	#choose status column as the target column. Ignore 'PPE' and 'MDVP:Fo(Hz)'
	rdd1=myloader.loadCSVwithHeader(sc,'data/csv_loader_test.csv','status',['PPE','MDVP:Fo(Hz)'])
	rdd1.foreach(print)