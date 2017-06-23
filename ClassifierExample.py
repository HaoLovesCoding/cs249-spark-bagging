from BaggingClassifier import BaggingClassifier
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.util import MLUtils
from pyspark import SparkContext, RDD, since
from pyspark.mllib.evaluation import MulticlassMetrics

if __name__ == "__main__":

	sc = SparkContext(appName = 'Test_baggingClassifier')
	data = MLUtils.loadLibSVMFile(sc, 'data/classifier_test.txt')
	(trainingData, testData) = data.randomSplit([0.7, 0.3])

	myclassifier=BaggingClassifier(n_estimators=10,sample_probability=0.9,features_num=600)

	singleclassifier=DecisionTree()
	#singleclassifier=GradientBoostedTrees()

	modelC = myclassifier.fit(trainingData,singleclassifier,{'numClasses':2,'categoricalFeaturesInfo':{},'impurity':'gini','maxDepth':5})
	#modelC = myclassifier.fit(trainingData,singleclassifier,{'categoricalFeaturesInfo':{}, 'numIterations':3})
	myclassifier.predict(testData,modelC)
	print('Bagging Test precision: '+str(myclassifier.precision))
	print('Bagging Test recall: '+str(myclassifier.recall))
	
	model=singleclassifier.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={}, impurity='gini', maxDepth=5)
	#model=singleclassifier.trainClassifier(trainingData,categoricalFeaturesInfo={}, numIterations=3)
	predictions = model.predict(testData.map(lambda x: x.features))
	labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
	metrics = MulticlassMetrics(labelsAndPredictions)
	print('Single Estimator Test precision: ' + str(metrics.precision()))
	print('Single Estimator Test precision: ' + str(metrics.recall()))