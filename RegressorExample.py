from BaggingRegressor import BaggingRegressor
from pyspark import SparkContext, RDD, since
from pyspark.mllib.util import MLUtils
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
if __name__=="__main__":
	sc = SparkContext(appName = 'testML')
	data = MLUtils.loadLibSVMFile(sc, 'data/regression_linear.txt')
	(trainingData, testData) = data.randomSplit([0.7, 0.3])

	myBagging=BaggingRegressor(n_estimators=10,sample_probability=0.8)
	singleRegressor=GradientBoostedTrees()

	models=myBagging.fit(trainingData,singleRegressor,{"categoricalFeaturesInfo":{}, "numIterations":3})
	myBagging.predict(testData,models)
	print('Bagging Test Mean Squared Error = '+str(myBagging.MSE))


	model = GradientBoostedTrees.trainRegressor(trainingData,
                                            categoricalFeaturesInfo={}, numIterations=3)
	# Evaluate model on test instances and compute test error
	predictions = model.predict(testData.map(lambda x: x.features))
	labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
	testMSE = labelsAndPredictions.map(lambda (v, p): (v - p) * (v - p)).sum() /\
	    float(testData.count())
	print('Test Mean Squared Error = ' + str(testMSE))