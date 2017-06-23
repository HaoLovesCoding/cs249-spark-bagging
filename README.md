##This project icnludes three API classes we made for Apache-Spark MLlib
###class BaggingClassifer
**Parameters:**
*n_estimators* - The number of estimators in the bagging ensemble
*sample_probability* - The probability of a data point being sampled
*features_num* - The number of features will be chosen for each estimators
*precision* - The precision for the estimation of test dataset
*recall* - The recall for the estimation of test dataset
*F1score* - The F1 score for the estimation of test dataset

**Methods:**
1 ###*def __init__(self,n_estimators=3,sample_probability=0.9,features_num=100)*
2 ###*def fit(self, data, classifier, argument)*
  *data* - The RDD of LabeledPoint, training dataset
  *classifier* - The single classifer making up the ensemble. Example: decision tree is the single classifier for random forest.
  *argument* - Dictionary format. The argument passed to the classifier.train() or classifier.trainClassifier(). 
  Example: {'numClasses':2,'categoricalFeaturesInfo':{},'impurity':'gini','maxDepth':5} is identical to
  trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={}, impurity='gini', maxDepth=5)
  *This method will return the trained models ensemble*
3.###*def predict(self,data,models)*
  *data* - The RDD of LabeledPoint, test dataset
  *models* - The trained model ensembles returned by fit() method
