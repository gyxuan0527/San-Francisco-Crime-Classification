#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pyspark.sql import SparkSession
from pyspark import SparkContext
import pandas as pd
import numpy as np
import io
spark = SparkSession.builder.master("local").getOrCreate()
sc = SparkContext.getOrCreate()


# In[ ]:


train = pd.read_csv('train.csv')
train.pop('Descript')
train.pop('Resolution')
train.pop('Address')
train["Dates"] = train["Dates"].fillna(0)
train["Category"] = train["Category"].fillna(0)
train["DayOfWeek"] = train["DayOfWeek"].fillna(0)
train["PdDistrict"] = train["PdDistrict"].fillna(0)
train["X"] = train["X"].fillna(0)
train["Y"] = train["Y"].fillna(0)
train=pd.get_dummies(data=train,columns=["DayOfWeek"])
train=pd.get_dummies(data=train,columns=["PdDistrict"])
train['Dates']=train['Dates'].str.split(" ",expand=True)[1]
train['Dates']=train['Dates'].str.split(":",expand=True)[0]
train=pd.get_dummies(data=train,columns=["Dates"])
train.head(5)


# In[ ]:


from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
from pyspark.ml.classification import DecisionTreeClassifier
import pyspark.sql 
from pyspark.sql import SQLContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
#(trainingData, testData) = train.randomSplit([0.75, 0.25])
df = spark.createDataFrame(train)
df.show()
vector_assembler = VectorAssembler(inputCols=["X", "Y","DayOfWeek_Friday","DayOfWeek_Monday","DayOfWeek_Saturday","DayOfWeek_Sunday","DayOfWeek_Thursday","DayOfWeek_Tuesday","DayOfWeek_Wednesday","PdDistrict_BAYVIEW", "PdDistrict_CENTRAL","PdDistrict_INGLESIDE","PdDistrict_NORTHERN","PdDistrict_PARK","PdDistrict_RICHMOND","PdDistrict_TARAVAL","PdDistrict_TENDERLOIN","PdDistrict_BAYVIEW","PdDistrict_CENTRAL","PdDistrict_INGLESIDE","PdDistrict_MISSION","PdDistrict_NORTHERN","PdDistrict_PARK","PdDistrict_RICHMOND","PdDistrict_SOUTHERN","PdDistrict_TARAVAL","PdDistrict_TENDERLOIN","Dates_00","Dates_01","Dates_02","Dates_03","Dates_04","Dates_05","Dates_06","Dates_07","Dates_08","Dates_09","Dates_10","Dates_11","Dates_12","Dates_13","Dates_14","Dates_15","Dates_16","Dates_17","Dates_18","Dates_19","Dates_20","Dates_21","Dates_22","Dates_23"],outputCol="features")
df_temp = vector_assembler.transform(df)
df_temp.show(3)
df = df_temp.drop("X", "Y","DayOfWeek_Friday","DayOfWeek_Monday","DayOfWeek_Saturday","DayOfWeek_Sunday","DayOfWeek_Thursday","DayOfWeek_Tuesday","DayOfWeek_Wednesday","PdDistrict_BAYVIEW", "PdDistrict_CENTRAL","PdDistrict_INGLESIDE","PdDistrict_NORTHERN","PdDistrict_PARK","PdDistrict_RICHMOND","PdDistrict_TARAVAL","PdDistrict_TENDERLOIN","PdDistrict_BAYVIEW","PdDistrict_CENTRAL","PdDistrict_INGLESIDE","PdDistrict_MISSION","PdDistrict_NORTHERN","PdDistrict_PARK","PdDistrict_RICHMOND","PdDistrict_SOUTHERN","PdDistrict_TARAVAL","PdDistrict_TENDERLOIN","Dates_00","Dates_01","Dates_02","Dates_03","Dates_04","Dates_05","Dates_06","Dates_07","Dates_08","Dates_09","Dates_10","Dates_11","Dates_12","Dates_13","Dates_14","Dates_15","Dates_16","Dates_17","Dates_18","Dates_19","Dates_20","Dates_21","Dates_22","Dates_23")
df.show(3)
l_indexer = StringIndexer(inputCol="Category", outputCol="labelIndex")
df = l_indexer.fit(df).transform(df)
df.show(3)
(training,testing) = df.randomSplit([0.75,0.25])
dt = DecisionTreeClassifier(labelCol="labelIndex", featuresCol="features")
model = dt.fit(training)
predictions = model.transform(testing)
predictions.select("prediction", "labelIndex").show(5)
evaluator = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction",metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g " % accuracy)
evaluator1 = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction",metricName="precisionByLabel")
precision = evaluator1.evaluate(predictions)
print("Precision = %g " % precision)
evaluator2 = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction",metricName="recallByLabel")
recall = evaluator2.evaluate(predictions)
print("Recall = %g " % recall)

