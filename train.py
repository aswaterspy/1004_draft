#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Part 2: supervised model training

Usage:

    $ spark-submit recommendation_train.py hdfs:/user/bm106/pub/project/cf_train.parquet hdfs:/path/to/save/model

'''


# We need sys to get the command line arguments
import sys
import numpy as np
# And pyspark.sql to get the spark session
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
# TODO: you may need to add imports here


def main(spark, data_file, model_file):
    '''Main routine for supervised training

    Parameters
    ----------
    spark : SparkSession object

    data_file : string, path to the parquet file to load

    model_file : string, path to store the serialized model file
    '''

    ###
    # TODO: YOUR CODE GOES HERE
    ###

    training_data = spark.read.parquet(data_file).sample(False, 0.001)
    indexer_id = StringIndexer(inputCol="user_id", outputCol="userindex").setHandleInvalid("skip")
    indexer_item = StringIndexer(inputCol="track_id", outputCol="itemindex").setHandleInvalid("skip")
    
    als = ALS(maxIter=5, userCol="userindex", itemCol="itemindex", ratingCol="count")
    pipeline = Pipeline(stages=[indexer_id, indexer_item, als])
    
#     valid_data = spark.read.parquet(test_file).sample(False, 0.02)
#     pipeline2 = Pipeline(stages=[indexer_id, indexer_item])
#     processed_test = pipeline2.fit(valid_data)
    
    
    paramGrid = ParamGridBuilder().addGrid(als.regParam, [0.5]).addGrid(als.rank, [10]).addGrid(als.alpha, [0.4]).build()


    crossval = CrossValidator(estimator=pipeline,estimatorParamMaps=paramGrid,
                              evaluator=RegressionEvaluator(metricName="rmse", labelCol="count",predictionCol="prediction"),
                              numFolds=5)
    
    
    model = crossval.fit(training_data)
    pip_model = model.bestModel
    pip_model.write().overwrite().save(model_file)

    print('Process Finished')
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('remommendation_train').getOrCreate()

    # Get the filename from the command line
    data_file = sys.argv[1]

    # And the location to store the trained model
    model_file = sys.argv[2]

    # Call our main routine
    main(spark, data_file, model_file)