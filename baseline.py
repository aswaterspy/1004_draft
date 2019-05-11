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
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.recommendation import ALS
from pyspark.sql.function import expr
# TODO: you may need to add imports here


def main(spark, train_file, test_file, model_file):
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

    df_train = spark.read.parquet(train_file).sample(False, 0.1)
    df_validation = spark.read.parquet(test_file)
    
    reg_list = [0.1]
    rank_list = [10]
    alhpa_list = [1]
    para_pool = []
    eva_list = []
    
    for reg in reg_list:
        for rank in rank_list:
            for alpha in alpha_list:
                als_model = ALS(maxIter=5, userCol="userindex", itemCol="itemindex", ratingCol="count", 
                                regParam=reg, rank=rank, alpha=alpha)
                model = als_model.fit(df_train)
                prediction = model.recommendForAllUsers(500).select('userindex', 'recommendations.itemindex')
                para_pool.append([reg, rank, alpha])
                testing_df = df_validation.groupBy('userindex').agg(expr('collect_list(itemindex) as item_list'))
                predictionAndLabels = prediction.join(testing_df, 'userindex')
                pred_df = predictionAndLabels.select(['itemindex','item_list']).rdd.map(list)

                metrics = RankingMetrics(pred_df)
                eva = metrics.meanAveragePrecision
                eva_list.append(eva)

                print('Complete Run with regParam: ', reg, ' rank: ', rank, ' alpha: ', alpha, '. MAP is ', eva)
    
    reg, rank, alpha = para_pool[eva.index(np.max(eva))]
    best_model = ALS(maxIter=5, userCol="userindex", itemCol="itemindex", ratingCol="count", 
                                regParam=reg, rank=rank, alpha=alpha)
    best_model.write().overwrite().save(model_file)      
    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('remommendation_train').getOrCreate()

    # Get the filename from the command line
    train_file = sys.argv[1]

    # And the location to store the trained model
    test_file = sys.argv[2]
    
    model_file = sys.argv[3]

    # Call our main routine
    main(spark, train_file, test_file, model_file)