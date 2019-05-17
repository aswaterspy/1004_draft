#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
ALS Model Testing
'''

# We need sys to get the command line arguments
import sys
import time

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import expr
from pyspark.sql import functions as F
# TODO: you may need to add imports here

def main(spark, model_file, train_data_file, test_data_file):

    time_a = time.time()
    start = time_a

    training_data = spark.read.parquet(train_data_file)
    indexer_id = StringIndexer(inputCol="user_id", outputCol="userindex").setHandleInvalid("skip")
    indexer_id_model = indexer_id.fit(training_data)
    indexer_item = StringIndexer(inputCol="track_id", outputCol="itemindex").setHandleInvalid("skip")
    indexer_item_model = indexer_item.fit(training_data)

    testing_data = spark.read.parquet(test_data_file)
    testing_data = indexer_id_model.transform(testing_data)
    testing_data = indexer_item_model.transform(testing_data)

    testing_data = testing_data.select('userindex','itemindex','count')

    print('Finished Indexing!')
    time_b = time.time()
    print(time_b - time_a)
    time_a = time_b

    model = ALSModel.load(model_file)
    prediction = model.recommendForAllUsers(500).select('userindex', 'recommendations.itemindex')
    print('Finished Prediction DF!')

    testing_df = testing_data.groupBy('userindex').agg(expr('collect_list(itemindex) as item_list'))
    print('Finished Label DF!')

    predictionAndLabels = prediction.join(testing_df, 'userindex')
    print('Joined Prediction and Labels!')
    time_b = time.time()
    print(time_b - time_a)
    time_a = time_b

    pred_df = predictionAndLabels.select(['itemindex','item_list']).rdd.map(list)
    metrics = RankingMetrics(pred_df)

    print('Ranking Metrics Calculated!')
    time_b = time.time()
    print(time_b - time_a)
    time_a = time_b

    eva = metrics.meanAveragePrecision
    print("Model on Testing Data gives MAP= ", eva)

    print('Process Finished!')
    print(time.time() - start)

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('remommendation_test').config("spark.sql.broadcastTimeout","36000").getOrCreate()

    # model output
    model_file = sys.argv[1]

    # train data input
    train_data_file = sys.argv[2]

    # test data input
    test_data_file = sys.argv[3]

    # Call our main routine
    main(spark, model_file, training_data_file, test_data_file)