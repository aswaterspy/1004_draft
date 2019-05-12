#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
ALS Model Train Baseline
$ spark-submit baseline.py hdfs:/user/bm106/pub/project/cf_train.parquet hdfs:/path/to/save/model
'''


# We need sys to get the command line arguments
import sys
import time

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import expr
# TODO: you may need to add imports here


def main(spark, train_data_file, test_data_file, model_file):

    time_a = time.time()
    start = time_a

    training_data = spark.read.parquet(train_data_file)
    training_data.createOrReplaceTempView('training_data')

    #make sure the partial history 110k users are in sample
    partial_history_rows = spark.sql("SELECT * FROM training_data WHERE user_id IN (SELECT user_id FROM training_data GROUP BY user_id ORDER BY max(__index_level_0__) DESC LIMIT 110000)")
    
    #full_history_rows = training_data.subtract(partial_history_rows)
    #full_history_rows = full_history_rows.sample(False, 0.02)
    training_data = partial_history_rows.sample(0.01)#.union(full_history_rows) 
    print('Finished Sampling!')
    time_b = time.time()
    print(time_a - time_b)
    time_a = time_b

    indexer_id = StringIndexer(inputCol="user_id", outputCol="userindex").setHandleInvalid("skip")
    indexer_id_model = indexer_id.fit(training_data)
    indexer_item = StringIndexer(inputCol="track_id", outputCol="itemindex").setHandleInvalid("skip")
    indexer_item_model = indexer_item.fit(training_data)

    training_data = indexer_id_model.transform(training_data)
    training_data = indexer_item_model.transform(training_data)
    training_data = training_data.select('userindex','itemindex','count')

    testing_data = spark.read.parquet(test_data_file)
    testing_data = indexer_id_model.transform(testing_data)
    testing_data = indexer_item_model.transform(testing_data)
    testing_data = testing_data.select('userindex','itemindex','count')
    print('Finished Indexing!')
    time_b = time.time()
    print(time_a - time_b)
    time_a = time_b

    result_dict = {}
    rank_list = [10]
    reg_param_list = [0.1]
    alpha_list = [1]

    for rank in rank_list:
        for reg_param in reg_param_list:
            for alpha in alpha_list:
                current_key = (rank,reg_param,alpha)
                als = ALS(maxIter=5, userCol="userindex", itemCol="itemindex", ratingCol="count", rank=rank, regParam=reg_param, alpha=alpha)
                model = als.fit(training_data)
                print('Finished Modeling with Param:', current_key)
                time_b = time.time()
                print(time_a - time_b)
                time_a = time_b

                prediction = model.recommendForAllUsers(500).select('userindex', 'recommendations.itemindex')
                
                testing_df = testing_data.groupBy('userindex').agg(expr('collect_list(itemindex) as item_list'))
                predictionAndLabels = prediction.join(testing_df, 'userindex')
                print('Joined Prediction and Labels!')
                time_b = time.time()
                print(time_a - time_b)
                time_a = time_b
                pred_df = predictionAndLabels.select(['itemindex','item_list']).rdd.map(list)

                metrics = RankingMetrics(pred_df)
                print('Ranking Metrics Calculated!')
                time_b = time.time()
                print(time_a - time_b)
                time_a = time_b
                eva = metrics.meanAveragePrecision

                result_dict[current_key] = eva

                print(current_key,"parameter combination has been trained! MAP= ", eva)
                time_b = time.time()
                print(time_a - time_b)
                time_a = time_b

    best_model_param = max(result_dict, key=result_dict.get)
    als = ALS(maxIter=5, userCol="userindex", itemCol="itemindex", ratingCol="count", rank=best_model_param[0], regParam=best_model_param[1], alpha=best_model_param[2])
    als.fit(training_data).write().overwrite().save(model_file)

    print('Process Finished!')
    print(time.time() - start)
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('remommendation_test').getOrCreate()

    # train data input
    train_data_file = sys.argv[1]

    # test data input
    test_data_file = sys.argv[2]

    # model output
    model_file = sys.argv[3]


    # Call our main routine
    main(spark, train_data_file, test_data_file, model_file)