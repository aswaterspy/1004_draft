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

    # Use Validation and Test user_id to filter Train data, to get the 110k mandatory users
    # Stored here hdfs:/user/dz584/cf_train_sample.parquet
    """
    training_data = spark.read.parquet('hdfs:/user/bm106/pub/project/cf_train.parquet')
    validation_data = spark.read.parquet('hdfs:/user/bm106/pub/project/cf_validation.parquet')
    testing_data = spark.read.parquet('hdfs:/user/bm106/pub/project/cf_test.parquet')

    validandtest_userid = validation_data.union(testing_data).select('user_id').distinct()
    validandtest_userid.createOrReplaceTempView('validandtest_userid')

    training_data.createOrReplaceTempView('training_data')
    training_data = spark.sql("SELECT * FROM training_data WHERE user_id IN (SELECT user_id FROM validandtest_userid GROUP BY user_id)")
    training_data.write.parquet("cf_train_sample.parquet")
    """

    training_data = spark.read.parquet(train_data_file)
    indexer_id = StringIndexer(inputCol="user_id", outputCol="userindex").setHandleInvalid("skip")
    indexer_id_model = indexer_id.fit(training_data)
    indexer_item = StringIndexer(inputCol="track_id", outputCol="itemindex").setHandleInvalid("skip")
    indexer_item_model = indexer_item.fit(training_data)

    training_data = indexer_id_model.transform(training_data)
    training_data = indexer_item_model.transform(training_data)

    testing_data = spark.read.parquet(test_data_file)
    testing_data = indexer_id_model.transform(testing_data)
    testing_data = indexer_item_model.transform(testing_data)

    training_data = training_data.select('userindex','itemindex','count')
    testing_data = testing_data.select('userindex','itemindex','count')

    print('Finished Indexing!')
    time_b = time.time()
    print(time_b - time_a)
    time_a = time_b

    result_dict = {}
    rank_list = [1000,1500]#[10,20,30,50]
    reg_param_list = [0.7,0.9]#[0.1,0.5]
    alpha_list = [1]#[1,1.5]

    for rank in rank_list:
        for reg_param in reg_param_list:
            for alpha in alpha_list:

                current_key = (rank,reg_param,alpha)
                als = ALS(maxIter=5, userCol="userindex", itemCol="itemindex", ratingCol="count", rank=rank, regParam=reg_param, alpha=alpha)
                model = als.fit(training_data)

                print('Finished Modeling with Param:', current_key)
                time_b = time.time()
                print(time_b - time_a)
                time_a = time_b

                prediction = model.recommendForAllUsers(500).select('userindex', 'recommendations.itemindex')
                print('Finished Prediction DF!')

                testing_df = testing_data.groupBy('userindex').agg(expr('collect_list(itemindex) as item_list'))
                print('Finished Label DF!')

                predictionAndLabels = prediction.join(testing_df, 'userindex')
                predandlabel_name = 'rk'+str(rank)+'reg'+str(reg_param)+'a'+str(alpha)
                predandlabel_name = predandlabel_name.replace(".","")+'.parquet'
                predictionAndLabels.write.parquet(predandlabel_name)

                print('Joined Prediction and Labels!')
                time_b = time.time()
                print(time_b - time_a)
                time_a = time_b

    #             pred_df = predictionAndLabels.select(['itemindex','item_list']).rdd.map(list)

    #             metrics = RankingMetrics(pred_df)

    #             print('Ranking Metrics Calculated!')
    #             time_b = time.time()
    #             print(time_b - time_a)
    #             time_a = time_b

    #             eva = metrics.meanAveragePrecision
    #             result_dict[current_key] = eva

    #             print(current_key,"parameter combination has been trained! MAP= ", eva)
    #             time_b = time.time()
    #             print(time_b - time_a)
    #             time_a = time_b

    # best_model_param = max(result_dict, key=result_dict.get)
    # als = ALS(maxIter=5, userCol="userindex", itemCol="itemindex", ratingCol="count", rank=best_model_param[0], regParam=best_model_param[1], alpha=best_model_param[2])
    # als.fit(training_data).write().overwrite().save(model_file)

    print('Process Finished!')
    print(time.time() - start)
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('remommendation_test').config("spark.sql.broadcastTimeout","36000").getOrCreate()

    # train data input
    train_data_file = sys.argv[1]

    # test data input
    test_data_file = sys.argv[2]

    # model output
    model_file = sys.argv[3]


    # Call our main routine
    main(spark, train_data_file, test_data_file, model_file)