#!/usr/bin/env python
# -*- coding: utf-8 -*-



# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
# TODO: you may need to add imports here


def main(spark, data_file):
    '''Main routine for supervised evaluation

    Parameters
    ----------
    spark : SparkSession object

    model_file : string, path to store the serialized model file

    data_file : string, path to the parquet file to load
    '''

    ###
    # TODO: YOUR CODE GOES HERE
    ###
    df = spark.read.parquet(data_file)
    #df = df.sample(0.0001)
    indexer_id = StringIndexer(inputCol="user_id", outputCol="userindex").setHandleInvalid("skip")
    df = indexer_id.fit(df).transform(df)
    indexer_item = StringIndexer(inputCol="track_id", outputCol="itemindex").setHandleInvalid("skip")
    df = indexer_item.fit(df).transform(df)
    df.select("userindex", "count", "itemindex").write.parquet("training_data.parquet")
    

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('string_conversion').getOrCreate()

    # And the location to store the trained model
    data_file = sys.argv[1]

    # Call our main routine
    main(spark, data_file)