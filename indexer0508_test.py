#!/usr/bin/env python
# -*- coding: utf-8 -*-



# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
# TODO: you may need to add imports here


def main(spark):
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
    df_index = spark.read.parquet('cf_train_indexed.parquet')
    #df = df.sample(0.0001)
    df_index.show()
    

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('string_conversion').getOrCreate()

    # Call our main routine
    main(spark)