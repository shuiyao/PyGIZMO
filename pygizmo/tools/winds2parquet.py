import numpy as np
import pandas as pd
import warnings
import os
import time
import glob
# from . import utils
import utils

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.window import Window
import pyspark.sql.functions as sF
from myinit import *

defaultSchema = utils.load_default_schema()

sparkDtypeDict = {
    "int64":LongType(),
    "int32":IntegerType(),
    "float32":FloatType(),
    "float64":DoubleType()
}

def spark_read_schema(record='phews'):
    schema = utils.load_default_schema()
    fields = []
    for col in schema[record]['columns']:
        dtypeString = schema[record]['dtypes'][col]
        fields.append(StructField(col, sparkDtypeDict[dtypeString], True))
    schemaSpark = StructType(fields)
    return schemaSpark

def spark_read_phews(schema):
    df = spark.read.options(delimiter=' ').csv(path_phews, schema)
    df = df.select('atime', 'PhEWKey', 'Mass', 'M_c', 'Z_c', 'dr', 'dv')
    return df

def spark_read_initwinds(schema):
    df = spark.read.options(delimiter=' ').csv(path_initwinds, schema)
    df = df.select('atime', 'PhEWKey', 'Mass', 'MinPotID', 'PID')\
           .withColumnRenamed('Mass', 'Minit')
    return df

spark = SparkSession.builder.appName('Winds2Parquet').getOrCreate()
sc = spark.sparkContext
sc.setLogLevel('WARN')

path_model = os.path.join("/home/shuiyao_umass_edu/proj/spark/data/")
path_phews = os.path.join(path_model, "WINDS/phews.*")
path_initwinds = os.path.join(path_model, "WINDS/initwinds.*")
AMIN, AMAX = 0.50, 0.505

tbeg = time.time()

# ---------------- 1. Filter out winds from initwinds ----------------
schema_initwinds = spark_read_schema('initwinds')
dfinit = spark_read_initwinds(schema_initwinds)
keys = dfinit.where("atime >= {} and atime < {}".format(AMIN, AMAX)).withColumnRenamed("atime", "ainit")

# ---------------- 2. Load phews data ----------------
schema_phews = spark_read_schema('phews')
df = spark_read_phews(schema_phews)
# Filter out PhEWs that are launched from between AMIN and AMAX
df = df.join(keys, "PhEWKey", 'inner')

# ---------------- 4. Group and Order ----------------
# Window Method
w = Window.partitionBy(df.PhEWKey).orderBy(df.atime)
df = df.withColumn("order", sF.rank().over(w))

# ---------------- 5. Create Nested Structure ----------------
# Schema:
# {UID}, {Init}, {Tracks}
# UID (Unique Identifiers): PhEWKey, PID
# {Init}: ainit, Z_c, MinPotId (-> galId)
# {Tracks}: ArrayType, atime, Mass, M_c, dr, dv

# df2 = df.groupBy("PhEWKey").agg(sF.collect_list(sF.struct(['atime','Mass','M_c','dr','dv'])).alias("track"))
df = df.groupBy("PhEWKey").agg(
    sF.first('PID').alias("PID"),
    sF.first('ainit').alias("ainit"),
    sF.first('Minit').alias("Minit"),    
    sF.first('Z_c').alias("Z_c"),
    sF.first('MinPotID').alias("MinPotID"),
    sF.struct([
    sF.collect_list('atime').alias('atime'),
    sF.collect_list('Mass').alias('Mass'),
    sF.collect_list('M_c').alias('M_c'),
    sF.collect_list('dr').alias('dr'),
    sF.collect_list('dv').alias('dv')]
    ).alias('track'),
    sF.max('order').alias("records")
    )

# ---------------- 6. Output ----------------
# df.repartition(128, df.PhEWKey).write.format("csv").mode("overwrite").option('header', True).save("sorted_phews_z1.csv")

fout = os.path.join(path_model, "phews.parquet")
df.repartition(128, df.PhEWKey).write.format("parquet").mode("overwrite").save(fout)

print("Cost: {}".format(time.time() - tbeg))


