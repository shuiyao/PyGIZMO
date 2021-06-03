'''
Utilities that provide Apache Spark support.
'''

from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.types import *
import pyspark.sql.functions as sF

spark = SparkSession.builder.appName("PyGIZMO").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel('WARN')

sparkTypeCast = {
    "int64":LongType(),
    "int32":IntegerType(),
    "float32":FloatType(),
    "float64":DoubleType()
}

def spark_read_schema(schema_json):
    '''
    Create a Spark CSV schema from JSON type schema definition.

    Parameters
    ----------
    schema_json: dict.
        Must have a 'columns' field that gives the order of fields
        Must have a 'dtypes' field that maps fieldname to numpy dtype
        Example: {'columns':['col1', 'col2'], 
                  'dtypes':{'col1':int32, 'col2':int64}}

    Returns
    -------
    schemaSpark: Spark schema.
    '''

    cols = schema_json['columns']
    dtypes = schema_json['dtypes']
    fields = []
    for col in cols:
        dtypeString = dtypes[col]
        fields.append(StructField(col, sparkTypeCast[dtypeString], True))
    schemaSpark = StructType(fields)
    return schemaSpark

