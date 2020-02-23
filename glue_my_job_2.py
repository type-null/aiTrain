import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from utils.pre_process import Preprocess

my_preprocessor = Preprocess()

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)
## @type: DataSource
## @args: [database = "hwk4-dev", table_name = "dev", transformation_ctx = "datasource0"]
## @return: datasource0
## @inputs: []
datasource0 = glueContext.create_dynamic_frame.from_catalog(database = "hwk4-dev", table_name = "dev", transformation_ctx = "datasource0")
## @type: ApplyMapping
## @args: [mapping = [("sentiment", "long", "sentiment", "long"), ("tweet", "string", "tweet", "string")], transformation_ctx = "applymapping1"]
## @return: applymapping1
## @inputs: [frame = datasource0]
applymapping1 = ApplyMapping.apply(frame = datasource0, mappings = [("sentiment", "long", "sentiment", "long"), ("tweet", "string", "tweet", "string")], transformation_ctx = "applymapping1")
## @type: Map
## @args: [f = map_function, transformation_ctx = "mapping1"]
## @return: mapping1
## @inputs: [frame = applymapping1]
def map_function(dynamicRecord):
    tweet = dynamicRecord['tweet']
    features = my_preprocessor.one_for_all(tweet)
    dynamicRecord['features'] = features
    return dynamicRecord
mapping1 = Map.apply(frame = applymapping1, f = map_function, transformation_ctx = "mapping1")

## @type: DataSink
## @args: [connection_type = "s3", connection_options = {"path": "s3://ai2020/hwk4"}, format = "json", transformation_ctx = "datasink2"]
## @return: datasink2
## @inputs: [frame = mapping1]
datasink2 = glueContext.write_dynamic_frame.from_options(frame = mapping1, connection_type = "s3", connection_options = {"path": "s3://ai2020/hwk4"}, format = "json", transformation_ctx = "datasink2")
job.commit()