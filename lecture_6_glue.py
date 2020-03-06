import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame

from utils.pre_process import Preprocess

from pyspark.sql import Row
from pyspark.ml.feature import CountVectorizer
from pyspark.mllib.clustering import LDA
from pyspark.mllib.linalg import Vectors

glueContext = GlueContext(sc)

my_text_processor = TextProcessor()
my_pre_processor = PreProcess()

english_word_regex = r'^[a-zA-Z0-9#!?.]*$'

def get_tokens(text):
    tokens = my_text_processor.tokenize_text(my_text_processor.clean_text(text.lower))
    return_tokens = []
    for token in tokens:
        if my_pre_processor.embedding.embedding_dictionary.get(token, 0) > 200 and re.match(english_word_regex, token):
            return_tokens.append(token)
    return return_tokens

datasource0 = glueContext.create_dynamic_frame.from_catalog(databse="twitter-data", table_name="twitter_state_selected", transformation_ctx='datasource0')

applymapping1 = ApplyMapping.apply(frame=datasource0, mappings=[("id", "long", "id", "long"),("text", "string", "text", "string")], transformation_ctx='applymapping1')

applymapping1_rdd = applymapping1.roDF().rdd

tokenized = applymapping1_rdd.map(lambda row: Row(id=row["id"], words=get_tokens(row["text"])))


doc_DF = spark.createDataFrame(tokenized)
vectorizer = CountVectorizer(inputCol="words", outputCol="vectors")
fitted_vectorizer = vectorizer.fit(doc_DF)
word_vectors = fitted_vectorizer.transform(doc_DF)

corpus = word_vectors.select("id", "vectors").rdd.map(lambda row: [row["id"], Vectors.fromML(row["vectors"])]).cache()

# Cluster the documents into three topics using LDA
ldaModel = LDA.train(corpus, k=20, maxIterations=50, optimizer="online")
topics = ldaModel.topicsMatrix()
vocabArray = fitted_vectorizer.vocabulary

wordNumbers = 20
topicIndices = sc.parallelize(ldaModel.describeTopics(maxTermPerTopic=wordNumbers))


def topic_render(topic):
    terms = topic[0]
    result = []
    for i in range(wordNumbers):
        term = vocabArray[terms[i]]
        result.append(term)
    return result


topics_final = topicIndices.map(lambda topic: topic_render(topic)).collect()

for topic in range(len(topics_final)):
    print("Topic" + str(topic) + ":")
    for term in topics_final[topic]:
        print(term)
    print("\n")
    