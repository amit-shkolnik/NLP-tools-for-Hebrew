"""
Spark NLP NER model for Hebrew 
https://nlp.johnsnowlabs.com/2020/12/09/hebrewner_cc_300d_he.html
"""
print("Start")
try:
    import sys, os, traceback
    #import sparknlp
    #from sparknlp.common import *
    from sparknlp.annotator import  * #
    #import spark
    from pyspark import SparkConf, SparkContext


    #os.environ['HADOOP_HOME'] = "C:/Apps/spark-3.3.0-bin-hadoop3"
    #sys.path.append("C:/Apps/spark-3.3.0-bin-hadoop3/bin")
    #os.environ["hadoop.home.dir"]=( "c:\hadoop\\bin\\\winutil\\\")


    conf = SparkConf().setAppName("PySpark App").setMaster("local[*]")
    sc = SparkContext(conf=conf)

    word_embeddings = WordEmbeddingsModel.pretrained("hebrew_cc_300d", "he")
    word_embeddings.setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")
    ner = NerDLModel.pretrained("hebrewner_cc_300d", "he") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")
    ner_converter = NerConverter().setInputCols(["sentence", "token", "ner"]).setOutputCol("ner_chunk")
    nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, ner, ner_converter])
    light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
    annotations = light_pipeline.fullAnnotate("""
    ב- 25 לאוגוסט עצר השב"כ את מוחמד אבו-ג'וייד , אזרח ירדני , 
    שגויס לארגון הפת"ח והופעל על ידי חיזבאללה. 
    אבו-ג'וייד התכוון להקים חוליות טרור בגדה ובקרב ערביי ישראל , לבצע פיגוע ברכבת ישראל בנהריה , 
    לפגוע במטרות ישראליות בירדן ולחטוף חיילים כדי לשחרר אסירים ביטחוניים
    """)

except Exception as err:
    print( sys.exc_info()[0])            
    print( traceback.format_exc())          
    print( str(err))


 


# import sys, traceback
# import nlu


# print("Start")
# try:
#     result =nlu.load("he.ner").predict("""ח והופעל על ידי חיזבאללה. 
#     אבו-ג'וייד התכוון להקים חוליות טרור בגדה ובקרב ערביי ישראל
#     , לבצע פיגוע ברכבת ישראל בנהריה ,
#     לפגוע במטרות ישראליות בירדן ולחטוף חיילים כדי לשחרר אסירים ביטחוניים.""")
# except Exception as err:
#     print( sys.exc_info()[0])            
#     print( traceback.format_exc())          
#     print( str(err))
# 
# 
#  